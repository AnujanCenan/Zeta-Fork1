"""
zeta_localize.py

Extends ZETA inference for a single ECG to produce:
  - Per-observation similarity scores (from the unimodal dot-product path)
  - Temporal localization intervals (from the cross-attention fusion path)

Output example:
  Condition: 1AVB  (final score: 0.73)
  Observation                              Score  Localization          Strength
  [P] pr interval >200ms                  0.82   120ms – 340ms  II     ████████  highlight strongly
  [P] normal p-wave morphology            0.71    80ms – 120ms  II     ██████    highlight moderately
  [N] pr interval <200ms                  0.18   diffuse               ░░        suppress

Usage:
  python zeta_localize.py \
      --ptbxl_path /path/to/ptbxl/records500/00000/00001_hr \
      --condition 1AVB \
      --checkpoint checkpoints/best.pt \
      --config configs/config.json \
      --observations configs/observations.json

  --ptbxl_path should be the value from the 'filename_hr' column of the
  PTB-XL CSV (without extension), relative to --ptbxl_root.

  Alternatively, pass --ptbxl_root and --ptbxl_record_id to look up a
  specific record by its ecg_id from ptbxl_database.csv.
"""

import argparse
import ast
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from pathlib import Path
from types import SimpleNamespace
from transformers import T5TokenizerFast

# ── repo imports (run from Zeta/ directory) ──────────────────────────────────
from models.cmelt import M3AEModel

# ── constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 500          # Hz
ECG_LENGTH       = 5000         # samples (10 s)
CONV_LAYERS      = [(256,2,2)] * 4  # must match config conv_feature_layers
NUM_ECG_TOKENS   = 312          # computed from CONV_LAYERS above
SAMPLES_PER_TOKEN = ECG_LENGTH / NUM_ECG_TOKENS   # ≈ 16.03 samples = ~32 ms
MS_PER_TOKEN     = (SAMPLES_PER_TOKEN / SAMPLE_RATE) * 1000  # ≈ 32.05 ms

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

SOFTMAX_TEMP = 0.5   # matches ZETA paper / get_diseases_probs()

# attention heatmap thresholds
DIFFUSE_ENTROPY_THRESHOLD = 0.85   # fraction of max entropy → call it "diffuse"
MIN_INTERVAL_TOKENS       = 3      # smallest meaningful interval (~96 ms)
PEAK_PERCENTILE           = 75     # tokens above this percentile form the interval


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> M3AEModel:
    with open(config_path) as f:
        cfg = SimpleNamespace(**json.load(f)["model"])

    model = M3AEModel(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"]
    # remove key that is only present during pre-training
    state.pop("ecg_encoder.mask_emb", None)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ECG encoding  (unimodal path — same as main.py)
# ─────────────────────────────────────────────────────────────────────────────

def encode_ecg(model: M3AEModel, ecg: np.ndarray, device: torch.device):
    """
    Returns
    -------
    ecg_vec   : (768,) normalised pooled vector  — used for similarity scoring
    ecg_seq   : (1, Lx+1, 768) full token sequence with CLS prepended
                — used for cross-attention localisation
    ecg_mask  : attention mask for ecg_seq, shape (1, Lx+1)

    Pooling strategy: mean over all sequence tokens (excluding CLS) after
    multi_modal_ecg_proj. Diagnostics show this gives the best separation
    between conditions. The trained unimodal_ecg_pooler saturates due to
    large proj magnitudes driving Tanh to ±1.
    """
    x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(device)  # (1,5000,12)
    x = x.permute(0, 2, 1)                                               # (1,12,5000)

    with torch.no_grad():
        feats, padding_mask = model.ecg_encoder.get_embeddings(x, padding_mask=None)
        # feats: (1, 312, 768) — no CLS yet

        # prepend CLS for the transformer (needed for get_output)
        cls_emb = model.class_embedding.repeat(1, 1, 1)          # (1,1,768)
        feats_with_cls = torch.cat([cls_emb, feats], dim=1)      # (1,313,768)
        feats_out = model.ecg_encoder.get_output(feats_with_cls, padding_mask)

        # project to shared space
        proj = model.multi_modal_ecg_proj(feats_out)             # (1,313,768)

        # mean pool over sequence tokens only (skip CLS at index 0)
        ecg_vec = proj[:, 1:, :].mean(dim=1).squeeze(0)          # (768,)
        ecg_vec = F.normalize(ecg_vec, dim=0)

    # build attention mask (all ones — no padding for fixed-length ECG)
    Lx_plus1 = feats_out.size(1)
    ecg_mask = torch.ones((1, Lx_plus1), dtype=torch.long, device=device)

    return ecg_vec, proj.detach(), ecg_mask


# ─────────────────────────────────────────────────────────────────────────────
# Text encoding  (unimodal path)
# ─────────────────────────────────────────────────────────────────────────────

def encode_text(model: M3AEModel,
                tokenizer: T5TokenizerFast,
                text: str,
                device: torch.device):
    """
    Returns
    -------
    text_vec  : (768,) normalised pooled vector
    text_seq  : (1, Lt, 768) full token sequence (post-proj, for cross-attention)
    text_mask : attention mask (1, Lt)

    Pooling strategy: mean over all tokens from the raw Flan-T5 output,
    BEFORE multi_modal_language_proj. Diagnostics show this gives the best
    separation between positive and negative observations (sep=0.0174),
    outperforming post-proj and CLS-based strategies.

    text_seq is still returned post-proj so the cross-attention fusion
    module operates in the shared 768-d space as intended.
    """
    enc = tokenizer(text, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        raw  = model.language_encoder(input_ids=enc["input_ids"])[0]  # (1, Lt, 768)
        proj = model.multi_modal_language_proj(raw)                    # (1, Lt, 768)

        # mean pool over all tokens from raw Flan-T5 output (pre-proj)
        text_vec = raw.mean(dim=1).squeeze(0)                          # (768,)
        text_vec = F.normalize(text_vec, dim=0)

    return text_vec, proj.detach(), enc["attention_mask"]


# ─────────────────────────────────────────────────────────────────────────────
# Similarity scoring  (ZETA pairwise softmax — matches get_diseases_probs)
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_score(ecg_vec: torch.Tensor,
                   pos_vecs: list[torch.Tensor],
                   neg_vecs: list[torch.Tensor]) -> tuple[list[float], list[float]]:
    """
    For each (pos_i, neg_i) pair compute softmax probability that pos wins.
    Mirrors get_diseases_probs() in main.py exactly.

    Returns per-observation positive scores and negative scores.
    """
    pos_scores, neg_scores = [], []
    for p_vec, n_vec in zip(pos_vecs, neg_vecs):
        sim_p = (ecg_vec @ p_vec).item()
        sim_n = (ecg_vec @ n_vec).item()
        # softmax with temperature 0.5
        exp_p = math.exp(sim_p / SOFTMAX_TEMP)
        exp_n = math.exp(sim_n / SOFTMAX_TEMP)
        total = exp_p + exp_n
        pos_scores.append(exp_p / total)
        neg_scores.append(exp_n / total)
    return pos_scores, neg_scores


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention localisation
# ─────────────────────────────────────────────────────────────────────────────

def get_cross_attention_heatmap(model: M3AEModel,
                                text_seq: torch.Tensor,
                                text_mask: torch.Tensor,
                                ecg_seq: torch.Tensor,
                                ecg_mask: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
    """
    Run one forward pass through multi_modal_language_layers
    (text-as-query, ECG-as-key/value) and collect cross-attention
    weights from each of the 6 layers.

    Returns heatmap of shape (Lx,) — one importance value per ECG token
    (excludes the CLS token at index 0).

    BertCrossLayer output tuple: (layer_output, self_attn, cross_attn)
    cross_attn shape: (batch, num_heads, Lt, Lx+1)
    """
    # add modality-type embeddings — must match the forward() in cmelt.py
    text_ids_dummy = torch.zeros(
        (1, text_seq.size(1)), dtype=torch.long, device=device
    )
    ecg_ids_dummy = torch.ones(
        (1, ecg_seq.size(1)), dtype=torch.long, device=device
    )

    with torch.no_grad():
        x = text_seq + model.modality_type_embeddings(text_ids_dummy)
        y = ecg_seq  + model.modality_type_embeddings(ecg_ids_dummy)

        # extended masks (additive, matching cmelt.py)
        ext_text_mask = model.language_encoder.get_extended_attention_mask(
            text_mask, text_mask.size()
        )
        ext_ecg_mask = model.language_encoder.get_extended_attention_mask(
            ecg_mask, ecg_mask.size()
        )

        all_cross_attn = []   # one tensor per layer

        for text_layer in model.multi_modal_language_layers:
            outputs = text_layer(
                x, y,
                attention_mask=ext_text_mask,
                encoder_attention_mask=ext_ecg_mask,
                output_attentions=True,
            )
            # outputs = (layer_out, self_attn_weights, cross_attn_weights)
            cross_attn = outputs[2]   # (1, 12_heads, Lt, Lx+1)
            all_cross_attn.append(cross_attn)
            x = outputs[0]   # feed updated text sequence to next layer

    # stack layers → (6, 1, 12, Lt, Lx+1)
    stacked = torch.stack(all_cross_attn, dim=0)

    # mean over layers, batch, heads, text-tokens → (Lx+1,)
    heatmap_with_cls = stacked.mean(dim=0).mean(dim=0).mean(dim=0).mean(dim=0)

    # drop CLS token at index 0 → (Lx,)  = (312,)
    heatmap = heatmap_with_cls[1:]

    return heatmap.cpu()


def heatmap_to_interval(heatmap: torch.Tensor, lead_axis: int = 1):
    """
    Convert a 1-D importance heatmap over ECG tokens into:
      - (start_ms, end_ms)  bounding interval
      - dominant_lead index
      - is_diffuse flag

    The heatmap here is aggregated over leads, so lead selection
    is a separate step (we pick the lead whose raw signal has
    highest variance in the predicted window — a proxy for
    diagnostic salience without requiring per-lead attention).
    """
    n_tokens = len(heatmap)

    # normalise to [0,1]
    h = heatmap.float()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    # diffuseness: compare actual entropy to maximum possible entropy
    probs = h / (h.sum() + 1e-8)
    entropy = -(probs * (probs + 1e-8).log()).sum().item()
    max_entropy = math.log(n_tokens)
    diffuse = (entropy / max_entropy) > DIFFUSE_ENTROPY_THRESHOLD

    # threshold: tokens above PEAK_PERCENTILE
    threshold = torch.quantile(h, PEAK_PERCENTILE / 100.0).item()
    active = (h >= threshold).nonzero(as_tuple=True)[0]

    if len(active) < MIN_INTERVAL_TOKENS or diffuse:
        return None, None, True   # diffuse

    start_token = active[0].item()
    end_token   = active[-1].item()

    start_ms = int(start_token * MS_PER_TOKEN)
    end_ms   = int((end_token + 1) * MS_PER_TOKEN)

    return start_ms, end_ms, False


def pick_dominant_lead(ecg: np.ndarray,
                       start_ms: int,
                       end_ms: int) -> int:
    """
    Within the predicted interval, return the lead index with
    highest signal variance — a proxy for diagnostic salience.
    """
    start_sample = int(start_ms / 1000 * SAMPLE_RATE)
    end_sample   = int(end_ms   / 1000 * SAMPLE_RATE)
    segment = ecg[start_sample:end_sample, :]   # (samples, 12)
    variances = segment.var(axis=0)
    return int(np.argmax(variances))


# ─────────────────────────────────────────────────────────────────────────────
# Strength label
# ─────────────────────────────────────────────────────────────────────────────

def strength_label(score: float, is_positive: bool) -> tuple[str, str]:
    """
    For positive observations: high score = strong evidence for condition.
    For negative observations: low score  = strong evidence for condition
                               (the negative feature is absent).
    Returns (bar, label).
    """
    # for negative obs, invert: a low pos-score means the neg obs is absent
    effective = score if is_positive else (1.0 - score)

    if effective >= 0.75:
        return "████████", "highlight strongly"
    elif effective >= 0.60:
        return "██████  ", "highlight moderately"
    elif effective >= 0.45:
        return "████    ", "highlight weakly"
    else:
        return "░░      ", "suppress"


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(ecg: np.ndarray,
        condition: str,
        model: M3AEModel,
        tokenizer: T5TokenizerFast,
        observations: dict,
        device: torch.device,
        ground_truth: dict | None = None):

    if condition not in observations:
        raise ValueError(f"Condition '{condition}' not found in observations JSON.")

    pos_texts = [t.lower() for t in observations[condition]["P"]]
    neg_texts = [t.lower() for t in observations[condition]["N"]]

    # ── encode ECG once ──────────────────────────────────────────────────────
    ecg_vec, ecg_seq, ecg_mask = encode_ecg(model, ecg, device)

    # ── encode all observation texts ─────────────────────────────────────────
    pos_vecs, pos_seqs, pos_masks = [], [], []
    for t in pos_texts:
        v, s, m = encode_text(model, tokenizer, t, device)
        pos_vecs.append(v); pos_seqs.append(s); pos_masks.append(m)

    neg_vecs, neg_seqs, neg_masks = [], [], []
    for t in neg_texts:
        v, s, m = encode_text(model, tokenizer, t, device)
        neg_vecs.append(v); neg_seqs.append(s); neg_masks.append(m)

    # ── similarity scores (ZETA path) ────────────────────────────────────────
    n_pairs = min(len(pos_vecs), len(neg_vecs))
    pos_scores, neg_scores = pairwise_score(
        ecg_vec, pos_vecs[:n_pairs], neg_vecs[:n_pairs]
    )
    final_score = float(np.mean(pos_scores))

    # ── cross-attention localisation for each observation ────────────────────
    def localise(text_seq, text_mask):
        heatmap = get_cross_attention_heatmap(
            model, text_seq, text_mask, ecg_seq, ecg_mask, device
        )
        start_ms, end_ms, diffuse = heatmap_to_interval(heatmap)
        if diffuse or start_ms is None:
            return None, None, None
        lead_idx = pick_dominant_lead(ecg, start_ms, end_ms)
        return start_ms, end_ms, lead_idx

    pos_locs = [localise(s, m) for s, m in zip(pos_seqs, pos_masks)]
    neg_locs = [localise(s, m) for s, m in zip(neg_seqs, neg_masks)]

    # ── print results ─────────────────────────────────────────────────────────
    print()

    # ground truth block
    if ground_truth is not None:
        fold = ground_truth["strat_fold"]
        fold_note = " (official test set)" if fold == 10 else f" (fold {fold})"
        print(f"Report:     {ground_truth['report']}{fold_note}")
        print()

        # check whether the queried condition appears in confirmed or uncertain
        all_codes = (
            [e[0] for e in ground_truth["confirmed"]] +
            [e[0] for e in ground_truth["uncertain"]]
        )
        if condition in [e[0] for e in ground_truth["confirmed"]]:
            gt_marker = f"✓  '{condition}' is CONFIRMED in this ECG (likelihood 100%)"
        elif condition in [e[0] for e in ground_truth["uncertain"]]:
            match = next(e for e in ground_truth["uncertain"] if e[0] == condition)
            gt_marker = f"~  '{condition}' is UNCERTAIN in this ECG (likelihood {match[1]:.0f}%)"
        else:
            gt_marker = f"✗  '{condition}' is NOT labelled in this ECG"
        print(f"Ground truth:  {gt_marker}")

        if ground_truth["confirmed"]:
            print("  Confirmed (100%):")
            for code, likelihood, desc, cats in ground_truth["confirmed"]:
                cat_parts = []
                if cats.get("diagnostic_class"):
                    cat_parts.append(f"class={cats['diagnostic_class']}")
                if cats.get("diagnostic_subclass"):
                    cat_parts.append(f"sub={cats['diagnostic_subclass']}")
                if cats.get("rhythm"):
                    cat_parts.append("rhythm")
                if cats.get("form"):
                    cat_parts.append("form")
                cat_str = ", ".join(cat_parts) if cat_parts else "—"
                print(f"    {code:<12} {desc:<40} [{cat_str}]")

        if ground_truth["uncertain"]:
            print("  Uncertain (<100%):")
            for code, likelihood, desc, cats in ground_truth["uncertain"]:
                print(f"    {code:<12} {desc:<40} [{likelihood:.0f}%]")
        print()

    print(f"Condition: {condition}  (final score: {final_score:.2f})")
    print()

    col_obs  = 45
    col_scr  = 7
    col_loc  = 22
    col_bar  = 10

    header = (
        f"  {'Observation':<{col_obs}}"
        f"{'Score':>{col_scr}}  "
        f"{'Localization':<{col_loc}}"
        f"{'Strength'}"
    )
    print(header)
    print("  " + "─" * (col_obs + col_scr + col_loc + col_bar + 20))

    def format_row(tag, text, score, loc, is_positive):
        start_ms, end_ms, lead_idx = loc
        if start_ms is not None:
            loc_str = f"{start_ms}ms – {end_ms}ms  {LEAD_NAMES[lead_idx]}"
        else:
            loc_str = "diffuse"
        bar, label = strength_label(score, is_positive)
        obs_str = f"[{tag}] {text}"
        print(
            f"  {obs_str:<{col_obs}}"
            f"{score:>{col_scr}.2f}  "
            f"{loc_str:<{col_loc}}"
            f"{bar}  {label}"
        )

    for i, (text, score, loc) in enumerate(zip(pos_texts, pos_scores, pos_locs)):
        format_row("P", text, score, loc, is_positive=True)

    print()

    for i, (text, score, loc) in enumerate(zip(neg_texts, neg_scores, neg_locs)):
        # neg_scores[i] is the softmax prob of the negative observation winning —
        # a LOW neg score means the neg feature is absent → supports the condition
        format_row("N", text, score, loc, is_positive=False)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_ptbxl_record(ptbxl_root: str, filename_hr: str) -> np.ndarray:
    """
    Load a single PTB-XL record from its .dat/.hea WFDB file pair.

    Parameters
    ----------
    ptbxl_root  : root directory of PTB-XL
                  (the folder containing records500/, ptbxl_database.csv, etc.)
    filename_hr : value from the 'filename_hr' column of ptbxl_database.csv,
                  e.g. 'records500/00000/00001_hr'  (no extension)

    Returns
    -------
    ecg : np.ndarray, shape (5000, 12), float32, normalised to [0, 1]
          Lead order: I, II, III, aVR, aVL, aVF, V1–V6
          (aVL/aVF swapped to match MIMIC/D-BETA training convention)
    """
    path = str(Path(ptbxl_root) / filename_hr)

    # wfdb.rdsamp returns (signal, fields)
    # signal shape: (T, 12) — already time-first, no transpose needed
    signal, fields = wfdb.rdsamp(path)

    ecg = signal.astype(np.float32)      # (T, 12)
    ecg = ecg[:5000, :]                  # guard: ensure exactly 5000 samples
    ecg = ecg.T                          # → (12, 5000) for normalisation

    # normalise to [0, 1] per recording — matches data_load.py
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-8)

    # swap aVL (index 4) and aVF (index 5) to match MIMIC lead order
    # PTB-XL order: I II III aVR aVL aVF V1-V6
    # MIMIC order:  I II III aVR aVF aVL V1-V6
    ecg[[4, 5]] = ecg[[5, 4]]

    return ecg.T   # → (5000, 12) time-first


def load_ptbxl_db(ptbxl_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ptbxl_database.csv and scp_statements.csv from the PTB-XL root.
    Both are returned indexed by their natural key.
    """
    db_path  = Path(ptbxl_root) / "ptbxl_database.csv"
    scp_path = Path(ptbxl_root) / "scp_statements.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")
    if not scp_path.exists():
        raise FileNotFoundError(f"scp_statements.csv not found at {scp_path}")
    db  = pd.read_csv(db_path,  index_col="ecg_id")
    scp = pd.read_csv(scp_path, index_col=0)
    return db, scp


def find_filename_hr(ptbxl_root: str, ecg_id: int) -> str:
    """
    Look up filename_hr for a given ecg_id from ptbxl_database.csv.
    """
    db, _ = load_ptbxl_db(ptbxl_root)
    if ecg_id not in db.index:
        raise ValueError(f"ecg_id {ecg_id} not found in ptbxl_database.csv")
    return db.loc[ecg_id, "filename_hr"]


# Likelihood threshold above which a diagnostic SCP code is considered positive.
# PTB-XL convention: annotators assign 100 = "definitely present",
# lower values = uncertain. We show both tiers separately.
CONFIRMED_THRESHOLD  = 100.0
UNCERTAIN_THRESHOLD  =   0.0   # > 0 means mentioned, even if not confirmed


def get_ground_truth(ptbxl_root: str, ecg_id: int) -> dict:
    """
    Read ground truth labels for a given ecg_id directly from
    ptbxl_database.csv and scp_statements.csv.

    Returns a dict with keys:
      'confirmed'  : list of (code, description, categories) where likelihood == 100
      'uncertain'  : list of (code, description, categories) where 0 < likelihood < 100
      'report'     : raw free-text report string from the database
      'strat_fold' : integer fold (10 = official test set)

    'categories' is itself a dict, e.g.:
      {'diagnostic_class': 'CD', 'diagnostic_subclass': '1AVB',
       'rhythm': False, 'form': False}
    """
    db, scp = load_ptbxl_db(ptbxl_root)
    if ecg_id not in db.index:
        raise ValueError(f"ecg_id {ecg_id} not found in ptbxl_database.csv")

    row = db.loc[ecg_id]
    raw_codes = ast.literal_eval(row["scp_codes"])   # {'NORM': 100.0, 'SR': 0.0, ...}

    confirmed, uncertain = [], []
    for code, likelihood in raw_codes.items():
        if likelihood <= UNCERTAIN_THRESHOLD:
            continue   # zero-likelihood codes are annotation artefacts, skip

        # look up description and category from scp_statements
        if code in scp.index:
            s = scp.loc[code]
            description = str(s["description"])
            categories  = {
                "diagnostic_class":    s["diagnostic_class"]
                                       if pd.notna(s["diagnostic_class"]) else None,
                "diagnostic_subclass": s["diagnostic_subclass"]
                                       if pd.notna(s["diagnostic_subclass"]) else None,
                "rhythm": pd.notna(s["rhythm"]),
                "form":   pd.notna(s["form"]),
            }
        else:
            description = "(unknown)"
            categories  = {}

        entry = (code, likelihood, description, categories)
        if likelihood >= CONFIRMED_THRESHOLD:
            confirmed.append(entry)
        else:
            uncertain.append(entry)

    return {
        "confirmed":  confirmed,
        "uncertain":  uncertain,
        "report":     str(row["report"]),
        "strat_fold": int(row["strat_fold"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZETA localisation for a single PTB-XL ECG")

    # --- ECG source (one of two options) ---
    ecg_group = parser.add_mutually_exclusive_group(required=True)
    ecg_group.add_argument(
        "--filename_hr",
        help=(
            "Value of 'filename_hr' from ptbxl_database.csv, "
            "e.g. 'records500/00000/00001_hr'. "
            "Used together with --ptbxl_root."
        )
    )
    ecg_group.add_argument(
        "--ecg_id",
        type=int,
        help=(
            "Integer ecg_id from ptbxl_database.csv. "
            "Used together with --ptbxl_root to look up filename_hr automatically."
        )
    )

    parser.add_argument(
        "--ptbxl_root",
        required=True,
        help="Root directory of the PTB-XL dataset (contains ptbxl_database.csv and records500/)."
    )

    parser.add_argument(
        "--ground_truth",
        action="store_true",
        help=(
            "Look up ground truth labels for this record directly from "
            "ptbxl_database.csv and scp_statements.csv. "
            "Requires --ecg_id and --ptbxl_root."
        )
    )
    parser.add_argument("--condition",    required=True,
                        help="Condition code from observations.json, e.g. 1AVB, AFIB, WPW")
    parser.add_argument("--checkpoint",   default="checkpoints/best.pt")
    parser.add_argument("--config",       default="configs/config.json")
    parser.add_argument("--observations", default="configs/observations.json")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    # --- resolve PTB-XL record ---
    if args.ecg_id is not None:
        print(f"Looking up ecg_id {args.ecg_id} in ptbxl_database.csv ...")
        filename_hr = find_filename_hr(args.ptbxl_root, args.ecg_id)
        print(f"  → filename_hr: {filename_hr}")
    else:
        filename_hr = args.filename_hr

    print(f"Loading ECG: {filename_hr} ...")
    ecg = load_ptbxl_record(args.ptbxl_root, filename_hr)
    print(f"  → shape {ecg.shape}, min {ecg.min():.3f}, max {ecg.max():.3f}")

    # --- load model ---
    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.config, args.checkpoint, device)

    tokenizer = T5TokenizerFast.from_pretrained(
        "google/flan-t5-base", do_lower_case=True
    )

    with open(args.observations) as f:
        observations = json.load(f)

    # --- optional ground truth lookup ---
    ground_truth = None
    if args.ground_truth:
        if args.ecg_id is None:
            print("Warning: --ground_truth requires --ecg_id; skipping GT lookup.")
        else:
            ground_truth = get_ground_truth(args.ptbxl_root, args.ecg_id)

    run(ecg, args.condition, model, tokenizer, observations, device, ground_truth)


if __name__ == "__main__":
    main()