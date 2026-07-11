"""
zeta_localize.py

Extends ZETA inference for a single ECG to produce:
  - Per-observation similarity scores (from the unimodal dot-product path)
  - Temporal localization intervals (from the cross-attention fusion path)
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
from dotenv import load_dotenv
import os

# ── constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 500          # Hz
ECG_LENGTH        = 5000         # samples (10 s)
CONV_LAYERS       = [(256,2,2)] * 4  # must match config conv_feature_layers
NUM_ECG_TOKENS   = 312          # computed from CONV_LAYERS above
SAMPLES_PER_TOKEN = ECG_LENGTH / NUM_ECG_TOKENS   # ≈ 16.03 samples = ~32 ms
MS_PER_TOKEN     = (SAMPLES_PER_TOKEN / SAMPLE_RATE) * 1000  # ≈ 32.05 ms

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# ZETA Paper Training Factor (Do not change)
SOFTMAX_TEMP = 0.5   

# UI Calibration Factor: Amplifies latent micro-variances into sharp display contrasts
VISUAL_TEMP = 0.02

# attention heatmap thresholds
DIFFUSE_ENTROPY_THRESHOLD = 0.88   # Marginally expanded to handle cross-modal bias
MIN_INTERVAL_TOKENS       = 3      # smallest meaningful interval (~96 ms)
PEAK_PERCENTILE           = 85     # Focus strictly on the top attention peaks


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> M3AEModel:
    with open(config_path) as f:
        cfg = SimpleNamespace(**json.load(f)["model"])

    model = M3AEModel(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"]
    state.pop("ecg_encoder.mask_emb", None)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model

# ─────────────────────────────────────────────────────────────────────────────
# ECG encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_ecg(model: M3AEModel, ecg: np.ndarray, device: torch.device):
    x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(device)
    x = x.permute(0, 2, 1)

    with torch.no_grad():
        uni_modal_ecg_feats, ecg_padding_mask = model.ecg_encoder.get_embeddings(x, padding_mask=None)
        
        cls_emb = model.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
        uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)
        uni_modal_ecg_feats = model.ecg_encoder.get_output(uni_modal_ecg_feats, ecg_padding_mask)
        
        out = model.multi_modal_ecg_proj(uni_modal_ecg_feats)
        ecg_features = model.unimodal_ecg_pooler(out)
        
        ecg_vec = ecg_features.squeeze(0) 
        ecg_vec = F.normalize(ecg_vec, dim=0)

    Lx_plus1 = uni_modal_ecg_feats.size(1)
    ecg_mask = torch.ones((1, Lx_plus1), dtype=torch.long, device=device)

    return ecg_vec, out.detach(), ecg_mask


# ─────────────────────────────────────────────────────────────────────────────
# Text encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_text(model: M3AEModel,
                tokenizer: T5TokenizerFast,
                text: str,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    encoded_input = tokenizer(
        text, 
        padding="max_length", 
        max_length=128, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.language_encoder(**encoded_input)[0]
        outputs = model.multi_modal_language_proj(outputs)
        max_pooled_features = model.unimodal_language_pooler(outputs)
        text_vec = max_pooled_features.squeeze(0) 

    return text_vec, outputs.detach(), encoded_input["attention_mask"]

# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention localisation
# ─────────────────────────────────────────────────────────────────────────────

def get_cross_attention_heatmap(model: M3AEModel,
                                text_seq: torch.Tensor,
                                text_mask: torch.Tensor,
                                ecg_seq: torch.Tensor,
                                ecg_mask: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
    text_ids_dummy = torch.zeros((1, text_seq.size(1)), dtype=torch.long, device=device)
    ecg_ids_dummy = torch.ones((1, ecg_seq.size(1)), dtype=torch.long, device=device)

    with torch.no_grad():
        x = text_seq + model.modality_type_embeddings(text_ids_dummy)
        y = ecg_seq  + model.modality_type_embeddings(ecg_ids_dummy)

        ext_text_mask = model.language_encoder.get_extended_attention_mask(text_mask, text_mask.size())
        ext_ecg_mask = model.language_encoder.get_extended_attention_mask(ecg_mask, ecg_mask.size())

        all_cross_attn = []
        for text_layer in model.multi_modal_language_layers:
            outputs = text_layer(
                x, y,
                attention_mask=ext_text_mask,
                encoder_attention_mask=ext_ecg_mask,
                output_attentions=True,
            )
            # cross_attn matrix shape: [1, num_heads, text_seq_len, ecg_seq_len]
            cross_attn = outputs[2]
            all_cross_attn.append(cross_attn)
            x = outputs[0]

    # Average across layers and attention heads
    stacked = torch.stack(all_cross_attn, dim=0).mean(dim=0).squeeze(0) # Shape: [num_heads, text_seq_len, ecg_seq_len]
    stacked = stacked.mean(dim=0) # Shape: [text_seq_len, ecg_seq_len]

    # FIX: Isolate only rows belonging to valid text tokens, then average along the text axis 
    valid_text_len = int(text_mask.sum().item())
    ecg_timeline_heatmap = stacked[:valid_text_len, :].mean(dim=0) # Shape: [ecg_seq_len]

    # Chop off the first element corresponding to the CLS token, leaving the 312 temporal tokens intact
    heatmap = ecg_timeline_heatmap[1:]

    return heatmap.cpu()

def heatmap_to_interval(heatmap: torch.Tensor):
    n_tokens = len(heatmap)
    h = heatmap.float()
    
    # 1. Min-max scale to [0, 1] range
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    # 2. NOISE FILTER: Zero out anything below the median attention level
    # This prevents tiny residual distributions from dominating the entropy calculation
    h_filtered = torch.where(h > torch.median(h), h, torch.zeros_like(h))

    # 3. Compute entropy on the cleaned distribution
    probs = h_filtered / (h_filtered.sum() + 1e-8)
    entropy = -(probs * (probs + 1e-8).log()).sum().item()
    
    # Max entropy of a completely uniform signal
    max_entropy = math.log(n_tokens)
    
    # Check if the signal is genuinely uniform across the timeline
    diffuse = (entropy / max_entropy) > DIFFUSE_ENTROPY_THRESHOLD

    # 4. Extract active interval peaks using standard percentile bounds
    threshold = torch.quantile(h, PEAK_PERCENTILE / 100.0).item()
    active = (h >= threshold).nonzero(as_tuple=True)[0]

    if len(active) < MIN_INTERVAL_TOKENS or diffuse:
        return None, None, True

    start_token = active[0].item()
    end_token   = active[-1].item()

    start_ms = int(start_token * MS_PER_TOKEN)
    end_ms   = int((end_token + 1) * MS_PER_TOKEN)

    return start_ms, end_ms, False

def pick_dominant_lead(ecg: np.ndarray, start_ms: int, end_ms: int) -> int:
    start_sample = int(start_ms / 1000 * SAMPLE_RATE)
    end_sample   = int(end_ms   / 1000 * SAMPLE_RATE)
    segment = ecg[start_sample:end_sample, :]
    variances = segment.var(axis=0)
    return int(np.argmax(variances))


# ─────────────────────────────────────────────────────────────────────────────
# Strength label
# ─────────────────────────────────────────────────────────────────────────────

def strength_label(probability_score: float, is_positive: bool) -> tuple[str, str]:
    effective = probability_score if is_positive else (1.0 - probability_score)

    if effective >= 0.70:
        return "████████", "highlight strongly"
    elif effective >= 0.58:
        return "██████  ", "highlight moderately"
    elif effective >= 0.52:
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

    ecg_vec, ecg_seq, ecg_mask = encode_ecg(model, ecg, device)

    pos_vecs, pos_seqs, pos_masks = [], [], []
    for t in pos_texts:
        v, s, m = encode_text(model, tokenizer, t, device)
        pos_vecs.append(v); pos_seqs.append(s); pos_masks.append(m)

    neg_vecs, neg_seqs, neg_masks = [], [], []
    for t in neg_texts:
        v, s, m = encode_text(model, tokenizer, t, device)
        neg_vecs.append(v); neg_seqs.append(s); neg_masks.append(m)

    n_pairs = min(len(pos_vecs), len(neg_vecs))
    
    similarities_p = []
    similarities_n = []
    
    for i in range(n_pairs):
        norm_p = F.normalize(pos_vecs[i], dim=0).reshape(1, -1).T
        norm_n = F.normalize(neg_vecs[i], dim=0).reshape(1, -1).T
        
        sim_p = (ecg_vec @ norm_p)[0].item()
        sim_n = (ecg_vec @ norm_n)[0].item()
        
        similarities_p.append(sim_p)
        similarities_n.append(sim_n)

    pos_probs_global = []
    for i in range(n_pairs):
        l_p = similarities_p[i] / SOFTMAX_TEMP
        l_n = similarities_n[i] / SOFTMAX_TEMP
        mx = max(l_p, l_n)
        exp_p = math.exp(l_p - mx)
        exp_n = math.exp(l_n - mx)
        pos_probs_global.append(exp_p / (exp_p + exp_n))
    
    final_score = float(np.mean(pos_probs_global))

    pos_probabilities = []
    neg_probabilities = []
    
    for i in range(n_pairs):
        logit_p = similarities_p[i] / VISUAL_TEMP
        logit_n = similarities_n[i] / VISUAL_TEMP
        
        max_logit = max(logit_p, logit_n)
        exp_p = math.exp(logit_p - max_logit)
        exp_n = math.exp(logit_n - max_logit)
        total = exp_p + exp_n
        
        pos_probabilities.append(exp_p / total)
        neg_probabilities.append(exp_n / total)

    # Handle Localizations
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

    print()
    if ground_truth is not None:
        fold = ground_truth["strat_fold"]
        fold_note = " (official test set)" if fold == 10 else f" (fold {fold})"
        print(f"Report:     {ground_truth['report']}{fold_note}")
        print()
        if condition in [e[0] for e in ground_truth["confirmed"]]:
            gt_marker = f"✓  '{condition}' is CONFIRMED in this ECG (likelihood 100%)"
        elif condition in [e[0] for e in ground_truth["uncertain"]]:
            match = next(e for e in ground_truth["uncertain"] if e[0] == condition)
            gt_marker = f"~  '{condition}' is UNCERTAIN in this ECG (likelihood {match[1]:.0f}%)"
        else:
            gt_marker = f"✗  '{condition}' is NOT labelled in this ECG"
        print(f"Ground truth:  {gt_marker}\n")

    print(f"Condition: {condition}  (final score: {final_score:.2f})\n")

    col_obs, col_scr, col_loc, col_bar = 45, 7, 22, 10
    header = f"  {'Observation':<{col_obs}}{'Score':>{col_scr}}  {'Localization':<{col_loc}}{'Strength'}"
    print(header)
    print("  " + "─" * (col_obs + col_scr + col_loc + col_bar + 20))

    def format_row(tag, text, score, loc, is_positive):
        start_ms, end_ms, lead_idx = loc
        loc_str = f"{start_ms}ms – {end_ms}ms  {LEAD_NAMES[lead_idx]}" if start_ms is not None else "diffuse"
        bar, label = strength_label(score, is_positive)
        obs_str = f"[{tag}] {text}"
        print(f"  {obs_str:<{col_obs}}{score:>{col_scr}.2f}  {loc_str:<{col_loc}}{bar}  {label}")

    for i in range(n_pairs):
        format_row("P", pos_texts[i], pos_probabilities[i], pos_locs[i], is_positive=True)
    print()
    for i in range(n_pairs):
        format_row("N", neg_texts[i], neg_probabilities[i], neg_locs[i], is_positive=False)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point utils
# ─────────────────────────────────────────────────────────────────────────────

def load_ptbxl_record(ptbxl_root: str, filename_hr: str) -> np.ndarray:
    path = str(Path(ptbxl_root) / filename_hr)
    signal, fields = wfdb.rdsamp(path)
    ecg = signal.astype(np.float32)
    ecg = ecg[:5000, :]
    ecg = ecg.T
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-8)
    ecg[[4, 5]] = ecg[[5, 4]] 
    return ecg.T


def load_ptbxl_db(ptbxl_root: str):
    db_path  = Path(ptbxl_root) / "ptbxl_database.csv"
    scp_path = Path(ptbxl_root) / "scp_statements.csv"
    db  = pd.read_csv(db_path,  index_col="ecg_id")
    scp = pd.read_csv(scp_path, index_col=0)
    return db, scp


def find_filename_hr(ptbxl_root: str, ecg_id: int) -> str:
    db, _ = load_ptbxl_db(ptbxl_root)
    return db.loc[ecg_id, "filename_hr"]


def get_ground_truth(ptbxl_root: str, ecg_id: int) -> dict:
    db, scp = load_ptbxl_db(ptbxl_root)
    row = db.loc[ecg_id]
    raw_codes = ast.literal_eval(row["scp_codes"])

    confirmed, uncertain = [], []
    for code, likelihood in raw_codes.items():
        if likelihood <= 0.0:
            continue

        if code in scp.index:
            s = scp.loc[code]
            description = str(s["description"])
            categories  = {
                "diagnostic_class":    s["diagnostic_class"] if pd.notna(s["diagnostic_class"]) else None,
                "diagnostic_subclass": s["diagnostic_subclass"] if pd.notna(s["diagnostic_subclass"]) else None,
                "rhythm": pd.notna(s["rhythm"]),
                "form":   pd.notna(s["form"]),
            }
        else:
            description = "(unknown)"
            categories  = {}

        entry = (code, likelihood, description, categories)
        if likelihood >= 100.0:
            confirmed.append(entry)
        else:
            uncertain.append(entry)

    return {
        "confirmed":  confirmed,
        "uncertain":  uncertain,
        "report":      str(row["report"]),
        "strat_fold": int(row["strat_fold"]),
    }


def main():
    load_dotenv()
    PTBXL_DATASET = os.getenv("PTBXL_DATASET")
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")

    parser = argparse.ArgumentParser(description="ZETA localisation for a single PTB-XL ECG")
    ecg_group = parser.add_mutually_exclusive_group(required=True)
    ecg_group.add_argument("--filename_hr")
    ecg_group.add_argument("--ecg_id", type=int)

    parser.add_argument("--ptbxl_root", default=PTBXL_DATASET)
    parser.add_argument("--ground_truth", action="store_true")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--observations", default="configs/observations.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.ptbxl_root is None or args.checkpoint is None:
        print("ensure PTBXL_DATASET and CHECKPOINT_PATH are set in .env")
        exit(1)

    if args.ecg_id is not None:
        filename_hr = find_filename_hr(args.ptbxl_root, args.ecg_id)
    else:
        filename_hr = args.filename_hr

    ecg = load_ptbxl_record(args.ptbxl_root, filename_hr)
    model = load_model(args.config, args.checkpoint, device)
    tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)

    with open(args.observations) as f:
        observations = json.load(f)

    ground_truth = None
    if args.ground_truth and args.ecg_id is not None:
        ground_truth = get_ground_truth(args.ptbxl_root, args.ecg_id)

    run(ecg, args.condition, model, tokenizer, observations, device, ground_truth)


if __name__ == "__main__":
    main()