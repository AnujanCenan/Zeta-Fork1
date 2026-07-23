"""
zeta_localise_2.0.py

Extends ZETA inference for a single ECG to produce:
  - Per-observation similarity scores (from the unimodal dot-product path)
  - Temporal localization INTERVALS (plural) for each observation, from the
    cross-attention fusion path, using multi-peak detection so that findings
    which recur across multiple heartbeats are reported as several distinct windows.
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
from scipy.signal import find_peaks, peak_widths
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
DIFFUSE_ENTROPY_THRESHOLD = 0.88   # kept only for diagnostic reporting
PROMINENCE_THRESHOLD      = 0.15   # baseline peak prominence floor
MIN_INTERVAL_TOKENS       = 3      # smallest meaningful interval (~96 ms)

# multi-peak detection knobs
MIN_PEAK_DISTANCE_TOKENS  = 14     # Increased to prevent overlapping sub-peaks inside one cardiac cycle
PEAK_WIDTH_REL_HEIGHT     = 0.4      # Tightened full-width baseline reference
MAX_INTERVALS_PER_OBS     = 8      # cap on number of reported intervals per observation


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
    x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(device)      # <1, 5000, 12>
    x = x.permute(0, 2, 1)      # <1, 12, 5000>

    with torch.no_grad():
        uni_modal_ecg_feats, ecg_padding_mask = model.ecg_encoder.get_embeddings(x, padding_mask=None)  # <1, 312, 256>
        
        cls_emb = model.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))        # <1, 1, 256>
        uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)          # <1, 313, 256>
        uni_modal_ecg_feats = model.ecg_encoder.get_output(uni_modal_ecg_feats, ecg_padding_mask)   # <1, 313, 256>
        
        out = model.multi_modal_ecg_proj(uni_modal_ecg_feats)       # <1, 313, 768>
        ecg_features = model.unimodal_ecg_pooler(out)               # <1, 768>
        
        ecg_vec = ecg_features.squeeze(0)       # <768>
        ecg_vec = F.normalize(ecg_vec, dim=0)   # <768> 

    Lx_plus1 = uni_modal_ecg_feats.size(1)      # <313>
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
        outputs = model.language_encoder(**encoded_input)[0]        # (1, 128, 768)
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
        for text_layer, ecg_layer in zip(model.multi_modal_language_layers, model.multi_modal_ecg_layers):
            text_outputs = text_layer(
                x, y,
                attention_mask=ext_text_mask,
                encoder_attention_mask=ext_ecg_mask,
                output_attentions=True,
            )
            ecg_outputs = ecg_layer(
                y, x,
                attention_mask=ext_ecg_mask,
                encoder_attention_mask=ext_text_mask,
                output_attentions=True,
            )

            # ecg_outputs[2] Shape: [1, num_heads, ecg_seq_len, text_seq_len] -> [1, 12, 313, 128]
            cross_attn = ecg_outputs[2].squeeze(0) # [12, 313, 128]
            
            # Apply focused softmax normalization over the ECG timeline axis (dim=1)
            cross_attn = F.softmax(cross_attn / 0.1, dim=1) 
            
            all_cross_attn.append(cross_attn)
            x, y = text_outputs[0], ecg_outputs[0]

    # Average across layers and attention heads -> Shape: [313, 128]
    stacked = torch.stack(all_cross_attn, dim=0).mean(dim=0).mean(dim=0) 

    # Isolate only the valid text words columns
    valid_text_len = int(text_mask.sum().item())
    
    # Average across the valid text dimension columns (dim=1) -> Shape: [313]
    ecg_timeline_heatmap = stacked[:, :valid_text_len].mean(dim=1) 

    # CORRECT SLICE: Remove index 0 (the ECG global [CLS] token)
    heatmap = ecg_timeline_heatmap[1:] # Now exactly 312 temporal tokens

    return heatmap.cpu()


def heatmap_to_intervals(heatmap: torch.Tensor, debug_label: str | None = None, verbose: bool = False):
    n_tokens = len(heatmap)
    h = heatmap.float()

    # 1. Min-max scale to [0, 1] range
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    h_np = h.numpy()

    # 2. Diffuseness metric reporting
    probs_full = h / (h.sum() + 1e-8)
    entropy = -(probs_full * (probs_full + 1e-8).log()).sum().item()
    max_entropy = math.log(n_tokens)
    entropy_ratio = entropy / max_entropy

    baseline = float(np.median(h_np))
    global_peak_val = float(h_np.max())
    global_prominence = global_peak_val - baseline

    diffuse = global_prominence < PROMINENCE_THRESHOLD

    if verbose:
        label = f" ({debug_label})" if debug_label else ""
        verdict = "DIFFUSE" if diffuse else "localized"
        print(
            f"    [debug]{label} entropy_ratio={entropy_ratio:.3f}  "
            f"peak={global_peak_val:.3f}  baseline(median)={baseline:.3f}  "
            f"prominence={global_prominence:.3f}  threshold={PROMINENCE_THRESHOLD:.3f}  -> {verdict}"
        )

    if diffuse:
        return [], True

    # 3. MULTI-PEAK DETECTION via scipy.signal.find_peaks.
    peak_indices, _properties = find_peaks(
        h_np,
        height=baseline + PROMINENCE_THRESHOLD,
        prominence=PROMINENCE_THRESHOLD,
        distance=MIN_PEAK_DISTANCE_TOKENS,
    )

    if verbose:
        label = f" ({debug_label})" if debug_label else ""
        print(f"    [debug]{label} find_peaks located {len(peak_indices)} peak(s) "
              f"(min_distance={MIN_PEAK_DISTANCE_TOKENS} tokens)")

    if len(peak_indices) == 0:
        return [], True

    # 4. Determine each peak's width at half-max position
    widths, _width_heights, left_ips, right_ips = peak_widths(
        h_np, peak_indices, rel_height=PEAK_WIDTH_REL_HEIGHT
    )

    intervals = []
    for i, peak_idx in enumerate(peak_indices):
        start_token = left_ips[i]
        end_token = right_ips[i]

        if (end_token - start_token) < MIN_INTERVAL_TOKENS:
            pad = (MIN_INTERVAL_TOKENS - (end_token - start_token)) / 2.0
            start_token -= pad
            end_token += pad

        start_token = max(0.0, start_token)
        end_token = min(float(n_tokens - 1), end_token)

        start_ms = int(start_token * MS_PER_TOKEN)
        end_ms = int((end_token + 1) * MS_PER_TOKEN)
        peak_val = float(h_np[peak_idx])
        intervals.append((start_ms, end_ms, peak_val))

    # 5. Merge any intervals that ended up overlapping/touching
    intervals.sort(key=lambda t: t[0])
    merged = [list(intervals[0])]
    for start_ms, end_ms, peak_val in intervals[1:]:
        if start_ms <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_ms)
            merged[-1][2] = max(merged[-1][2], peak_val)
        else:
            merged.append([start_ms, end_ms, peak_val])

    # 6. Cap to the strongest MAX_INTERVALS_PER_OBS peaks
    merged = sorted(merged, key=lambda t: t[2], reverse=True)[:MAX_INTERVALS_PER_OBS]
    merged.sort(key=lambda t: t[0])

    return [tuple(m) for m in merged], False


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


def format_locations(locs) -> str:
    if not locs:
        return "diffuse"

    MAX_SHOWN = 6 
    parts = [f"{s}ms-{e}ms ({LEAD_NAMES[l]})" for s, e, l in locs[:MAX_SHOWN]]
    remaining = len(locs) - MAX_SHOWN
    text = ", ".join(parts)
    if remaining > 0:
        text += f" +{remaining} more"
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(ecg: np.ndarray,
        condition: str,
        model: M3AEModel,
        tokenizer: T5TokenizerFast,
        observations: dict,
        device: torch.device,
        ground_truth: dict | None = None,
        verbose: bool = False):

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

    def localise(text_seq, text_mask, debug_label=None):
        heatmap = get_cross_attention_heatmap(
            model, text_seq, text_mask, ecg_seq, ecg_mask, device
        )
        intervals, diffuse = heatmap_to_intervals(heatmap, debug_label=debug_label, verbose=verbose)
        if diffuse or not intervals:
            return None
        locs = []
        for start_ms, end_ms, _peak_val in intervals:
            lead_idx = pick_dominant_lead(ecg, start_ms, end_ms)
            locs.append((start_ms, end_ms, lead_idx))
        return locs

    if verbose:
        print("\n  [debug] localization diagnostics (prominence gates the diffuse/localized call):")
    pos_locs = [localise(s, m, debug_label=f"P: {pos_texts[i]}") for i, (s, m) in enumerate(zip(pos_seqs, pos_masks))]
    neg_locs = [localise(s, m, debug_label=f"N: {neg_texts[i]}") for i, (s, m) in enumerate(zip(neg_seqs, neg_masks))]

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

    col_obs, col_scr, col_loc, col_bar = 45, 7, 40, 10
    header = f"  {'Observation':<{col_obs}}{'Score':>{col_scr}}  {'Localization(s)':<{col_loc}}{'Strength'}"
    print(header)
    print("  " + "─" * (col_obs + col_scr + col_loc + col_bar + 20))

    def format_row(tag, text, score, locs, is_positive):
        loc_str = format_locations(locs)
        bar, label = strength_label(score, is_positive)
        obs_str = f"[{tag}] {text}"
        print(f"  {obs_str:<{col_obs}}{score:>{col_scr}.2f}  {loc_str:<{col_loc}}{bar}  {label}")

    for i in range(n_pairs):
        format_row("P", pos_texts[i], pos_probabilities[i], pos_locs[i], is_positive=True)
    print()
    for i in range(n_pairs):
        format_row("N", neg_texts[i], neg_probabilities[i], neg_locs[i], is_positive=False)
    print()

import matplotlib.pyplot as plt

def plot_ecg_12_leads(ecg: np.ndarray, title: str = "12-Lead ECG Signal"):
    if ecg.shape[0] == 12:
        ecg = ecg.T
        
    n_samples, n_leads = ecg.shape
    time_ms = np.arange(n_samples) * (1000.0 / SAMPLE_RATE)
    
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.92)
    
    for i in range(n_leads):
        ax = axes[i]
        ax.plot(time_ms, ecg[:, i], color='crimson', linewidth=1.0)
        
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=20, 
                      fontsize=11, fontweight='bold', verticalalignment='center')
        
        ax.grid(True, which='both', color='pink', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(True, which='minor', color='mistyrose', linestyle=':', linewidth=0.5)
        
    plt.xlabel("Time (ms)", fontsize=12, fontweight='bold')
    plt.xlim(0, time_ms[-1])
    plt.subplots_adjust(hspace=0.2)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point utils
# ─────────────────────────────────────────────────────────────────────────────

def load_ptbxl_record(ptbxl_root: str, filename_hr: str) -> np.ndarray:
    path = str(Path(ptbxl_root) / filename_hr)
    signal, fields = wfdb.rdsamp(path)
    ecg = signal.astype(np.float32)[:5000, :]
    
    # FIXED: Min-max normalize per lead independently so precordial leads (V1/V2) 
    # maintain explicit spatial morphology details rather than being blanketed out by high-voltage limb leads.
    ecg_min = ecg.min(axis=0, keepdims=True)
    ecg_max = ecg.max(axis=0, keepdims=True)
    ecg = (ecg - ecg_min) / (ecg_max - ecg_min + 1e-8)
    
    # If the network requires channel index adjustments based on custom checkpoint history mapping, 
    # swap specific lines here.
    return ecg


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
        "report":       str(row["report"]),
        "strat_fold": int(row["strat_fold"]),
    }


def main():
    load_dotenv()
    PTBXL_DATASET = os.getenv("PTBXL_DATASET")
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")

    parser = argparse.ArgumentParser(description="ZETA localisation (v2.0) for a single PTB-XL ECG")
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
    parser.add_argument("--debug", action="store_true")

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
    plot_ecg_12_leads(ecg, title=f"ECG Visualization Plot: ID {args.ecg_id or filename_hr}")
    model = load_model(args.config, args.checkpoint, device)
    tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)

    with open(args.observations) as f:
        observations = json.load(f)

    ground_truth = None
    if args.ground_truth and args.ecg_id is not None:
        ground_truth = get_ground_truth(args.ptbxl_root, args.ecg_id)

    run(ecg, args.condition, model, tokenizer, observations, device, ground_truth=ground_truth, verbose=args.debug)

if __name__ == "__main__":
    main()