"""
zeta_localise_multi_figs.py

Extends ZETA inference for a single ECG to produce:
  - Per-observation similarity scores (from the unimodal dot-product path)
  - Temporal localization INTERVALS (plural) for each observation.
  - Multi-plot vertical tracking where each positive observation feature is 
    rendered on its own independent figure canvas with high-resolution 
    millisecond grid grids.
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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# ZETA Paper Training Factor
SOFTMAX_TEMP = 0.5   

# UI Calibration Factor: Amplifies latent micro-variances into sharp display contrasts
VISUAL_TEMP = 0.02

# attention heatmap thresholds
PROMINENCE_THRESHOLD      = 0.30   # Elevated to ensure strict anchoring onto high-energy beats
MIN_INTERVAL_TOKENS       = 3      # smallest meaningful interval (~96 ms)

# multi-peak detection knobs
MIN_PEAK_DISTANCE_TOKENS  = 14     # Set to prevent overlapping sub-peaks inside one cardiac cycle
PEAK_WIDTH_REL_HEIGHT     = 0.4      # Tightened full-width baseline reference
MAX_INTERVALS_PER_OBS     = 8      # cap on number of reported intervals per observation


# ─────────────────────────────────────────────────────────────────────────────
# Model loading & encoding functions
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


def encode_text(model: M3AEModel, tokenizer: T5TokenizerFast, text: str, device: torch.device):
    encoded_input = tokenizer(
        text, padding="max_length", max_length=128, truncation=True, return_tensors="pt"
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
    text_ids_dummy = torch.ones((1, text_seq.size(1)), dtype=torch.long, device=device)        
    ecg_ids_dummy = torch.ones((1, ecg_seq.size(1)), dtype=torch.long, device=device)           

    with torch.no_grad():
        x = text_seq + model.modality_type_embeddings(text_ids_dummy)
        y = ecg_seq  + model.modality_type_embeddings(ecg_ids_dummy)

        ext_text_mask = model.language_encoder.get_extended_attention_mask(text_mask, text_mask.size())     
        ext_ecg_mask = model.language_encoder.get_extended_attention_mask(ecg_mask, ecg_mask.size())

        all_cross_attn = []
        for text_layer, ecg_layer in zip(model.multi_modal_language_layers, model.multi_modal_ecg_layers):
            text_outputs = text_layer(x, y, attention_mask=ext_text_mask, encoder_attention_mask=ext_ecg_mask, output_attentions=True)
            ecg_outputs = ecg_layer(y, x, attention_mask=ext_ecg_mask, encoder_attention_mask=ext_text_mask, output_attentions=True)

            cross_attn = ecg_outputs[2].squeeze(0)
            cross_attn = F.softmax(cross_attn / 0.1, dim=2) 
            all_cross_attn.append(cross_attn)
            x, y = text_outputs[0], ecg_outputs[0]

    stacked = torch.stack(all_cross_attn, dim=0).mean(dim=0).mean(dim=0) 
    valid_text_len = int(text_mask.sum().item())
    ecg_timeline_heatmap = stacked[:, :valid_text_len].mean(dim=1) 
    heatmap = ecg_timeline_heatmap[1:] 
    return heatmap.cpu()


def heatmap_to_intervals(heatmap: torch.Tensor, debug_label: str | None = None, verbose: bool = False):
    n_tokens = len(heatmap)
    h = heatmap.float()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    h_np = h.numpy()

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
        print(f"    [debug]{label} entropy_ratio={entropy_ratio:.3f}  peak={global_peak_val:.3f}  baseline(median)={baseline:.3f}  prominence={global_prominence:.3f}  threshold={PROMINENCE_THRESHOLD:.3f}  -> {verdict}")

    if diffuse: return [], True

    peak_indices, _properties = find_peaks(h_np, height=baseline + PROMINENCE_THRESHOLD, prominence=PROMINENCE_THRESHOLD, distance=MIN_PEAK_DISTANCE_TOKENS)
    if verbose:
        label = f" ({debug_label})" if debug_label else ""
        print(f"    [debug]{label} find_peaks located {len(peak_indices)} peak(s) (min_distance={MIN_PEAK_DISTANCE_TOKENS} tokens)")

    if len(peak_indices) == 0: return [], True

    widths, _width_heights, left_ips, right_ips = peak_widths(h_np, peak_indices, rel_height=PEAK_WIDTH_REL_HEIGHT)

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

    intervals.sort(key=lambda t: t[0])
    merged = [list(intervals[0])]
    for start_ms, end_ms, peak_val in intervals[1:]:
        if start_ms <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_ms)
            merged[-1][2] = max(merged[-1][2], peak_val)
        else:
            merged.append([start_ms, end_ms, peak_val])

    merged = sorted(merged, key=lambda t: t[2], reverse=True)[:MAX_INTERVALS_PER_OBS]
    merged.sort(key=lambda t: t[0])
    return [tuple(m) for m in merged], False


def pick_dominant_lead(ecg: np.ndarray, start_ms: int, end_ms: int, text_phrase: str = "") -> int:
    start_sample = int(start_ms / 1000 * SAMPLE_RATE)
    end_sample   = int(end_ms   / 1000 * SAMPLE_RATE)
    segment = ecg[start_sample:end_sample, :]
    variances = segment.var(axis=0)
    
    text_lower = text_phrase.lower()
    targeted_leads = []
    for idx, name in enumerate(LEAD_NAMES):
        if name.lower() in text_lower: targeted_leads.append(idx)
            
    if targeted_leads:
        sub_variances = [(variances[i], i) for i in targeted_leads]
        return max(sub_variances, key=lambda item: item[0])[1]
        
    return int(np.argmax(variances))


# ─────────────────────────────────────────────────────────────────────────────
# Dedicated Feature Canvas Visualizer (High-Resolution MS Axes)
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_feature_heatmap(ecg_data: np.ndarray, 
                                intervals: list, 
                                feature_label: str,
                                plot_color: str = 'crimson'):
    """
    Plots a 12-lead ECG signal stack in millisecond x-coordinates, overlaying 
    translucent highlight bands for a single isolated observation string.
    """
    if ecg_data.shape[0] != 12 and ecg_data.shape[1] == 12:
        ecg_data = ecg_data.T
        
    num_leads, signal_length = ecg_data.shape
    
    # NEW: Establish timeline completely in milliseconds (0 to 10000 ms)
    total_duration_ms = int((signal_length / SAMPLE_RATE) * 1000)
    time_axis_ms = np.linspace(0, total_duration_ms, signal_length)
    
    fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(15, 16), sharex=True)
    fig.suptitle(f"ZETA Fine Grounding Layer\nFeature Target: \"{feature_label}\"", 
                 fontsize=14, fontweight='bold', y=0.96)
    
    for idx, ax in enumerate(axes):
        # Plot signal using milliseconds timeline
        ax.plot(time_axis_ms, ecg_data[idx], color='black', linewidth=0.9)
        ax.set_ylabel(LEAD_NAMES[idx], rotation=0, labelpad=15, fontsize=11, fontweight='bold', va='center')
        
        # Style grid lanes to match a high-resolution sub-second tracking sheet
        ax.grid(True, which='major', color='darkgray', linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.3)
        ax.tick_params(axis='y', labelsize=8)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        # Draw translucent spans using native ms coordinates
        if intervals:
            for start_ms, end_ms, *_ in intervals:
                ax.axvspan(
                    xmin=start_ms, 
                    xmax=end_ms, 
                    color=plot_color, 
                    alpha=0.22,
                    zorder=0
                )

    # NEW: Configure granular Millisecond ticks on the shared X-Axis
    # Places major markings every 500ms and minor alignment lines every 100ms
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(500))
    axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(100))
    
    axes[-1].set_xlabel("Time (milliseconds)", fontsize=12, fontweight='bold')
    axes[-1].set_xlim(0, total_duration_ms)
    
    # Rotate markings to ensure tight sub-second values remain perfectly readable
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    # Add a clean single top indicator legend bar
    legend_bar = [plt.Rectangle((0, 0), 1, 1, color=plot_color, alpha=0.4, label=f"Detected Active Region")]
    axes[0].legend(handles=legend_bar, loc='upper right', bbox_to_anchor=(1.0, 1.9), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Execution runtime path
# ─────────────────────────────────────────────────────────────────────────────

def strength_label(probability_score: float, is_positive: bool) -> tuple[str, str]:
    effective = probability_score if is_positive else (1.0 - probability_score)
    if effective >= 0.70: return "████████", "highlight strongly"
    elif effective >= 0.58: return "██████  ", "highlight moderately"
    elif effective >= 0.52: return "████    ", "highlight weakly"
    else: return "░░      ", "suppress"

def format_locations(locs) -> str:
    if not locs: return "diffuse"
    MAX_SHOWN = 2
    parts = [f"{s}ms-{e}ms ({LEAD_NAMES[l]})" for s, e, l in locs[:MAX_SHOWN]]
    remaining = len(locs) - MAX_SHOWN
    text = ", ".join(parts)
    if remaining > 0: text += f" +{remaining} more"
    return text

def run(ecg: np.ndarray, condition: str, model: M3AEModel, tokenizer: T5TokenizerFast,
        observations: dict, device: torch.device, ground_truth: dict | None = None, verbose: bool = False):

    if condition not in observations: raise ValueError(f"Condition '{condition}' not found.")

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
    similarities_p, similarities_n = [], []
    for i in range(n_pairs):
        sim_p = (ecg_vec @ F.normalize(pos_vecs[i], dim=0).reshape(1, -1).T)[0].item()
        sim_n = (ecg_vec @ F.normalize(neg_vecs[i], dim=0).reshape(1, -1).T)[0].item()
        similarities_p.append(sim_p); similarities_n.append(sim_n)

    pos_probs_global = []
    for i in range(n_pairs):
        l_p, l_n = similarities_p[i] / SOFTMAX_TEMP, similarities_n[i] / SOFTMAX_TEMP
        mx = max(l_p, l_n)
        pos_probs_global.append(math.exp(l_p - mx) / (math.exp(l_p - mx) + math.exp(l_n - mx)))
    
    final_score = float(np.mean(pos_probs_global))
    pos_probabilities, neg_probabilities = [], []
    for i in range(n_pairs):
        logit_p, logit_n = similarities_p[i] / VISUAL_TEMP, similarities_n[i] / VISUAL_TEMP
        mx = max(logit_p, logit_n)
        total = math.exp(logit_p - mx) + math.exp(logit_n - mx)
        pos_probabilities.append(math.exp(logit_p - mx) / total)
        neg_probabilities.append(math.exp(logit_n - mx) / total)

    def localise(text_seq, text_mask, text_phrase, debug_label=None):
        heatmap = get_cross_attention_heatmap(model, text_seq, text_mask, ecg_seq, ecg_mask, device)
        intervals, diffuse = heatmap_to_intervals(heatmap, debug_label=debug_label, verbose=verbose)
        if diffuse or not intervals: return None
        return [(s, e, pick_dominant_lead(ecg, s, e, text_phrase=text_phrase)) for s, e, _ in intervals]

    if verbose: print("\n  [debug] localization diagnostics:")
    pos_locs = [localise(s, m, pos_texts[i], debug_label=f"P: {pos_texts[i]}") for i, (s, m) in enumerate(zip(pos_seqs, pos_masks))]
    neg_locs = [localise(s, m, neg_texts[i], debug_label=f"N: {neg_texts[i]}") for i, (s, m) in enumerate(zip(neg_seqs, neg_masks))]

    print()
    print(f"Condition: {condition}  (final score: {final_score:.2f})\n")
    col_obs, col_scr, col_loc = 45, 7, 40
    print(f"  {'Observation':<{col_obs}}{'Score':>{col_scr}}  {'Localization(s)':<{col_loc}}Strength")
    print("  " + "─" * (col_obs + col_scr + col_loc + 30))

    for i in range(n_pairs):
        format_row = lambda t, txt, s, l, p: print(f"  [{t}] {txt:<{col_obs-4}}{s:>{col_scr}.2f}  {format_locations(l):<{col_loc}}{strength_label(s, p)[0]}  {strength_label(s, p)[1]}")
        format_row("P", pos_texts[i], pos_probabilities[i], pos_locs[i], True)
    print()
    for i in range(n_pairs):
        format_row("N", neg_texts[i], neg_probabilities[i], neg_locs[i], False)
    print()

    # Palette to distribute single clear tones to independent graphs
    palette = ['crimson', 'dodgerblue', 'forestgreen', 'darkorchid', 'darkorange', 'teal']

    # NEW: Loop through positive observations and spawn completely isolated figures
    for i, label_phrase in enumerate(pos_texts):
        if pos_locs[i] is not None:
            plot_single_feature_heatmap(
                ecg_data=ecg,
                intervals=pos_locs[i],
                feature_label=label_phrase,
                plot_color=palette[i % len(palette)]
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point file loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_ptbxl_record(ptbxl_root: str, filename_hr: str) -> np.ndarray:
    path = str(Path(ptbxl_root) / filename_hr)
    signal, _ = wfdb.rdsamp(path)
    ecg = signal.astype(np.float32)[:5000, :]
    return (ecg - ecg.min(axis=0)) / (ecg.max(axis=0) - ecg.min(axis=0) + 1e-8)

def load_ptbxl_db(ptbxl_root: str):
    return pd.read_csv(Path(ptbxl_root) / "ptbxl_database.csv", index_col="ecg_id"), pd.read_csv(Path(ptbxl_root) / "scp_statements.csv", index_col=0)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    ecg_group = parser.add_mutually_exclusive_group(required=True)
    ecg_group.add_argument("--filename_hr")
    ecg_group.add_argument("--ecg_id", type=int)
    parser.add_argument("--ptbxl_root", default=os.getenv("PTBXL_DATASET"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--ground_truth", action="store_true")
    parser.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH"))
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--observations", default="configs/observations.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.ecg_id is not None:
        db, _ = load_ptbxl_db(args.ptbxl_root)
        filename_hr = db.loc[args.ecg_id, "filename_hr"]
    else:
        filename_hr = args.filename_hr

    ecg = load_ptbxl_record(args.ptbxl_root, filename_hr)
    model = load_model(args.config, args.checkpoint, device)
    tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)

    with open(args.observations) as f: observations = json.load(f)
    run(ecg, args.condition, model, tokenizer, observations, device, verbose=args.debug)

if __name__ == "__main__":
    main()