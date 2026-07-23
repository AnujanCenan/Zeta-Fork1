"""
zeta_localise_multi_figs.py

Extends ZETA inference for a single ECG to produce:
  - Per-observation similarity scores (from the unimodal dot-product path)
  - Temporal localization INTERVALS (plural) for each observation.
  - Multi-plot vertical tracking where each positive observation feature is 
    rendered on its own independent figure canvas showing ONLY relevant channels
    annotated with exact millisecond bounds positioned ABOVE the plot area.
  - Model-intrinsic multi-lead attribution mapping.
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
#
# FIX: lowered back from 0.30 toward 0.15. 0.30 was compensating for the
# artificial peakiness introduced by the now-removed softmax(x/0.1)
# re-sharpening (fix 2 above). With that distortion gone, the true
# entropy/prominence values on real attention are unverified against this
# threshold -- treat 0.15 as a starting point and use --debug to check
# actual prominence values on your data before trusting it.
PROMINENCE_THRESHOLD      = 0.15
MIN_INTERVAL_TOKENS       = 3      # restored from 1 -> 3 (~96ms) as a sane floor

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
# Cross-attention localization and Multi-Lead Selection Matrix
# ─────────────────────────────────────────────────────────────────────────────

def pick_relevant_leads(model: M3AEModel,
                        text_seq_features: torch.Tensor,
                        text_mask: torch.Tensor,
                        ecg_data: np.ndarray,
                        start_ms: int,
                        end_ms: int,
                        device: torch.device,
                        text_phrase: str = "",
                        rel_threshold: float = 0.85) -> list[int]:
    text_lower = text_phrase.lower()
    explicit_leads = [idx for idx, name in enumerate(LEAD_NAMES) if name.lower() in text_lower]
    
    if explicit_leads:
        return explicit_leads

    start_sample = int(start_ms / 1000 * SAMPLE_RATE)
    end_sample   = int(end_ms   / 1000 * SAMPLE_RATE)
    segment = ecg_data[start_sample:end_sample, :]
    
    variances = segment.var(axis=0)
    max_variance = np.max(variances)
    
    if max_variance == 0:
        return [0]
        
    relevant_indices = [idx for idx, v in enumerate(variances) if v >= (max_variance * rel_threshold)]
    return relevant_indices


def get_cross_attention_heatmap(model: M3AEModel,
                                text_seq: torch.Tensor,
                                text_mask: torch.Tensor,
                                ecg_seq: torch.Tensor,
                                ecg_mask: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
    """
    Expected dimensions:
    - text_seq: (1, 128, 768)
    - text_mask: (1, 128)
    - ecg_seq: (1, 313, 768)
    - ecg_mask (1, 313)
    """
    # FIX 1: modality ids must match the convention the model was actually
    # trained with (see cmelt.py forward): text tokens = modality 0,
    # ECG tokens = modality 1. Previously both were set to `ones`, tagging
    # the text stream as if it were ECG.
    text_ids_dummy = torch.zeros((1, text_seq.size(1)), dtype=torch.long, device=device)
    ecg_ids_dummy = torch.ones((1, ecg_seq.size(1)), dtype=torch.long, device=device)

    with torch.no_grad():
        x = text_seq + model.modality_type_embeddings(text_ids_dummy)
        y = ecg_seq  + model.modality_type_embeddings(ecg_ids_dummy)

        ext_text_mask = model.language_encoder.get_extended_attention_mask(text_mask, text_mask.size())
        ext_ecg_mask = model.language_encoder.get_extended_attention_mask(ecg_mask, ecg_mask.size())

        all_cross_attn = []
        # Both branches are advanced together every layer, exactly mirroring
        # M3AEModel.forward in cmelt.py (the ECG branch has its own
        # cross-attention layer cross-attending back onto the text branch;
        # leaving it frozen would mean later layers attend against ECG
        # representations the model never saw associated with that layer's
        # text features during training).
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

            # FIX 2 + FIX 3:
            #  - Use the TEXT branch's cross-attention (text token = query,
            #    ECG tokens = key/value). Shape: [1, heads, text_len, ecg_len].
            #    This forces the softmax to discriminate across the full
            #    312-token ECG timeline FOR THIS SPECIFIC TEXT, which is
            #    what actually makes different observations produce
            #    different heatmaps (the reversed, ecg-as-query direction
            #    only has to discriminate over a handful of text tokens,
            #    and is much less text-specific).
            #  - `text_outputs[2]` is already softmax-normalized by the
            #    underlying BertAttention module -- no extra
            #    softmax(x / temperature) is applied on top of it. Doing so
            #    (as the previous version did, with temperature 0.1) takes
            #    an already-normalized distribution and re-sharpens tiny,
            #    non-meaningful differences into near-total certainty.
            cross_attn = text_outputs[2]
            all_cross_attn.append(cross_attn)

            # advance both branches together (co-evolution fix)
            x, y = text_outputs[0], ecg_outputs[0]

    # Average across layers and attention heads
    stacked = torch.stack(all_cross_attn, dim=0).mean(dim=0).squeeze(0)  # [heads, text_len, ecg_len]
    stacked = stacked.mean(dim=0)  # [text_len, ecg_len]

    valid_text_len = int(text_mask.sum().item())
    ecg_timeline_heatmap = stacked[:valid_text_len, :].mean(dim=0)  # [ecg_len]

    # Chop off the first element corresponding to the CLS token
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


def contrastive_diff(heatmap_target: torch.Tensor, heatmap_other: torch.Tensor) -> torch.Tensor:
    """
    Given the RAW (pre-min-max-scaled) attention heatmaps for a paired
    positive/negative observation, returns where `heatmap_target` exceeds
    `heatmap_other`, clipped at zero.

    Rationale: since each `get_cross_attention_heatmap` output is the mean
    of several per-text-token softmax distributions over the 312 ECG
    tokens, both P and N heatmaps individually sum to roughly 1 across the
    ECG timeline, so a direct subtraction is a fair, like-for-like
    comparison (not an artifact of one being scaled differently). If the
    fusion layers' cross-attention is dominated by generic ECG-side
    salience (e.g. QRS complexes) rather than by the specific text query --
    which is what our CRBBB debug run suggested (all 6 P/N observations
    landing on ~identical intervals with an identical peak count) -- that
    shared generic component should mostly cancel out in the subtraction,
    leaving (if anything survives) whatever small residual is actually
    conditioned on this text versus its paired opposite.
    """
    diff = heatmap_target - heatmap_other
    return torch.clamp(diff, min=0.0)



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


# ─────────────────────────────────────────────────────────────────────────────
# Dedicated Feature Canvas Visualizer (External Time Labels Above Plot Area)
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_feature_heatmap(ecg_data: np.ndarray, 
                                intervals: list, 
                                relevant_leads: list[int],
                                feature_label: str,
                                plot_color: str = 'crimson'):
    """
    Plots an ECG stack containing ONLY relevant leads, with text annotations 
    positioned safely outside and ABOVE the coordinate box area.
    """
    if ecg_data.shape[0] != 12 and ecg_data.shape[1] == 12:
        ecg_data = ecg_data.T
        
    num_leads, signal_length = ecg_data.shape
    total_duration_ms = int((signal_length / SAMPLE_RATE) * 1000)
    time_axis_ms = np.linspace(0, total_duration_ms, signal_length)
    
    if not relevant_leads:
        relevant_leads = list(range(12))
        
    num_active_rows = len(relevant_leads)
    
    # Expanded vertical row padding context slightly to safeguard external tags
    dyn_height = max(4, int(num_active_rows * 1.65))
    fig, axes = plt.subplots(nrows=num_active_rows, ncols=1, figsize=(15, dyn_height), sharex=True)
    
    if num_active_rows == 1:
        axes = [axes]
        
    fig.suptitle(f"ZETA Heatmap\nFeature Target: \"{feature_label}\"", 
                 fontsize=13, fontweight='bold', y=0.97)
    
    for row_idx, lead_idx in enumerate(relevant_leads):
        ax = axes[row_idx]
        ax.plot(time_axis_ms, ecg_data[lead_idx], color='black', linewidth=0.9)
        ax.set_ylabel(LEAD_NAMES[lead_idx], rotation=0, labelpad=15, fontsize=11, fontweight='bold', va='center')
        
        ax.grid(True, which='major', color='darkgray', linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.3)
        ax.tick_params(axis='y', labelsize=8)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        if intervals:
            for start_ms, end_ms, *_ in intervals:
                # 1. Shaded baseline boundary segment
                ax.axvspan(xmin=start_ms, xmax=end_ms, color=plot_color, alpha=0.22, zorder=0)
                
                # 2. Anchor position horizontally over mid-point
                text_x = (start_ms + end_ms) / 2
                
                # 3. Position float badge above the top boundary ceiling (y=1.05)
                ax.text(
                    x=text_x, 
                    y=1.05, 
                    s=f"{start_ms}-{end_ms} ms",
                    transform=ax.get_xaxis_transform(), 
                    color='black',
                    fontsize=8,
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    clip_on=False, # Allows text element rendering outside axis window grids
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor=plot_color, linewidth=0.5, boxstyle='round,pad=0.2')
                )

    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(500))
    axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(100))
    
    axes[-1].set_xlabel("Time (milliseconds)", fontsize=12, fontweight='bold')
    axes[-1].set_xlim(0, total_duration_ms)
    
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    legend_bar = [plt.Rectangle((0, 0), 1, 1, color=plot_color, alpha=0.4, label=f"Detected Active Region")]
    axes[0].legend(handles=legend_bar, loc='upper right', bbox_to_anchor=(1.0, 1.8), fontsize=10, frameon=True)

    # Tight layout grid tracking boundary constraints
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Execution runtime path
# ─────────────────────────────────────────────────────────────────────────────

def format_locations(locs) -> str:
    if not locs: return "diffuse"
    MAX_SHOWN = 2
    
    parts = []
    for s, e, lead_list in locs[:MAX_SHOWN]:
        lead_str = ", ".join([LEAD_NAMES[l] for l in lead_list])
        parts.append(f"{s}ms-{e}ms ({lead_str})")
        
    remaining = len(locs) - MAX_SHOWN
    text = "; ".join(parts)
    if remaining > 0: text += f" +{remaining} more"
    return text

def run(ecg: np.ndarray, condition: str, model: M3AEModel, tokenizer: T5TokenizerFast,
        observations: dict, device: torch.device, ground_truth: dict | None = None, verbose: bool = False,
        localise_mode: str = "contrastive"):

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

    def build_locs(text_seq, text_mask, text_phrase, intervals, diffuse):
        if diffuse or not intervals:
            return None
        return [(s, e, pick_relevant_leads(model, text_seq, text_mask, ecg, s, e, device, text_phrase=text_phrase))
                for s, e, _ in intervals]

    def localise_pair(i):
        """
        Computes localization for the i-th P/N pair. In "contrastive" mode
        (the new method being tried here), both raw heatmaps are computed
        first and then differenced (P-minus-N, clipped at zero, and vice
        versa) before peak-finding -- see contrastive_diff() for the
        rationale. In "independent" mode (the previous behavior), each
        text's heatmap is localized on its own, with no reference to its
        paired opposite; kept available via --localise_mode independent
        so the two approaches can be compared directly on the same ECG.
        """
        heatmap_pos_raw = get_cross_attention_heatmap(model, pos_seqs[i], pos_masks[i], ecg_seq, ecg_mask, device)
        heatmap_neg_raw = get_cross_attention_heatmap(model, neg_seqs[i], neg_masks[i], ecg_seq, ecg_mask, device)

        if localise_mode == "contrastive":
            heatmap_for_pos = contrastive_diff(heatmap_pos_raw, heatmap_neg_raw)
            heatmap_for_neg = contrastive_diff(heatmap_neg_raw, heatmap_pos_raw)
            pos_label_suffix, neg_label_suffix = " (vs N)", " (vs P)"
        else:  # "independent" -- original behavior, no differencing
            heatmap_for_pos = heatmap_pos_raw
            heatmap_for_neg = heatmap_neg_raw
            pos_label_suffix, neg_label_suffix = "", ""

        pos_intervals, pos_diffuse = heatmap_to_intervals(
            heatmap_for_pos, debug_label=f"P: {pos_texts[i]}{pos_label_suffix}", verbose=verbose
        )
        neg_intervals, neg_diffuse = heatmap_to_intervals(
            heatmap_for_neg, debug_label=f"N: {neg_texts[i]}{neg_label_suffix}", verbose=verbose
        )

        pos_locs_i = build_locs(pos_seqs[i], pos_masks[i], pos_texts[i], pos_intervals, pos_diffuse)
        neg_locs_i = build_locs(neg_seqs[i], neg_masks[i], neg_texts[i], neg_intervals, neg_diffuse)
        return pos_locs_i, neg_locs_i

    if verbose:
        print(f"\n  [debug] localization diagnostics (mode={localise_mode}):")
    pos_locs, neg_locs = [], []
    for i in range(n_pairs):
        p_locs, n_locs = localise_pair(i)
        pos_locs.append(p_locs)
        neg_locs.append(n_locs)
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
    print()


    print(f"Condition: {condition}  (final score: {final_score:.2f})\n")
    col_obs, col_scr, col_loc = 45, 7, 45
    print(f"  {'Observation':<{col_obs}}{'Score':>{col_scr}}  {'Localization(s)':<{col_loc}}")
    print("  " + "─" * (col_obs + col_scr + col_loc + 30))

    for i in range(n_pairs):
        format_row = lambda t, txt, s, l, p: print(f"  [{t}] {txt:<{col_obs-4}}{s:>{col_scr}.2f}  {format_locations(l):<{col_loc}}")
        format_row("P", pos_texts[i], pos_probabilities[i], pos_locs[i], True)
    print()
    for i in range(n_pairs):
        format_row("N", neg_texts[i], neg_probabilities[i], neg_locs[i], False)
    print()

    palette = ['crimson', 'dodgerblue', 'forestgreen', 'darkorchid', 'darkorange', 'teal']

    for i, label_phrase in enumerate(pos_texts):
        if pos_locs[i] is not None:
            unique_leads = sorted(list(set([l for interval in pos_locs[i] for l in interval[2]])))
            
            plot_single_feature_heatmap(
                ecg_data=ecg,
                intervals=pos_locs[i],
                relevant_leads=unique_leads,
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

    # FIX 4: restored from the working version --
    #  (a) transpose to (12, 5000) so the lead-order correction below
    #      operates on the LEAD axis, not the time axis;
    #  (b) use GLOBAL min-max normalization across the whole 12-lead block,
    #      not per-lead (axis=0) normalization -- per-lead normalization
    #      erases genuine inter-lead amplitude differences that findings
    #      like "wide/dominant/narrow S wave" depend on, and doesn't match
    #      how the model was trained;
    #  (c) restore the aVL/aVF lead-order correction, which was silently
    #      dropped. Confirm this matches your data source's actual lead
    #      ordering if you're unsure why it's needed.
    ecg = ecg.T  # (12, 5000)
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-8)
    ecg[[4, 5]] = ecg[[5, 4]]
    return ecg.T  # back to (5000, 12)

def load_ptbxl_db(ptbxl_root: str):
    return pd.read_csv(Path(ptbxl_root) / "ptbxl_database.csv", index_col="ecg_id"), pd.read_csv(Path(ptbxl_root) / "scp_statements.csv", index_col=0)


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
    parser = argparse.ArgumentParser()
    ecg_group = parser.add_mutually_exclusive_group(required=True)
    ecg_group.add_argument("--filename_hr")
    ecg_group.add_argument("--ecg_id", type=int)
    parser.add_argument("--ptbxl_root", default=os.getenv("PTBXL_DATASET"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH"))
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--observations", default="configs/observations.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ground_truth", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--localise_mode", choices=["contrastive", "independent"], default="contrastive",
                         help="'contrastive' (default) localizes each observation on its P-minus-N "
                              "attention difference (see contrastive_diff()); 'independent' reproduces "
                              "the previous behavior of localizing each text's heatmap on its own, with "
                              "no reference to its paired opposite -- useful for direct before/after comparison.")

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

    ground_truth = None
    if args.ground_truth and args.ecg_id is not None:
        ground_truth = get_ground_truth(args.ptbxl_root, args.ecg_id)

    with open(args.observations) as f: observations = json.load(f)
    run(ecg, args.condition, model, tokenizer, observations, device, ground_truth, verbose=args.debug, localise_mode=args.localise_mode)

if __name__ == "__main__":
    main()
