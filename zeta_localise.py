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

# ZETA Contrastive Scaling Temperature Factor matching the authors' i / 0.5
SOFTMAX_TEMP = 0.5   

# attention heatmap thresholds
DIFFUSE_ENTROPY_THRESHOLD = 0.85   # fraction of max entropy -> call it "diffuse"
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
    state.pop("ecg_encoder.mask_emb", None)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ECG encoding (Using Native Trained Pooler)
# ─────────────────────────────────────────────────────────────────────────────

def encode_ecg(model: M3AEModel, ecg: np.ndarray, device: torch.device):
    x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(device)
    x = x.permute(0, 2, 1)

    with torch.no_grad():
        feats, padding_mask = model.ecg_encoder.get_embeddings(x, padding_mask=None)
        cls_emb = model.class_embedding.repeat(len(feats), 1, 1)
        feats_with_cls = torch.cat([cls_emb, feats], dim=1)
        feats_out = model.ecg_encoder.get_output(feats_with_cls, padding_mask)
        
        # Project sequence into the shared multimodal space
        proj = model.multi_modal_ecg_proj(feats_out)
        
        # Use the native trained pooler module layer
        ecg_vec = model.unimodal_ecg_pooler(proj).squeeze(0)
        ecg_vec = F.normalize(ecg_vec, dim=0)

    Lx_plus1 = feats_out.size(1)
    ecg_mask = torch.ones((1, Lx_plus1), dtype=torch.long, device=device)

    return ecg_vec, proj.detach(), ecg_mask


# ─────────────────────────────────────────────────────────────────────────────
# Text encoding (Using Native Trained Pooler)
# ─────────────────────────────────────────────────────────────────────────────

def encode_text(model: M3AEModel,
                tokenizer: T5TokenizerFast,
                text: str,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc = tokenizer(text, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        raw  = model.language_encoder(input_ids=enc["input_ids"])[0]
        
        # Project sequence into the shared multimodal space
        proj = model.multi_modal_language_proj(raw)
        
        # Use the native trained pooler module layer
        text_vec = model.unimodal_language_pooler(proj).squeeze(0)
        text_vec = F.normalize(text_vec, dim=0)

    return text_vec, proj.detach(), enc["attention_mask"]


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
            cross_attn = outputs[2]
            all_cross_attn.append(cross_attn)
            x = outputs[0]

    stacked = torch.stack(all_cross_attn, dim=0)
    heatmap_with_cls = stacked.mean(dim=0).mean(dim=0).mean(dim=0).mean(dim=0)
    heatmap = heatmap_with_cls[1:]

    return heatmap.cpu()


def heatmap_to_interval(heatmap: torch.Tensor, lead_axis: int = 1):
    n_tokens = len(heatmap)
    h = heatmap.float()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    probs = h / (h.sum() + 1e-8)
    entropy = -(probs * (probs + 1e-8).log()).sum().item()
    max_entropy = math.log(n_tokens)
    diffuse = (entropy / max_entropy) > DIFFUSE_ENTROPY_THRESHOLD

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
    """
    Determines UI highlight levels using absolute binary probability thresholds.
    """
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

    # 1. Encode Modalities using native trained model poolers
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
    
    # 2. Compute Temperature-Scaled Pairwise Softmax
    pos_probabilities = []
    neg_probabilities = []
    
    for i in range(n_pairs):
        sim_p = (ecg_vec @ pos_vecs[i]).item()
        sim_n = (ecg_vec @ neg_vecs[i]).item()
        
        # Scale by target temperature parameters
        logit_p = sim_p / SOFTMAX_TEMP
        logit_n = sim_n / SOFTMAX_TEMP
        
        max_logit = max(logit_p, logit_n)
        exp_p = math.exp(logit_p - max_logit)
        exp_n = math.exp(logit_n - max_logit)
        total = exp_p + exp_n
        
        pos_probabilities.append(exp_p / total)
        neg_probabilities.append(exp_n / total)

    # Aggregate structural observations using mean score composition
    final_score = float(np.mean(pos_probabilities))

    # 3. Handle Localizations
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

    # 4. Interface Print Dashboard Logs
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