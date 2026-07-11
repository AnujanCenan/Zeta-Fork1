"""
Diagnostic v3 - inspect checkpoint weights directly
Run from Zeta/ directory.
"""
import json, torch, numpy as np
import torch.nn.functional as F
from types import SimpleNamespace
from models.cmelt import M3AEModel
from dotenv import load_dotenv
import os

load_dotenv()
CKPT = os.getenv("CHECKPOINT_PATH")

ck = torch.load(CKPT, map_location="cpu")["model"]
ck.pop("ecg_encoder.mask_emb", None)

# ── inspect pooler weights ────────────────────────────────────────────────────
print("=== Pooler weight stats ===")
for key in sorted(ck.keys()):
    if "pooler" in key or "class_embed" in key:
        t = ck[key]
        print(f"  {key}: shape={list(t.shape)}  mean={t.mean():.4f}  std={t.std():.4f}  min={t.min():.4f}  max={t.max():.4f}")

# ── check if unimodal poolers are identical (would indicate untrained) ────────
print("\n=== Are unimodal poolers identical? ===")
ecg_w  = ck["unimodal_ecg_pooler.dense.weight"]
lang_w = ck["unimodal_language_pooler.dense.weight"]
print(f"  ECG pooler weight == Lang pooler weight: {torch.allclose(ecg_w, lang_w)}")
print(f"  ECG pooler weight std:  {ecg_w.std():.6f}")
print(f"  Lang pooler weight std: {lang_w.std():.6f}")

# ── check class_embedding ─────────────────────────────────────────────────────
print("\n=== class_embedding ===")
cls = ck["class_embedding"]
print(f"  shape={list(cls.shape)}  mean={cls.mean():.4f}  std={cls.std():.4f}")
print(f"  first 10: {cls.squeeze()[:10].tolist()}")

# ── load model and inspect what CLS token looks like after transformer ────────
print("\n=== CLS token after transformer (first 10 dims) ===")
with open("configs/config.json") as f:
    cfg = SimpleNamespace(**json.load(f)["model"])
model = M3AEModel(cfg)
model.load_state_dict(ck, strict=True)
model.eval()

import wfdb
from pathlib import Path
PTBXL = os.getenv("PTBXL_DATASET")
path = str(Path(PTBXL) / "records500/00000/00102_hr")
sig, _ = wfdb.rdsamp(path)
ecg = sig.astype(np.float32)[:5000, :].T
ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-8)
ecg[[4,5]] = ecg[[5,4]]

x = torch.tensor(ecg).unsqueeze(0)  # (1,12,5000) cpu
with torch.no_grad():
    feats, pmask = model.ecg_encoder.get_embeddings(x, padding_mask=None)
    print(f"  feats after conv+transformer (pre-CLS): shape={feats.shape}")
    print(f"  feats[0,0,:10] = {feats[0,0,:10].tolist()}")

    cls_emb = model.class_embedding.repeat(1,1,1)
    print(f"\n  class_embedding[:10] = {cls_emb[0,0,:10].tolist()}")

    feats_with_cls = torch.cat([cls_emb, feats], dim=1)
    feats_out = model.ecg_encoder.get_output(feats_with_cls, pmask)
    print(f"\n  CLS token after transformer[:10] = {feats_out[0,0,:10].tolist()}")

    proj = model.multi_modal_ecg_proj(feats_out)
    print(f"\n  CLS token after proj[:10] = {proj[0,0,:10].tolist()}")

    pooled = model.unimodal_ecg_pooler(proj)
    print(f"\n  After pooler (Tanh)[:10] = {pooled[0,:10].tolist()}")
    print(f"  Pooled std across 768 dims: {pooled.std():.6f}")

    normed = F.normalize(pooled.squeeze(0), dim=0)
    print(f"\n  After L2 norm[:10] = {normed[:10].tolist()}")
    print(f"  Normed std: {normed.std():.6f}")