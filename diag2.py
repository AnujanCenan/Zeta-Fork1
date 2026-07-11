"""
Diagnostic v2 — compare CLS pooler vs max pooling for text,
and check raw dot products for both.
Run from Zeta/ directory.
"""
import json, torch, numpy as np, wfdb, math
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
from transformers import T5TokenizerFast
from models.cmelt import M3AEModel
from dotenv import load_dotenv
import os

load_dotenv()
PTBXL = os.getenv("PTBXL_DATASET")
CKPT  = os.getenv("CHECKPOINT_PATH")

# ── load model ───────────────────────────────────────────────────────────────
with open("configs/config.json") as f:
    cfg = SimpleNamespace(**json.load(f)["model"])
model = M3AEModel(cfg)
ck = torch.load(CKPT, map_location="cpu")["model"]
ck.pop("ecg_encoder.mask_emb", None)
model.load_state_dict(ck, strict=True)
model.eval()
device = torch.device("cuda")
model.to(device)

# ── load ECG 102 (confirmed 1AVB) ────────────────────────────────────────────
path = str(Path(PTBXL) / "records500/00000/00102_hr")
sig, _ = wfdb.rdsamp(path)
ecg = sig.astype(np.float32)[:5000, :].T   # (12, 5000)
ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-8)
ecg[[4,5]] = ecg[[5,4]]

x = torch.tensor(ecg).unsqueeze(0).to(device)  # (1,12,5000)
with torch.no_grad():
    feats, pmask = model.ecg_encoder.get_embeddings(x, padding_mask=None)
    cls = model.class_embedding.repeat(1,1,1)
    feats = torch.cat([cls, feats], dim=1)
    feats = model.ecg_encoder.get_output(feats, pmask)
    proj  = model.multi_modal_ecg_proj(feats)
    ecg_vec = model.unimodal_ecg_pooler(proj).squeeze(0)
    ecg_vec = F.normalize(ecg_vec, dim=0)

print(f"ECG vec norm: {ecg_vec.norm():.4f}")
print(f"ECG first 5: {ecg_vec[:5].cpu().tolist()}")

# ── encode texts with BOTH pooling methods ───────────────────────────────────
tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)

texts = [
    "pr interval >200ms",
    "pr interval <200ms",
    "normal p-wave morphology",
    "abnormal p wave morphology",
]

print("\n{'text': (cls_sim, max_sim)}")
print("-" * 70)

for text in texts:
    enc = tokenizer(text, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        raw  = model.language_encoder(input_ids=enc["input_ids"])[0]
        proj = model.multi_modal_language_proj(raw)

        # CLS pooler (training path)
        cls_vec = model.unimodal_language_pooler(proj).squeeze(0)
        cls_vec = F.normalize(cls_vec, dim=0)
        cls_sim = (ecg_vec @ cls_vec).item()

        # Max pooling (main.py path)
        max_vec, _ = torch.max(proj, dim=1)
        max_vec = F.normalize(max_vec.squeeze(0), dim=0)
        max_sim = (ecg_vec @ max_vec).item()

    print(f"  '{text}'")
    print(f"    CLS pooler sim: {cls_sim:.4f}  | max pool sim: {max_sim:.4f}")

# ── softmax scores for 1AVB pos/neg pair using CLS ───────────────────────────
print("\n--- Softmax scores (τ=0.5) for 1AVB key pair using CLS pooler ---")
pairs = [
    ("pr interval >200ms",  "pr interval <200ms"),
    ("normal p-wave morphology", "abnormal p wave morphology"),
]
for pos_t, neg_t in pairs:
    for t, label in [(pos_t, "pos"), (neg_t, "neg")]:
        enc = tokenizer(t, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            raw  = model.language_encoder(input_ids=enc["input_ids"])[0]
            proj = model.multi_modal_language_proj(raw)
            vec  = model.unimodal_language_pooler(proj).squeeze(0)
            vec  = F.normalize(vec, dim=0)
        if label == "pos":
            sim_p = (ecg_vec @ vec).item()
            p_vec = vec
        else:
            sim_n = (ecg_vec @ vec).item()
    exp_p = math.exp(sim_p / 0.5)
    exp_n = math.exp(sim_n / 0.5)
    score = exp_p / (exp_p + exp_n)
    print(f"  [{pos_t}] vs [{neg_t}]")
    print(f"    sim_p={sim_p:.4f}  sim_n={sim_n:.4f}  → softmax score={score:.4f}")