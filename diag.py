"""
Minimal diagnostic — run from Zeta/ directory.
Checks raw dot-product similarities before softmax to see if
the ECG and text embeddings are actually different from each other.
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

# ── load ECG 102 ─────────────────────────────────────────────────────────────
path = str(Path(PTBXL) / "records500/00000/00102_hr")
sig, _ = wfdb.rdsamp(path)
ecg = sig.astype(np.float32)[:5000, :].T          # (12, 5000)
ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-8)
ecg[[4,5]] = ecg[[5,4]]

# test BOTH orientations
for label, tensor in [
    ("(1,12,5000) leads-first", torch.tensor(ecg).unsqueeze(0).to(device)),
    ("(1,5000,12) time-first",  torch.tensor(ecg.T).unsqueeze(0).to(device)),
]:
    try:
        with torch.no_grad():
            feats, pmask = model.ecg_encoder.get_embeddings(tensor, padding_mask=None)
            cls = model.class_embedding.repeat(1,1,1)
            feats = torch.cat([cls, feats], dim=1)
            feats = model.ecg_encoder.get_output(feats, pmask)
            proj  = model.multi_modal_ecg_proj(feats)
            vec   = model.unimodal_ecg_pooler(proj).squeeze(0)
            vec   = F.normalize(vec, dim=0)
        print(f"\n{label}: OK — ecg_vec norm={vec.norm():.4f}, shape={vec.shape}")
        print(f"  first 5 values: {vec[:5].cpu().tolist()}")
    except Exception as e:
        print(f"\n{label}: FAILED — {e}")

# ── encode two contrasting texts ─────────────────────────────────────────────
tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)

texts = ["pr interval >200ms", "pr interval <200ms"]
vecs = []
for t in texts:
    enc = tokenizer(t, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        raw  = model.language_encoder(input_ids=enc["input_ids"])[0]
        proj = model.multi_modal_language_proj(raw)
        v, _ = torch.max(proj, dim=1)
        v    = F.normalize(v.squeeze(0), dim=0)
    vecs.append(v)
    print(f"\nText '{t}': norm={v.norm():.4f}")
    print(f"  first 5 values: {v[:5].cpu().tolist()}")

print(f"\nDot product between the two text vecs: {(vecs[0]@vecs[1]).item():.4f}")
print("(should be < 1.0 if they're meaningfully different)")

# ── raw similarities for correct orientation ──────────────────────────────────
print("\n--- Raw dot products (leads-first input) ---")
tensor = torch.tensor(ecg).unsqueeze(0).to(device)
with torch.no_grad():
    feats, pmask = model.ecg_encoder.get_embeddings(tensor, padding_mask=None)
    cls = model.class_embedding.repeat(1,1,1)
    feats = torch.cat([cls, feats], dim=1)
    feats = model.ecg_encoder.get_output(feats, pmask)
    proj  = model.multi_modal_ecg_proj(feats)
    ecg_vec = model.unimodal_ecg_pooler(proj).squeeze(0)
    ecg_vec = F.normalize(ecg_vec, dim=0)

for text, v in zip(texts, vecs):
    sim = (ecg_vec @ v).item()
    exp_p = math.exp(sim / 0.5)
    print(f"  sim(ecg, '{text}') = {sim:.4f}")