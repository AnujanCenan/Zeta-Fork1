"""
Diagnostic v4 - try different pooling strategies to find one that
produces meaningful, non-saturated ECG and text embeddings.
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

with open("configs/config.json") as f:
    cfg = SimpleNamespace(**json.load(f)["model"])
model = M3AEModel(cfg)
ck = torch.load(CKPT, map_location="cpu")["model"]
ck.pop("ecg_encoder.mask_emb", None)
model.load_state_dict(ck, strict=True)
model.eval()
device = torch.device("cuda")
model.to(device)

tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base", do_lower_case=True)

# ── load two ECGs: confirmed 1AVB (102) and NORM (1) ─────────────────────────
def load_ecg(ecg_id_path):
    path = str(Path(PTBXL) / ecg_id_path)
    sig, _ = wfdb.rdsamp(path)
    ecg = sig.astype(np.float32)[:5000, :].T
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-8)
    ecg[[4,5]] = ecg[[5,4]]
    return torch.tensor(ecg).unsqueeze(0).to(device)  # (1,12,5000)

ecg_1avb  = load_ecg("records500/00000/00102_hr")   # confirmed 1AVB
ecg_norm  = load_ecg("records500/00000/00001_hr")   # NORM

def get_ecg_intermediates(x):
    with torch.no_grad():
        feats, pmask = model.ecg_encoder.get_embeddings(x, padding_mask=None)
        # feats: (1, 312, 768) — no CLS yet
        cls = model.class_embedding.repeat(1,1,1)
        feats_cls = torch.cat([cls, feats], dim=1)   # (1, 313, 768)
        out = model.ecg_encoder.get_output(feats_cls, pmask)  # (1, 313, 768)
        proj = model.multi_modal_ecg_proj(out)        # (1, 313, 768)
    return out, proj   # pre-proj and post-proj

out_1avb, proj_1avb = get_ecg_intermediates(ecg_1avb)
out_norm, proj_norm = get_ecg_intermediates(ecg_norm)

# ── try pooling strategies ────────────────────────────────────────────────────
strategies = {
    "CLS pre-proj":         lambda out, proj: out[:, 0, :],
    "mean pre-proj":        lambda out, proj: out[:, 1:, :].mean(dim=1),
    "CLS post-proj":        lambda out, proj: proj[:, 0, :],
    "mean post-proj":       lambda out, proj: proj[:, 1:, :].mean(dim=1),
    "CLS post-proj+pooler": lambda out, proj: model.unimodal_ecg_pooler(proj),
}

texts_pos = ["pr interval >200ms", "normal p-wave morphology",
             "stable pr interval duration"]
texts_neg = ["pr interval <200ms", "abnormal p wave morphology",
             "instable pr interval duration"]

print(f"{'Strategy':<25} {'1AVB_std':>10} {'NORM_std':>10} {'cos_sim(1AVB,NORM)':>20} {'avg_sep':>10}")
print("-" * 80)

for name, pool_fn in strategies.items():
    with torch.no_grad():
        v1 = F.normalize(pool_fn(out_1avb, proj_1avb).squeeze(0), dim=0)
        vn = F.normalize(pool_fn(out_norm, proj_norm).squeeze(0), dim=0)

    # std of raw (pre-norm) vector
    raw1 = pool_fn(out_1avb, proj_1avb).squeeze(0)
    std1 = raw1.std().item()

    cos_ecg = (v1 @ vn).item()

    # text similarities
    seps = []
    for pt, nt in zip(texts_pos, texts_neg):
        for t in [pt, nt]:
            enc = tokenizer(t, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                raw_t  = model.language_encoder(input_ids=enc["input_ids"])[0]
                proj_t = model.multi_modal_language_proj(raw_t)
                # use mean of sequence for text (no CLS)
                tv = F.normalize(proj_t[:, 1:, :].mean(dim=1).squeeze(0), dim=0)
            if t == pt:
                sp = (v1 @ tv).item()
            else:
                sn = (v1 @ tv).item()
        seps.append(abs(sp - sn))

    print(f"  {name:<23} {std1:>10.4f} {'—':>10} {cos_ecg:>20.4f} {np.mean(seps):>10.4f}")

# ── now test text strategies similarly ───────────────────────────────────────
print("\n\n--- Text embedding strategies ---")
print(f"{'Strategy':<30} {'std(>200ms)':>12} {'sim_p':>8} {'sim_n':>8} {'sep':>8}")
print("-" * 70)

# use mean-pre-proj for ECG (likely best from above)
with torch.no_grad():
    out_e, proj_e = get_ecg_intermediates(ecg_1avb)
    ecg_v = F.normalize(out_e[:, 1:, :].mean(dim=1).squeeze(0), dim=0)

text_strategies = {
    "CLS pre-proj":   lambda r, p: r[:, 0, :],
    "mean pre-proj":  lambda r, p: r[:, 1:, :].mean(dim=1),
    "CLS post-proj":  lambda r, p: p[:, 0, :],
    "mean post-proj": lambda r, p: p[:, 1:, :].mean(dim=1),
    "max post-proj":  lambda r, p: p.max(dim=1).values,
    "CLS+pooler":     lambda r, p: model.unimodal_language_pooler(p),
}

for name, tfn in text_strategies.items():
    enc_p = tokenizer("pr interval >200ms", truncation=True, return_tensors="pt").to(device)
    enc_n = tokenizer("pr interval <200ms", truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        rp = model.language_encoder(input_ids=enc_p["input_ids"])[0]
        pp = model.multi_modal_language_proj(rp)
        vp = F.normalize(tfn(rp, pp).squeeze(0), dim=0)

        rn = model.language_encoder(input_ids=enc_n["input_ids"])[0]
        pn = model.multi_modal_language_proj(rn)
        vn = F.normalize(tfn(rn, pn).squeeze(0), dim=0)

    raw_std = tfn(rp, pp).squeeze(0).std().item()
    sp = (ecg_v @ vp).item()
    sn = (ecg_v @ vn).item()
    print(f"  {name:<28} {raw_std:>12.4f} {sp:>8.4f} {sn:>8.4f} {abs(sp-sn):>8.4f}")