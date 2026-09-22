"""Fair CNN-1 baseline on the enriched labels.

CNN-1's shipped thresholds were tuned against the ORIGINAL valid set, whose gold
density is 3.4 concepts/image.  Applying them unchanged to a 5.1-concept enriched
gold handicaps CNN-1: it emits ~2.1 concepts/image and is scored against ~5.1.

This re-tunes tau on the ENRICHED valid set (weights frozen -- only the operating
point moves) and re-evaluates on enriched test.  It separates "retraining on
enriched labels helped" from "simply predicting more concepts helped", which is
the control CNN-2 must be measured against.
"""
import os, json, numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import transforms as T
from roco_cui_classifier import CUINet, samples_f1, tune_global_threshold, tune_per_label

torch.set_num_threads(int(os.environ.get("NT", "2")))
ENR = "/home/matei/rocov2_enriched"
th  = json.load(open("cui_clf_ckpt/thresholds.json"))
vocab = th["vocab"]; C = len(vocab); S_IMG = int(th.get("img_size", 224))
vidx = {c: i for i, c in enumerate(vocab)}

m = CUINet(C); m.load_state_dict(torch.load("cui_clf_ckpt/best.pt", map_location="cpu")); m.eval()
tf = T.Compose([T.Resize(int(S_IMG*1.14)), T.CenterCrop(S_IMG), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def score_split(split):
    gold = {str(r.ID): set(str(r.CUIs).split(';'))
            for r in pd.read_csv(f"{ENR}/{split}_concepts.csv").itertuples()}
    ids = sorted(gold)
    Sm = np.zeros((len(ids), C), dtype=np.float32)
    Gm = np.zeros((len(ids), C), dtype=bool)
    B = 32
    for s in range(0, len(ids), B):
        chunk = ids[s:s+B]; imgs = []
        for i in chunk:
            try: imgs.append(tf(Image.open(f"rocov2/{split}/{i}.jpg").convert("RGB")))
            except Exception: imgs.append(torch.zeros(3, S_IMG, S_IMG))
        with torch.no_grad():
            Sm[s:s+len(chunk)] = torch.sigmoid(m(torch.stack(imgs))).numpy()
        for j, i in enumerate(chunk):
            for c in gold[i] & set(vocab): Gm[s+j, vidx[c]] = True
        if s % 1600 == 0: print(f"  {split}: {s}/{len(ids)}", flush=True)
    return Sm, Gm

print("scoring enriched VALID (to re-tune tau) ...", flush=True)
Sv, Gv = score_split("valid")
print("scoring enriched TEST  (to evaluate)   ...", flush=True)
St, Gt = score_split("test")
# persist the score matrices so any further analysis is free
np.savez_compressed("cnn1_enriched_scores.npz", Sv=Sv, Gv=Gv, St=St, Gt=Gt)
print("  saved score matrices -> cnn1_enriched_scores.npz", flush=True)

tau_old = np.asarray(th["per_label_tau"])
f_old = samples_f1(St >= tau_old, Gt)

tg, f_g_val = tune_global_threshold(Sv, Gv)
tau_new, f_val_pl = tune_per_label(Sv, Gv, tg)
f_new_g  = samples_f1(St >= tg, Gt)
f_new_pl = samples_f1(St >= tau_new, Gt)

out = {"frozen_tau_test_f1": float(f_old),
       "recal_global_tau": float(tg), "recal_global_test_f1": float(f_new_g),
       "recal_perlabel_test_f1": float(f_new_pl),
       "recal_valid_f1_global": float(f_g_val), "recal_valid_f1_perlabel": float(f_val_pl),
       "pred_per_img_frozen": float((St >= tau_old).sum(1).mean()),
       "pred_per_img_recal": float((St >= tau_new).sum(1).mean()),
       "gold_per_img": float(Gt.sum(1).mean())}
json.dump(out, open("cnn1_recalibrated.json", "w"), indent=1)
print("\n=== CNN-1 on ENRICHED test ===")
print(f"  frozen tau (old baseline)    : {f_old:.4f}   [{out['pred_per_img_frozen']:.2f} pred/img]")
print(f"  recalibrated global tau      : {f_new_g:.4f}")
print(f"  recalibrated per-label tau   : {f_new_pl:.4f}   [{out['pred_per_img_recal']:.2f} pred/img]")
print(f"  enriched gold density        : {out['gold_per_img']:.2f} concepts/img")
