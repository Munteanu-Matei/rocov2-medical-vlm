#!/usr/bin/env python
# Evaluate the trained CUI classifier on a HELD-OUT split.
#
# The thresholds are FROZEN -- read from thresholds.json exactly as tuned on the
# validation set, never re-tuned here.  Re-tuning on test would reintroduce the
# optimism this script exists to avoid: the val number (0.5586) is measured on the
# same data that chose both the checkpoint and the thresholds, so it is an upper
# bound, not an estimate of generalisation.
#
# The label space also comes from thresholds.json, NOT rebuilt from train_concepts.csv.
# per_label_tau[i] is meaningless unless index i means the same concept it meant during
# tuning, and rebuilding the vocab risks a silent reordering.
#
#   SPLIT=test  python eval_test.py
#   SPLIT=valid python eval_test.py     # sanity check: must reproduce the val number
import os, json, collections
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from roco_cui_classifier import CUINet, RocoCUI, samples_f1, MEAN, STD

ROCO_DIR = os.environ.get("ROCO_DIR", "./rocov2")
OUT_DIR  = os.environ.get("OUT_DIR", "./cui_clf_ckpt")
SPLIT    = os.environ.get("SPLIT", "test")
BATCH    = int(os.environ.get("BATCH", "32"))
WORKERS  = int(os.environ.get("WORKERS", "0"))
CKPT     = os.environ.get("CKPT", os.path.join(OUT_DIR, "best.pt"))
THRESH   = os.environ.get("THRESH", os.path.join(OUT_DIR, "thresholds.json"))
SAVE     = os.environ.get("SAVE_PREDS", "")        # optional CSV of predicted CUIs

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def main():
    th    = json.load(open(THRESH))
    VOCAB = th["vocab"]
    IDX   = {c: i for i, c in enumerate(VOCAB)}
    C     = len(VOCAB)
    tau_pl = np.asarray(th["per_label_tau"], dtype=np.float64)
    tau_g  = float(th["global_tau"])
    size   = int(th.get("img_size", 224))
    assert tau_pl.shape == (C,), f"threshold/vocab mismatch: {tau_pl.shape} vs {C}"
    print(f"device = {device} | split = {SPLIT} | vocab = {C} concepts", flush=True)
    print(f"frozen thresholds from {THRESH}  (global tau={tau_g:.2f})", flush=True)

    # ── gold, keeping the FULL concept set per image, not just in-vocab ──────
    df = pd.read_csv(os.path.join(ROCO_DIR, f"{SPLIT}_concepts.csv"))
    gold_all = {r.ID: [c for c in str(r.CUIs).split(";") if c and c != "nan"]
                for r in df.itertuples()}

    img_dir = os.path.join(ROCO_DIR, SPLIT)
    recs = [{"id": i, "path": os.path.join(img_dir, f"{i}.jpg"),
             "y": [IDX[c] for c in cs if c in IDX]}
            for i, cs in gold_all.items()
            if os.path.isfile(os.path.join(img_dir, f"{i}.jpg"))]
    recs.sort(key=lambda r: r["id"])            # NO shuffle: evaluation is order-free
    print(f"{SPLIT}: {len(recs)} images with an image file "
          f"({len(gold_all) - len(recs)} listed but missing)", flush=True)

    n_gold_full = np.array([len(gold_all[r["id"]]) for r in recs], dtype=np.float64)
    n_gold_voc  = np.array([len(r["y"]) for r in recs], dtype=np.float64)
    oov = n_gold_full.sum() - n_gold_voc.sum()
    print(f"gold concepts: {n_gold_full.sum():.0f} total, "
          f"{oov:.0f} ({100*oov/max(1,n_gold_full.sum()):.1f}%) outside the "
          f"{C}-concept vocabulary and therefore unpredictable", flush=True)

    eval_tf = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size),
                         T.ToTensor(), T.Normalize(MEAN, STD)])
    dl = DataLoader(RocoCUI(recs, eval_tf, C), batch_size=BATCH, shuffle=False,
                    num_workers=WORKERS, pin_memory=(device == "cuda"))

    model = CUINet(C).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    print(f"loaded {CKPT}", flush=True)

    S, G = [], []
    with torch.no_grad():
        for x, y in tqdm(dl, desc="scoring", mininterval=10):
            S.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
            G.append(y.numpy().astype(bool))
    S = np.concatenate(S); G = np.concatenate(G)

    def report(name, pred):
        tp = (pred & G).sum(1).astype(np.float64)
        npd = pred.sum(1).astype(np.float64)
        # in-vocab gold: directly comparable to the validation number
        f_voc = float(np.where(npd + n_gold_voc > 0,
                               2 * tp / np.maximum(npd + n_gold_voc, 1), 1.0).mean())
        # full gold: counts unpredictable out-of-vocab concepts as misses, which is
        # what the ImageCLEF leaderboard measures against
        f_all = float(np.where(npd + n_gold_full > 0,
                               2 * tp / np.maximum(npd + n_gold_full, 1), 1.0).mean())
        print(f"  {name:<22} samples-F1 = {f_voc:.4f} (in-vocab gold) | "
              f"{f_all:.4f} (full gold) | {npd.mean():.2f} concepts/image", flush=True)
        return f_voc, f_all

    print(f"\nresults on {SPLIT} ({len(recs)} images):", flush=True)
    g_voc, g_all = report("global threshold", S >= tau_g)
    p_voc, p_all = report("per-label thresholds", S >= tau_pl)

    out = {"split": SPLIT, "n_images": len(recs), "n_concepts": C,
           "checkpoint": CKPT, "thresholds": THRESH,
           "global_tau": tau_g,
           "f1_global_invocab": g_voc, "f1_global_fullgold": g_all,
           "f1_perlabel_invocab": p_voc, "f1_perlabel_fullgold": p_all,
           "val_f1_perlabel_reference": th.get("val_samples_f1_perlabel")}
    res_p = os.path.join(OUT_DIR, f"{SPLIT}_results.json")
    json.dump(out, open(res_p, "w"), indent=1)
    print(f"\nsaved -> {res_p}", flush=True)

    if SAVE:
        pred = S >= tau_pl
        with open(SAVE, "w") as f:
            f.write("ID,CUIs\n")
            for i, r in enumerate(recs):
                f.write(f"{r['id']},{';'.join(VOCAB[j] for j in np.flatnonzero(pred[i]))}\n")
        print(f"predictions -> {SAVE}", flush=True)

    print("\nImageCLEF 2025 reference (test split): AUEB 1st = 0.5888 | 5th = 0.5225",
          flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
