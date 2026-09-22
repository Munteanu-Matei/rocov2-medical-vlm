"""Does CNN-1 memorise its own training set?

If it does, feeding a TRAINING image back to it reproduces the labels it was
fitted on, so the veto can never discover a NEW label -- it just re-asserts the
original annotation.  Decisive test: samples-F1 on held-in TRAIN images vs the
known held-out VALID F1 (0.5586).  A large gap = memorisation.
"""
import os, json, random, numpy as np, torch, pandas as pd
from PIL import Image
from torchvision import transforms as T
from roco_cui_classifier import CUINet

N = int(os.environ.get("N", "4000"))
th = json.load(open("cui_clf_ckpt/thresholds.json"))
vocab = th["vocab"]; idx = {c: i for i, c in enumerate(vocab)}
tau = np.asarray(th["per_label_tau"]); C = len(vocab); S = int(th.get("img_size", 224))

m = CUINet(C); m.load_state_dict(torch.load("cui_clf_ckpt/best.pt", map_location="cpu")); m.eval()
tf = T.Compose([T.Resize(int(S*1.14)), T.CenterCrop(S), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def run(split, concepts_csv):
    gold = {str(r.ID): set(str(r.CUIs).split(';')) for r in pd.read_csv(concepts_csv).itertuples()}
    ids = sorted(gold); random.seed(0); ids = random.sample(ids, min(N, len(ids)))
    f1s, npred = [], []
    B = 32
    for s in range(0, len(ids), B):
        chunk = ids[s:s+B]; imgs = []
        for i in chunk:
            p = f"rocov2/{split}/{i}.jpg"
            try: imgs.append(tf(Image.open(p).convert("RGB")))
            except Exception: imgs.append(torch.zeros(3, S, S))
        with torch.no_grad():
            sc = torch.sigmoid(m(torch.stack(imgs))).numpy()
        for j, i in enumerate(chunk):
            pred = {vocab[k] for k in np.where(sc[j] >= tau)[0]}
            g = gold[i] & set(vocab)
            npred.append(len(pred))
            if not pred and not g: f1s.append(1.0); continue
            tp = len(pred & g)
            f1s.append(0.0 if tp == 0 else 2*tp/(len(pred)+len(g)))
        if s % 640 == 0: print(f"  {split}: {s}/{len(ids)}", flush=True)
    return float(np.mean(f1s)), float(np.mean(npred))

torch.set_num_threads(2)
print(f"CNN-1, frozen thresholds, n={N} per split\n")
f_tr, p_tr = run("train", "rocov2/train_concepts.csv")
print(f"\n  TRAIN (model was FITTED on these): samples-F1 = {f_tr:.4f} | {p_tr:.2f} concepts/img")
f_va, p_va = run("valid", "rocov2/valid_concepts.csv")
print(f"  VALID (held out)                 : samples-F1 = {f_va:.4f} | {p_va:.2f} concepts/img")
print(f"\n  generalisation gap = {f_tr-f_va:+.4f}")
json.dump({"n":N,"train_f1":f_tr,"valid_f1":f_va,"gap":f_tr-f_va,
           "train_pred_per_img":p_tr,"valid_pred_per_img":p_va},
          open("memorization_test.json","w"), indent=1)
