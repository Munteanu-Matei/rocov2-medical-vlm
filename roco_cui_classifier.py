#!/usr/bin/env python
# Stage 1 -- baseline multi-LABEL CUI classifier for ROCOv2.
#
# WHY MULTI-LABEL, NOT MULTI-CLASS: ROCOv2 images carry 3.35 concepts on average
# (max 21), so concepts must be predicted INDEPENDENTLY -- sigmoid per class + BCE,
# never softmax + cross-entropy (which would force exactly one label per image).
#
# ARCHITECTURE (justified by ImageCLEFmedical 2025 concept-detection results, where
# ImageNet CNNs beat both a medical transformer and an 8B VLM):
#   DenseNet-121 (ImageNet-pretrained, in BOTH top-2 teams)
#     -> GeM pooling (learnable p; AUEB credit it -- interpolates avg<->max so both
#        global properties like modality and small local findings survive)
#     -> Linear(1024 -> n_concepts)
#   Loss: BCEWithLogits + pos_weight  (~3 positives among ~1571 classes)
#   Thresholds: global sweep, then PER-LABEL coordinate ascent (AUEB's winning trick)
#
# PRIMARY METRIC: samples-average F1 -- per-image F1 between predicted and gold
# concept sets, averaged over images. This is ImageCLEF's official metric.
# Reference: AUEB won 2025 with 0.5888; ImageCLEF baseline range ~0.52-0.59.
#
# RUNS ON: cuda | mps (Apple Silicon) | cpu -- auto-detected.
#   pip install torch torchvision pandas pillow tqdm
#
#   smoke : N_TRAIN=2000 EPOCHS=1 python roco_cui_classifier.py
#   full  : python roco_cui_classifier.py 2>&1 | tee cui_clf.log
#   resume: RESUME=1 python roco_cui_classifier.py
import os, json, math, time, random, collections
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

# Torch's default "file_descriptor" sharing strategy burns one FD per tensor passed
# worker -> main.  Across 15 epochs x (train + val) loaders that exhausts the process
# FD limit and DataLoader startup dies with [Errno 24] Too many open files, hours in.
# "file_system" shares through /dev/shm-style files instead and does not accumulate.
torch.multiprocessing.set_sharing_strategy("file_system")

# ── config (env-overridable) ────────────────────────────────────────────────
ROCO_DIR   = os.environ.get("ROCO_DIR", "./rocov2")     # set to your local copy
OUT_DIR    = os.environ.get("OUT_DIR", "./cui_clf_ckpt")
MIN_FREQ   = int(os.environ.get("MIN_FREQ", "10"))      # ImageCLEF's own curation rule -> 1571
IMG_SIZE   = int(os.environ.get("IMG_SIZE", "224"))
BATCH      = int(os.environ.get("BATCH", "32"))
EPOCHS     = int(os.environ.get("EPOCHS", "15"))
LR         = float(os.environ.get("LR", "1e-4"))        # AdamW, ImageCLEF teams use 1e-4..1e-3
WEIGHT_DEC = 1e-4
POSW_CLAMP = float(os.environ.get("POSW_CLAMP", "50"))
N_TRAIN    = int(os.environ.get("N_TRAIN", "0"))        # 0 = all 59,958
N_VAL      = int(os.environ.get("N_VAL", "0"))          # 0 = all 9,904
WORKERS    = int(os.environ.get("WORKERS", "4"))
# Validation defaults to 0 workers ON PURPOSE.  It is only ~310 forward-only
# batches, so the parallelism buys ~1 min/epoch, but spawning a second worker
# pool mid-run is where this script has twice wedged when detached from a
# terminal.  Zero workers = no spawn = nothing to deadlock.  Override if wanted.
VAL_WORKERS = int(os.environ.get("VAL_WORKERS", "0"))
SEED       = int(os.environ.get("SEED", "42"))
RESUME     = os.environ.get("RESUME", "0") == "1"
PATIENCE   = int(os.environ.get("PATIENCE", "4"))

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# ── label space, built from TRAIN ONLY (never valid/test -> no leakage) ──────
def read_concepts(split):
    df = pd.read_csv(os.path.join(ROCO_DIR, f"{split}_concepts.csv"))
    return {r.ID: [c for c in str(r.CUIs).split(";") if c and c != "nan"]
            for r in df.itertuples()}

def build_records(split, id2cuis, limit=0):
    img_dir = os.path.join(ROCO_DIR, split)
    recs = [{"id": i, "path": os.path.join(img_dir, f"{i}.jpg"),
             "y": [IDX[c] for c in cs if c in IDX]}
            for i, cs in id2cuis.items()
            if os.path.isfile(os.path.join(img_dir, f"{i}.jpg"))]
    recs.sort(key=lambda r: r["id"])            # deterministic before shuffling
    random.Random(SEED).shuffle(recs)
    return recs[:limit] if limit else recs

# ── data ────────────────────────────────────────────────────────────────────
from torchvision import transforms as T
from torchvision.models import densenet121, DenseNet121_Weights

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]     # ImageNet stats
# NOTE: deliberately NO horizontal flip. Laterality is clinically meaningful and the
# label space itself contains directionality concepts (AP / PA / sagittal) -- mirroring
# an image would make its label wrong.
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([
    T.Resize(int(IMG_SIZE * 1.14)), T.CenterCrop(IMG_SIZE),
    T.ToTensor(), T.Normalize(MEAN, STD),
])

class RocoCUI(Dataset):
    # n_classes is passed IN rather than read from a global: with the "spawn"
    # start method (macOS default) DataLoader workers re-import this module
    # WITHOUT running main(), so module-level globals set in main() do not exist
    # there.  Anything __getitem__ needs must travel with the pickled instance.
    def __init__(self, recs, tf, n_classes):
        self.recs, self.tf, self.C = recs, tf, n_classes
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        try:
            img = Image.open(r["path"]).convert("RGB")   # radiographs are 1-channel
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        y = torch.zeros(self.C); y[r["y"]] = 1.0
        return self.tf(img), y

# ── model: DenseNet-121 + GeM + linear head ─────────────────────────────────
class GeM(nn.Module):
    """Generalised-mean pooling:  (mean(x^p))^(1/p), with p LEARNED.
    p=1 -> average pooling, p->inf -> max pooling. Lets the net choose how much to
    favour strong local activations (small findings) vs global context (modality)."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(p)); self.eps = eps
    def forward(self, x):
        # p is CLAMPED: nothing in the loss stops the learned p drifting toward 0,
        # and pow(1/p) then explodes -> NaN weights hours into a run. p>=1 keeps
        # the pooling between average (p=1) and max (large p), which is the range
        # the parameter is meant to interpolate anyway.
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        return F.avg_pool2d(x, x.shape[-2:]).pow(1.0 / p).flatten(1)

class CUINet(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = m.features               # conv stack, ImageNet-pretrained
        self.pool = GeM()
        self.head = nn.Linear(1024, n_out)       # DenseNet-121 final channels = 1024
    def forward(self, x):
        return self.head(self.pool(F.relu(self.features(x), inplace=True)))

# ── metric: samples-average F1 (ImageCLEF's official primary metric) ─────────
def samples_f1(pred_bool, gold_bool):
    """Per-image F1 between predicted and gold concept SETS, averaged over images."""
    tp = (pred_bool & gold_bool).sum(1).astype(np.float64)
    np_ = pred_bool.sum(1); ng = gold_bool.sum(1)
    denom = np_ + ng
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0)   # both empty -> 1
    return float(f1.mean())

@torch.no_grad()
def collect_scores(dl):
    """Return (scores [N,C] as probabilities, gold [N,C] bool)."""
    model.eval(); S, G = [], []
    for x, y in tqdm(dl, desc="scoring", leave=False):
        s = torch.sigmoid(model(x.to(device))).float().cpu().numpy()
        S.append(s); G.append(y.numpy().astype(bool))
    return np.concatenate(S), np.concatenate(G)

def tune_global_threshold(S, G):
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = samples_f1(S >= t, G)
        if f > best_f: best_t, best_f = float(t), f
    return best_t, best_f

def tune_per_label(S, G, tau0, sweeps=2, n_cand=20):
    """Coordinate ascent on per-label thresholds (AUEB's method).
    Optimise one concept's threshold at a time, holding the rest fixed, and loop.
    Incremental: changing tau[c] only affects images where prediction for c flips."""
    n, c_ = S.shape
    tau = np.full(c_, tau0, dtype=np.float64)
    pred = S >= tau
    tp = (pred & G).sum(1).astype(np.float64)
    npd = pred.sum(1).astype(np.float64)
    ng  = G.sum(1).astype(np.float64)
    def score(tp, npd):
        d = npd + ng
        return float(np.where(d > 0, 2 * tp / np.maximum(d, 1), 1.0).mean())
    cur = score(tp, npd)
    for s in range(sweeps):
        for c in tqdm(range(c_), desc=f"threshold sweep {s+1}/{sweeps}", leave=False):
            col, g = S[:, c], G[:, c]
            cands = np.unique(np.quantile(col, np.linspace(0.01, 0.99, n_cand)))
            base_pred = pred[:, c].copy()
            best_t, best_s = tau[c], cur
            for t in cands:
                newp = col >= t
                flip = newp != base_pred
                if not flip.any(): continue
                d_np = np.where(newp[flip], 1.0, -1.0)
                d_tp = np.where(newp[flip] & g[flip], 1.0,
                        np.where((~newp[flip]) & base_pred[flip] & g[flip], -1.0, 0.0))
                tp2, np2 = tp.copy(), npd.copy()
                tp2[flip] += d_tp; np2[flip] += d_np
                sc = score(tp2, np2)
                if sc > best_s: best_t, best_s = float(t), sc
            if best_t != tau[c]:                       # commit the improvement
                newp = col >= best_t
                flip = newp != base_pred
                npd[flip] += np.where(newp[flip], 1.0, -1.0)
                tp[flip]  += np.where(newp[flip] & g[flip], 1.0,
                             np.where((~newp[flip]) & base_pred[flip] & g[flip], -1.0, 0.0))
                pred[:, c] = newp; tau[c] = best_t; cur = best_s
        print(f"  after sweep {s+1}: samples-F1 = {cur:.4f}", flush=True)
    return tau, cur


# ── everything below RUNS; it lives inside main() so that DataLoader worker
#    processes (spawn) can import this module without re-executing training ──
def main():
    global VOCAB, IDX, C, model     # module-level funcs above read these

    print(f"device = {device}", flush=True)

    cui_train, cui_valid = read_concepts("train"), read_concepts("valid")
    freq  = collections.Counter(c for v in cui_train.values() for c in v)
    VOCAB = sorted([c for c, n in freq.items() if n >= MIN_FREQ])
    IDX   = {c: i for i, c in enumerate(VOCAB)}
    C     = len(VOCAB)
    print(f"label space: {C} concepts (>= {MIN_FREQ} occurrences)", flush=True)

    train_recs = build_records("train", cui_train, N_TRAIN)
    val_recs   = build_records("valid", cui_valid, N_VAL)
    print(f"train {len(train_recs)} | valid {len(val_recs)}", flush=True)

    pin = (device == "cuda")
    # persistent_workers: spawn the worker pool ONCE instead of tearing it down and
    # rebuilding it every epoch.  Fewer pipes churned (the [Errno 24] crash happened
    # in exactly that per-epoch respawn) and it skips the spawn cost each epoch.
    keep = (WORKERS > 0)
    train_dl = DataLoader(RocoCUI(train_recs, train_tf, C), batch_size=BATCH, shuffle=True,
                          num_workers=WORKERS, pin_memory=pin, drop_last=True,
                          persistent_workers=keep)
    val_dl   = DataLoader(RocoCUI(val_recs, eval_tf, C), batch_size=BATCH, shuffle=False,
                          num_workers=VAL_WORKERS, pin_memory=pin,
                          persistent_workers=(VAL_WORKERS > 0))

    model = CUINet(C).to(device)

    # pos_weight = #neg/#pos per class, clamped so ultra-rare concepts don't dominate
    cnt = collections.Counter(i for r in train_recs for i in r["y"])
    N   = len(train_recs)
    posw = torch.tensor([min((N - max(1, cnt.get(i, 0))) / max(1, cnt.get(i, 0)), POSW_CLAMP)
                         for i in range(C)], dtype=torch.float32, device=device)
    print(f"pos_weight: min {posw.min():.1f}  max {posw.max():.1f}", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=posw)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    # ── train ───────────────────────────────────────────────────────────────────
    state_p = os.path.join(OUT_DIR, "state.json")
    hist_p  = os.path.join(OUT_DIR, "history.json")
    last_p  = os.path.join(OUT_DIR, "last.pt")
    start_ep, best_f1, bad = 0, -1.0, 0
    hist = []
    # last.pt carries the FULL training state (weights + AdamW moments + cosine LR
    # position); best.pt stays a bare state_dict so inference can load it directly.
    if RESUME and os.path.isfile(state_p):
        st = json.load(open(state_p)); start_ep, best_f1, bad = st["epoch"], st["best_f1"], st["bad"]
        ck = torch.load(last_p, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])    # without this AdamW restarts cold
        sched.load_state_dict(ck["sched"])    # without this cosine LR restarts at LR
        if sched.T_max != EPOCHS:             # EPOCHS is the authority for THIS run
            sched.T_max = EPOCHS
            # CosineAnnealingLR.step() updates the lr RECURSIVELY from its current
            # value, so retargeting T_max alone cannot lift an lr that the old
            # (shorter) schedule already annealed to ~0 -- it just multiplies zero.
            # Re-seed it from the closed form; the recursion is exact from there.
            lr_now = LR * (1 + math.cos(math.pi * start_ep / EPOCHS)) / 2
            for g in optim.param_groups: g["lr"] = lr_now
            print(f"[resume] EPOCHS {st.get('t_max', '?')} -> {EPOCHS}, "
                  f"cosine retargeted (lr {lr_now:.2e})", flush=True)
        if os.path.isfile(hist_p):            # else the earlier epochs are overwritten
            hist = json.load(open(hist_p))
        print(f"[resume] from epoch {start_ep} (best samples-F1 {best_f1:.4f}, "
              f"lr {optim.param_groups[0]['lr']:.2e})", flush=True)
    t0 = time.time()
    for ep in range(start_ep, EPOCHS):
        model.train(); tot, nb = 0.0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {ep+1}/{EPOCHS}", mininterval=10):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            tot += loss.item(); nb += 1
        sched.step()
        S, G = collect_scores(val_dl)
        t_g, f_g = tune_global_threshold(S, G)
        hist.append({"epoch": ep + 1, "train_loss": tot / max(1, nb),
                     "val_samples_f1": f_g, "global_tau": t_g,
                     "hours": (time.time() - t0) / 3600})
        print(f"[epoch {ep+1}] loss {tot/max(1,nb):.4f} | val samples-F1 {f_g:.4f} "
              f"(tau={t_g:.2f}) | best {max(best_f1,0):.4f}", flush=True)
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                    "sched": sched.state_dict()}, last_p)
        if f_g > best_f1:
            best_f1, bad = f_g, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
            print("  [ckpt] new best -> best.pt", flush=True)
        else:
            bad += 1; print(f"  no improvement ({bad}/{PATIENCE})", flush=True)
        json.dump({"epoch": ep + 1, "best_f1": best_f1, "bad": bad, "t_max": EPOCHS},
                  open(state_p, "w"))
        json.dump(hist, open(hist_p, "w"), indent=1)
        if bad >= PATIENCE:
            print("[early stop]", flush=True); break

    # ── final: per-label threshold tuning on validation ─────────────────────────
    print("\nloading best checkpoint and tuning PER-LABEL thresholds ...", flush=True)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt"), map_location=device))
    S, G = collect_scores(val_dl)
    t_g, f_g = tune_global_threshold(S, G)
    print(f"  global threshold   : tau={t_g:.2f}  samples-F1 = {f_g:.4f}", flush=True)
    tau, f_pl = tune_per_label(S, G, t_g)
    print(f"  per-label thresholds: samples-F1 = {f_pl:.4f}  (+{f_pl-f_g:.4f})", flush=True)

    json.dump({"vocab": VOCAB, "min_freq": MIN_FREQ, "img_size": IMG_SIZE,
               "global_tau": t_g, "per_label_tau": tau.tolist(),
               "val_samples_f1_global": f_g, "val_samples_f1_perlabel": f_pl},
              open(os.path.join(OUT_DIR, "thresholds.json"), "w"))
    print(f"\nsaved -> {OUT_DIR}/best.pt, thresholds.json, history.json", flush=True)
    print("ImageCLEF 2025 reference (test split): AUEB 1st = 0.5888 | 5th = 0.5225", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()          # no-op unless frozen; required by the spawn idiom
    main()
