#!/usr/bin/env python
# LoRA fine-tuning of Qwen3-VL-4B-Instruct on ROCOv2 WITH UMLS CONCEPT GUIDANCE (Model B).
#
# Model B = Model A (qwen3vl_roco_finetune.py) + an auxiliary multi-label concept head on
# the MERGER output. Everything else -- data subset, seed, resolution, LoRA config, LRs,
# schedule, prompt, decoding -- is identical, so the ONLY difference is the CUI guidance.
#
# Precedent: AI Stat Lab (ImageCLEFmedical Caption 2025, 3rd place, CEUR Vol-4038 paper_196).
# They mean-pool the Q-Former output and attach two linear classifiers (2,478 CUIs + 21
# semantic types), training with L_total = L_caption + lambda * L_cls at lambda = 0.1.
# Their Table 2 ablation (dual encoder, #1673 -> #1695) is the only clean isolation of this
# variable in the literature: ROUGE-1 +0.0062, UMLS F1 +0.0048, AlignScore -0.0053.
#
# Two documented deviations from them:
#   1. BCE-with-pos_weight instead of multi-label margin loss. BCE is what every top
#      ImageCLEF-2025 concept-detection team used (AUEB, DeepLens, UIT-Oggy); pos_weight
#      handles the 3-of-1571 imbalance explicitly, and we never need calibrated
#      probabilities since the head is discarded at inference. Set CUI_LOSS=margin to
#      reproduce their exact choice.
#   2. Coarse head = ROCOv2's 18 MANUALLY CURATED concepts (modality / body region /
#      directionality) instead of 21 UMLS semantic types, which ROCOv2 does not ship.
#      Same role (small, dense, low-tail), hand-curated rather than auto-extracted.
#
# The head sits BEFORE the LLM and the ViT is frozen, so the auxiliary gradient reaches
# ONLY the merger -- i.e. this tests connector *content*, orthogonal to the earlier
# connector *bandwidth* result. Nothing is consumed at inference: the heads are dropped,
# so evaluation uses qwen3vl_roco_zeroshot_v1.py unchanged (no CUI detector, hence none of
# the error-propagation that sank the concept-injection methods at ImageCLEF).
#
#   probe : PROBE=1 python qwen3vl_roco_finetune_cui.py
#   train : python qwen3vl_roco_finetune_cui.py 2>&1 | tee ft_cui_run.log
#   ablate: LAMBDA_CUI=0 python qwen3vl_roco_finetune_cui.py   # == Model A, same code path
#   resume: RESUME=1 python qwen3vl_roco_finetune_cui.py
import os
os.chdir("/home/matei")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import math
import random
import time
import collections

import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if device == "cuda" else torch.float32
print(f"device={device} dtype={DTYPE}", flush=True)

# ── config (identical to Model A except the CUI block) ───────────────────────
MODEL_ID      = os.environ.get("MODEL_ID", "/home/matei/qwen3-vl-4b-instruct")
RUN_TAG       = os.environ.get("RUN_TAG", "qwen3vl_ft_cui")
OUT_DIR       = f"/home/matei/{RUN_TAG}_ckpt"
PROBE         = os.environ.get("PROBE", "0") == "1"
PROBE_STEPS   = int(os.environ.get("PROBE_STEPS", "20"))
RESUME        = os.environ.get("RESUME", "0") == "1"

N_TRAIN       = int(os.environ.get("N_TRAIN", "12000"))
N_VAL         = int(os.environ.get("N_VAL", "500"))
SEED          = int(os.environ.get("SEED", "42"))          # same seed -> same 12k subset as A
MIN_PIXELS    = 256 * 32 * 32
MAX_PIXELS    = int(os.environ.get("MAX_VIS_TOKENS", "768")) * 32 * 32
MAX_CAP_TOK   = int(os.environ.get("MAX_CAP_TOK", "48"))

EPOCHS        = float(os.environ.get("EPOCHS", "2"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM    = int(os.environ.get("GRAD_ACCUM", "4"))
LR_LORA       = float(os.environ.get("LR_LORA", "1e-4"))
LR_MERGER     = float(os.environ.get("LR_MERGER", "2e-5"))
LR_HEAD       = float(os.environ.get("LR_HEAD", "1e-4"))   # randomly init'd -> LoRA-scale LR
WEIGHT_DECAY  = 0.01
WARMUP_FRAC   = 0.03
MAX_GRAD_NORM = 1.0
TRAIN_MERGER  = os.environ.get("TRAIN_MERGER", "1") == "1"
LORA_R        = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA    = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT  = 0.05
EVAL_EVERY    = int(os.environ.get("EVAL_EVERY", "500"))
PATIENCE      = int(os.environ.get("PATIENCE", "3"))

# ---- CUI guidance (the ONLY block that differs from Model A) ----
LAMBDA_CUI    = float(os.environ.get("LAMBDA_CUI", "0.1"))  # 0 -> exactly Model A
MIN_CUI_FREQ  = int(os.environ.get("MIN_CUI_FREQ", "10"))   # ImageCLEF's own curation rule
USE_MANUAL    = os.environ.get("USE_MANUAL", "1") == "1"    # the 18-class coarse head
CUI_LOSS      = os.environ.get("CUI_LOSS", "bce")           # "bce" | "margin"
POSW_CLAMP    = float(os.environ.get("POSW_CLAMP", "50"))   # cap pos_weight on rare classes

if TRAIN_MERGER is False and LAMBDA_CUI > 0:
    print("[warn] LAMBDA_CUI>0 but TRAIN_MERGER=0: the auxiliary gradient has nowhere to go "
          "(ViT frozen, head is post-merger). Enable TRAIN_MERGER.", flush=True)

# ── the V1 prompt (identical to Model A and to the eval script) ──────────────
SYSTEM_PROMPT = (
    "You are an expert radiologist writing captions for a radiology teaching archive. "
    "You are given one medical image. Write a single concise caption that describes only "
    "what is directly visible.\n"
    "Rules:\n"
    "1. Describe only what is observable in this image: the imaging modality, the anatomical "
    "region, and any clearly visible findings. Do not infer diagnoses, patient history, or "
    "findings that are not directly visible.\n"
    "2. If a finding cannot be determined from the image, do not state it.\n"
    "3. Be terse and clinical: one sentence, radiology-report style.\n"
    "4. Output only the caption -- no preamble, no \"This image shows\", no disclaimers."
)
USER_PROMPT = "Provide the caption for this image."

# ── model + processor ────────────────────────────────────────────────────────
from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup

processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
processor.tokenizer.padding_side = "right"
MERGE_SIZE = getattr(processor.image_processor, "merge_size", 2)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=DTYPE, device_map={"": 0} if device == "cuda" else None)
model.config.use_cache = False

# Grab the merger module BEFORE the PEFT wrapper: the same object stays in the graph, so a
# forward hook on it still fires and its output is still connected to autograd.
merger_module = model.model.visual.merger
D_MERGER = merger_module.linear_fc2.out_features     # == LLM hidden size
print(f"merger output dim: {D_MERGER}", flush=True)

for p in model.parameters():
    p.requires_grad = False

# Scope of merger training. "main" = the MLP vision-language merger only (27.3M) --
# this is what Qwen's Stage 0 trains, what AI Stat Lab train (one Q-Former), and what
# the ROCOv2 paper adapts as "mm_proj". "all" adds the 3 DeepStack mergers (+81.8M),
# which costs ~1 GB more in grads+Adam and does not fit alongside 768 visual tokens.
MERGER_SCOPE = os.environ.get("MERGER_SCOPE", "main")      # "main" | "all"
MERGER_PREFIXES = (("model.visual.merger",) if MERGER_SCOPE == "main"
                   else ("model.visual.merger", "model.visual.deepstack_merger_list"))
merger_params = []
if TRAIN_MERGER:
    for n, p in model.named_parameters():
        if n.startswith(MERGER_PREFIXES):
            p.requires_grad = True
            p.data = p.data.float()
            merger_params.append(p)

from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)
# PEFT's _mark_only_adapters_as_trainable() re-freezes EVERY non-LoRA parameter,
# including the mergers unfrozen above. Re-assert them -- merger_params holds the same
# Parameter objects, so this restores the flags on the modules actually in the graph.
for p in merger_params:
    p.requires_grad = True
assert all(p.requires_grad for p in merger_params), "merger re-freeze not repaired"
for n, p in model.named_parameters():
    if "lora_" in n:
        p.data = p.data.float()
        p.requires_grad = True

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]

# ── label spaces, built from TRAIN only ──────────────────────────────────────
import pandas as pd
ROCO_DIR = "/home/matei/rocov2"

def read_concepts(fname):
    df = pd.read_csv(os.path.join(ROCO_DIR, fname))
    return {r.ID: [c for c in str(r.CUIs).split(";") if c and c != "nan"]
            for r in df.itertuples()}

cui_train = read_concepts("train_concepts.csv")
cui_valid = read_concepts("valid_concepts.csv")
man_train = read_concepts("train_concepts_manual.csv") if USE_MANUAL else {}
man_valid = read_concepts("valid_concepts_manual.csv") if USE_MANUAL else {}

_freq = collections.Counter(c for v in cui_train.values() for c in v)
CUI_VOCAB = sorted([c for c, n in _freq.items() if n >= MIN_CUI_FREQ])
CUI2IDX   = {c: i for i, c in enumerate(CUI_VOCAB)}
MAN_VOCAB = sorted({c for v in man_train.values() for c in v}) if USE_MANUAL else []
MAN2IDX   = {c: i for i, c in enumerate(MAN_VOCAB)}
N_CUI, N_MAN = len(CUI_VOCAB), len(MAN_VOCAB)
print(f"label spaces: {N_CUI} CUIs (>= {MIN_CUI_FREQ} occurrences) | {N_MAN} curated concepts",
      flush=True)

# pos_weight = #neg/#pos per class, clamped so ultra-rare classes don't dominate the loss.
def pos_weight_for(vocab, idx, id2list, n_images):
    cnt = collections.Counter(c for v in id2list.values() for c in v if c in idx)
    w = torch.ones(len(vocab))
    for c, i in idx.items():
        p = max(1, cnt.get(c, 0))
        w[i] = min((n_images - p) / p, POSW_CLAMP)
    return w

# ── auxiliary heads (the intervention) ───────────────────────────────────────
# Mean-pool the merger's visual tokens -> two linear classifiers, exactly as AI Stat Lab
# do on their Q-Former output. fp32 so AdamW updates are stable.
head_cui = nn.Linear(D_MERGER, N_CUI).to(device, torch.float32)
head_man = nn.Linear(D_MERGER, N_MAN).to(device, torch.float32) if N_MAN else None
head_params = list(head_cui.parameters()) + (list(head_man.parameters()) if head_man else [])

_pw_cui = pos_weight_for(CUI_VOCAB, CUI2IDX, cui_train, len(cui_train)).to(device)
_pw_man = (pos_weight_for(MAN_VOCAB, MAN2IDX, man_train, len(man_train)).to(device)
           if head_man else None)
bce_cui = nn.BCEWithLogitsLoss(pos_weight=_pw_cui, reduction="mean")
bce_man = nn.BCEWithLogitsLoss(pos_weight=_pw_man, reduction="mean") if head_man else None
margin_loss = nn.MultiLabelMarginLoss()

# Forward hook: capture the merger output so we can pool it for the heads.
# Cleared before every forward; we read it immediately after, so the recomputation that
# gradient checkpointing triggers during backward cannot overwrite what we used.
_captured = []
merger_module.register_forward_hook(lambda m, i, o: _captured.append(o))

def pooled_visual(grid_thw):
    """Mean-pool merger output per image -> [B, D_MERGER]."""
    out = _captured[0]                                   # [total_merged_tokens, D]
    if out.dim() == 3:                                   # already [B, T, D]
        return out.mean(dim=1)
    per_img = (grid_thw.prod(dim=-1) // (MERGE_SIZE ** 2)).tolist()
    return torch.stack([c.mean(0) for c in torch.split(out, per_img, dim=0)])

def concept_targets(ids, vocab_n, idx, id2list):
    """Multi-hot targets [B, n] + a mask marking images that actually have labels."""
    y = torch.zeros(len(ids), vocab_n)
    m = torch.zeros(len(ids))
    for b, _id in enumerate(ids):
        got = [idx[c] for c in id2list.get(_id, []) if c in idx]
        if got:
            y[b, got] = 1.0
            m[b] = 1.0                                   # 493 train images have no manual concept
    return y.to(device), m.to(device)

def aux_loss(pooled, ids, split_cui, split_man):
    """L_cls = L_cui + L_man, each masked to images that carry that label type."""
    parts = {}
    p32 = pooled.float()
    y, m = concept_targets(ids, N_CUI, CUI2IDX, split_cui)
    if m.sum() > 0:
        logit = head_cui(p32)
        if CUI_LOSS == "margin":                         # AI Stat Lab's exact choice
            tgt = torch.full_like(logit, -1, dtype=torch.long)
            for b in range(len(ids)):
                pos = y[b].nonzero(as_tuple=True)[0]
                tgt[b, :len(pos)] = pos
            parts["cui"] = margin_loss(logit, tgt)
        else:
            per = nn.functional.binary_cross_entropy_with_logits(
                logit, y, pos_weight=_pw_cui, reduction="none").mean(dim=1)
            parts["cui"] = (per * m).sum() / m.sum()
    if head_man is not None:
        y2, m2 = concept_targets(ids, N_MAN, MAN2IDX, split_man)
        if m2.sum() > 0:
            per = nn.functional.binary_cross_entropy_with_logits(
                head_man(p32), y2, pos_weight=_pw_man, reduction="none").mean(dim=1)
            parts["man"] = (per * m2).sum() / m2.sum()
    return parts

# ── data (same loader, same seed, same subset as Model A) ────────────────────
def load_split(split, csv):
    img_dir = os.path.join(ROCO_DIR, split)
    caps = pd.read_csv(os.path.join(ROCO_DIR, csv)).dropna(subset=["Caption"]).reset_index(drop=True)
    return [{"id": r.ID, "path": os.path.join(img_dir, f"{r.ID}.jpg"), "caption": str(r.Caption)}
            for r in caps.itertuples() if os.path.isfile(os.path.join(img_dir, f"{r.ID}.jpg"))]

train_all = load_split("train", "train_captions.csv")
val_all   = load_split("valid", "valid_captions.csv")
rng = random.Random(SEED)
rng.shuffle(train_all)
rng.shuffle(val_all)
train_recs = train_all[:N_TRAIN] if N_TRAIN else train_all
val_recs   = val_all[:N_VAL]
print(f"train {len(train_recs)} (of {len(train_all)}) | val {len(val_recs)}", flush=True)

_DUMMY = Image.new("RGB", (32, 32))
_msgs = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user",   "content": [{"type": "image", "image": _DUMMY},
                                   {"type": "text",  "text": USER_PROMPT}]},
]
PROMPT_TEXT = processor.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
EOS = processor.tokenizer.eos_token or "<|im_end|>"

def build_example(rec):
    img = Image.open(rec["path"]).convert("RGB")
    cap = " ".join(str(rec["caption"]).split())
    cap_ids = processor.tokenizer(cap, add_special_tokens=False)["input_ids"][:MAX_CAP_TOK]
    cap = processor.tokenizer.decode(cap_ids)
    full = processor(text=[PROMPT_TEXT + cap + EOS], images=[img], return_tensors="pt")
    plen = processor(text=[PROMPT_TEXT], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels
    return full

# ── optimiser: LoRA / merger / heads ─────────────────────────────────────────
groups = [{"params": lora_params, "lr": LR_LORA, "weight_decay": WEIGHT_DECAY}]
if merger_params:
    groups.append({"params": merger_params, "lr": LR_MERGER, "weight_decay": WEIGHT_DECAY})
if LAMBDA_CUI > 0 and head_params:
    groups.append({"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY})
optim = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

n_lora   = sum(p.numel() for p in lora_params)
n_merger = sum(p.numel() for p in merger_params)
n_head   = sum(p.numel() for p in head_params)
print(f"trainable: LoRA {n_lora/1e6:.1f}M + merger {n_merger/1e6:.1f}M + heads {n_head/1e6:.1f}M "
      f"= {(n_lora+n_merger+n_head)/1e6:.1f}M", flush=True)
if device == "cuda":
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

steps_per_epoch = max(1, math.ceil(len(train_recs) / (BATCH_SIZE * GRAD_ACCUM)))
total_steps     = max(1, int(steps_per_epoch * EPOCHS))
sched = get_cosine_schedule_with_warmup(optim, int(WARMUP_FRAC * total_steps), total_steps)
print(f"{steps_per_epoch} optimiser steps/epoch x {EPOCHS} epochs = {total_steps} total", flush=True)

def forward_losses(rec, split_cui, split_man):
    """Returns (total, caption_loss, {aux parts}) -- aux is skipped when LAMBDA_CUI == 0."""
    ex = {k: v.to(model.device) for k, v in build_example(rec).items()}
    _captured.clear()
    with torch.autocast("cuda", dtype=DTYPE) if device == "cuda" else torch.enable_grad():
        out = model(**ex)
    l_cap = out.loss
    parts = {}
    if LAMBDA_CUI > 0 and _captured:
        pooled = pooled_visual(ex["image_grid_thw"])
        parts = aux_loss(pooled, [rec["id"]], split_cui, split_man)
    total = l_cap + LAMBDA_CUI * sum(parts.values()) if parts else l_cap
    return total, l_cap, parts

@torch.no_grad()
def validate():
    """Validation on CAPTION loss only -- identical criterion to Model A, so epoch
    selection is comparable between the two models."""
    model.eval()
    tot, n = 0.0, 0
    for rec in tqdm(val_recs, desc="val", leave=False, mininterval=15):
        try:
            _, l_cap, _ = forward_losses(rec, cui_valid, man_valid)
            tot += l_cap.item(); n += 1
        except Exception as e:
            print(f"[warn] val skip {rec['id']}: {e}", flush=True)
    model.train()
    return tot / max(1, n)

def save_ckpt(tag):
    d = os.path.join(OUT_DIR, tag)
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)
    if merger_params:
        msd = {n: p.detach().to(torch.float32).cpu()
               for n, p in model.named_parameters()
               if any(k in n for k in ("visual.merger", "visual.deepstack_merger_list"))}
        torch.save(msd, os.path.join(d, "merger.pt"))
    if LAMBDA_CUI > 0 and head_params:      # not needed for inference; saved for analysis
        torch.save({"head_cui": head_cui.state_dict(),
                    "head_man": head_man.state_dict() if head_man else None,
                    "cui_vocab": CUI_VOCAB, "man_vocab": MAN_VOCAB},
                   os.path.join(d, "concept_heads.pt"))
    print(f"[ckpt] saved -> {d}", flush=True)

def _peek():
    ex = build_example(train_recs[0])
    lab = ex["labels"][0]
    print(f"[mask check] seq={lab.numel()} supervised={(lab != -100).sum().item()} "
          f"-> '{processor.tokenizer.decode(lab[lab != -100])[:70]}...'", flush=True)
    r = train_recs[0]
    print(f"[label check] {r['id']}: {len(cui_train.get(r['id'], []))} CUIs, "
          f"{len(man_train.get(r['id'], []))} curated -> "
          f"{[c for c in man_train.get(r['id'], [])]}", flush=True)

# ── probe ────────────────────────────────────────────────────────────────────
if PROBE:
    _peek()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    _w0_merger = merger_params[0].detach().clone() if merger_params else None
    _w0_lora   = lora_params[0].detach().clone() if lora_params else None
    _w0_lora_all = [p.detach().clone() for p in lora_params]
    _lora_name0 = [n for n, p in model.named_parameters() if p is lora_params[0]][0]
    t0 = time.time()
    last = {}
    for i in tqdm(range(PROBE_STEPS), desc="probe"):
        total, l_cap, parts = forward_losses(train_recs[i], cui_train, man_train)
        (total / GRAD_ACCUM).backward()
        if i == 0 and merger_params:      # PROOF the auxiliary/merger path is connected
            got = sum(p.grad is not None for p in merger_params)
            print(f"\n[grad check] merger params receiving gradient: {got}/{len(merger_params)}",
                  flush=True)
            assert got == len(merger_params), "merger is NOT receiving gradient"
        last = {"caption": l_cap.item(),
                **{f"aux_{k}": v.item() for k, v in parts.items()}}
        if (i + 1) % GRAD_ACCUM == 0:
            # Clip the caption path SEPARATELY from the heads. Clipping them jointly would fold the
            # heads' (large, randomly-initialised) gradient norm into the global norm and scale the
            # caption-path gradients down too -- an effective LR cut Model A does not experience,
            # i.e. a second uncontrolled difference between the two models.
            n_cap = torch.nn.utils.clip_grad_norm_(lora_params + merger_params, MAX_GRAD_NORM)
            n_head = (torch.nn.utils.clip_grad_norm_(head_params, MAX_GRAD_NORM)
                      if head_params else torch.tensor(0.))
            if i + 1 == GRAD_ACCUM:      # report once: how big is the effect actually?
                print(f"\n[grad norms] caption {float(n_cap):.2f} | heads {float(n_head):.2f}",
                      flush=True)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
    dt = (time.time() - t0) / PROBE_STEPS
    peak = torch.cuda.max_memory_allocated() / 1e9
    aux_sum = sum(v for k, v in last.items() if k.startswith("aux_"))
    if _w0_merger is not None:
        d = (merger_params[0].detach() - _w0_merger).abs().max().item()
        print(f"  merger updated : {d:.3e}  {'YES' if d > 0 else 'NO -- NOT TRAINING'}")
    if _w0_lora is not None:
        d = (lora_params[0].detach() - _w0_lora).abs().max().item()
        n_grad = sum(p.grad is not None for p in lora_params)
        n_chg  = sum((p.detach() - w).abs().max().item() > 0
                     for p, w in zip(lora_params, _w0_lora_all))
        print(f"  LoRA grads     : {n_grad}/{len(lora_params)} params have .grad")
        print(f"  LoRA changed   : {n_chg}/{len(lora_params)} params moved "
              f"(param[0] delta {d:.3e}, name {_lora_name0})")
    print("\n==== PROBE (Model B, CUI-guided) ====", flush=True)
    print(f"  MAX_VIS_TOKENS : {MAX_PIXELS // (32*32)}")
    print(f"  lambda         : {LAMBDA_CUI}   loss: {CUI_LOSS}")
    print(f"  label spaces   : {N_CUI} CUIs | {N_MAN} curated")
    print(f"  peak VRAM      : {peak:.2f} GB / 12.6 GB")
    print(f"  s/sample       : {dt:.2f}")
    print(f"  -> 1 epoch over {len(train_recs)} imgs = {dt*len(train_recs)/3600:.1f} h")
    print(f"  losses         : {last}")
    if aux_sum:
        print(f"  weighted aux / caption = {LAMBDA_CUI*aux_sum/last['caption']:.1%} "
              f"(want roughly 5-20%; tune LAMBDA_CUI if far off)", flush=True)
    raise SystemExit(0)

# ── training loop (identical control flow to Model A) ────────────────────────
_peek()
os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"cui_vocab": CUI_VOCAB, "man_vocab": MAN_VOCAB, "lambda": LAMBDA_CUI,
           "min_cui_freq": MIN_CUI_FREQ, "loss": CUI_LOSS},
          open(os.path.join(OUT_DIR, "concept_config.json"), "w"), indent=1)
state_path = os.path.join(OUT_DIR, "state.json")
start_step, best_val, bad_evals = 0, float("inf"), 0
if RESUME and os.path.exists(state_path):
    st = json.load(open(state_path))
    start_step, best_val, bad_evals = st["step"], st["best_val"], st["bad_evals"]
    print(f"[resume] from optimiser step {start_step} (best val {best_val:.4f})", flush=True)

model.train()
gstep, seen, t0 = 0, 0, time.time()
history, stop = [], False
run_cap, run_aux, run_n = 0.0, 0.0, 0
for epoch in range(math.ceil(EPOCHS)):
    if stop:
        break
    rng.shuffle(train_recs)
    pbar = tqdm(train_recs, desc=f"epoch {epoch+1}", mininterval=15)
    for rec in pbar:
        try:
            total, l_cap, parts = forward_losses(rec, cui_train, man_train)
            (total / GRAD_ACCUM).backward()
            run_cap += l_cap.item()
            run_aux += sum(v.item() for v in parts.values())
            run_n   += 1
        except torch.OutOfMemoryError:
            print(f"[warn] OOM on {rec['id']} -- skipped", flush=True)
            optim.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"[warn] skip {rec['id']}: {e}", flush=True)
            optim.zero_grad(set_to_none=True); continue
        seen += 1
        if seen % GRAD_ACCUM:
            continue

        torch.nn.utils.clip_grad_norm_(lora_params + merger_params, MAX_GRAD_NORM)
        if head_params:      # separate clip -> caption path behaves exactly as in Model A
            torch.nn.utils.clip_grad_norm_(head_params, MAX_GRAD_NORM)
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        gstep += 1
        if gstep <= start_step:
            continue
        pbar.set_postfix(cap=f"{run_cap/max(1,run_n):.3f}",
                         aux=f"{run_aux/max(1,run_n):.3f}", step=gstep)

        if gstep % EVAL_EVERY == 0 or gstep == total_steps:
            vl = validate()
            mc, ma = run_cap / max(1, run_n), run_aux / max(1, run_n)
            history.append({"step": gstep, "val_caption_loss": vl, "train_caption": mc,
                            "train_aux": ma, "weighted_aux_frac": (LAMBDA_CUI * ma / mc) if mc else 0,
                            "hours": (time.time() - t0) / 3600})
            print(f"\n[eval] step {gstep} | val {vl:.4f} | best {best_val:.4f} | "
                  f"train cap {mc:.3f} aux {ma:.3f} | weighted aux {LAMBDA_CUI*ma/max(mc,1e-9):.1%}",
                  flush=True)
            run_cap = run_aux = 0.0; run_n = 0
            if vl < best_val:
                best_val, bad_evals = vl, 0
                save_ckpt("best")
            else:
                bad_evals += 1
                print(f"[eval] no improvement ({bad_evals}/{PATIENCE})", flush=True)
            save_ckpt("last")
            json.dump({"step": gstep, "best_val": best_val, "bad_evals": bad_evals},
                      open(state_path, "w"))
            json.dump(history, open(f"/home/matei/{RUN_TAG}_history.json", "w"), indent=1)
            if bad_evals >= PATIENCE:
                print("[early stop] validation caption loss stopped improving", flush=True)
                stop = True; break
        if gstep >= total_steps:
            stop = True; break

save_ckpt("last")
print(f"\ndone in {(time.time()-t0)/3600:.1f} h | best val caption loss {best_val:.4f}", flush=True)
print(f"evaluate (heads are dropped -- inference is unchanged):\n"
      f"  LORA_PATH={OUT_DIR}/best RUN_TAG={RUN_TAG}_test \\\n"
      f"    python qwen3vl_roco_zeroshot_v1.py", flush=True)
