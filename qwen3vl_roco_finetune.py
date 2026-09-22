#!/usr/bin/env python
# LoRA fine-tuning of Qwen3-VL-4B-Instruct on ROCOv2 captioning.
#
# Design follows the literature closest to this task -- the ROCOv2 LoRA study
# (PMC12730038) and Swin-Qwen3 (Front. Radiol. 2026) -- both of which use a SINGLE
# training stage, a FROZEN vision encoder, and LoRA on the LLM plus the multimodal
# projector ("mm_proj" there = `visual.merger` here), in bf16 with NO quantization.
# Qwen3-VL's own report (Table 1) trains the merger alone only to BOOTSTRAP a fresh
# ViT-LLM pairing; our model is already aligned, so that stage does not transfer.
#
# What is trainable:
#   * LoRA adapters (r=16, a=32) on the LLM q/k/v/o_proj + gate/up/down_proj   lr 1e-4
#   * visual.merger + visual.deepstack_merger_list.{0,1,2}  (the projectors)   lr 2e-5
# Everything else -- all ViT blocks, patch_embed, embeddings, lm_head -- is frozen, so
# no ViT activations are stored and memory stays inside 12 GB.
#
# Precision: base in bf16, trainable params in fp32 master weights, forward under
# bf16 autocast (standard mixed precision -- keeps AdamW updates from underflowing).
#
#   probe : PROBE=1 python qwen3vl_roco_finetune.py          # ~20 steps, VRAM + s/step
#   probe@512 : PROBE=1 MAX_VIS_TOKENS=512 python qwen3vl_roco_finetune.py
#   train : python qwen3vl_roco_finetune.py 2>&1 | tee ft_run.log
#   resume: RESUME=1 python qwen3vl_roco_finetune.py
import os
os.chdir("/home/matei")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import math
import random
import time

import torch
from PIL import Image
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if device == "cuda" else torch.float32
print(f"device={device} dtype={DTYPE}", flush=True)

# ── config (env-overridable) ─────────────────────────────────────────────────
MODEL_ID      = os.environ.get("MODEL_ID", "/home/matei/qwen3-vl-4b-instruct")
RUN_TAG       = os.environ.get("RUN_TAG", "qwen3vl_ft")
OUT_DIR       = f"/home/matei/{RUN_TAG}_ckpt"
PROBE         = os.environ.get("PROBE", "0") == "1"
PROBE_STEPS   = int(os.environ.get("PROBE_STEPS", "20"))
RESUME        = os.environ.get("RESUME", "0") == "1"

# Data. 12,000-image subsample matches the size used for the BLIP-2 / BioMedVQA
# fine-tunes; 768 visual tokens matches the V1 zero-shot eval so the zero-shot ->
# fine-tuned delta needs no baseline re-run.
N_TRAIN       = int(os.environ.get("N_TRAIN", "12000"))
N_VAL         = int(os.environ.get("N_VAL", "500"))
SEED          = int(os.environ.get("SEED", "42"))
MIN_PIXELS    = 256 * 32 * 32
MAX_PIXELS    = int(os.environ.get("MAX_VIS_TOKENS", "768")) * 32 * 32
MAX_CAP_TOK   = int(os.environ.get("MAX_CAP_TOK", "48"))   # ROCO captions avg ~21 words

# Optimisation. lr 1e-4 for LoRA is what BOTH comparable papers use on this task;
# merger gets a gentler 2e-5 (differential LR, mirroring the LLaVA recipe).
EPOCHS        = float(os.environ.get("EPOCHS", "2"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "1"))     # 12 GB -> 1 (see zero-shot runs)
GRAD_ACCUM    = int(os.environ.get("GRAD_ACCUM", "4"))     # effective batch 4 (Swin-Qwen3)
LR_LORA       = float(os.environ.get("LR_LORA", "1e-4"))
LR_MERGER     = float(os.environ.get("LR_MERGER", "2e-5"))
WEIGHT_DECAY  = 0.01
WARMUP_FRAC   = 0.03
MAX_GRAD_NORM = 1.0
TRAIN_MERGER  = os.environ.get("TRAIN_MERGER", "1") == "1"
LORA_R        = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA    = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT  = 0.05
EVAL_EVERY    = int(os.environ.get("EVAL_EVERY", "500"))   # optimiser steps
PATIENCE      = int(os.environ.get("PATIENCE", "3"))       # early stop on val loss

# ── the V1 prompt (identical to qwen3vl_roco_zeroshot_v1.py) ─────────────────
# Training with the SAME prompt used at evaluation keeps train/test conditions
# identical and lets the V1 eval script be reused verbatim.
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
processor.tokenizer.padding_side = "right"   # training (generation uses left; batch=1 anyway)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=DTYPE, device_map={"": 0} if device == "cuda" else None)
model.config.use_cache = False               # incompatible with gradient checkpointing

# ── freeze everything, then re-enable only what we train ─────────────────────
for p in model.parameters():
    p.requires_grad = False

# The projectors: `visual.merger` (main) + the three DeepStack mergers, which the
# report describes as feeding LLM layers 1-3. These are the vision->language bridge
# and the module the ROCOv2 paper adapts as "mm_proj".
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
            p.data = p.data.float()          # fp32 master weights for stable AdamW
            merger_params.append(p)

# LoRA on the LLM only. These suffixes exist solely under model.language_model.*;
# the ViT uses qkv/proj/linear_fc1/2, so there is no accidental match in the vision tower.
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
for n, p in model.named_parameters():        # LoRA in fp32 too (peft casts in forward)
    if "lora_" in n:
        p.data = p.data.float()
        p.requires_grad = True

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()           # else no grad flows through checkpointed blocks

lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
n_lora   = sum(p.numel() for p in lora_params)
n_merger = sum(p.numel() for p in merger_params)
n_total  = sum(p.numel() for p in model.parameters())
print(f"trainable: LoRA {n_lora/1e6:.1f}M + merger {n_merger/1e6:.1f}M "
      f"= {(n_lora+n_merger)/1e6:.1f}M / {n_total/1e9:.2f}B "
      f"({100*(n_lora+n_merger)/n_total:.2f}%)", flush=True)
if device == "cuda":
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ── data ─────────────────────────────────────────────────────────────────────
import pandas as pd
ROCO_DIR = "/home/matei/rocov2"

def load_split(split, csv):
    img_dir = os.path.join(ROCO_DIR, split)
    caps = pd.read_csv(os.path.join(ROCO_DIR, csv)).dropna(subset=["Caption"]).reset_index(drop=True)
    return [{"id": r.ID, "path": os.path.join(img_dir, f"{r.ID}.jpg"), "caption": str(r.Caption)}
            for r in caps.itertuples() if os.path.isfile(os.path.join(img_dir, f"{r.ID}.jpg"))]

train_all = load_split("train", "train_captions.csv")
val_all   = load_split("valid", "valid_captions.csv")
rng = random.Random(SEED)                     # fixed seed -> reproducible subsample
rng.shuffle(train_all)
rng.shuffle(val_all)
train_recs = train_all[:N_TRAIN] if N_TRAIN else train_all
val_recs   = val_all[:N_VAL]
print(f"train {len(train_recs)} (of {len(train_all)}) | val {len(val_recs)}", flush=True)

# Prompt text is identical for every image (built once, as in the zero-shot scripts).
_DUMMY = Image.new("RGB", (32, 32))
_msgs = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user",   "content": [{"type": "image", "image": _DUMMY},
                                   {"type": "text",  "text": USER_PROMPT}]},
]
PROMPT_TEXT = processor.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
EOS = processor.tokenizer.eos_token or "<|im_end|>"

def build_example(rec):
    """One training example: prompt+caption tokenised, loss masked to caption only."""
    img = Image.open(rec["path"]).convert("RGB")          # radiographs -> RGB
    cap = " ".join(str(rec["caption"]).split())
    cap_ids = processor.tokenizer(cap, add_special_tokens=False)["input_ids"][:MAX_CAP_TOK]
    cap = processor.tokenizer.decode(cap_ids)             # truncate to the token budget
    full = processor(text=[PROMPT_TEXT + cap + EOS], images=[img], return_tensors="pt")
    # Prompt length WITH the same image, so the expanded <|image_pad|> seats are counted.
    plen = processor(text=[PROMPT_TEXT], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100                               # mask prompt/system/image tokens
    labels[full["attention_mask"] == 0] = -100            # mask padding
    full["labels"] = labels
    return full

@torch.no_grad()
def _peek_masking():
    ex = build_example(train_recs[0])
    lab = ex["labels"][0]
    kept = (lab != -100).sum().item()
    print(f"[mask check] seq={lab.numel()} tokens, supervised={kept} "
          f"-> '{processor.tokenizer.decode(lab[lab != -100])[:70]}...'", flush=True)

# ── optimiser: two param groups (LoRA fast, merger slow) ─────────────────────
groups = [{"params": lora_params, "lr": LR_LORA, "weight_decay": WEIGHT_DECAY}]
if merger_params:
    groups.append({"params": merger_params, "lr": LR_MERGER, "weight_decay": WEIGHT_DECAY})
optim = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

steps_per_epoch = max(1, math.ceil(len(train_recs) / (BATCH_SIZE * GRAD_ACCUM)))
total_steps     = max(1, int(steps_per_epoch * EPOCHS))
sched = get_cosine_schedule_with_warmup(optim, int(WARMUP_FRAC * total_steps), total_steps)
print(f"{steps_per_epoch} optimiser steps/epoch x {EPOCHS} epochs = {total_steps} total", flush=True)

def forward_loss(rec):
    ex = {k: v.to(model.device) for k, v in build_example(rec).items()}
    with torch.autocast("cuda", dtype=DTYPE) if device == "cuda" else torch.enable_grad():
        return model(**ex).loss

@torch.no_grad()
def validate():
    model.eval()
    tot, n = 0.0, 0
    for rec in tqdm(val_recs, desc="val", leave=False, mininterval=15):
        try:
            tot += forward_loss(rec).item(); n += 1
        except Exception as e:
            print(f"[warn] val skip {rec['id']}: {e}", flush=True)
    model.train()
    return tot / max(1, n)

def save_ckpt(tag):
    d = os.path.join(OUT_DIR, tag)
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)                              # LoRA adapter
    if merger_params:                                     # mergers are full weights, not LoRA
        msd = {n: p.detach().to(torch.float32).cpu()
               for n, p in model.named_parameters()
               if any(k in n for k in ("visual.merger", "visual.deepstack_merger_list"))}
        torch.save(msd, os.path.join(d, "merger.pt"))
    print(f"[ckpt] saved -> {d}", flush=True)

# ── probe mode: measure VRAM + s/step, then exit ─────────────────────────────
if PROBE:
    _peek_masking()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    _w0_merger = merger_params[0].detach().clone() if merger_params else None
    _w0_lora   = lora_params[0].detach().clone() if lora_params else None
    _w0_lora_all = [p.detach().clone() for p in lora_params]
    _lora_name0 = [n for n, p in model.named_parameters() if p is lora_params[0]][0]
    t0 = time.time()
    for i in tqdm(range(PROBE_STEPS), desc="probe"):
        loss = forward_loss(train_recs[i]) / GRAD_ACCUM
        loss.backward()
        if i == 0 and merger_params:      # PROOF the auxiliary/merger path is connected
            got = sum(p.grad is not None for p in merger_params)
            print(f"\n[grad check] merger params receiving gradient: {got}/{len(merger_params)}",
                  flush=True)
            assert got == len(merger_params), "merger is NOT receiving gradient"
        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + merger_params, MAX_GRAD_NORM)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
    dt = (time.time() - t0) / PROBE_STEPS
    peak = torch.cuda.max_memory_allocated() / 1e9
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
    print("\n==== PROBE ====", flush=True)
    print(f"  MAX_VIS_TOKENS : {MAX_PIXELS // (32*32)}")
    print(f"  train_merger   : {TRAIN_MERGER}")
    print(f"  peak VRAM      : {peak:.2f} GB / 12.6 GB")
    print(f"  s/sample       : {dt:.2f}")
    print(f"  -> 1 epoch over {len(train_recs)} imgs = {dt*len(train_recs)/3600:.1f} h")
    print(f"  -> {EPOCHS} epochs = {dt*len(train_recs)*EPOCHS/3600:.1f} h", flush=True)
    raise SystemExit(0)

# ── training loop ────────────────────────────────────────────────────────────
_peek_masking()
os.makedirs(OUT_DIR, exist_ok=True)
state_path = os.path.join(OUT_DIR, "state.json")
start_step, best_val, bad_evals = 0, float("inf"), 0
if RESUME and os.path.exists(state_path):
    st = json.load(open(state_path))
    start_step, best_val, bad_evals = st["step"], st["best_val"], st["bad_evals"]
    print(f"[resume] from optimiser step {start_step} (best val {best_val:.4f})", flush=True)

model.train()
gstep, seen, t0 = 0, 0, time.time()
history = []
stop = False
for epoch in range(math.ceil(EPOCHS)):
    if stop:
        break
    rng.shuffle(train_recs)
    pbar = tqdm(train_recs, desc=f"epoch {epoch+1}", mininterval=15)
    for rec in pbar:
        try:
            loss = forward_loss(rec) / GRAD_ACCUM
            loss.backward()
        except torch.OutOfMemoryError:
            print(f"[warn] OOM on {rec['id']} -- skipped", flush=True)
            optim.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[warn] skip {rec['id']}: {e}", flush=True)
            optim.zero_grad(set_to_none=True); continue
        seen += 1
        if seen % GRAD_ACCUM:
            continue

        torch.nn.utils.clip_grad_norm_(lora_params + merger_params, MAX_GRAD_NORM)
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        gstep += 1
        if gstep <= start_step:                        # fast-forward on resume
            continue
        pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM:.3f}", step=gstep)

        if gstep % EVAL_EVERY == 0 or gstep == total_steps:
            vl = validate()
            history.append({"step": gstep, "val_loss": vl,
                            "hours": (time.time() - t0) / 3600})
            print(f"\n[eval] step {gstep} | val_loss {vl:.4f} | best {best_val:.4f}", flush=True)
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
                print("[early stop] validation loss stopped improving", flush=True)
                stop = True
                break
        if gstep >= total_steps:
            stop = True
            break

save_ckpt("last")
print(f"\ndone in {(time.time()-t0)/3600:.1f} h | best val_loss {best_val:.4f}", flush=True)
print(f"evaluate with:\n  LORA_PATH={OUT_DIR}/best RUN_TAG={RUN_TAG}_test \\\n"
      f"    python qwen3vl_roco_zeroshot_v1.py", flush=True)
