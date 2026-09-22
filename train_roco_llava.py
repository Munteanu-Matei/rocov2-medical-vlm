#!/usr/bin/env python
# LLaVA-style medical captioner on ROCOv2:
#   BioMedCLIP ViT-B/16 (frozen) -> 196 patch tokens -> 2-layer MLP -> Qwen2.5-1.5B (frozen + LoRA)
# Two-stage LLaVA-Med recipe, BOTH stages on the ROCOv2 train split:
#   Stage 1: train ONLY the MLP connector (align vision -> Qwen space)
#   Stage 2: add LoRA on Qwen; per-epoch checkpoint + validation (deberta BERTScore) + epoch selection
# Fits the 12 GB TITAN V (measured ~7.8 GB peak at batch 2, full 196 tokens).
#   smoke: STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 ROCO_TRAIN_N=32 VAL_LOSS_N=8 VAL_GEN_N=8 python train_roco_llava.py
#   full : python train_roco_llava.py
#   RUN_TAG (default vL<VISION_LAYER>, e.g. "vL-2") tags every output path so reruns never collide.
import os
os.chdir("/home/matei")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"   # ignore the stale HF token -> anonymous downloads
import json, time, random
import torch
import torch.nn as nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16                # fp16 -> AdamW-eps NaN (Week 3); bf16 has fp32 range
print(f"device={device} dtype={dtype}", flush=True)

# ── config (env-overridable) ─────────────────────────────────────────────────
STAGE1_EPOCHS = int(os.environ.get("STAGE1_EPOCHS", 2))   # connector alignment
STAGE2_EPOCHS = int(os.environ.get("STAGE2_EPOCHS", 3))   # MLP + LoRA
TRAIN_N    = int(os.environ.get("ROCO_TRAIN_N", 12000))
VAL_LOSS_N = int(os.environ.get("VAL_LOSS_N", 1000))
VAL_GEN_N  = int(os.environ.get("VAL_GEN_N", 500))
BATCH      = int(os.environ.get("BATCH", 2))
ACCUM      = int(os.environ.get("ACCUM", 4))              # effective batch = 8
MLP_LR_S1, MLP_LR_S2, LORA_LR = 1e-3, 2e-5, 2e-4          # LLaVA-1.5-style LRs
CAP_PROMPT, LLM_MAXLEN = "a photo of", 96
# Which ViT layer the patch tokens come from. -2 (penultimate) is the LLaVA standard:
# BioMedCLIP is CLS-pooled (verified: pooled output == CLS exactly), so the LAST block is
# optimised solely to assemble the CLS and its patch tokens lose local specificity.
VISION_LAYER = int(os.environ.get("VISION_LAYER", -2))
# Tags every output path for this run so a rerun (e.g. a different VISION_LAYER) never
# overwrites a previous run's checkpoints/history/val-preds. Override to set it by hand.
RUN_TAG = os.environ.get("RUN_TAG", f"vL{VISION_LAYER}")
SAVE_DIR = f"/home/matei/roco_llava_checkpoints_{RUN_TAG}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── BioMedCLIP vision encoder (frozen) + deterministic transform ────────────
import open_clip
bmc, _pt, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
vision_encoder = bmc.visual.to(device).eval()
for p in vision_encoder.parameters():
    p.requires_grad = False
image_processor = preprocess_val

# ── Qwen2.5-1.5B decoder ─────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
qwen = AutoModelForCausalLM.from_pretrained(QWEN, dtype=dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained(QWEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── the LLaVA-style model ────────────────────────────────────────────────────
class LlavaBioMed(nn.Module):
    def __init__(self, ve, qwen):
        super().__init__()
        self.ve = ve
        H = qwen.config.hidden_size
        self.mlp = nn.Sequential(nn.Linear(768, H), nn.GELU(), nn.Linear(H, H))   # 2-layer MLP connector
        self.qwen = qwen

    def visual_tokens(self, images):
        with torch.no_grad():
            if VISION_LAYER == -1:
                # last block (+ final norm), drop the CLS at index 0
                feats = self.ve.trunk.forward_features(images)[:, 1:, :]      # [B,196,768]
            else:
                # LLaVA standard: take block VISION_LAYER (e.g. -2 = penultimate).
                # get_intermediate_layers returns patch tokens only (CLS already stripped).
                feats = self.ve.trunk.get_intermediate_layers(images, n=abs(VISION_LAYER))[0]
        return self.mlp(feats.to(dtype))                              # [B,196,H]

    def forward(self, images, input_ids, attention_mask, labels):
        vis = self.visual_tokens(images)
        txt = self.qwen.get_input_embeddings()(input_ids)
        ie  = torch.cat([vis, txt], dim=1)
        att = torch.cat([torch.ones(vis.shape[:2], device=ie.device, dtype=attention_mask.dtype), attention_mask], 1)
        lab = torch.cat([torch.full(vis.shape[:2], -100, device=labels.device, dtype=labels.dtype), labels], 1)
        return self.qwen(inputs_embeds=ie, attention_mask=att, labels=lab)

    @torch.no_grad()
    def generate(self, images, prompt_ids, prompt_att, **gk):
        vis = self.visual_tokens(images)
        txt = self.qwen.get_input_embeddings()(prompt_ids)
        ie  = torch.cat([vis, txt], dim=1)
        att = torch.cat([torch.ones(vis.shape[:2], device=ie.device, dtype=prompt_att.dtype), prompt_att], 1)
        return self.qwen.generate(inputs_embeds=ie, attention_mask=att, **gk)

model = LlavaBioMed(vision_encoder, qwen).to(device)
model.mlp = model.mlp.to(dtype)

def set_mode(train):
    # gradient checkpointing (train) vs KV cache (generation)
    if train:
        model.train()
        model.qwen.config.use_cache = False
        model.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.eval()
        model.qwen.gradient_checkpointing_disable()
        model.qwen.config.use_cache = True

# ── ROCOv2 captioning data ───────────────────────────────────────────────────
import pandas as pd
from torch.utils.data import Dataset, DataLoader
ROCO_DIR = "/home/matei/rocov2"

def load_roco(split, n=None, seed=0):
    caps = pd.read_csv(os.path.join(ROCO_DIR, f"{split}_captions.csv")).dropna(subset=["Caption"]).reset_index(drop=True)
    d = os.path.join(ROCO_DIR, split)
    recs = [{"id": r.ID, "path": os.path.join(d, f"{r.ID}.jpg"), "caption": str(r.Caption)}
            for r in caps.itertuples() if os.path.isfile(os.path.join(d, f"{r.ID}.jpg"))]
    if n is not None:
        random.Random(seed).shuffle(recs); recs = recs[:n]
    return recs

train_recs    = load_roco("train", TRAIN_N)
val_loss_recs = load_roco("valid", VAL_LOSS_N)
val_gen_recs  = load_roco("valid", VAL_GEN_N)
print(f"train {len(train_recs)} | val-loss {len(val_loss_recs)} | val-gen {len(val_gen_recs)}", flush=True)

PROMPT_LEN = len(tokenizer(CAP_PROMPT).input_ids)

class ROCOCaptionDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        image = image_processor(Image.open(r["path"]).convert("RGB"))
        full  = f"{CAP_PROMPT} {r['caption']}{tokenizer.eos_token}"
        t = tokenizer(full, padding="max_length", max_length=LLM_MAXLEN, truncation=True, return_tensors="pt")
        ids, att = t.input_ids.squeeze(0), t.attention_mask.squeeze(0)
        labels = ids.clone()
        labels[att == 0] = -100
        labels[:min(PROMPT_LEN, LLM_MAXLEN)] = -100
        return {"images": image, "input_ids": ids, "attention_mask": att, "labels": labels}

train_loader = DataLoader(ROCOCaptionDataset(train_recs), batch_size=BATCH, shuffle=True, num_workers=2)
val_loader   = DataLoader(ROCOCaptionDataset(val_loss_recs), batch_size=BATCH, shuffle=False, num_workers=2)

# ── validation generation (same decoding as the other ROCO runs) + metrics ──
from tqdm.auto import tqdm
GEN_KWARGS = dict(max_new_tokens=40, min_new_tokens=8, num_beams=5, no_repeat_ngram_size=3,
                  length_penalty=1.0, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

def caption_records(recs, batch_size=8):
    p = tokenizer(CAP_PROMPT, return_tensors="pt").to(device)
    preds = []
    for i in tqdm(range(0, len(recs), batch_size), desc="val gen", mininterval=30, leave=False):
        chunk = recs[i:i+batch_size]
        pix = torch.stack([image_processor(Image.open(r["path"]).convert("RGB")) for r in chunk]).to(device)
        B = pix.shape[0]
        gen = model.generate(pix, p.input_ids.expand(B, -1), p.attention_mask.expand(B, -1), **GEN_KWARGS)
        preds.extend(s.strip() for s in tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds

@torch.no_grad()
def validation_loss(loader):
    set_mode(False)
    tot = n = 0
    for b in tqdm(loader, desc="val loss", mininterval=30, leave=False):
        out = model(b["images"].to(device), b["input_ids"].to(device),
                    b["attention_mask"].to(device), b["labels"].to(device))
        k = b["images"].shape[0]; tot += float(out.loss) * k; n += k
    return tot / n

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None
from bert_score import BERTScorer
BERT_MODEL = "microsoft/deberta-xlarge-mnli"
_bert = None

def _norm(s):
    return " ".join(str(s).lower().split())

def compute_metrics(preds, refs):
    global _bert
    preds = [p if str(p).strip() else "." for p in preds]
    refs  = [r if str(r).strip() else "." for r in refs]
    gts = {i: [_norm(refs[i])]  for i in range(len(refs))}
    res = {i: [_norm(preds[i])] for i in range(len(preds))}
    bleu, _  = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    meteor = sum(meteor_score([_norm(refs[i]).split()], _norm(preds[i]).split())
                 for i in range(len(preds))) / len(preds)
    if _bert is None:
        _bert = BERTScorer(model_type=BERT_MODEL, batch_size=8)
        _bert._tokenizer.model_max_length = 512
    _, _, F = _bert.score(preds, refs, batch_size=8, verbose=False)
    return {"BLEU-1": bleu[0], "BLEU-4": bleu[3], "METEOR": meteor, "ROUGE-L": rouge,
            "CIDEr": cider, "BERTScore-F1": F.mean().item()}

def train_epochs(optimizer, n_epochs, tag_prefix, do_val, history):
    from torch.nn.utils import clip_grad_norm_
    for epoch in range(n_epochs):
        set_mode(True)
        optimizer.zero_grad(set_to_none=True)
        running, t0 = 0.0, time.time()
        bar = tqdm(train_loader, desc=f"{tag_prefix} epoch {epoch+1}/{n_epochs}", mininterval=30)
        for step, b in enumerate(bar):
            out = model(b["images"].to(device), b["input_ids"].to(device),
                        b["attention_mask"].to(device), b["labels"].to(device))
            (out.loss / ACCUM).backward()
            if (step + 1) % ACCUM == 0:
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            running += out.loss.item()
        train_loss = running / len(train_loader)
        print(f"\n{tag_prefix} epoch {epoch+1} | {(time.time()-t0)/60:.1f} min | train {train_loss:.4f}", flush=True)
        if not do_val:
            continue
        path = save_ckpt(f"{tag_prefix}_epoch{epoch+1}")
        try:
            vloss = validation_loss(val_loader)
            set_mode(False)
            vpreds = caption_records(val_gen_recs)
            with open(f"/home/matei/roco_llava_val_preds_{tag_prefix}_epoch{epoch+1}_{RUN_TAG}.json", "w") as f:
                json.dump([{"ref": r, "pred": p} for r, p in zip(val_refs, vpreds)], f, indent=1)
            vm = compute_metrics(vpreds, val_refs)
            history.append({"epoch": epoch + 1, "stage": tag_prefix, "train_loss": train_loss,
                            "val_loss": vloss, "path": path, **vm})
            print(f"  val {vloss:.4f} | BERTScore {vm['BERTScore-F1']:.4f} | CIDEr {vm['CIDEr']:.4f} | "
                  f"ROUGE-L {vm['ROUGE-L']:.4f} | BLEU-1 {vm['BLEU-1']:.4f}", flush=True)
            print(f"  REF : {val_refs[0][:90]}", flush=True)
            print(f"  PRED: {vpreds[0][:90]}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            history.append({"epoch": epoch + 1, "stage": tag_prefix, "train_loss": train_loss,
                            "path": path, "val_error": str(e)})
            print(f"  VALIDATION FAILED (checkpoint is safe): {e}", flush=True)
        with open(f"/home/matei/roco_llava_finetune_history_{RUN_TAG}.json", "w") as f:
            json.dump(history, f, indent=1)

def save_ckpt(tag):
    lora_sd = {k: v for k, v in model.qwen.state_dict().items() if "lora" in k.lower()}
    p = os.path.join(SAVE_DIR, f"roco_llava_{tag}.pt")
    # record the vision layer so the eval script reproduces the exact feature path
    torch.save({"mlp": model.mlp.state_dict(), "qwen_lora": lora_sd,
                "vision_layer": VISION_LAYER}, p)
    return p

# ── smoke: one forward is finite ─────────────────────────────────────────────
from torch.optim import AdamW
set_mode(True)
for p in model.qwen.parameters():
    p.requires_grad = False                       # Stage 1: Qwen frozen, only MLP trains
_b = next(iter(train_loader))
_o = model(_b["images"].to(device), _b["input_ids"].to(device), _b["attention_mask"].to(device), _b["labels"].to(device))
assert torch.isfinite(_o.loss), "loss not finite"
model.zero_grad(set_to_none=True)
print(f"smoke OK | loss={float(_o.loss):.4f} | MLP-only trainable "
      f"{sum(p.numel() for p in model.mlp.parameters())/1e6:.1f}M", flush=True)

history = []
val_refs = [r["caption"] for r in val_gen_recs]

# ── STAGE 1: align the MLP connector (Qwen frozen) ───────────────────────────
print("\n===== STAGE 1: MLP connector alignment (Qwen frozen) =====", flush=True)
opt1 = AdamW(model.mlp.parameters(), lr=MLP_LR_S1)
train_epochs(opt1, STAGE1_EPOCHS, "s1", do_val=False, history=history)
save_ckpt("s1_final")

# ── STAGE 2: inject LoRA on Qwen; train MLP + LoRA, validate each epoch ──────
print("\n===== STAGE 2: MLP + LoRA on Qwen =====", flush=True)
from peft import LoraConfig, get_peft_model
model.qwen = get_peft_model(model.qwen, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM"))
lora_params = [p for p in model.qwen.parameters() if p.requires_grad]
print(f"stage-2 trainable: MLP {sum(p.numel() for p in model.mlp.parameters())/1e6:.1f}M "
      f"+ LoRA {sum(p.numel() for p in lora_params)/1e6:.1f}M", flush=True)
opt2 = AdamW([{"params": model.mlp.parameters(), "lr": MLP_LR_S2},
              {"params": lora_params,            "lr": LORA_LR}])
train_epochs(opt2, STAGE2_EPOCHS, "s2", do_val=True, history=history)

# ── epoch selection on VALIDATION (test untouched) ───────────────────────────
scored = [h for h in history if "BERTScore-F1" in h]
best = max(scored, key=lambda r: r["BERTScore-F1"]) if scored else None
print("\n" + "=" * 78, flush=True)
print("VALIDATION SUMMARY (test has not been touched)")
for h in scored:
    print(f"  {h['stage']} epoch {h['epoch']} | val_loss {h['val_loss']:.4f} | "
          f"BERTScore {h['BERTScore-F1']:.4f} | CIDEr {h['CIDEr']:.4f} | ROUGE-L {h['ROUGE-L']:.4f}")
if best is None:
    print("\nNo epoch scored; recompute from roco_llava_val_preds_*.json."); import sys; sys.exit(0)
print(f"\nBEST epoch by validation BERTScore: {best['stage']} epoch {best['epoch']}  ->  {best['path']}")
print("\nFINAL STEP -- score it ONCE on test (same decoding + deberta BERTScore):")
print(f"  CKPT_PATH={best['path']} RUN_TAG={RUN_TAG}_finetuned \\")
print(f"    /home/matei/miniconda3/envs/vlm/bin/python /home/matei/caption_roco_llava.py")
print("=" * 78, flush=True)
