#!/usr/bin/env python
# Fine-tune the VQA-RAD (LoRA-ViT) BLIP-2 model on a ROCOv2 train subset for CAPTIONING.
# Resumes from vqa_lora_final.pt (Sarah: "further train it on a subset").
# Per epoch: checkpoint + validation loss + validation generation/metrics.
# Test is NEVER touched here -- score the selected epoch with caption_roco.py.
#
#   smoke : ROCO_TRAIN_N=40 EPOCHS=1 VAL_GEN_N=16 VAL_LOSS_N=16 python train_roco_caption.py
#   full  : python train_roco_caption.py
import os
os.chdir("/home/matei")
import json, time, random
import torch
import torch.nn as nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"device={device} dtype={dtype}", flush=True)

# ── config (env-overridable) ─────────────────────────────────────────────────
TRAIN_N    = int(os.environ.get("ROCO_TRAIN_N", 8000))
EPOCHS     = int(os.environ.get("EPOCHS", 3))
VAL_LOSS_N = int(os.environ.get("VAL_LOSS_N", 1000))
VAL_GEN_N  = int(os.environ.get("VAL_GEN_N", 500))
BATCH, ACCUM = 2, 4
LR, LORA_LR, WD = 1e-5, 1e-4, 0.05
CAP_PROMPT, LLM_MAXLEN = "a photo of", 96
RESUME_FROM = os.environ.get("RESUME_FROM", "/home/matei/vqa_checkpoints_lora/vqa_lora_final.pt")
SAVE_DIR = "/home/matei/roco_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── load BLIP-2 + graft Q-Former text weights ────────────────────────────────
from transformers import Blip2Processor, Blip2ForConditionalGeneration

CKPT = "/home/matei/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(CKPT)
blip2 = Blip2ForConditionalGeneration.from_pretrained(CKPT, torch_dtype=dtype, low_cpu_mem_usage=True)
_QF_TEXT = torch.load("/home/matei/blip2_qformer_text_weights.pt", map_location="cpu")
qformer_word_emb = nn.Embedding.from_pretrained(_QF_TEXT["word_embeddings.weight"], freeze=False)
qformer_pos_emb  = nn.Embedding.from_pretrained(_QF_TEXT["position_embeddings.weight"], freeze=False)
qformer_text_ffn = _QF_TEXT["text_ffn"]

# ── model: query-only captioning (forward_caption = training twin) ───────────
import copy

class Blip2QFormerVQA(nn.Module):
    def __init__(self, blip2, word_emb, pos_emb, text_ffn=None):
        super().__init__()
        self.vision_model = blip2.vision_model
        self.query_tokens = blip2.query_tokens
        self.qformer      = blip2.qformer
        for _i, _layer in enumerate(self.qformer.encoder.layer):
            if not hasattr(_layer, 'intermediate'):
                _layer.intermediate = copy.deepcopy(_layer.intermediate_query)
                _layer.output       = copy.deepcopy(_layer.output_query)
                if text_ffn is not None:
                    _layer.intermediate.load_state_dict(text_ffn[_i]['intermediate'])
                    _layer.output.load_state_dict(text_ffn[_i]['output'])
        self.language_projection = blip2.language_projection
        self.language_model      = blip2.language_model
        self.qformer_word_emb    = word_emb
        self.qformer_pos_emb     = pos_emb
        self.n_query             = blip2.config.num_query_tokens

    def _qformer_features(self, pixel_values):
        image_embeds = self.vision_model(pixel_values).last_hidden_state
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        B = pixel_values.shape[0]
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_atts   = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=query_tokens.device)
        out = self.qformer(
            query_embeds=query_tokens, query_length=self.n_query, attention_mask=query_atts,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
        )
        query_output = out.last_hidden_state[:, :self.n_query, :].to(self.language_projection.weight.dtype)
        return self.language_projection(query_output)

    def forward_caption(self, pixel_values, llm_ids, llm_att, labels):
        soft = self._qformer_features(pixel_values)
        soft_att = torch.ones(soft.shape[:-1], dtype=torch.long, device=soft.device)
        text_embeds    = self.language_model.get_input_embeddings()(llm_ids)
        inputs_embeds  = torch.cat([soft, text_embeds], dim=1)
        attention_mask = torch.cat([soft_att, llm_att], dim=1)
        soft_labels = torch.full((pixel_values.shape[0], self.n_query), -100,
                                 dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([soft_labels, labels], dim=1)
        return self.language_model(inputs_embeds=inputs_embeds,
                                   attention_mask=attention_mask, labels=full_labels)

    @torch.no_grad()
    def generate_caption(self, pixel_values, prompt_ids=None, prompt_att=None, **gen_kwargs):
        soft = self._qformer_features(pixel_values)
        soft_att = torch.ones(soft.shape[:-1], dtype=torch.long, device=soft.device)
        if prompt_ids is not None:
            text_embeds    = self.language_model.get_input_embeddings()(prompt_ids)
            inputs_embeds  = torch.cat([soft, text_embeds], dim=1)
            attention_mask = torch.cat([soft_att, prompt_att], dim=1)
        else:
            inputs_embeds, attention_mask = soft, soft_att
        return self.language_model.generate(inputs_embeds=inputs_embeds,
                                            attention_mask=attention_mask, **gen_kwargs)

# ── instantiate, re-inject LoRA, resume from the VQA-RAD checkpoint ──────────
from peft import LoraConfig, inject_adapter_in_model

model = Blip2QFormerVQA(blip2, qformer_word_emb, qformer_pos_emb, qformer_text_ffn)
for p in model.vision_model.parameters():
    p.requires_grad = False
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["qkv", "projection", "fc1", "fc2"], bias="none")
inject_adapter_in_model(lora_cfg, model.vision_model)
model = model.to(device)
model.qformer_word_emb = model.qformer_word_emb.to(device, model.query_tokens.dtype)
model.qformer_pos_emb  = model.qformer_pos_emb.to(device, model.query_tokens.dtype)

ckpt = torch.load(RESUME_FROM, map_location=device)
model.qformer.load_state_dict(ckpt["qformer"])
model.language_projection.load_state_dict(ckpt["language_projection"])
with torch.no_grad():
    model.query_tokens.copy_(ckpt["query_tokens"].to(device, model.query_tokens.dtype))
model.qformer_word_emb.load_state_dict(ckpt["qformer_word_emb"])
model.qformer_pos_emb.load_state_dict(ckpt["qformer_pos_emb"])
model.vision_model.load_state_dict(ckpt["vit_lora"], strict=False)
print(f"resumed from {RESUME_FROM}", flush=True)

def set_trainable(m, flag):
    for p in m.parameters():
        p.requires_grad = flag

set_trainable(model.language_model, False)
set_trainable(model.qformer, True)
set_trainable(model.language_projection, True)
model.query_tokens.requires_grad = True
model.qformer_word_emb.weight.requires_grad = False   # unused in captioning
model.qformer_pos_emb.weight.requires_grad  = False

def set_mode(train: bool):
    # gradient checkpointing OFF for generation (it forces use_cache=False -> slow beams)
    if train:
        model.train()
        model.vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.language_model.gradient_checkpointing_enable()
        model.language_model.config.use_cache = False
    else:
        model.eval()
        model.vision_model.gradient_checkpointing_disable()
        model.language_model.gradient_checkpointing_disable()
        model.language_model.config.use_cache = True

set_mode(True)
_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
_to = sum(p.numel() for p in model.parameters())
print(f"trainable {_tr/1e6:.1f}M / total {_to/1e6:.1f}M", flush=True)

# ── data ─────────────────────────────────────────────────────────────────────
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader

ROCO_DIR = "/home/matei/rocov2"

def load_roco(split, n=None, seed=0):
    caps = pd.read_csv(os.path.join(ROCO_DIR, f"{split}_captions.csv")).dropna(subset=["Caption"]).reset_index(drop=True)
    d = os.path.join(ROCO_DIR, split)
    recs = [{"id": r.ID, "path": os.path.join(d, f"{r.ID}.jpg"), "caption": str(r.Caption)}
            for r in caps.itertuples() if os.path.isfile(os.path.join(d, f"{r.ID}.jpg"))]
    if n is not None:
        random.Random(seed).shuffle(recs)
        recs = recs[:n]
    return recs

train_recs    = load_roco("train", TRAIN_N)
val_loss_recs = load_roco("valid", VAL_LOSS_N)
val_gen_recs  = load_roco("valid", VAL_GEN_N)
print(f"train {len(train_recs)} | val-loss {len(val_loss_recs)} | val-gen {len(val_gen_recs)}", flush=True)

PROMPT_LEN = processor.tokenizer(CAP_PROMPT, return_tensors="pt").input_ids.shape[1]
PAD_ID = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None \
         else processor.tokenizer.eos_token_id

class ROCOCaptionDataset(TorchDataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        image = Image.open(r["path"]).convert("RGB")
        pixel = processor(images=image, return_tensors="pt").pixel_values[0]
        full  = f"{CAP_PROMPT} {r['caption']}{processor.tokenizer.eos_token}"
        llm   = processor.tokenizer(full, truncation=True, max_length=LLM_MAXLEN, return_tensors="pt")
        ids, att = llm.input_ids[0], llm.attention_mask[0]
        labels = ids.clone()
        labels[:min(PROMPT_LEN, labels.shape[0])] = -100
        return {"pixel_values": pixel, "llm_ids": ids, "llm_att": att, "labels": labels}

def collate_fn(batch):
    L, n = max(b["llm_ids"].shape[0] for b in batch), len(batch)
    ids = torch.full((n, L), PAD_ID, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    lab = torch.full((n, L), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        k = b["llm_ids"].shape[0]
        ids[i, :k], att[i, :k], lab[i, :k] = b["llm_ids"], b["llm_att"], b["labels"]
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "llm_ids": ids, "llm_att": att, "labels": lab}

train_loader = DataLoader(ROCOCaptionDataset(train_recs), batch_size=BATCH, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(ROCOCaptionDataset(val_loss_recs), batch_size=BATCH, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)

# ── validation: generation (IDENTICAL config to caption_roco.py) + metrics ───
from tqdm.auto import tqdm

GEN_KWARGS = dict(max_new_tokens=40, min_new_tokens=8, num_beams=5,
                  no_repeat_ngram_size=3, length_penalty=1.0,
                  eos_token_id=processor.tokenizer.eos_token_id,
                  pad_token_id=processor.tokenizer.eos_token_id)

def _load_pixels(paths):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return processor(images=imgs, return_tensors="pt").pixel_values.to(device, dtype)

def caption_records(recs, prompt=CAP_PROMPT, batch_size=8):
    base = processor.tokenizer(prompt, return_tensors="pt").to(device) if prompt else None
    preds = []
    for i in tqdm(range(0, len(recs), batch_size), desc="val captioning", mininterval=30, leave=False):
        chunk = recs[i:i+batch_size]
        pix = _load_pixels([r["path"] for r in chunk])
        B = pix.shape[0]
        if base is not None:
            pid, pat = base.input_ids.expand(B, -1), base.attention_mask.expand(B, -1)
        else:
            pid = pat = None
        gen = model.generate_caption(pix, pid, pat, **GEN_KWARGS)
        preds.extend(s.strip() for s in processor.tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds

@torch.no_grad()
def validation_loss(loader):
    set_mode(False)
    tot = n = 0
    for b in tqdm(loader, desc="val loss", mininterval=30, leave=False):
        out = model.forward_caption(b["pixel_values"].to(device, dtype), b["llm_ids"].to(device),
                                    b["llm_att"].to(device), b["labels"].to(device))
        k = b["llm_ids"].shape[0]
        tot += float(out.loss) * k
        n += k
    return tot / n

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bertscore

def _norm(s):
    return " ".join(str(s).lower().split())

def compute_caption_metrics(preds, refs):
    # A partially-trained model can emit a caption that is empty after decoding;
    # BERTScore's empty-string branch then crashes (build_inputs_with_special_tokens).
    # Replace empty/whitespace-only strings with "." so every scorer stays happy.
    preds = [p if str(p).strip() else "." for p in preds]
    refs  = [r if str(r).strip() else "." for r in refs]
    gts = {i: [_norm(refs[i])]  for i in range(len(refs))}
    res = {i: [_norm(preds[i])] for i in range(len(preds))}
    bleu, _  = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    meteor = sum(meteor_score([_norm(refs[i]).split()], _norm(preds[i]).split())
                 for i in range(len(preds))) / len(preds)
    _, _, F = bertscore(preds, refs, lang="en", verbose=False)
    return {"BLEU-1": bleu[0], "BLEU-2": bleu[1], "BLEU-3": bleu[2], "BLEU-4": bleu[3],
            "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider, "BERTScore-F1": F.mean().item()}

# ── smoke check: LoRA adapters must receive gradients (use_reentrant regression) ──
set_mode(True)
_b = next(iter(train_loader))
_o = model.forward_caption(_b["pixel_values"].to(device, dtype), _b["llm_ids"].to(device),
                           _b["llm_att"].to(device), _b["labels"].to(device))
assert torch.isfinite(_o.loss), "loss is not finite"
_o.loss.backward()
_named = dict(model.named_parameters())
_lora  = [n for n, p in _named.items() if "lora" in n.lower() and p.requires_grad]
_grad  = [n for n in _lora if _named[n].grad is not None]
assert len(_lora) > 0 and len(_grad) == len(_lora), "LoRA adapters received no gradient!"
model.zero_grad(set_to_none=True)
print(f"smoke OK | loss={float(_o.loss):.4f} | LoRA grads {len(_grad)}/{len(_lora)}", flush=True)

# ── training ────────────────────────────────────────────────────────────────
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

def save_ckpt(tag):
    ck = {
        "qformer":             model.qformer.state_dict(),
        "language_projection": model.language_projection.state_dict(),
        "query_tokens":        model.query_tokens.detach().cpu(),
        "qformer_word_emb":    model.qformer_word_emb.state_dict(),
        "qformer_pos_emb":     model.qformer_pos_emb.state_dict(),
        "vit_lora":            {k: v for k, v in model.vision_model.state_dict().items() if "lora" in k.lower()},
    }
    p = os.path.join(SAVE_DIR, f"roco_caption_{tag}.pt")
    torch.save(ck, p)
    return p

_no_decay_keys = ("bias", "norm", "query_tokens")
_lora_p, _decay, _no_decay = [], [], []
for _n, _p in model.named_parameters():
    if not _p.requires_grad:
        continue
    if "lora" in _n.lower():
        _lora_p.append(_p)
    elif _p.ndim <= 1 or any(k in _n.lower() for k in _no_decay_keys):
        _no_decay.append(_p)
    else:
        _decay.append(_p)
optimizer = AdamW([
    {"params": _decay,    "weight_decay": WD,  "lr": LR},
    {"params": _no_decay, "weight_decay": 0.0, "lr": LR},
    {"params": _lora_p,   "weight_decay": 0.0, "lr": LORA_LR},
])
_opt_steps = max(1, (len(train_loader) // ACCUM) * EPOCHS)
scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(0.1 * _opt_steps)), _opt_steps)
print(f"optimizer steps: {_opt_steps}", flush=True)

history  = []
val_refs = [r["caption"] for r in val_gen_recs]

for epoch in range(EPOCHS):
    set_mode(True)
    optimizer.zero_grad(set_to_none=True)
    running, t0 = 0.0, time.time()
    bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS}", mininterval=30)
    for step, b in enumerate(bar):
        out = model.forward_caption(b["pixel_values"].to(device, dtype), b["llm_ids"].to(device),
                                    b["llm_att"].to(device), b["labels"].to(device))
        (out.loss / ACCUM).backward()
        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        running += out.loss.item()
        bar.set_postfix({"loss": out.loss.item()})
    train_loss = running / len(train_loader)

    # checkpoint FIRST -- an epoch's weights must never be lost to a scoring bug
    path = save_ckpt(f"epoch{epoch+1}")
    print(f"\nepoch {epoch+1} | {(time.time()-t0)/60:.0f} min | train {train_loss:.4f} | -> {path}", flush=True)
    try:
        vloss = validation_loss(val_loader)
        set_mode(False)
        vpreds = caption_records(val_gen_recs)
        # persist raw val predictions so metrics can be recomputed offline if needed
        with open(f"/home/matei/roco_val_preds_epoch{epoch+1}.json", "w") as f:
            json.dump([{"ref": r, "pred": p} for r, p in zip(val_refs, vpreds)], f, indent=1)
        vm = compute_caption_metrics(vpreds, val_refs)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": vloss, "path": path, **vm})
        print(f"  val {vloss:.4f} | BERTScore {vm['BERTScore-F1']:.4f} | CIDEr {vm['CIDEr']:.4f} | "
              f"ROUGE-L {vm['ROUGE-L']:.4f} | BLEU-1 {vm['BLEU-1']:.4f}", flush=True)
        print(f"  REF : {val_refs[0][:95]}", flush=True)
        print(f"  PRED: {vpreds[0][:95]}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "path": path, "val_error": str(e)})
        print(f"  VALIDATION FAILED (checkpoint is safe, training continues): {e}", flush=True)
    with open("/home/matei/roco_finetune_history.json", "w") as f:
        json.dump(history, f, indent=1)

# ── epoch selection on VALIDATION (test untouched) ───────────────────────────
scored = [h for h in history if "BERTScore-F1" in h]   # skip any epoch whose validation failed
best = max(scored, key=lambda r: r["BERTScore-F1"]) if scored else None
print("\n" + "=" * 78, flush=True)
print("VALIDATION SUMMARY (test has not been touched)")
for h in history:
    if "BERTScore-F1" in h:
        print(f"  epoch {h['epoch']} | val_loss {h['val_loss']:.4f} | BERTScore {h['BERTScore-F1']:.4f} "
              f"| CIDEr {h['CIDEr']:.4f} | ROUGE-L {h['ROUGE-L']:.4f}")
    else:
        print(f"  epoch {h['epoch']} | validation failed: {h.get('val_error', '?')}  (checkpoint {h['path']})")
if best is None:
    print("\nNo epoch was scored; recompute metrics from the saved roco_val_preds_epoch*.json files.")
    import sys; sys.exit(0)
print(f"\nBEST epoch by validation BERTScore: {best['epoch']}  ->  {best['path']}")
print("\nFINAL STEP -- score it ONCE on test, same code path as zero-shot:")
print(f"  CKPT_PATH={best['path']} RUN_TAG=finetuned \\")
print(f"    /home/matei/miniconda3/envs/vlm/bin/python /home/matei/caption_roco.py")
print("=" * 78, flush=True)
