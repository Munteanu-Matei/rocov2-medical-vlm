#!/usr/bin/env python
# Fine-tune the VQA-RAD BioMedVQA model (BioMedCLIP vision encoder -> linear
# translator -> GPT-2) on ROCOv2 CAPTIONING. Resumes from biomed_vqa_final.pt.
# Per epoch: checkpoint + validation loss + validation generation/metrics
# (deberta-xlarge-mnli BERTScore, comparable to the ROCOv2 leaderboard).
# Epoch is selected on VALIDATION; TEST is scored separately with
# caption_roco_biomed.py (CKPT_PATH=<best>), never here.
#   smoke: ROCO_TRAIN_N=40 EPOCHS=1 VAL_LOSS_N=16 VAL_GEN_N=16 python train_roco_biomed.py
#   full : python train_roco_biomed.py
import os
os.chdir("/home/matei")
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # models are cached; avoid flaky hub calls
import json, time, random
import torch
import torch.nn as nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

# ── config (env-overridable) ─────────────────────────────────────────────────
TRAIN_N    = int(os.environ.get("ROCO_TRAIN_N", 12000))
EPOCHS     = int(os.environ.get("EPOCHS", 3))
VAL_LOSS_N = int(os.environ.get("VAL_LOSS_N", 1000))
VAL_GEN_N  = int(os.environ.get("VAL_GEN_N", 500))
BATCH      = int(os.environ.get("BATCH", 8))
LR, WD     = 2e-5, 0.01                      # same as the VQA-RAD fine-tuning
CAP_PROMPT, LLM_MAXLEN = "a photo of", 96    # 96 keeps 97% of captions untruncated
RESUME_FROM = os.environ.get("CKPT_PATH", "/home/matei/biomed_vqa_checkpoints/biomed_vqa_final.pt")
SAVE_DIR = "/home/matei/roco_biomed_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── BioMedCLIP vision encoder + deterministic transform ─────────────────────
import open_clip
biomedclip_model, _pt, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
vision_encoder = biomedclip_model.visual
image_processor = preprocess_val

# ── model (same architecture as biomed_captioner_v2.ipynb) ───────────────────
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BioMedVQA(nn.Module):
    def __init__(self, vision_encoder, text_model_name="gpt2"):
        super().__init__()
        self.vision_encoder = vision_encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        self.llm = GPT2LMHeadModel.from_pretrained(text_model_name)
        # train only the last 6 GPT-2 blocks + final norm + LM head (same as VQA-RAD)
        TRAINABLE_BLOCKS = {6, 7, 8, 9, 10, 11}
        for name, param in self.llm.named_parameters():
            block = next((int(x) for x in name.split(".") if x.isdigit()), None)
            param.requires_grad = (block in TRAINABLE_BLOCKS) or ("ln_f" in name) or ("lm_head" in name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.translator = nn.Linear(512, self.llm.config.hidden_size)

    def forward(self, images, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            image_features = self.vision_encoder(images)
        translated = self.translator(image_features).unsqueeze(1)               # [B,1,768]
        text_emb   = self.llm.transformer.wte(input_ids)                        # [B,L,768]
        inputs_embeds = torch.cat([translated, text_emb], dim=1)
        img_att = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attn    = torch.cat([img_att, attention_mask], dim=1)
        if labels is not None:
            img_lab = torch.full((labels.shape[0], 1), -100, device=labels.device)  # image position: no loss
            labels  = torch.cat([img_lab, labels], dim=1)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)

    @torch.no_grad()
    def generate(self, images, input_ids, attention_mask, **gk):
        image_features = self.vision_encoder(images)
        translated = self.translator(image_features).unsqueeze(1)
        text_emb   = self.llm.transformer.wte(input_ids)
        inputs_embeds = torch.cat([translated, text_emb], dim=1)
        img_att = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attn    = torch.cat([img_att, attention_mask], dim=1)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attn, **gk)

model = BioMedVQA(vision_encoder).to(device)
ckpt = torch.load(RESUME_FROM, map_location=device)
model.translator.load_state_dict(ckpt["translator"])
model.llm.load_state_dict(ckpt["llm"])
print(f"resumed from {RESUME_FROM}", flush=True)
tokenizer = model.tokenizer
_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable {_tr/1e6:.1f}M params", flush=True)

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
        random.Random(seed).shuffle(recs)
        recs = recs[:n]
    return recs

train_recs    = load_roco("train", TRAIN_N)
val_loss_recs = load_roco("valid", VAL_LOSS_N)
val_gen_recs  = load_roco("valid", VAL_GEN_N)
print(f"train {len(train_recs)} | val-loss {len(val_loss_recs)} | val-gen {len(val_gen_recs)}", flush=True)

PROMPT_LEN = len(tokenizer(CAP_PROMPT).input_ids)   # GPT-2 has no BOS -> 3 tokens for "a photo of"

class ROCOCaptionDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        image = image_processor(Image.open(r["path"]).convert("RGB"))
        full  = f"{CAP_PROMPT} {r['caption']}{tokenizer.eos_token}"
        tok = tokenizer(full, padding="max_length", max_length=LLM_MAXLEN, truncation=True, return_tensors="pt")
        ids, att = tok.input_ids.squeeze(0), tok.attention_mask.squeeze(0)
        labels = ids.clone()
        labels[att == 0] = -100                        # ignore padding (real eos keeps att==1 -> supervised)
        labels[:min(PROMPT_LEN, LLM_MAXLEN)] = -100    # ignore the "a photo of" prompt
        return {"images": image, "input_ids": ids, "attention_mask": att, "labels": labels}

train_loader = DataLoader(ROCOCaptionDataset(train_recs), batch_size=BATCH, shuffle=True, num_workers=2)
val_loader   = DataLoader(ROCOCaptionDataset(val_loss_recs), batch_size=BATCH, shuffle=False, num_workers=2)

# ── validation generation (identical decoding to caption_roco_biomed.py) ─────
from tqdm.auto import tqdm
GEN_KWARGS = dict(max_new_tokens=40, min_new_tokens=8, num_beams=5, no_repeat_ngram_size=3,
                  length_penalty=1.0, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

def caption_records(recs, batch_size=16):
    tok = tokenizer(CAP_PROMPT, return_tensors="pt").to(device)
    preds = []
    for i in tqdm(range(0, len(recs), batch_size), desc="val gen", mininterval=30, leave=False):
        chunk = recs[i:i+batch_size]
        pix = torch.stack([image_processor(Image.open(r["path"]).convert("RGB")) for r in chunk]).to(device)
        B = pix.shape[0]
        gen = model.generate(pix, tok.input_ids.expand(B, -1), tok.attention_mask.expand(B, -1), **GEN_KWARGS)
        preds.extend(s.strip() for s in tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds

@torch.no_grad()
def validation_loss(loader):
    model.eval()
    tot = n = 0
    for b in tqdm(loader, desc="val loss", mininterval=30, leave=False):
        out = model(b["images"].to(device), b["input_ids"].to(device),
                    b["attention_mask"].to(device), b["labels"].to(device))
        k = b["images"].shape[0]
        tot += float(out.loss) * k; n += k
    return tot / n

# ── metrics: BLEU/ROUGE-L/CIDEr/METEOR + deberta BERTScore (comparable) ──────
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

# ── smoke: one forward is finite, and only the expected params train ─────────
_b = next(iter(train_loader))
_o = model(_b["images"].to(device), _b["input_ids"].to(device), _b["attention_mask"].to(device), _b["labels"].to(device))
assert torch.isfinite(_o.loss), "loss is not finite"
print(f"smoke OK | loss={float(_o.loss):.4f}", flush=True)

# ── training ────────────────────────────────────────────────────────────────
from torch.optim import AdamW

def save_ckpt(tag):
    p = os.path.join(SAVE_DIR, f"roco_biomed_{tag}.pt")
    torch.save({"translator": model.translator.state_dict(), "llm": model.llm.state_dict()}, p)
    return p

optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
history  = []
val_refs = [r["caption"] for r in val_gen_recs]

for epoch in range(EPOCHS):
    model.train()
    running, t0 = 0.0, time.time()
    for b in tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS}", mininterval=30):
        optimizer.zero_grad()
        out = model(b["images"].to(device), b["input_ids"].to(device),
                    b["attention_mask"].to(device), b["labels"].to(device))
        out.loss.backward()
        optimizer.step()
        running += out.loss.item()
    train_loss = running / len(train_loader)

    path = save_ckpt(f"epoch{epoch+1}")                 # checkpoint FIRST
    print(f"\nepoch {epoch+1} | {(time.time()-t0)/60:.1f} min | train {train_loss:.4f} | -> {path}", flush=True)
    try:
        vloss = validation_loss(val_loader)
        model.eval()
        vpreds = caption_records(val_gen_recs)
        with open(f"/home/matei/roco_biomed_val_preds_epoch{epoch+1}.json", "w") as f:
            json.dump([{"ref": r, "pred": p} for r, p in zip(val_refs, vpreds)], f, indent=1)
        vm = compute_metrics(vpreds, val_refs)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": vloss, "path": path, **vm})
        print(f"  val {vloss:.4f} | BERTScore {vm['BERTScore-F1']:.4f} | CIDEr {vm['CIDEr']:.4f} | "
              f"ROUGE-L {vm['ROUGE-L']:.4f} | BLEU-1 {vm['BLEU-1']:.4f}", flush=True)
        print(f"  REF : {val_refs[0][:90]}", flush=True)
        print(f"  PRED: {vpreds[0][:90]}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "path": path, "val_error": str(e)})
        print(f"  VALIDATION FAILED (checkpoint is safe): {e}", flush=True)
    with open("/home/matei/roco_biomed_finetune_history.json", "w") as f:
        json.dump(history, f, indent=1)

# ── epoch selection on VALIDATION (test untouched) ───────────────────────────
scored = [h for h in history if "BERTScore-F1" in h]
best = max(scored, key=lambda r: r["BERTScore-F1"]) if scored else None
print("\n" + "=" * 78, flush=True)
print("VALIDATION SUMMARY (test has not been touched)")
for h in history:
    if "BERTScore-F1" in h:
        print(f"  epoch {h['epoch']} | val_loss {h['val_loss']:.4f} | BERTScore {h['BERTScore-F1']:.4f} "
              f"| CIDEr {h['CIDEr']:.4f} | ROUGE-L {h['ROUGE-L']:.4f}")
    else:
        print(f"  epoch {h['epoch']} | validation failed: {h.get('val_error','?')}  (checkpoint {h['path']})")
if best is None:
    print("\nNo epoch was scored; recompute from roco_biomed_val_preds_epoch*.json.")
    import sys; sys.exit(0)
print(f"\nBEST epoch by validation BERTScore: {best['epoch']}  ->  {best['path']}")
print("\nFINAL STEP -- score it ONCE on test (same script/decoding as the zero-shot run):")
print(f"  CKPT_PATH={best['path']} RUN_TAG=biomed_finetuned \\")
print(f"    /home/matei/miniconda3/envs/vlm/bin/python /home/matei/caption_roco_biomed.py")
print("=" * 78, flush=True)
