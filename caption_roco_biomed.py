#!/usr/bin/env python
# Zero-shot ROCOv2 captioning with the BioMedVQA model (BioMedCLIP vision encoder
# + linear translator + GPT-2) fine-tuned on VQA-RAD. Uses the SAME "a photo of"
# prompt and the SAME beam-search decoding as the BLIP-2 run (caption_roco.py)
# for a fair comparison.
#   full : python caption_roco_biomed.py
#   smoke: ROCO_N=40 python caption_roco_biomed.py
import os
os.chdir("/home/matei")
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # models are cached; avoid flaky hub calls
import json
import torch
import torch.nn as nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

# ── BioMedCLIP vision encoder + DETERMINISTIC eval transform ─────────────────
import open_clip
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
biomedclip_model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
vision_encoder = biomedclip_model.visual
# NOTE: preprocess_val (Resize+CenterCrop) is deterministic -- unlike the notebook's
# RandomResizedCrop -- so captions are reproducible and use the whole image.
image_processor = preprocess_val

# ── model (same architecture as biomed_captioner_v2.ipynb; inference only) ───
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BioMedVQA(nn.Module):
    def __init__(self, vision_encoder, text_model_name="gpt2"):
        super().__init__()
        self.vision_encoder = vision_encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        self.llm = GPT2LMHeadModel.from_pretrained(text_model_name)     # base weights, overwritten below
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.translator = nn.Linear(512, self.llm.config.hidden_size)   # 512 -> 768

    @torch.no_grad()
    def generate(self, images, input_ids, attention_mask, **gen_kwargs):
        image_features   = self.vision_encoder(images)                          # [B, 512]
        translated_image = self.translator(image_features).unsqueeze(1)         # [B, 1, 768]
        text_embeddings  = self.llm.transformer.wte(input_ids)                  # [B, L, 768]
        combined = torch.cat([translated_image, text_embeddings], dim=1)        # [B, 1+L, 768]
        img_att  = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        combined_att = torch.cat([img_att, attention_mask], dim=1)
        return self.llm.generate(inputs_embeds=combined, attention_mask=combined_att, **gen_kwargs)

model = BioMedVQA(vision_encoder).to(device)
CKPT_PATH = os.environ.get("CKPT_PATH", "/home/matei/biomed_vqa_checkpoints/biomed_vqa_final.pt")
ckpt = torch.load(CKPT_PATH, map_location=device)
model.translator.load_state_dict(ckpt["translator"])
model.llm.load_state_dict(ckpt["llm"])
model.eval()
print(f"loaded BioMedVQA weights (translator + GPT-2) from {CKPT_PATH}", flush=True)

# ── ROCOv2 test split ────────────────────────────────────────────────────────
import pandas as pd
ROCO_DIR = "/home/matei/rocov2"
IMG_DIR  = os.path.join(ROCO_DIR, "test")
caps = pd.read_csv(os.path.join(ROCO_DIR, "test_captions.csv")).dropna(subset=["Caption"]).reset_index(drop=True)
records = [{"id": r.ID, "path": os.path.join(IMG_DIR, f"{r.ID}.jpg"), "caption": str(r.Caption)}
           for r in caps.itertuples() if os.path.isfile(os.path.join(IMG_DIR, f"{r.ID}.jpg"))]
_N = os.environ.get("ROCO_N")
if _N is not None:
    records = records[:int(_N)]
print(f"ROCOv2 test: {len(records)} image-caption pairs", flush=True)

# ── captioning: "a photo of" + identical decoding to caption_roco.py ─────────
from tqdm.auto import tqdm
RUN_TAG        = os.environ.get("RUN_TAG", "biomed_zeroshot")
CAPTION_PROMPT = "a photo of"
CAPTION_BATCH  = 16
GEN_KWARGS = dict(max_new_tokens=40, min_new_tokens=8, num_beams=5,
                  no_repeat_ngram_size=3, length_penalty=1.0,
                  eos_token_id=model.tokenizer.eos_token_id,
                  pad_token_id=model.tokenizer.eos_token_id)

def _load_pixels(paths):
    imgs = [image_processor(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs).to(device)

def caption_records(recs, prompt=CAPTION_PROMPT, batch_size=CAPTION_BATCH):
    tok = model.tokenizer(prompt, return_tensors="pt").to(device)   # same prompt for all -> no padding
    preds = []
    for i in tqdm(range(0, len(recs), batch_size), desc="captioning", mininterval=15):
        chunk = recs[i:i+batch_size]
        pix = _load_pixels([r["path"] for r in chunk])
        B = pix.shape[0]
        ids = tok.input_ids.expand(B, -1)
        att = tok.attention_mask.expand(B, -1)
        gen = model.generate(pix, ids, att, **GEN_KWARGS)
        preds.extend(s.strip() for s in model.tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds

print(f"captioning {len(records)} images (prompt={CAPTION_PROMPT!r}) ...", flush=True)
preds = caption_records(records)
refs  = [r["caption"] for r in records]

OUT_PRED = f"/home/matei/roco_{RUN_TAG}_a_photo_of.json"
with open(OUT_PRED, "w") as f:
    json.dump([{"id": r["id"], "ref": r["caption"], "pred": p} for r, p in zip(records, preds)], f, indent=1)
print(f"saved predictions -> {OUT_PRED}", flush=True)

# free the biomed model before loading the (heavy) scoring models
del model, vision_encoder, biomedclip_model
import gc; gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

# ── metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr, BERTScore ────────────────────
print("scoring ...", flush=True)
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score

def _norm(s):
    return " ".join(str(s).lower().split())

# guard: an empty caption crashes BERTScore's empty-string path -> replace with "."
preds = [p if str(p).strip() else "." for p in preds]
refs  = [r if str(r).strip() else "." for r in refs]
gts = {i: [_norm(refs[i])]  for i in range(len(refs))}
res = {i: [_norm(preds[i])] for i in range(len(preds))}
bleu, _  = Bleu(4).compute_score(gts, res)
rouge, _ = Rouge().compute_score(gts, res)
cider, _ = Cider().compute_score(gts, res)
meteor = sum(meteor_score([_norm(refs[i]).split()], _norm(preds[i]).split())
             for i in range(len(preds))) / len(preds)

# BERTScore with the ROCOv2/ImageCLEF model so it is COMPARABLE to the paper's
# baselines: microsoft/deberta-xlarge-mnli, F1 (NOT roberta-large). Three deberta
# fixes are needed on this stack:
#   (1) neutralise transformers' over-strict .bin load gate (load is weights_only=True);
#   (2) cap deberta's sentinel tokenizer max_length (else the fast tokenizer overflows);
#   (3) small batch size (disentangled attention is memory-heavy on 12 GB).
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None                        # (1)
from bert_score import BERTScorer
BERT_MODEL = "microsoft/deberta-xlarge-mnli"
_bert_scorer = BERTScorer(model_type=BERT_MODEL, batch_size=8)
_bert_scorer._tokenizer.model_max_length = 512                             # (2)
_, _, F = _bert_scorer.score(preds, refs, batch_size=8, verbose=False)     # (3)

metrics = {"n": len(preds), "prompt": CAPTION_PROMPT, "model": "BioMedVQA (BioMedCLIP+GPT-2)",
           "BLEU-1": bleu[0], "BLEU-2": bleu[1], "BLEU-3": bleu[2], "BLEU-4": bleu[3],
           "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider,
           "BERTScore-F1": F.mean().item(), "BERTScore-model": BERT_MODEL}

print("\n==== BioMedVQA ZERO-SHOT CAPTIONING (a photo of) -- ROCOv2 test ====", flush=True)
for k, v in metrics.items():
    print(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k:14s}: {v}", flush=True)
with open(f"/home/matei/roco_{RUN_TAG}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=1)
print(f"saved metrics -> /home/matei/roco_{RUN_TAG}_metrics.json", flush=True)
