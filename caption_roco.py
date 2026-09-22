#!/usr/bin/env python
# Zero-shot ROCOv2 captioning with the VQA-RAD (LoRA-ViT) model, "a photo of" prompt.
# Headless version of blip-2_roco_captioning_zeroshot.ipynb for tmux.
#   full run:   python caption_roco.py
#   smoke test: ROCO_N=40 python caption_roco.py     (first 40 images only)
import os
os.chdir("/home/matei")
import json
import torch
import torch.nn as nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"device={device} dtype={dtype}", flush=True)

# ── load BLIP-2 + graft Q-Former text weights ────────────────────────────────
from typing import Any
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BertTokenizer

CKPT = "/home/matei/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(CKPT)
blip2 = Blip2ForConditionalGeneration.from_pretrained(CKPT, torch_dtype=dtype, low_cpu_mem_usage=True)
qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_QF_TEXT = torch.load("/home/matei/blip2_qformer_text_weights.pt", map_location="cpu")
qformer_word_emb = nn.Embedding.from_pretrained(_QF_TEXT["word_embeddings.weight"], freeze=False)
qformer_pos_emb  = nn.Embedding.from_pretrained(_QF_TEXT["position_embeddings.weight"], freeze=False)
qformer_text_ffn = _QF_TEXT["text_ffn"]

# ── model (identical to the notebook; captioning = query-only path) ──────────
import copy

class Blip2QFormerVQA(nn.Module):
    def __init__(self, blip2, word_emb, pos_emb, text_ffn=None):
        super().__init__()
        self.vision_model        = blip2.vision_model
        self.query_tokens        = blip2.query_tokens
        self.qformer             = blip2.qformer
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

    def _qformer_features(self, pixel_values, q_ids=None, q_att=None):
        image_embeds = self.vision_model(pixel_values).last_hidden_state
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        B = pixel_values.shape[0]
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_atts   = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=query_tokens.device)
        if q_ids is not None:
            seq_len  = q_ids.shape[1]
            pos_ids  = torch.arange(seq_len, device=q_ids.device).unsqueeze(0)
            text_emb = (self.qformer_word_emb(q_ids) + self.qformer_pos_emb(pos_ids)).to(query_tokens.dtype)
            query_embeds   = torch.cat([query_tokens, text_emb], dim=1)
            attention_mask = torch.cat([query_atts, q_att], dim=1)
        else:
            query_embeds   = query_tokens
            attention_mask = query_atts
        out = self.qformer(
            query_embeds=query_embeds, query_length=self.n_query, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
        )
        query_output = out.last_hidden_state[:, :self.n_query, :].to(self.language_projection.weight.dtype)
        return self.language_projection(query_output)

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
        return self.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs)

# ── instantiate, re-inject ViT LoRA, load the VQA-RAD checkpoint ─────────────
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

LOAD_PATH = os.environ.get("CKPT_PATH", "/home/matei/vqa_checkpoints_lora/vqa_lora_final.pt")
RUN_TAG   = os.environ.get("RUN_TAG", "zeroshot")
ckpt = torch.load(LOAD_PATH, map_location=device)
model.qformer.load_state_dict(ckpt["qformer"])
model.language_projection.load_state_dict(ckpt["language_projection"])
with torch.no_grad():
    model.query_tokens.copy_(ckpt["query_tokens"].to(device, model.query_tokens.dtype))
model.qformer_word_emb.load_state_dict(ckpt["qformer_word_emb"])
model.qformer_pos_emb.load_state_dict(ckpt["qformer_pos_emb"])
model.vision_model.load_state_dict(ckpt["vit_lora"], strict=False)
model.eval()
print(f"loaded VQA-RAD (LoRA-ViT) weights from {LOAD_PATH}", flush=True)

# ── ROCOv2 test split ────────────────────────────────────────────────────────
import pandas as pd
ROCO_DIR = "/home/matei/rocov2"
IMG_DIR  = os.path.join(ROCO_DIR, "test")
caps = pd.read_csv(os.path.join(ROCO_DIR, "test_captions.csv")).dropna(subset=["Caption"]).reset_index(drop=True)
def roco_image_path(_id):
    return os.path.join(IMG_DIR, f"{_id}.jpg")
records = [{"id": r.ID, "path": roco_image_path(r.ID), "caption": str(r.Caption)}
           for r in caps.itertuples() if os.path.isfile(roco_image_path(r.ID))]

_N = os.environ.get("ROCO_N")
if _N is not None:
    records = records[:int(_N)]
print(f"ROCOv2 test: {len(records)} image-caption pairs", flush=True)

# ── captioning ("a photo of", beam search — fixed config) ───────────────────
from tqdm.auto import tqdm
CAPTION_PROMPT = "a photo of"
CAPTION_BATCH  = 8
GEN_KWARGS = dict(max_new_tokens=40, min_new_tokens=8, num_beams=5,
                  no_repeat_ngram_size=3, length_penalty=1.0,
                  eos_token_id=processor.tokenizer.eos_token_id,
                  pad_token_id=processor.tokenizer.eos_token_id)

def _load_pixels(paths):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return processor(images=imgs, return_tensors="pt").pixel_values.to(device, dtype)

def caption_records(recs, prompt=CAPTION_PROMPT, batch_size=CAPTION_BATCH):
    base = processor.tokenizer(prompt, return_tensors="pt").to(device) if prompt else None
    preds = []
    for i in tqdm(range(0, len(recs), batch_size), desc="captioning", mininterval=15):
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

print(f"captioning {len(records)} images (prompt={CAPTION_PROMPT!r}) ...", flush=True)
preds = caption_records(records)
refs  = [r["caption"] for r in records]

OUT_PRED = f"/home/matei/roco_{RUN_TAG}_a_photo_of.json"
with open(OUT_PRED, "w") as f:
    json.dump([{"id": r["id"], "ref": r["caption"], "pred": p} for r, p in zip(records, preds)], f, indent=1)
print(f"saved predictions -> {OUT_PRED}", flush=True)

# ── metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr, BERTScore ───────────────────
print("scoring ...", flush=True)
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bertscore

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
_, _, F = bertscore(preds, refs, lang="en", verbose=False)
metrics = {"n": len(preds), "prompt": CAPTION_PROMPT,
           "BLEU-1": bleu[0], "BLEU-2": bleu[1], "BLEU-3": bleu[2], "BLEU-4": bleu[3],
           "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider, "BERTScore-F1": F.mean().item()}

print("\n==== ZERO-SHOT CAPTIONING (a photo of) -- ROCOv2 test ====", flush=True)
for k, v in metrics.items():
    print(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k:14s}: {v}", flush=True)
with open(f"/home/matei/roco_{RUN_TAG}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=1)
print(f"saved metrics -> /home/matei/roco_{RUN_TAG}_metrics.json", flush=True)
