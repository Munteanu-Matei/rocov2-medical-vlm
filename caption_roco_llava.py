#!/usr/bin/env python
# Score the LLaVA-style BioMedCLIP->MLP->Qwen model on the ROCOv2 TEST set.
# Same "a photo of" prompt, same beam decoding, and same deberta-xlarge-mnli
# BERTScore as the other ROCO runs, so results are directly comparable.
#   CKPT_PATH=<roco_llava_...pt> RUN_TAG=llava_finetuned python caption_roco_llava.py
#   smoke: ROCO_N=40 CKPT_PATH=<...> python caption_roco_llava.py
import os
os.chdir("/home/matei")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
import json
import torch
import torch.nn as nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
print(f"device={device} dtype={dtype}", flush=True)

# Which ViT layer the patch tokens come from. This is READ FROM THE CHECKPOINT below, so
# the eval always reproduces the feature path the model was TRAINED with. Checkpoints
# saved before this field existed were trained on the last block -> default -1.
VISION_LAYER = -1

# ── BioMedCLIP (frozen) + deterministic transform ───────────────────────────
import open_clip
bmc, _pt, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
vision_encoder = bmc.visual.to(device).eval()
image_processor = preprocess_val

# ── Qwen2.5-1.5B ─────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
qwen = AutoModelForCausalLM.from_pretrained(QWEN, dtype=dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained(QWEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
qwen.config.use_cache = True

class LlavaBioMed(nn.Module):
    def __init__(self, ve, qwen):
        super().__init__()
        self.ve = ve
        H = qwen.config.hidden_size
        self.mlp = nn.Sequential(nn.Linear(768, H), nn.GELU(), nn.Linear(H, H))
        self.qwen = qwen
    def visual_tokens(self, images):
        with torch.no_grad():
            if VISION_LAYER == -1:
                feats = self.ve.trunk.forward_features(images)[:, 1:, :]   # last block, drop CLS
            else:
                # LLaVA standard (-2 = penultimate); CLS already stripped by timm
                feats = self.ve.trunk.get_intermediate_layers(images, n=abs(VISION_LAYER))[0]
        return self.mlp(feats.to(dtype))
    @torch.no_grad()
    def generate(self, images, prompt_ids, prompt_att, **gk):
        vis = self.visual_tokens(images)
        txt = self.qwen.get_input_embeddings()(prompt_ids)
        ie  = torch.cat([vis, txt], dim=1)
        att = torch.cat([torch.ones(vis.shape[:2], device=ie.device, dtype=prompt_att.dtype), prompt_att], 1)
        return self.qwen.generate(inputs_embeds=ie, attention_mask=att, **gk)

model = LlavaBioMed(vision_encoder, qwen).to(device)
model.mlp = model.mlp.to(dtype)

# inject the SAME LoRA structure, then load the trained MLP + LoRA weights
from peft import LoraConfig, get_peft_model
model.qwen = get_peft_model(model.qwen, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM"))

CKPT_PATH = os.environ.get("CKPT_PATH", "/home/matei/roco_llava_checkpoints/roco_llava_s2_final.pt")
RUN_TAG   = os.environ.get("RUN_TAG", "llava_finetuned")
ckpt = torch.load(CKPT_PATH, map_location=device)
# reproduce the exact feature path this checkpoint was trained with.
# priority: explicit env override > value stored in the checkpoint > -1 (pre-field checkpoints)
_vl = os.environ.get("VISION_LAYER")
VISION_LAYER = int(_vl) if _vl is not None else int(ckpt.get("vision_layer", -1))
model.mlp.load_state_dict(ckpt["mlp"])
model.qwen.load_state_dict(ckpt["qwen_lora"], strict=False)   # LoRA adapters only (base already loaded)
model.eval()
print(f"loaded LLaVA weights (MLP + Qwen LoRA) from {CKPT_PATH} | vision_layer={VISION_LAYER}", flush=True)

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

# ── captioning ("a photo of", same beam decoding as the other runs) ─────────
from tqdm.auto import tqdm
CAPTION_PROMPT = "a photo of"
CAPTION_BATCH  = 8
GEN_KWARGS = dict(max_new_tokens=40, min_new_tokens=8, num_beams=5, no_repeat_ngram_size=3,
                  length_penalty=1.0, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

def caption_records(recs, batch_size=CAPTION_BATCH):
    p = tokenizer(CAPTION_PROMPT, return_tensors="pt").to(device)
    preds = []
    for i in tqdm(range(0, len(recs), batch_size), desc="captioning", mininterval=15):
        chunk = recs[i:i+batch_size]
        pix = torch.stack([image_processor(Image.open(r["path"]).convert("RGB")) for r in chunk]).to(device)
        B = pix.shape[0]
        gen = model.generate(pix, p.input_ids.expand(B, -1), p.attention_mask.expand(B, -1), **GEN_KWARGS)
        preds.extend(s.strip() for s in tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds

print(f"captioning {len(records)} images (prompt={CAPTION_PROMPT!r}) ...", flush=True)
preds = caption_records(records)
refs  = [r["caption"] for r in records]

OUT_PRED = f"/home/matei/roco_{RUN_TAG}_a_photo_of.json"
with open(OUT_PRED, "w") as f:
    json.dump([{"id": r["id"], "ref": r["caption"], "pred": p} for r, p in zip(records, preds)], f, indent=1)
print(f"saved predictions -> {OUT_PRED}", flush=True)

# free the model before the heavy scoring model
del model, vision_encoder, bmc, qwen
import gc; gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

# ── metrics: BLEU/ROUGE-L/CIDEr/METEOR + deberta BERTScore (comparable) ──────
print("scoring ...", flush=True)
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score

def _norm(s):
    return " ".join(str(s).lower().split())

preds = [p if str(p).strip() else "." for p in preds]
refs  = [r if str(r).strip() else "." for r in refs]
gts = {i: [_norm(refs[i])]  for i in range(len(refs))}
res = {i: [_norm(preds[i])] for i in range(len(preds))}
bleu, _  = Bleu(4).compute_score(gts, res)
rouge, _ = Rouge().compute_score(gts, res)
cider, _ = Cider().compute_score(gts, res)
meteor = sum(meteor_score([_norm(refs[i]).split()], _norm(preds[i]).split())
             for i in range(len(preds))) / len(preds)

import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None
from bert_score import BERTScorer
BERT_MODEL = "microsoft/deberta-xlarge-mnli"
_bert_scorer = BERTScorer(model_type=BERT_MODEL, batch_size=8)
_bert_scorer._tokenizer.model_max_length = 512
_, _, F = _bert_scorer.score(preds, refs, batch_size=8, verbose=False)

metrics = {"n": len(preds), "prompt": CAPTION_PROMPT, "model": "LLaVA-BioMedCLIP-Qwen2.5-1.5B",
           "BLEU-1": bleu[0], "BLEU-2": bleu[1], "BLEU-3": bleu[2], "BLEU-4": bleu[3],
           "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider,
           "BERTScore-F1": F.mean().item(), "BERTScore-model": BERT_MODEL}

print(f"\n==== LLaVA (BioMedCLIP+Qwen2.5-1.5B) CAPTIONING -- ROCOv2 test ====", flush=True)
for k, v in metrics.items():
    print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}", flush=True)
with open(f"/home/matei/roco_{RUN_TAG}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=1)
print(f"saved metrics -> /home/matei/roco_{RUN_TAG}_metrics.json", flush=True)
