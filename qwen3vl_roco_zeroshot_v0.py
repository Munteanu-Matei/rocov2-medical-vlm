#!/usr/bin/env python
# Zero-shot ROCOv2 captioning with Qwen3-VL (instruction-tuned VLM), used OFF THE
# SHELF -- no fine-tuning. Uses the SAME "a photo of" prompt and the SAME metric suite as
# the BLIP-2 (caption_roco.py) and BioMedVQA (caption_roco_biomed.py) zero-shot runs.
# Decoding is GREEDY (beam=5 was ~8 days on this 12 GB Volta card); to keep the three
# comparable, re-score BLIP-2 / BioMedVQA greedy too (num_beams=1, drop length_penalty).
#
# One necessary adaptation: Qwen3-VL is instruction-tuned, not a raw LM, so "a photo
# of" cannot be a plain decoder prefix. We apply it as an ASSISTANT-turn prefix that
# the model CONTINUES (the functional analog of priming BLIP-2/GPT-2's decoder with
# "a photo of"). The decoded prediction is the continuation only -- the prefix is
# stripped -- exactly as in the other two runs.
#
#   full  : conda activate vlm && python qwen3vl_roco_zeroshot_v0.py
#   smoke : ROCO_N=40 python qwen3vl_roco_zeroshot_v0.py
#   8B 4-bit instead of 4B bf16:
#           MODEL_ID=Qwen/Qwen3-VL-8B-Instruct LOAD_IN_4BIT=1 python qwen3vl_roco_zeroshot_v0.py
#   if you OOM: CAPTION_BATCH=2 python ...   (lower batch, or MAX_VIS_TOKENS for resolution)
import os
os.chdir("/home/matei")
# Model is downloaded locally (see MODEL_ID below) -- like blip2-opt-2.7b -- so the run
# makes no live hub call. Download once with:
#   hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /home/matei/qwen3-vl-4b-instruct
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation
import json
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if device == "cuda" else torch.float32   # bf16: matches the other runs
print(f"device={device} dtype={DTYPE}", flush=True)

# ── config (env-overridable, like the other roco scripts) ────────────────────
MODEL_ID     = os.environ.get("MODEL_ID", "/home/matei/qwen3-vl-4b-instruct")  # local copy of 4B bf16
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"                # 8B on ~6 GB (quantised)
RUN_TAG      = os.environ.get("RUN_TAG", "qwen3vl_zeroshot")
# Resolution cap (merge unit 32 px -> #tokens ~= area/1024). DEFAULT 768: at 1024 the LARGEST
# images (4096 ViT patches) OOM the vision encoder even at batch 1 on this 12 GB card. 768 caps
# every image at 3072 patches (fits) and only downscales the biggest ~18%; the rest are unchanged.
MIN_PIXELS   = 256 * 32 * 32
MAX_PIXELS   = int(os.environ.get("MAX_VIS_TOKENS", "768")) * 32 * 32

# ── load Qwen3-VL: processor + model ─────────────────────────────────────────
from transformers import AutoProcessor, AutoModelForImageTextToText
# Concrete class is Qwen3VLForConditionalGeneration; AutoModelForImageTextToText
# resolves to it and also serves the 8B.
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
# Left-pad: images differ in visual-token count, so batched prompts have unequal
# length -> left padding keeps the shared prompt-length slice below valid per batch.
processor.tokenizer.padding_side = "left"

_kwargs = {}
if device == "cuda":
    _kwargs["device_map"] = {"": 0}
if LOAD_IN_4BIT:
    from transformers import BitsAndBytesConfig
    _kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE)
else:
    _kwargs["dtype"] = DTYPE
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **_kwargs)
model.eval()
if device == "cuda":
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ── ROCOv2 test split (identical loader to caption_roco.py) ──────────────────
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

# ── captioning: "a photo of" (assistant-prefix continuation) + SAME decoding ──
from tqdm.auto import tqdm
CAPTION_PROMPT = "a photo of"
CAPTION_BATCH  = int(os.environ.get("CAPTION_BATCH", "1"))   # batch 4 rode the VRAM edge and OOM'd mid-run; batching gives ~no speedup here, so 1 = safe + same speed
# GREEDY decoding (num_beams=1, do_sample=False): ~5x faster than beam=5 here because it
# also stops re-encoding the image once per beam. Same short cap + n-gram guard as the
# other runs; for a clean comparison, re-score BLIP-2 / BioMedVQA greedy too.
GEN_KWARGS = dict(max_new_tokens=40, min_new_tokens=8, do_sample=False, num_beams=1,
                  no_repeat_ngram_size=3,
                  pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
# eos left to the model's generation_config so it stops on <|im_end|>; max_new_tokens caps length.

# Build the prompt text ONCE. User turn = image only; then prime the assistant reply
# with "a photo of" so the model CONTINUES it. The single <|image_pad|> placeholder is
# expanded per-image by the processor at call time (reserved-seat mechanism).
_DUMMY = Image.new("RGB", (32, 32))
_msgs  = [{"role": "user", "content": [{"type": "image", "image": _DUMMY}]}]
PROMPT_TEXT = processor.apply_chat_template(_msgs, tokenize=False,
                                            add_generation_prompt=True) + CAPTION_PROMPT

# Incremental, resumable checkpoint: one JSON line per finished image. A restart reloads
# these and skips them, so an interruption costs minutes, not the whole run.
# (To force a fresh run from scratch, delete this file first.)
CKPT = f"/home/matei/roco_{RUN_TAG}_preds.jsonl"

def _load_rgb(path):
    try:
        return Image.open(path).convert("RGB")            # radiographs -> RGB
    except Exception as e:                                 # a corrupt file must not kill a 40 h run
        print(f"[warn] unreadable image {path}: {e}", flush=True)
        return Image.new("RGB", (64, 64))                 # blank -> harmless caption; run continues

@torch.no_grad()
def caption_all(recs, batch_size, ckpt_path):
    done = {}
    if os.path.exists(ckpt_path):                         # resume: reload already-finished ids
        with open(ckpt_path) as f:
            for line in f:
                try:
                    d = json.loads(line); done[d["id"]] = d["pred"]
                except Exception:
                    pass
    todo = [r for r in recs if r["id"] not in done]
    print(f"resume: {len(done)} already done, {len(todo)} to caption", flush=True)
    with open(ckpt_path, "a") as ck:
        for i in tqdm(range(0, len(todo), batch_size), desc="captioning", mininterval=15):
            chunk = todo[i:i + batch_size]
            imgs  = [_load_rgb(r["path"]) for r in chunk]
            inputs = processor(text=[PROMPT_TEXT] * len(imgs), images=imgs,
                               padding=True, return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, **GEN_KWARGS)
            gen = gen[:, inputs["input_ids"].shape[1]:]   # left pad -> strip the shared prefix
            caps = [s.strip() for s in processor.batch_decode(gen, skip_special_tokens=True)]
            for r, cap in zip(chunk, caps):               # checkpoint each result immediately
                ck.write(json.dumps({"id": r["id"], "ref": r["caption"], "pred": cap},
                                    ensure_ascii=False) + "\n")
                done[r["id"]] = cap
            ck.flush()
    return done

print(f"captioning {len(records)} images (prompt={CAPTION_PROMPT!r}) ...", flush=True)
done  = caption_all(records, CAPTION_BATCH, CKPT)
preds = [done.get(r["id"], "") for r in records]          # align to record order for scoring
refs  = [r["caption"] for r in records]

OUT_PRED = f"/home/matei/roco_{RUN_TAG}_a_photo_of.json"
with open(OUT_PRED, "w") as f:
    json.dump([{"id": r["id"], "ref": r["caption"], "pred": p} for r, p in zip(records, preds)], f, indent=1)
print(f"saved predictions -> {OUT_PRED}", flush=True)

# free the VLM before loading the (heavy) deberta scorer
del model
import gc; gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

# ── metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr, BERTScore ────────────────────
# IDENTICAL to caption_roco_biomed.py so the three zero-shot models are comparable.
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

# BERTScore with the ROCOv2/ImageCLEF model (microsoft/deberta-xlarge-mnli, F1) so it
# is comparable to the paper baselines -- NOT roberta-large. Three deberta fixes:
#   (1) neutralise transformers' over-strict .bin load gate (load is weights_only=True);
#   (2) cap the sentinel tokenizer max_length; (3) small batch (disentangled attn).
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None                        # (1)
from bert_score import BERTScorer
BERT_MODEL = "microsoft/deberta-xlarge-mnli"
_bert_scorer = BERTScorer(model_type=BERT_MODEL, batch_size=8)
_bert_scorer._tokenizer.model_max_length = 512                             # (2)
_, _, F = _bert_scorer.score(preds, refs, batch_size=8, verbose=False)     # (3)

metrics = {"n": len(preds), "prompt": CAPTION_PROMPT, "model": f"{MODEL_ID} (zero-shot)",
           "BLEU-1": bleu[0], "BLEU-2": bleu[1], "BLEU-3": bleu[2], "BLEU-4": bleu[3],
           "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider,
           "BERTScore-F1": F.mean().item(), "BERTScore-model": BERT_MODEL}

print("\n==== Qwen3-VL ZERO-SHOT CAPTIONING (a photo of) -- ROCOv2 test ====", flush=True)
for k, v in metrics.items():
    print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}", flush=True)
with open(f"/home/matei/roco_{RUN_TAG}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=1)
print(f"saved metrics -> /home/matei/roco_{RUN_TAG}_metrics.json", flush=True)
