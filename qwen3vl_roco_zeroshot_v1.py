#!/usr/bin/env python
# Zero-shot ROCOv2 captioning with Qwen3-VL -- V1: a RADIOLOGY-CONSTRAINED prompt adapted
# to this instruction-tuned model (vs V0's bare "a photo of"). V1 shows Qwen at its best;
# it is NOT the strict same-prompt comparison to BLIP-2/BioMedVQA (that is V0) -- report both.
#
# Prompt = a system turn (expert radiologist; describe only what is visible; terse clinical
# register; no preamble) + a user turn (image + "Provide the caption for this image."). The
# model answers the instruction normally; the decoded prediction is that whole reply (the
# prompt is sliced off). Everything else -- greedy decoding, batch 1, 1024 vis-token cap,
# resumable checkpoint, and the identical metric suite -- matches V0, so the ONLY V0->V1
# difference is the prompt.
#
#   full  : conda activate vlm && python qwen3vl_roco_zeroshot_v1.py
#   smoke : ROCO_N=8 python qwen3vl_roco_zeroshot_v1.py
#   faster: MAX_VIS_TOKENS=768 python qwen3vl_roco_zeroshot_v1.py   (less detail, quicker)
#   resume: re-run the same command (skips finished images)
#   NOTE  : compare V0 and V1 at the SAME MAX_VIS_TOKENS, else prompt vs resolution confound.
import os
os.chdir("/home/matei")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation
import json
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if device == "cuda" else torch.float32   # bf16 (fp16 overflow risk on Volta)
print(f"device={device} dtype={DTYPE}", flush=True)

# ── config (env-overridable) ─────────────────────────────────────────────────
MODEL_ID     = os.environ.get("MODEL_ID", "/home/matei/qwen3-vl-4b-instruct")  # local copy of 4B bf16
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"
RUN_TAG      = os.environ.get("RUN_TAG", "qwen3vl_v1")
# Resolution cap (merge unit 32 px -> #tokens ~= area/1024). DEFAULT 768: at 1024 the LARGEST
# images (clamped to 4096 ViT patches) OOM the vision encoder even at batch 1 on this 12 GB card.
# 768 caps every image at 3072 patches (fits) and only downscales the biggest ~18% of images;
# the other ~82% (<=768^2) are unchanged. Keep it EQUAL to the V0 run.
MIN_PIXELS   = 256 * 32 * 32
MAX_PIXELS   = int(os.environ.get("MAX_VIS_TOKENS", "768")) * 32 * 32

# ── load Qwen3-VL: processor + model ─────────────────────────────────────────
from transformers import AutoProcessor, AutoModelForImageTextToText
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
processor.tokenizer.padding_side = "left"   # left-pad for batched generation (see V0 notes)

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

# ── optional: load a fine-tuned checkpoint (LoRA adapter + trained mergers) ──
# Zero-shot when LORA_PATH is unset -> this file scores BOTH the zero-shot and the
# fine-tuned model through exactly the same code path, so the delta is clean.
#   LORA_PATH=/home/matei/qwen3vl_ft_ckpt/best RUN_TAG=qwen3vl_ft_test python ...
LORA_PATH = os.environ.get("LORA_PATH")
if LORA_PATH:
    # Mergers first (plain weights, not LoRA), before the PEFT wrapper renames modules.
    merger_file = os.path.join(LORA_PATH, "merger.pt")
    if os.path.isfile(merger_file):
        msd = torch.load(merger_file, map_location="cpu")
        params = dict(model.named_parameters())
        def _norm(k):                       # strip any peft "base_model.model." prefixes
            i = k.find("model.visual")
            return k[i:] if i >= 0 else k
        n_loaded = 0
        for k, v in msd.items():
            tgt = params.get(_norm(k))
            if tgt is not None and tgt.shape == v.shape:
                tgt.data.copy_(v.to(tgt.dtype))
                n_loaded += 1
        print(f"[ft] loaded {n_loaded}/{len(msd)} merger tensors from {merger_file}", flush=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()        # fold LoRA into the base -> no inference overhead
    print(f"[ft] merged LoRA adapter from {LORA_PATH}", flush=True)

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

# ── V1 prompt: radiology-constrained system turn + a caption instruction ─────
from tqdm.auto import tqdm
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
USER_PROMPT   = "Provide the caption for this image."
PROMPT_ID     = "V1_radiology_constrained"
CAPTION_BATCH = int(os.environ.get("CAPTION_BATCH", "1"))   # batch 1: safe on 12 GB, ~no speed cost
# GREEDY decoding, IDENTICAL knobs to V0 so the only V0->V1 change is the prompt.
GEN_KWARGS = dict(max_new_tokens=40, min_new_tokens=8, do_sample=False, num_beams=1,
                  no_repeat_ngram_size=3,
                  pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)

# Build the prompt text ONCE (system + user are identical for every image). Unlike V0 we do
# NOT prime the assistant reply -- the model answers the instruction and we keep its whole
# reply. The single <|image_pad|> placeholder is expanded per-image by the processor.
_DUMMY = Image.new("RGB", (32, 32))
_msgs  = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user",   "content": [{"type": "image", "image": _DUMMY},
                                   {"type": "text",  "text": USER_PROMPT}]},
]
PROMPT_TEXT = processor.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)

# Resumable checkpoint: one JSON line per finished image; a restart skips them.
# (Delete this file to force a fresh run.)
CKPT = f"/home/matei/roco_{RUN_TAG}_preds.jsonl"

def _load_rgb(path):
    try:
        return Image.open(path).convert("RGB")            # radiographs -> RGB
    except Exception as e:                                 # a corrupt file must not kill a long run
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
            gen = gen[:, inputs["input_ids"].shape[1]:]   # left pad -> strip the shared prompt
            caps = [s.strip() for s in processor.batch_decode(gen, skip_special_tokens=True)]
            for r, cap in zip(chunk, caps):
                ck.write(json.dumps({"id": r["id"], "ref": r["caption"], "pred": cap},
                                    ensure_ascii=False) + "\n")
                done[r["id"]] = cap
            ck.flush()
    return done

print(f"captioning {len(records)} images (prompt={PROMPT_ID}) ...", flush=True)
done  = caption_all(records, CAPTION_BATCH, CKPT)
preds = [done.get(r["id"], "") for r in records]
refs  = [r["caption"] for r in records]

OUT_PRED = f"/home/matei/roco_{RUN_TAG}_captions.json"
with open(OUT_PRED, "w") as f:
    json.dump([{"id": r["id"], "ref": r["caption"], "pred": p} for r, p in zip(records, preds)], f, indent=1)
print(f"saved predictions -> {OUT_PRED}", flush=True)

# free the VLM before loading the (heavy) deberta scorer
del model
import gc; gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

# ── metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr, BERTScore (IDENTICAL to V0) ──
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

_tag = f"fine-tuned: {LORA_PATH}" if LORA_PATH else "zero-shot"
metrics = {"n": len(preds), "prompt": PROMPT_ID, "model": f"{MODEL_ID} ({_tag}, V1)",
           "BLEU-1": bleu[0], "BLEU-2": bleu[1], "BLEU-3": bleu[2], "BLEU-4": bleu[3],
           "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider,
           "BERTScore-F1": F.mean().item(), "BERTScore-model": BERT_MODEL}

print("\n==== Qwen3-VL V1 (radiology-constrained prompt) -- ROCOv2 test ====", flush=True)
for k, v in metrics.items():
    print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}", flush=True)
with open(f"/home/matei/roco_{RUN_TAG}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=1)
print(f"saved metrics -> /home/matei/roco_{RUN_TAG}_metrics.json", flush=True)
