#!/usr/bin/env python
"""Standalone BLIP-2 VQA-RAD fine-tuning WITH LoRA-tuned ViT.
Generated from blip-2_fine_tuned_VQA-RAD_LoRA-ViT.ipynb.
Run headless inside tmux:  python /home/matei/train_lora.py
Checkpoints -> /home/matei/vqa_checkpoints_lora/ (every epoch + final)."""
import os
os.chdir("/home/matei")            # dataset paths are relative to here


# ======================================================================
# cell: c-setup
# ======================================================================
# %pip install -U transformers accelerate datasets pillow tqdm peft

import torch
import torch.nn as nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# bfloat16: same memory as fp16 but fp32's dynamic range — avoids the fp16
# optimizer-underflow NaN and OPT's fp16 overflow. (TITAN V supports it.)
dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

print(f"device = {device} | dtype = {dtype}")


# ======================================================================
# cell: c-data
# ======================================================================
# ── Load VQA-RAD (canonical split; add the templated "framed" questions to TRAIN) ──
import json, os
from datasets import Dataset, DatasetDict

dataset_dir = "VQA-RAD_dataset"
image_dir   = os.path.join(dataset_dir, "VQA_RAD Image Folder")

with open(os.path.join(dataset_dir, "VQA_RAD Dataset Public.json")) as f:
    records = json.load(f)

def load_records(recs, add_framed):
    rows = {"image_path": [], "question": [], "answer": [], "question_type": [], "answer_type": [], "image_organ": []}
    seen = set()
    def add_row(img_path, question, rec):
        # skip exact-duplicate (image, question, answer) rows
        key = (img_path, " ".join(str(question).strip().lower().split()), str(rec["answer"]).strip().lower())
        if key in seen:
            return
        seen.add(key)
        rows["image_path"].append(img_path)
        rows["question"].append(str(question))
        rows["answer"].append(str(rec["answer"]))
        rows["question_type"].append(str(rec.get("question_type", "OTHER")))
        rows["answer_type"].append(str(rec.get("answer_type", "OTHER")).strip().upper())
        rows["image_organ"].append(str(rec.get("image_organ", "")))
    for rec in recs:
        img_path = os.path.join(image_dir, rec["image_name"])
        if not os.path.isfile(img_path):
            continue
        add_row(img_path, rec["question"], rec)              # the record's own question (free-form or paraphrase)
        # Paraphrase ("rephrase") questions are ALREADY separate records, so re-adding the
        # question_rephrase field only duplicates them -> we do NOT. The templated "framed"
        # questions exist ONLY in this field (never as records), so we add them, but to TRAIN
        # only: adding a test question's framing to train would leak the test set.
        if add_framed:
            frame = rec.get("question_frame", "NULL")
            if frame and frame != "NULL":
                add_row(img_path, frame, rec)
    return Dataset.from_dict(rows)

dataset = DatasetDict({
    "train": load_records([r for r in records if not r["phrase_type"].startswith("test_")], add_framed=True),
    "test":  load_records([r for r in records if     r["phrase_type"].startswith("test_")], add_framed=False),
})
print(f"train {len(dataset['train'])} | test {len(dataset['test'])}")


# ======================================================================
# cell: a6ec327c
# ======================================================================
# ── Load BLIP-2 and the Q-Former question-embedding table ────────────────────
from typing import Any
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration, BertTokenizer
)

CKPT = "/home/matei/blip2-opt-2.7b"         
processor = Blip2Processor.from_pretrained(CKPT)
blip2 = Blip2ForConditionalGeneration.from_pretrained(
    CKPT, torch_dtype=dtype, low_cpu_mem_usage=True
)

# The Q-Former processes the question with BERT tokenisation. The blip2-opt-2.7b
# checkpoint dropped the Q-Former's text weights, so we graft the GENUINELY
# PRETRAINED ones extracted from the BLIP-2 retrieval checkpoint (blip2-itm-vit-g)
# via extract_itm.py -> blip2_qformer_text_weights.pt, then fine-tune them.
# bert-base-uncased's tokeniser is token-id-identical to the Q-Former's for the
# real vocab (ids 0..30521, verified), so we keep it and only swap the weights.
qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

_QF_TEXT = torch.load("/home/matei/blip2_qformer_text_weights.pt", map_location="cpu")
qformer_word_emb = nn.Embedding.from_pretrained(_QF_TEXT["word_embeddings.weight"], freeze=False)      # [30523, 768]
qformer_pos_emb  = nn.Embedding.from_pretrained(_QF_TEXT["position_embeddings.weight"], freeze=False)  # [512,   768]
qformer_text_ffn = _QF_TEXT["text_ffn"]   # 12 layers of pretrained text feed-forward, grafted in cell 5

# The sub-configs are real objects at runtime (OPTConfig / Blip2QFormerConfig),
# but the type stub annotates them as `dict | None`; `Any` silences that.
cfg: Any = blip2.config
d_llm     = cfg.text_config.hidden_size
d_qformer = cfg.qformer_config.hidden_size
n_query   = cfg.num_query_tokens
print(f"d_llm={d_llm} | d_qformer={d_qformer} | num_query_tokens={n_query}")
assert qformer_word_emb.weight.shape[1] == d_qformer, "BERT hidden must equal Q-Former hidden (768)"


# ======================================================================
# cell: c-module
# ======================================================================
import copy

class Blip2QFormerVQA(nn.Module):
    # Wraps the pretrained BLIP-2 submodules and adds the question->Q-Former path.
    def __init__(self, blip2, word_emb, pos_emb, text_ffn=None):
        super().__init__()
        self.vision_model        = blip2.vision_model
        self.query_tokens        = blip2.query_tokens            # nn.Parameter [1, 32, 768]
        self.qformer             = blip2.qformer
        # The blip2-opt-2.7b checkpoint was exported with use_qformer_text_input=False,
        # so each Q-Former layer has only the *query* feed-forward (intermediate_query/
        # output_query) and lacks the *text* feed-forward (intermediate/output) that the
        # question tokens need. We recreate that module (deepcopy gives the right shape)
        # and load the GENUINELY PRETRAINED text weights from blip2-itm-vit-g (text_ffn),
        # then fine-tune it (self.qformer is set trainable). Everything else — query path,
        # cross-attention, shared self-attention, projection — stays OPT-matched.
        for _i, _layer in enumerate(self.qformer.encoder.layer):
            if not hasattr(_layer, 'intermediate'):
                _layer.intermediate = copy.deepcopy(_layer.intermediate_query)
                _layer.output       = copy.deepcopy(_layer.output_query)
                if text_ffn is not None:
                    _layer.intermediate.load_state_dict(text_ffn[_i]['intermediate'])
                    _layer.output.load_state_dict(text_ffn[_i]['output'])
        self.language_projection = blip2.language_projection     # 768 -> d_llm
        self.language_model      = blip2.language_model
        self.qformer_word_emb    = word_emb
        self.qformer_pos_emb     = pos_emb
        self.n_query             = blip2.config.num_query_tokens

    def _qformer_features(self, pixel_values, q_ids, q_att):
        # 1. image features. The ViT is LoRA-tuned here, so we do NOT wrap it in
        #    torch.no_grad() — gradients must reach the LoRA adapters. Activation
        #    memory is instead controlled by gradient checkpointing on the ViT
        #    (enabled in the instantiation cell).
        image_embeds = self.vision_model(pixel_values).last_hidden_state
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)

        B = pixel_values.shape[0]
        query_tokens = self.query_tokens.expand(B, -1, -1)                       # [B, 32, 768]
        query_atts   = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=query_tokens.device)

        # 2. question embeddings (word + position), matching Blip2TextEmbeddings
        seq_len  = q_ids.shape[1]
        pos_ids  = torch.arange(seq_len, device=q_ids.device).unsqueeze(0)
        # match the question embeddings to the Q-Former's working dtype (kept fp32 in fp16 loads)
        text_emb = (self.qformer_word_emb(q_ids) + self.qformer_pos_emb(pos_ids)).to(query_tokens.dtype)   # [B, Lq, 768]

        # 3. concat queries + question; queries come FIRST (query_length boundary)
        query_embeds   = torch.cat([query_tokens, text_emb], dim=1)              # [B, 32+Lq, 768]
        attention_mask = torch.cat([query_atts, q_att], dim=1)                   # [B, 32+Lq]

        # 4. cross-attention restricted to the first n_query positions
        out = self.qformer(
            query_embeds=query_embeds,
            query_length=self.n_query,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        )

        # cast Q-Former output to the LLM/projection dtype before crossing into the LLM region
        query_output = out.last_hidden_state[:, :self.n_query, :].to(self.language_projection.weight.dtype)  # [B, 32, 768]
        
        return self.language_projection(query_output)                          # [B, 32, d_llm]

    def forward(self, pixel_values, q_ids, q_att, llm_ids, llm_att, labels):
        soft = self._qformer_features(pixel_values, q_ids, q_att)              # [B, 32, d_llm]
        soft_att = torch.ones(soft.shape[:-1], dtype=torch.long, device=soft.device)

        text_embeds   = self.language_model.get_input_embeddings()(llm_ids)    # [B, L, d_llm]
        inputs_embeds = torch.cat([soft, text_embeds], dim=1)
        attention_mask = torch.cat([soft_att, llm_att], dim=1)

        # -100 for the 32 visual/query positions so they contribute no loss
        B = pixel_values.shape[0]
        soft_labels = torch.full((B, self.n_query), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([soft_labels, labels], dim=1)

        return self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=full_labels
        )

    @torch.no_grad()
    def generate(self, pixel_values, q_ids, q_att, llm_ids, llm_att, **gen_kwargs):
        soft = self._qformer_features(pixel_values, q_ids, q_att)
        soft_att = torch.ones(soft.shape[:-1], dtype=torch.long, device=soft.device)
        text_embeds   = self.language_model.get_input_embeddings()(llm_ids)
        inputs_embeds = torch.cat([soft, text_embeds], dim=1)
        attention_mask = torch.cat([soft_att, llm_att], dim=1)
        # generate() with inputs_embeds (no input_ids) returns ONLY new tokens
        return self.language_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs
        )


# ======================================================================
# cell: c-freeze
# ======================================================================
# ── Instantiate, LoRA-tune the ViT, set what trains ─────────────────────────
from peft import LoraConfig, inject_adapter_in_model

model = Blip2QFormerVQA(blip2, qformer_word_emb, qformer_pos_emb, qformer_text_ffn)

def set_trainable(module, flag):
    for p in module.parameters():
        p.requires_grad = flag

# LoRA on the ViT: freeze the 986M base weights, then inject low-rank adapters into
# the attention (qkv, projection) and MLP (fc1, fc2) linear layers. Only the adapters
# (~15M params) become trainable, so the encoder can adapt to radiology cheaply. This
# is the ONLY modelling change vs the frozen-ViT notebook.
set_trainable(model.vision_model, False)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["qkv", "projection", "fc1", "fc2"], bias="none",
)
inject_adapter_in_model(lora_cfg, model.vision_model)   # in-place; leaves only adapters trainable

model = model.to(device)                                # moves base + new adapters to GPU
model.qformer_word_emb = model.qformer_word_emb.to(device, model.query_tokens.dtype)
model.qformer_pos_emb  = model.qformer_pos_emb.to(device, model.query_tokens.dtype)

# what trains: Q-Former + projection + query tokens + question embeddings + ViT LoRA
set_trainable(model.language_model, False)   # LLM frozen (paper)
set_trainable(model.qformer, True)           # Q-Former trained (paper)
set_trainable(model.language_projection, True)
model.query_tokens.requires_grad = True
model.qformer_word_emb.weight.requires_grad = True
model.qformer_pos_emb.weight.requires_grad  = True

# Gradient checkpointing to fit the ViT backward in 12 GB.
# ViT: use_reentrant=False is REQUIRED — the input pixels carry no gradient but the
# LoRA adapters inside do, and reentrant checkpointing would silently drop them.
model.vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.language_model.gradient_checkpointing_enable()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable/1e6:.1f}M / total {total/1e6:.1f}M")


# ======================================================================
# cell: c-dataset
# ======================================================================
# ── Dataset / DataLoader ─────────────────────────────────────────────────────
from torch.utils.data import Dataset as TorchDataset, DataLoader

Q_MAXLEN   = 32   # Q-Former question length
LLM_MAXLEN = 64   # LLM  question+answer length

class VQARADDataset(TorchDataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[idx]
        image  = Image.open(row["image_path"]).convert("RGB")
        pixel  = processor(images=image, return_tensors="pt").pixel_values[0]  # pyright: ignore[reportCallIssue]

        question = row["question"]
        answer   = str(row["answer"])

        # question -> Q-Former (BERT tokeniser)
        qf = qformer_tokenizer(question, padding="max_length", truncation=True,
                               max_length=Q_MAXLEN, return_tensors="pt")

        # "Question: q Answer: answer</s>" -> LLM (OPT tokeniser)
        prompt_text = f"Question: {question} Answer:"
        full_text   = f"{prompt_text} {answer}{processor.tokenizer.eos_token}"
        llm = processor.tokenizer(full_text, padding="max_length", truncation=True,
                                  max_length=LLM_MAXLEN, return_tensors="pt")

        input_ids = llm.input_ids[0]
        llm_att   = llm.attention_mask[0]
        labels    = input_ids.clone()
        labels[llm_att == 0] = -100                       # ignore padding
        # ignore the "Question: ... Answer:" prompt tokens (train only on the answer)
        prompt_len = processor.tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
        labels[:min(prompt_len, LLM_MAXLEN)] = -100

        return {
            "pixel_values": pixel,
            "q_ids":  qf.input_ids[0],
            "q_att":  qf.attention_mask[0],
            "llm_ids": input_ids,
            "llm_att": llm_att,
            "labels":  labels,
        }

train_loader = DataLoader(VQARADDataset(dataset["train"]), batch_size=2, shuffle=True)
_b = next(iter(train_loader))
print({k: tuple(v.shape) for k, v in _b.items()})


# ======================================================================
# cell: c-train
# ======================================================================
# ── Training loop ────────────────────────────────────────────────────────────
import os, math
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

EPOCHS       = 5      # paper (Table 8) uses 5. NOTE: the ViT backward makes this
                      # notebook markedly slower than the frozen-ViT run.
ACCUM_STEPS  = 4      # effective batch = 2 * 4 = 8. If you OOM, use batch_size=1 / ACCUM_STEPS=8.
LR           = 1e-5   # paper's VQA fine-tuning LR (Q-Former, projection, embeddings)
LORA_LR      = 1e-4   # LoRA adapters train well at a higher LR than full fine-tuning
WEIGHT_DECAY = 0.05   # paper's value
SAVE_DIR     = "/home/matei/vqa_checkpoints_lora"   # separate from the frozen-ViT run
os.makedirs(SAVE_DIR, exist_ok=True)

def save_ckpt(tag):
    # Trainable parts only: Q-Former + projection + query tokens + embeddings + ViT LoRA adapters.
    ckpt = {
        "qformer":             model.qformer.state_dict(),
        "language_projection": model.language_projection.state_dict(),
        "query_tokens":        model.query_tokens.detach().cpu(),
        "qformer_word_emb":    model.qformer_word_emb.state_dict(),
        "qformer_pos_emb":     model.qformer_pos_emb.state_dict(),
        "vit_lora":            {k: v for k, v in model.vision_model.state_dict().items() if "lora" in k.lower()},
    }
    p = os.path.join(SAVE_DIR, f"vqa_lora_{tag}.pt")
    torch.save(ckpt, p)
    return p

# Three parameter groups: ViT LoRA adapters (higher LR, no decay); matmul weights
# (weight decay); biases/norms/embeddings/query tokens (no decay).
_no_decay_keys = ("bias", "norm", "word_emb", "pos_emb", "query_tokens")
_lora, _decay, _no_decay = [], [], []
for _n, _p in model.named_parameters():
    if not _p.requires_grad:
        continue
    if "lora" in _n.lower():
        _lora.append(_p)
    elif _p.ndim <= 1 or any(k in _n.lower() for k in _no_decay_keys):
        _no_decay.append(_p)
    else:
        _decay.append(_p)
optimizer = AdamW([
    {"params": _decay,    "weight_decay": WEIGHT_DECAY, "lr": LR},
    {"params": _no_decay, "weight_decay": 0.0,          "lr": LR},
    {"params": _lora,     "weight_decay": 0.0,          "lr": LORA_LR},
])

# Warmup + cosine decay, stepped once per OPTIMIZER update (not per micro-batch).
opt_steps_per_epoch = len(train_loader) // ACCUM_STEPS
total_opt_steps     = opt_steps_per_epoch * EPOCHS
warmup_steps        = max(1, int(0.1 * total_opt_steps))
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)
print(f"optimizer steps: {total_opt_steps} total | {warmup_steps} warmup")

model.train()
for epoch in range(EPOCHS):
    running = 0.0
    optimizer.zero_grad()
    bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(bar):
        pixel  = batch["pixel_values"].to(device, dtype)
        q_ids  = batch["q_ids"].to(device)
        q_att  = batch["q_att"].to(device)
        llm_id = batch["llm_ids"].to(device)
        llm_at = batch["llm_att"].to(device)
        labels = batch["labels"].to(device)

        out  = model(pixel, q_ids, q_att, llm_id, llm_at, labels)
        loss = out.loss / ACCUM_STEPS
        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running += out.loss.item()
        bar.set_postfix({"loss": out.loss.item(), "lr": scheduler.get_last_lr()[0]})

    ckpt_path = save_ckpt(f"epoch{epoch+1}")     # checkpoint written to disk every epoch
    print(f"epoch {epoch+1} | avg loss {running/len(train_loader):.4f} | saved -> {ckpt_path}")

final_path = save_ckpt("final")
print(f"training complete | final model -> {final_path}")
