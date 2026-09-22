# Medical Image Captioning and Concept Annotation on ROCOv2

**Matei-Ioan Munteanu** — Research internship, LIX (École Polytechnique), Summer 2026

This project studies two connected problems on
[ROCOv2](https://doi.org/10.1038/s41597-024-03496-6), a dataset of 79,789
radiology images from PubMed Central, each with a caption and a set of UMLS
medical concepts:

1. **Medical image captioning.** Generating a radiology caption from an image.
   Six vision–language systems were built, adapted, and compared under one
   evaluation protocol, from a 124M-parameter model up to a 4B-parameter
   Qwen3-VL.
2. **Concept annotation quality.** ROCOv2's concept labels are incomplete: an
   image is often tagged with one concept when it clearly shows several. A
   multi-label image classifier and a caption-based label-enrichment pipeline
   were built to recover the missing labels. Retraining on the enriched labels
   produced a better classifier.

Every result below is on the full 9,927-image ROCOv2 test set and can be
reproduced with the commands in [Reproducing the results](#reproducing-the-results).

---

## Contents

- [Highlights](#highlights)
- [Results](#results)
- [Project overview](#project-overview)
- [Findings worth knowing](#findings-worth-knowing)
- [Repository layout](#repository-layout)
- [Environment and hardware](#environment-and-hardware)
- [Reproducing the results](#reproducing-the-results)
- [Operational notes](#operational-notes)
- [Limitations](#limitations)
- [References](#references)

---

## Highlights

- **Built a LLaVA-style medical captioner** (BioMedCLIP → MLP connector →
  Qwen2.5-1.5B with LoRA). It fixed the mode collapse of the earlier
  single-visual-token model: widening the visual input from 1 to 196 tokens
  raised caption diversity from about 8% to about 80% unique captions.
- **Fine-tuned Qwen3-VL-4B with LoRA** on a 12 GB GPU, at native image
  resolution with gradient checkpointing and mixed precision.
- **Found a silent PEFT defect**: `get_peft_model` re-froze the
  vision–language connector that had been deliberately unfrozen. A 45-hour
  training run had in fact trained only the LoRA adapters. Fixed with an
  invariant check and a gradient probe.
- **Diagnosed why an auxiliary-loss experiment could not test its own
  hypothesis.** A fixed loss weight let the auxiliary signal fade from 3.5% to
  1.7% of the main loss during training. Replaced it with a per-step normalised
  weight that holds the signal constant, then re-ran the experiment properly.
- **Designed a caption → UMLS concept matcher** (alias expansion,
  lemmatisation, token n-gram index, case-sensitive abbreviations, negation
  scopes). It adds about 49% more concept labels to every split. On the concepts
  where a curated reference exists, it keeps precision very high.
- **Showed the enriched labels are useful downstream.** A classifier retrained
  on them beats the original classifier, including after the original's
  decision thresholds are re-tuned on the same enriched target. That control
  rules out "it simply predicts more labels" as the explanation.

---

## Results

### Captioning — ROCOv2 test set (n = 9,927)

| Model | Setting | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | BERTScore |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B, concept-guided, adaptive λ, enriched labels | fine-tuned | 0.1570 | 0.0292 | 0.1550 | 0.1921 | 0.1970 | **0.6766** |
| Qwen3-VL-4B, LoRA, connector frozen | fine-tuned | 0.1564 | 0.0293 | 0.1544 | 0.1914 | 0.1950 | 0.6764 |
| Qwen3-VL-4B, LoRA, connector trained | fine-tuned | 0.1569 | 0.0291 | 0.1544 | 0.1916 | 0.1924 | 0.6764 |
| Qwen3-VL-4B, concept-guided, fixed λ = 0.1 | fine-tuned | 0.1582 | 0.0293 | 0.1554 | 0.1927 | 0.1973 | 0.6763 |
| LLaVA-style BioMedCLIP → Qwen2.5-1.5B (last ViT block) | fine-tuned | 0.1754 | 0.0316 | 0.1638 | 0.1886 | 0.1885 | 0.6708 |
| LLaVA-style BioMedCLIP → Qwen2.5-1.5B (penultimate ViT block) | fine-tuned | 0.1678 | 0.0303 | 0.1581 | 0.1839 | 0.1773 | 0.6695 |
| Qwen3-VL-4B, radiology system prompt (V1) | zero-shot | 0.1603 | 0.0195 | 0.1336 | 0.1582 | 0.1245 | 0.6539 |
| BLIP-2 (OPT-2.7B), VQA-RAD-adapted | fine-tuned | 0.0815 | 0.0120 | 0.1113 | 0.1543 | 0.0931 | 0.6441 |
| BioMedCLIP + GPT-2 | fine-tuned | 0.0719 | 0.0115 | 0.1076 | 0.1484 | 0.0864 | 0.6378 |
| Qwen3-VL-4B, "a photo of" (V0) | zero-shot | 0.1479 | 0.0129 | 0.1342 | 0.1407 | 0.0381 | 0.5980 |
| BLIP-2 (OPT-2.7B), VQA-RAD-adapted | zero-shot | 0.1516 | 0.0091 | 0.0984 | 0.1252 | 0.0350 | 0.5195 |
| BioMedCLIP + GPT-2 | zero-shot | 0.0574 | 0.0029 | 0.0570 | 0.0983 | 0.0103 | 0.4965 |

BERTScore is the F1 of `microsoft/deberta-xlarge-mnli` without baseline
rescaling, the configuration used by the ImageCLEFmedical leaderboard. Qwen3-VL
rows use greedy decoding; all other rows use beam search (see
[Limitations](#limitations)).

**Reading the table.** The fine-tuned Qwen3-VL variants lie within 0.0003
BERTScore of each other. With one training run per configuration, that is not a
ranking: the connector, the concept supervision, and the loss weighting made no
measurable difference to captioning. The largest single improvement in the
Qwen3-VL study came from the system prompt (V0 → V1), at no training cost.

### Concept detection — samples-averaged F1 (ImageCLEF's official metric)

| Classifier | Original test labels | Enriched test labels | Predicted concepts / image |
|---|---|---|---|
| CNN-1, trained on original labels | **0.5593** | 0.4571 | 2.12 |
| CNN-2, trained on enriched labels | — | **0.4876** | 3.62 |

Per-label decision thresholds, in-vocabulary gold, 1,571 concepts. The
ImageCLEFmedical 2025 concept-detection task was won with 0.5888 on the
original labels.

The fair comparison is the **enriched** column, because both classifiers face
the same target there. CNN-2 still outperforms CNN-1 after CNN-1's thresholds
are re-tuned on the enriched validation set. That control was run with
`recalibrate_cnn1.py`, and roughly half of CNN-2's apparent gain turns out to
come from re-thresholding alone.

---

## Project overview

### Stage 1 — Visual question answering on VQA-RAD

The internship started by adapting BLIP-2 (OPT-2.7B) to medical VQA on VQA-RAD.
BLIP-2's Q-Former text weights were grafted from `blip2-itm-vit-g` to restore
its text pathway, and two variants were trained: frozen vision encoder, and
LoRA on the vision encoder. A second model, BioMedVQA (BioMedCLIP encoder →
linear translator → GPT-2), was also trained on VQA-RAD. The resulting
checkpoints are the starting points for Stage 2.

### Stage 2 — Captioning on ROCOv2

All systems share one protocol: the same splits, the same canonical
`"a photo of"` prompt, the same metric suite, and validation-based checkpoint
selection. The test split is never used for model selection.

| System | Architecture | What was trained |
|---|---|---|
| BLIP-2 | ViT-g → Q-Former (32 queries) → OPT-2.7B | ViT LoRA, fine-tuned on a 12k ROCOv2 subset |
| BioMedVQA | BioMedCLIP → 1 visual token → GPT-2 | Translator + GPT-2, fine-tuned on a 12k subset |
| LLaVA-style | BioMedCLIP → **196 patch tokens** → 2-layer MLP → Qwen2.5-1.5B | Two stages: connector only, then LoRA; full 59,958-image train split |
| Qwen3-VL-4B | SigLIP-2 ViT → 2×2 MLP merger → Qwen3 LLM | Zero-shot (two prompts), then LoRA + merger on a 12k subset |
| Qwen3-VL-4B + concepts | as above + auxiliary UMLS concept heads on the merger output | As above, plus 1,571-CUI and 18-concept classification heads |

### Stage 3 — Improving concept annotation

1. **CNN-1**: DenseNet-121 with GeM pooling and a 1,571-way sigmoid head,
   trained with BCE and per-class `pos_weight`. Decision thresholds are
   calibrated with a global sweep followed by per-label coordinate ascent.
   There is deliberately no horizontal flip, because laterality is clinically
   meaningful.
2. **Why the classifier alone cannot fix the labels.** It is trained on the
   incomplete labels, and its thresholds are calibrated against them, so it
   learns to reproduce the gaps.
3. **Caption → CUI matcher**: the caption is an independent view of each image
   and often states what the labels omit. The matcher expands each concept into
   its UMLS aliases (via scispaCy), lemmatises both sides, matches token n-grams
   through an inverted index, handles short abbreviations case-sensitively
   against a curated allowlist, and rejects matches inside NegEx-style negation
   scopes.
4. **Enrichment**: the same caption-only rule is applied to train, validation,
   and test. A classifier veto on the training split was tried and dropped
   (see [Findings](#findings-worth-knowing)).
5. **CNN-2**: the same classifier retrained on the enriched labels, evaluated
   against the enriched test set and a threshold-recalibrated CNN-1.

---

## Findings worth knowing

**PEFT silently re-freezes non-LoRA parameters.** `get_peft_model` calls
`_mark_only_adapters_as_trainable()`, which sets `requires_grad=False` on every
parameter without `lora_` in its name, including modules unfrozen beforehand.
It emits no warning. The fix re-asserts the flags after wrapping and asserts
the invariant, and a probe counts the parameters that actually receive a
gradient.

**A constant auxiliary-loss weight does not give a constant auxiliary
signal.** With `L = L_caption + λ·L_aux`, what matters is the ratio
`λ·L_aux / L_caption`. The auxiliary task (predicting ~3 concepts) converged
fast, so `L_aux` fell 56% while `L_caption` fell 9%, and the signal decayed to
1.7%. The replacement weight is recomputed each step:
`λ_t = clamp(0.10 · L_caption.detach() / L_aux.detach(), 0.05, 5.0)`. Detaching
both operands matters: otherwise the optimiser minimises the ratio itself.

**A detector with low recall cannot validate new labels.** The classifier veto
rejected 96.9% of caption-derived candidates. The first hypothesis,
memorisation, was tested and refuted: the model performs about the same on the
images it was trained on as on held-out images. The real reason is that it
recalls fewer than half of the labels already known to be correct, so its
rejections carry no information about whether a candidate is wrong.

**Case-sensitivity only protects if it replaces the permissive path.** The
matcher initially indexed short aliases both case-sensitively *and* as
lowercase lemmas, so `CT` still matched *Carpal Tunnel Syndrome* in every
lowercase "ct" and turned CT scans into spurious labels. The fix: short aliases
get only the case-sensitive entry, and only for allowlisted
(abbreviation, concept) pairs.

**Choose checkpoints on generation metrics, not validation loss.** For the
LLaVA-style captioner, teacher-forced validation loss rose from the first epoch
while BERTScore and CIDEr kept improving. Selecting on loss would have discarded
the best checkpoints.

---

## Repository layout

| Script | Purpose | Environment |
|---|---|---|
| `extract_itm.py` | Extract Q-Former text weights from `blip2-itm-vit-g` for grafting | `vlm` |
| `train.py` | BLIP-2 VQA-RAD fine-tuning, frozen ViT | `vlm` |
| `train_lora.py` | BLIP-2 VQA-RAD fine-tuning, LoRA on the ViT | `vlm` |
| `train_roco_caption.py` | Fine-tune the VQA-RAD BLIP-2 model for ROCOv2 captioning | `vlm` |
| `caption_roco.py` | Caption the ROCOv2 test set with BLIP-2 | `vlm` |
| `train_roco_biomed.py` | Fine-tune BioMedVQA for ROCOv2 captioning | `vlm` |
| `caption_roco_biomed.py` | Caption the ROCOv2 test set with BioMedVQA | `vlm` |
| `train_roco_llava.py` | Two-stage training of the LLaVA-style captioner | `vlm` |
| `caption_roco_llava.py` | Caption the ROCOv2 test set with the LLaVA-style captioner | `vlm` |
| `rescore_bertscore.py`, `imageclef_bertscore.py` | Add leaderboard-comparable BERTScore to saved BLIP-2 / BioMedVQA predictions | `vlm` |
| `qwen3vl_roco_zeroshot_v0.py` | Qwen3-VL zero-shot, `"a photo of"` prompt | `vlm` |
| `qwen3vl_roco_zeroshot_v1.py` | Qwen3-VL zero-shot, radiology prompt; also evaluates fine-tuned checkpoints via `LORA_PATH` | `vlm` |
| `qwen3vl_roco_finetune.py` | Qwen3-VL LoRA + connector fine-tuning | `vlm` |
| `qwen3vl_roco_finetune_cui.py` | As above + auxiliary concept heads, fixed λ | `vlm` |
| `qwen3vl_roco_finetune_cui_v2.py` | As above with normalised λ and configurable label directory | `vlm` |
| `roco_full_metrics.py` | Score any saved prediction file (relevance and factuality metrics) | `vlm` |
| `install_factuality_metrics.sh` | Install scispaCy and AlignScore for the factuality metrics | `vlm` |
| `roco_cui_classifier.py` | Train the DenseNet-121 CUI classifier and calibrate thresholds | `roco_env` |
| `eval_test.py` | Evaluate a trained classifier on a split | `roco_env` / `vlm` |
| `cui_caption_matcher.py` | Caption → CUI matching and label enrichment | `roco_env` |
| `recalibrate_cnn1.py` | *Supplementary*: re-tune CNN-1's thresholds on enriched validation labels | `roco_env` |
| `test_memorization.py` | *Supplementary*: compare CNN-1 on training vs held-out images | `roco_env` |
| `score_audit.py` | *Supplementary*: weighted precision + Wilson interval from the manual audit sheet | any |

Each Python script has a notebook twin (`*.ipynb`) with the same logic, used for
interactive development. The scripts are the versions intended for
long unattended runs.

---

## Environment and hardware

| | |
|---|---|
| GPU | NVIDIA TITAN V, 12 GB (compute capability 7.0) |
| CPU | Intel Xeon W-2123, 8 logical cores |
| Also used | MacBook Air (Apple M4, MPS) for CNN-1 |
| OS | Linux, CUDA driver 12.4 |

Two conda environments, both Python 3.10. The exact package versions are in
`requirements_vlm.txt` and `requirements_roco_env.txt`.

| Environment | Used for | Key packages |
|---|---|---|
| `vlm` | All vision–language models, captioning metrics | `torch 2.5.1+cu121`, `transformers 5.12.1`, `peft 0.19.1`, `open_clip`, `pycocoevalcap`, `bert_score` |
| `roco_env` | Concept matcher, classifier on CPU | `spacy 3.7.5`, `scispacy 0.6.2`, `negspacy 1.1.0`, `torch`, `torchvision` |

```bash
conda create -n vlm python=3.10 -y && conda run -n vlm pip install -r requirements_vlm.txt
conda create -n roco_env python=3.10 -y && conda run -n roco_env pip install -r requirements_roco_env.txt
conda run -n roco_env pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz
```

The `torch` build in `roco_env` targets a newer CUDA than the server driver, so
it cannot use the GPU. CPU jobs launched from `roco_env` therefore never compete
with GPU training.

---

## Reproducing the results

**Paths.** The scripts assume the project root is `/home/matei` (several call
`os.chdir("/home/matei")`). To run elsewhere, replace that path throughout, or
run from a user account with the same home directory.

**Long runs.** Most steps take many hours. Run them inside `tmux` so they
survive disconnects, and log to a file:

```bash
tmux new -d -s job
tmux send-keys -t job 'cd /home/matei && conda activate vlm && <command> 2>&1 | tee job.log' C-m
```

Each step lists the result file it produces, so the output can be checked
against the tables above.

### 0. Data and model weights

| Resource | Location | Source |
|---|---|---|
| ROCOv2 | `rocov2/` with `train/`, `valid/`, `test/` image folders and the `*_captions.csv`, `*_concepts.csv`, `*_concepts_manual.csv`, `cui_mapping.csv` files | Zenodo (ROCOv2 release) |
| VQA-RAD | `VQA-RAD_dataset/` (`VQA_RAD Dataset Public.json` + image folder) | OSF (VQA-RAD release) |
| BLIP-2 | `blip2-opt-2.7b/` | `Salesforce/blip2-opt-2.7b` |
| Qwen3-VL-4B-Instruct | `qwen3-vl-4b-instruct/` | `Qwen/Qwen3-VL-4B-Instruct` |
| BioMedCLIP, Qwen2.5-1.5B-Instruct, DeBERTa-xlarge-MNLI, `blip2-itm-vit-g` | Hugging Face cache (downloaded automatically on first use) | Hugging Face Hub |

Qwen3-VL and BLIP-2 are loaded from local directories, so long runs have no
network dependency. If a download fails with HTTP 401, a stale token in
`~/.cache/huggingface/token` is the usual cause.

### 1. VQA-RAD models (starting checkpoints)

```bash
conda activate vlm
python extract_itm.py      # -> blip2_qformer_text_weights.pt
python train.py            # BLIP-2, frozen ViT     -> vqa_checkpoints/vqa_finetuned_final.pt
python train_lora.py       # BLIP-2, LoRA on ViT    -> vqa_checkpoints_lora/vqa_lora_final.pt
```

The BioMedVQA VQA-RAD model is trained in `biomed_captioner_v2.ipynb`, which
writes `biomed_vqa_checkpoints/biomed_vqa_final.pt`. The zero-shot BLIP-2
VQA-RAD baseline, including the three closed-ended answer parsers, is in
`blip-2_vqa_rad_SERVER.ipynb`. Evaluation of the fine-tuned VQA-RAD models is
in `blip-2_fine_tuned_VQA-RAD.ipynb` and `blip-2_fine_tuned_VQA-RAD_LoRA-ViT.ipynb`.

### 2. BLIP-2 captioning

```bash
# zero-shot
CKPT_PATH=/home/matei/vqa_checkpoints_lora/vqa_lora_final.pt RUN_TAG=zeroshot python caption_roco.py

# fine-tune (2 epochs on a 12k subset), then score the selected epoch
python train_roco_caption.py                     # -> roco_checkpoints/roco_caption_epoch{1,2}.pt
CKPT_PATH=/home/matei/roco_checkpoints/roco_caption_epoch2.pt RUN_TAG=finetuned python caption_roco.py

# add the leaderboard-comparable BERTScore to both
python rescore_bertscore.py
```

Results are in `roco_zeroshot_metrics.json` and `roco_finetuned_metrics.json`.
**Use the `BERTScore-F1-imageclef` field.** The plain `BERTScore-F1` field in
these two files uses a different scoring model and is not comparable with the
other rows.

### 3. BioMedVQA captioning

```bash
python caption_roco_biomed.py                    # zero-shot -> roco_biomed_zeroshot_metrics.json
python train_roco_biomed.py                      # -> roco_biomed_checkpoints/roco_biomed_epoch{1,2,3}.pt
CKPT_PATH=/home/matei/roco_biomed_checkpoints/roco_biomed_epoch3.pt RUN_TAG=biomed_finetuned \
  python caption_roco_biomed.py                  # -> roco_biomed_finetuned_metrics.json
```

### 4. LLaVA-style captioner

`VISION_LAYER` selects which BioMedCLIP block feeds the connector. **Its default
is `-2`**, so the last-block model must set `-1` explicitly. `RUN_TAG` (default
`vL<VISION_LAYER>`) sets the checkpoint directory.

```bash
# last ViT block: stage 1 (connector, 1 epoch) + stage 2 (LoRA, 4 epochs), ~39 h
VISION_LAYER=-1 python train_roco_llava.py      # -> roco_llava_checkpoints_vL-1/roco_llava_s2_epoch{1..4}.pt
CKPT_PATH=/home/matei/roco_llava_checkpoints_vL-1/roco_llava_s2_epoch3.pt RUN_TAG=llava_finetuned \
  python caption_roco_llava.py                   # -> roco_llava_finetuned_metrics.json

# penultimate ViT block
VISION_LAYER=-2 python train_roco_llava.py      # -> roco_llava_checkpoints_vL-2/...
CKPT_PATH=/home/matei/roco_llava_checkpoints_vL-2/roco_llava_s2_epoch2.pt RUN_TAG=llava_vL2_ep2 \
  python caption_roco_llava.py                   # -> roco_llava_vL2_ep2_metrics.json
```

Epochs 3 (last block) and 2 (penultimate block) were chosen on **validation**
BERTScore, with CIDEr breaking a tie, from `roco_llava_finetune_history*.json`.
The original last-block run predates `RUN_TAG`, so its checkpoints on the server
are in `roco_llava_checkpoints/` without a suffix. `caption_roco_llava.py` reads
the ViT layer back from the checkpoint, so evaluation always uses the same
feature path as training.

### 5. Qwen3-VL zero-shot

Each run takes about 38 h (~14 s/image, greedy decoding, `MAX_VIS_TOKENS=768`,
batch size 1).

```bash
python qwen3vl_roco_zeroshot_v0.py               # "a photo of"      -> roco_qwen3vl_zeroshot_metrics.json
python qwen3vl_roco_zeroshot_v1.py               # radiology prompt  -> roco_qwen3vl_v1_metrics.json
```

### 6. Qwen3-VL fine-tuning

All three variants use the same controlled settings: 12,000-image subset
(seed 42), 1 epoch = 3,000 optimiser steps, 250 validation images, evaluation
every 750 steps. **The scripts default to 2 epochs and 500 validation images, so
set these explicitly.** Each training run takes about 46 h; each test
evaluation about 38 h.

```bash
COMMON="EPOCHS=1 N_VAL=250 EVAL_EVERY=750"

# (a) LoRA + connector (merger) trained
env $COMMON python qwen3vl_roco_finetune.py                        # -> qwen3vl_ft_ckpt/best

# (b) connector-frozen control (LoRA only)
env $COMMON TRAIN_MERGER=0 RUN_TAG=qwen3vl_ft_frozen python qwen3vl_roco_finetune.py

# (c) + auxiliary concept heads, fixed lambda = 0.1
env $COMMON python qwen3vl_roco_finetune_cui.py                    # -> qwen3vl_ft_cui_ckpt/best

# (d) + normalised lambda, supervised with the ENRICHED labels (run step 9 first)
env $COMMON RUN_TAG=qwen3vl_ft_cui_v2 ROCO_DIR=/home/matei/rocov2_enriched \
  python qwen3vl_roco_finetune_cui_v2.py                           # -> qwen3vl_ft_cui_v2_ckpt/best
```

Before a long run, check the setup with `PROBE=1` (a few steps; prints trainable
parameters, gradient checks, peak VRAM, and the auxiliary-loss fraction).

**Evaluate** each checkpoint with the V1 script, which merges the adapter and
loads the trained connector. The concept heads are dropped at inference.

```bash
LORA_PATH=/home/matei/qwen3vl_ft_ckpt/best RUN_TAG=qwen3vl_ft_test python qwen3vl_roco_zeroshot_v1.py
SKIP_UMLS=1 SKIP_ALIGN=1 python roco_full_metrics.py roco_qwen3vl_ft_test_captions.json
# -> roco_qwen3vl_ft_test_captions_fullmetrics.json
```

Repeat with `qwen3vl_ft_cui_ckpt` / `RUN_TAG=qwen3vl_ft_cui_test` and
`qwen3vl_ft_cui_v2_ckpt` / `RUN_TAG=qwen3vl_ft_cui_v2_test`.

Two things to expect when evaluating:
- **The first log line must read `resume: 0 already done, 9927 to caption`.**
  The script resumes from `roco_<RUN_TAG>_preds.jsonl`. If that file is left
  over from an earlier run under the same tag, it silently re-scores the old
  predictions.
- **An out-of-memory traceback at the end of generation is expected.** The V1
  script tries to load DeBERTa while the 4B model still holds the GPU. Captions
  are saved before that point, and `roco_full_metrics.py` scores them in a
  fresh process.

### 7. Factuality metrics (optional)

`roco_full_metrics.py` can also compute UMLS concept-F1 and AlignScore. These
were not part of the reported results.

```bash
bash install_factuality_metrics.sh
python roco_full_metrics.py roco_qwen3vl_ft_test_captions.json
```

### 8. CNN-1 — concept classifier on original labels

```bash
conda activate roco_env
python roco_cui_classifier.py                    # -> cui_clf_ckpt/{best.pt, thresholds.json, history.json}
SPLIT=test python eval_test.py                   # -> cui_clf_ckpt/test_results.json   (0.5593)
```

Defaults: 15 epochs max, batch 32, AdamW 1e-4, cosine schedule, early stopping
with patience 4, 224 px. The script uses CUDA, then Apple MPS, then CPU,
whichever is available. `RESUME=1` continues from `last.pt` with the full
optimiser and scheduler state.

### 9. Label enrichment

The first run loads the UMLS knowledge base (about 30 min) and caches the alias
dictionary to `enriched/alias_cache.json`.

```bash
conda activate roco_env
export OMP_NUM_THREADS=3                          # leave CPU for concurrent GPU jobs
python cui_caption_matcher.py --mode train --no-veto   # -> enriched/train_concepts_enriched_noveto.csv
python cui_caption_matcher.py --mode valid             # -> enriched/valid_concepts_enriched.csv
python cui_caption_matcher.py --mode test              # -> enriched/test_concepts_enriched.csv
```

Each run writes a `*_enrichment_meta*.json` with the counts reported above.
`--mode train` without `--no-veto` reproduces the abandoned classifier-veto
variant. It requires `cui_clf_ckpt/` and writes to a separate file.
`--limit N` runs a smoke test, but **it writes to the same output path** as a
full run.

Build the enriched dataset directory. Only the three concept files are
replaced; everything else is a symbolic link to the original data:

```bash
mkdir -p rocov2_enriched
for f in rocov2/*; do
  case "$(basename "$f")" in train_concepts.csv|valid_concepts.csv|test_concepts.csv) ;;
    *) ln -s "/home/matei/$f" "rocov2_enriched/$(basename "$f")" ;; esac
done
ln -s /home/matei/enriched/train_concepts_enriched_noveto.csv rocov2_enriched/train_concepts.csv
ln -s /home/matei/enriched/valid_concepts_enriched.csv        rocov2_enriched/valid_concepts.csv
ln -s /home/matei/enriched/test_concepts_enriched.csv         rocov2_enriched/test_concepts.csv
```

### 10. CNN-2 and the comparison on enriched labels

```bash
conda activate roco_env
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4   # CPU training; 4 threads was faster than 6 here

# CNN-2: same script and hyperparameters, enriched labels, separate output dir (~2 h/epoch on CPU)
ROCO_DIR=/home/matei/rocov2_enriched OUT_DIR=/home/matei/cui_clf_ckpt_enriched \
  WORKERS=2 python roco_cui_classifier.py
ROCO_DIR=/home/matei/rocov2_enriched OUT_DIR=/home/matei/cui_clf_ckpt_enriched \
  SPLIT=test python eval_test.py                   # -> cui_clf_ckpt_enriched/test_results.json (0.4876)

# CNN-1 on the enriched test set. eval_test.py writes into OUT_DIR, so point it at a
# directory of symlinks to CNN-1's files; otherwise CNN-1's original result is overwritten.
mkdir -p cnn1_on_enriched
ln -sf /home/matei/cui_clf_ckpt/best.pt         cnn1_on_enriched/best.pt
ln -sf /home/matei/cui_clf_ckpt/thresholds.json cnn1_on_enriched/thresholds.json
ROCO_DIR=/home/matei/rocov2_enriched OUT_DIR=/home/matei/cnn1_on_enriched \
  SPLIT=test python eval_test.py                   # -> cnn1_on_enriched/test_results.json (0.4571)
```

### 11. Supplementary analyses

These support conclusions in the write-up. They are not part of the main
evaluation pipeline.

```bash
python recalibrate_cnn1.py     # CNN-1 thresholds re-tuned on enriched valid -> cnn1_recalibrated.json
                               # also saves score matrices to cnn1_enriched_scores.npz
python test_memorization.py    # CNN-1 on training vs held-out images -> memorization_test.json
python score_audit.py          # precision of enrichment from enriched/AUDIT_SHEET.csv (fill correct_YN first)
```

---

## Operational notes

Problems that cost real time during the project. Each one fails silently.

| Symptom | Cause | Prevention |
|---|---|---|
| A fine-tuning run matches a variant it should differ from | PEFT re-froze the connector | Check the `[grad check]` probe line reports all merger parameters receiving gradient |
| Evaluation finishes instantly with the previous model's scores | Stale `roco_<RUN_TAG>_preds.jsonl` | Confirm `resume: 0 already done`; use a new `RUN_TAG` per model |
| A new run overwrites an old checkpoint | `RUN_TAG` / `OUT_DIR` defaults point at a previous model | Always set them explicitly |
| Training runs but every sample is skipped | An exception inside the per-sample handler | `grep -c 'warn] skip' <log>` must print 0; smoke-test the real training loop, not only `PROBE=1` |
| Matcher fixes have no effect | Old `enriched/alias_cache.json` is reloaded | Delete the cache after changing alias construction |
| `roco_full_metrics.py` fails with `No module named spacy` | Run under `vlm` without the factuality stack | Use `SKIP_UMLS=1 SKIP_ALIGN=1`, or run step 7 first |
| A chained `python … \| tee log && next` continues after a crash | The pipeline's exit status is `tee`'s | Put `set -o pipefail` before the chain |
| DataLoader dies hours in with `Errno 24` | File-descriptor sharing strategy | Already handled in `roco_cui_classifier.py` (`file_system` strategy) |

---

## Limitations

- **One run per configuration.** Seed variance was not measured. It could
  exceed the differences between the fine-tuned Qwen3-VL variants.
- **Decoding differs across the captioning table.** Qwen3-VL uses greedy
  decoding (beam search measured at ~8 days per test pass on this GPU); the
  other systems use beam search.
- **BERTScore rewards clinical register more than clinical correctness.**
  Reading the outputs shows prompts can raise scores while the named finding is
  still wrong. Factuality metrics were implemented but not reported.
- **Enriched labels are not ground truth.** The same matcher produces training
  and test enrichment, so a systematic matcher error helps CNN-2 on the enriched
  test set. The CNN-1 vs CNN-2 comparison is tilted in CNN-2's favour by an
  amount related to the matcher's error rate.
- **The enrichment's precision is established for modality concepts** (against
  ROCOv2's manually curated labels) **but not yet independently for rare
  anatomical concepts.** A blind manual audit (`enriched/AUDIT_SHEET.csv`) is in
  progress.
- **Training budgets.** The Qwen3-VL variants use 12,000 of 59,958 training
  images and had not converged at 3,000 steps.

---

## References

- Rückert et al., *ROCOv2: Radiology Objects in COntext Version 2, an updated multimodal image dataset*, Scientific Data 11, 688 (2024).
- Lau et al., *A dataset of clinically generated visual questions and answers about radiology images* (VQA-RAD), Scientific Data 5 (2018).
- Li et al., *BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models*, ICML 2023.
- Zhang et al., *BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image–text pairs*, arXiv:2303.00915.
- Liu et al., *Visual Instruction Tuning* (LLaVA), NeurIPS 2023; Li et al., *LLaVA-Med*, NeurIPS 2023 Datasets and Benchmarks.
- Qwen Team, *Qwen2.5 Technical Report*, arXiv:2412.15115; *Qwen3-VL Technical Report*.
- Hu et al., *LoRA: Low-rank adaptation of large language models*, ICLR 2022.
- Zhang et al., *BERTScore: Evaluating text generation with BERT*, ICLR 2020.
- Huang et al., *Densely connected convolutional networks*, CVPR 2017; Radenović et al., *Fine-tuning CNN image retrieval with no human annotation* (GeM pooling), TPAMI 2019.
- Neumann et al., *ScispaCy: Fast and robust models for biomedical natural language processing*, BioNLP 2019.
- Chapman et al., *A simple algorithm for identifying negated findings and diseases in discharge summaries* (NegEx), J. Biomed. Inform. 2001.
- ImageCLEFmedical 2025 Caption task overview and participant papers, CEUR Workshop Proceedings Vol-4038.
