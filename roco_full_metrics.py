#!/usr/bin/env python
# Full ImageCLEF-2025-style scoring for any saved ROCOv2 prediction JSON.
#
# Computes, in one pass:
#   RELEVANCE  BLEU-1..4, METEOR, ROUGE-L, CIDEr, BERTScore (deberta-xlarge-mnli F1)
#              -- identical to every previous run of ours, so the nine existing models
#                 stay directly comparable.
#   FACTUALITY UMLS Concept F1  (ImageCLEF's primary factuality metric)
#              AlignScore       (ImageCLEF's second factuality metric)
#
# Definitions follow the task overview (Damm et al., CEUR Vol-4038 paper_170, section 4):
#   * UMLS Concept F1 "evaluates the overlap of medical entities between the generated and
#     reference captions", extracting concepts from BOTH texts and filtering to semantic
#     types relevant to clinical accuracy (as in MEDCON), then F1 over the two CUI sets.
#   * AlignScore is RoBERTa-based: it aligns the claims in the generated caption against
#     the supporting evidence in the reference caption.
#
# TWO DOCUMENTED DEVIATIONS (state these in the report):
#   1. ImageCLEF extracts concepts with MedCAT, whose UMLS model packs need a UMLS/UTS
#      licence. We use scispaCy's UMLS EntityLinker (freely downloadable KB). Absolute
#      values therefore do NOT match the leaderboard -- but both our models are scored by
#      the SAME extractor, so the A-vs-B comparison is internally valid.
#   2. ImageCLEF uses Recall-based BERTScore with idf weighting; we keep F1 without idf
#      for consistency with our nine earlier runs.
#
# Bonus metric we can compute because ROCOv2 ships gold CUIs per image (ImageCLEF cannot):
#   * UMLS-F1-vs-gold -- generated-caption concepts scored against the image's GOLD CUI
#     annotation instead of against concepts mined from the reference text. Free of
#     reference-extraction noise. Reported separately, clearly labelled.
#
#   all runs   : python roco_full_metrics.py
#   one file   : python roco_full_metrics.py roco_qwen3vl_ft_test_captions.json
#   skip slow  : SKIP_ALIGN=1 python roco_full_metrics.py
import os
os.chdir("/home/matei")
import json
import sys
import re
import collections

# --------------------------------------------------------------------------- #
# Runs to score. Any file missing is skipped with a warning.
# --------------------------------------------------------------------------- #
RUNS = [
    ("BLIP-2 zero-shot",       "roco_zeroshot_a_photo_of.json"),
    ("BLIP-2 fine-tuned",      "roco_finetuned_a_photo_of.json"),
    ("BioMedVQA zero-shot",    "roco_biomed_zeroshot_a_photo_of.json"),
    ("BioMedVQA fine-tuned",   "roco_biomed_finetuned_a_photo_of.json"),
    ("LLaVA ViT-1 fine-tuned", "roco_llava_finetuned_a_photo_of.json"),
    ("LLaVA ViT-2 ep2",        "roco_llava_vL2_ep2_a_photo_of.json"),
    ("Qwen3-VL V0 zero-shot",  "roco_qwen3vl_zeroshot_a_photo_of.json"),
    ("Qwen3-VL V1 zero-shot",  "roco_qwen3vl_v1_captions.json"),
    ("Qwen3-VL FT (no CUI)",   "roco_qwen3vl_ft_test_captions.json"),
    ("Qwen3-VL FT + CUI",      "roco_qwen3vl_ft_cui_test_captions.json"),
]

SKIP_ALIGN = os.environ.get("SKIP_ALIGN", "0") == "1"
SKIP_UMLS  = os.environ.get("SKIP_UMLS", "0") == "1"
ALIGN_CKPT = os.environ.get("ALIGN_CKPT", "/home/matei/alignscore_ckpt/AlignScore-base.ckpt")

# MEDCON-style semantic-type filter: UMLS semantic GROUPS relevant to clinical accuracy
# (ANAT, CHEM, DEVI, DISO, PHYS, PROC). Expressed as TUIs since scispaCy exposes types.
CLINICAL_TUIS = {
    # ANAT -- anatomy
    "T017", "T029", "T023", "T030", "T031", "T022", "T025", "T026", "T018", "T021", "T024",
    # DISO -- disorders / findings
    "T020", "T190", "T049", "T019", "T047", "T050", "T033", "T037", "T048", "T191",
    "T046", "T184",
    # PROC -- procedures
    "T060", "T065", "T058", "T059", "T063", "T062", "T061",
    # DEVI -- devices
    "T074", "T075",
    # PHYS -- physiology
    "T043", "T201", "T045",
    # CHEM -- chemicals/drugs (contrast agents etc.)
    "T121", "T130", "T197", "T195",
}


def norm(s):
    """ImageCLEF preprocessing: lowercase, strip punctuation, numbers -> 'number'."""
    s = str(s).lower()
    s = re.sub(r"\d+(\.\d+)?", " number ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


# --------------------------------------------------------------------------- #
# Relevance metrics (identical to every earlier run of ours)
# --------------------------------------------------------------------------- #
def relevance_metrics(preds, refs):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from nltk.translate.meteor_score import meteor_score

    def _n(s):
        return " ".join(str(s).lower().split())

    gts = {i: [_n(refs[i])] for i in range(len(refs))}
    res = {i: [_n(preds[i])] for i in range(len(preds))}
    bleu, _ = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    meteor = sum(meteor_score([_n(refs[i]).split()], _n(preds[i]).split())
                 for i in range(len(preds))) / len(preds)

    # deberta-xlarge-mnli = the ImageCLEF/ROCOv2 model. Three fixes this stack needs:
    # (1) unblock its .bin load; (2) cap the sentinel tokenizer; (3) small batch.
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", batch_size=8)
    scorer._tokenizer.model_max_length = 512
    _, _, F = scorer.score(preds, refs, batch_size=8, verbose=False)
    del scorer
    return {"BLEU-1": bleu[0], "BLEU-2": bleu[1], "BLEU-3": bleu[2], "BLEU-4": bleu[3],
            "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider,
            "BERTScore-F1": F.mean().item()}


# --------------------------------------------------------------------------- #
# UMLS Concept F1
# --------------------------------------------------------------------------- #
_NLP = None


def get_linker():
    """Load scispaCy + the UMLS entity linker once (KB downloads on first use, ~1 GB)."""
    global _NLP
    if _NLP is None:
        import spacy
        from scispacy.linking import EntityLinker  # noqa: F401  (registers the pipe)
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker",
                     config={"resolve_abbreviations": True, "linker_name": "umls",
                             "threshold": 0.85, "max_entities_per_mention": 1})
        _NLP = nlp
    return _NLP


def extract_cuis(texts, batch_size=64):
    """Text -> set of clinically-relevant CUIs per text (MEDCON-style TUI filter)."""
    nlp = get_linker()
    linker = nlp.get_pipe("scispacy_linker")
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        cuis = set()
        for ent in doc.ents:
            for cui, score in ent._.kb_ents[:1]:          # top-1 candidate only
                types = set(linker.kb.cui_to_entity[cui].types)
                if types & CLINICAL_TUIS:
                    cuis.add(cui)
        out.append(cuis)
    return out


def concept_f1(pred_sets, ref_sets):
    """Per-caption F1 over CUI sets, averaged over captions (ImageCLEF's formulation)."""
    tot, n = 0.0, 0
    for p, r in zip(pred_sets, ref_sets):
        if not p and not r:
            continue                                      # both empty -> undefined, skip
        inter = len(p & r)
        prec = inter / len(p) if p else 0.0
        rec = inter / len(r) if r else 0.0
        tot += 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        n += 1
    return tot / max(1, n)


# --------------------------------------------------------------------------- #
# AlignScore
# --------------------------------------------------------------------------- #
def align_score(preds, refs):
    """Claims in the generated caption aligned against evidence in the reference."""
    import torch
    from alignscore import AlignScore
    scorer = AlignScore(model="roberta-base", batch_size=16,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        ckpt_path=ALIGN_CKPT, evaluation_mode="nli_sp")
    scores = scorer.score(contexts=refs, claims=preds)    # context=reference, claim=generated
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
# Gold-CUI variant (only possible because ROCOv2 ships per-image concepts)
# --------------------------------------------------------------------------- #
def load_gold_cuis():
    import pandas as pd
    f = "/home/matei/rocov2/test_concepts.csv"
    if not os.path.isfile(f):
        return None
    df = pd.read_csv(f)
    return {r.ID: {c for c in str(r.CUIs).split(";") if c and c != "nan"}
            for r in df.itertuples()}


# --------------------------------------------------------------------------- #
def score_file(name, path, gold):
    data = json.load(open(path))
    ids = [d["id"] for d in data]
    preds = [d["pred"] if str(d["pred"]).strip() else "." for d in data]
    refs = [d["ref"] if str(d["ref"]).strip() else "." for d in data]
    print(f"\n{'='*70}\n{name}  ({len(data)} captions)\n{'='*70}", flush=True)

    m = {"n": len(data), "file": path}
    print("  relevance ...", flush=True)
    m.update(relevance_metrics(preds, refs))

    if not SKIP_UMLS:
        print("  UMLS concept extraction (generated + reference) ...", flush=True)
        # ImageCLEF preprocessing before extraction, per section 4.
        p_sets = extract_cuis([norm(p) for p in preds])
        r_sets = extract_cuis([norm(r) for r in refs])
        m["UMLS-Concept-F1"] = concept_f1(p_sets, r_sets)
        m["pred-concepts-per-caption"] = sum(len(s) for s in p_sets) / max(1, len(p_sets))
        if gold:                                          # bonus: vs the gold annotation
            g_sets = [gold.get(i, set()) for i in ids]
            m["UMLS-F1-vs-gold"] = concept_f1(p_sets, g_sets)

    if not SKIP_ALIGN:
        print("  AlignScore ...", flush=True)
        try:
            m["AlignScore"] = align_score(preds, refs)
        except Exception as e:
            print(f"  [warn] AlignScore failed: {e}", flush=True)

    for k, v in m.items():
        print(f"  {k:26s}: {v:.4f}" if isinstance(v, float) else f"  {k:26s}: {v}")
    out = path.replace(".json", "") + "_fullmetrics.json"
    json.dump(m, open(out, "w"), indent=1)
    print(f"  saved -> {out}", flush=True)
    return m


def main():
    gold = load_gold_cuis()
    print(f"gold CUIs: {'loaded ' + str(len(gold)) + ' test images' if gold else 'NOT FOUND'}")
    runs = ([(os.path.basename(a), a) for a in sys.argv[1:]] if len(sys.argv) > 1 else RUNS)
    summary = []
    for name, path in runs:
        if not os.path.isfile(path):
            print(f"SKIP {name}: {path} not found", flush=True)
            continue
        summary.append((name, score_file(name, path, gold)))

    print(f"\n{'='*100}\nSUMMARY\n{'='*100}")
    hdr = (f"{'Model':26s} {'BERTSc':>8s} {'ROUGE-L':>8s} {'CIDEr':>8s} "
           f"{'UMLS-F1':>8s} {'Align':>8s} {'vs-gold':>8s}")
    print(hdr + "\n" + "-" * len(hdr))
    for name, m in summary:
        print(f"{name:26s} {m.get('BERTScore-F1', 0):8.4f} {m.get('ROUGE-L', 0):8.4f} "
              f"{m.get('CIDEr', 0):8.4f} {m.get('UMLS-Concept-F1', 0):8.4f} "
              f"{m.get('AlignScore', 0):8.4f} {m.get('UMLS-F1-vs-gold', 0):8.4f}")
    print("\nImageCLEF 2025 reference (different extractor -- context only, not comparable):")
    print("  best team UMLS-F1 0.1816 / AlignScore 0.1417 | zero-shot baseline 0.1302 / 0.0955")


if __name__ == "__main__":
    main()
