#!/usr/bin/env python
# Add the ImageCLEF/ROCOv2-comparable BERTScore (deberta-xlarge-mnli F1) to every
# ROCO captioning run, RE-USING THE SAVED PREDICTIONS -- no model is re-run.
# It reads roco_<tag>_a_photo_of.json (id/ref/pred) and updates roco_<tag>_metrics.json
# with a "BERTScore-F1-imageclef" field.
#   python rescore_bertscore.py
import os
os.chdir("/home/matei")
import json
from imageclef_bertscore import imageclef_bertscore_f1, IMAGECLEF_MODEL

RUNS = [
    ("BLIP-2 zero-shot",  "roco_zeroshot_a_photo_of.json",        "roco_zeroshot_metrics.json"),
    ("BLIP-2 fine-tuned", "roco_finetuned_a_photo_of.json",       "roco_finetuned_metrics.json"),
    ("BioMedVQA zero-shot","roco_biomed_zeroshot_a_photo_of.json","roco_biomed_zeroshot_metrics.json"),
]

summary = []
for name, predfile, mfile in RUNS:
    if not os.path.isfile(predfile):
        print(f"SKIP {name}: {predfile} not found", flush=True)
        continue
    d = json.load(open(predfile))
    preds = [x["pred"] for x in d]
    refs  = [x["ref"]  for x in d]
    print(f"[{name}] scoring {len(d)} pairs with {IMAGECLEF_MODEL} ...", flush=True)
    f1 = imageclef_bertscore_f1(preds, refs)
    print(f"[{name}] ImageCLEF BERTScore-F1 = {f1:.4f}  (n={len(d)})", flush=True)
    summary.append((name, len(d), f1))

    m = json.load(open(mfile)) if os.path.isfile(mfile) else {"n": len(d)}
    m["BERTScore-F1-imageclef"] = f1
    m["BERTScore-imageclef-model"] = IMAGECLEF_MODEL
    with open(mfile, "w") as f:
        json.dump(m, f, indent=1)
    print(f"[{name}] updated {mfile}", flush=True)

print("\n" + "=" * 66, flush=True)
print(f"  ImageCLEF-comparable BERTScore ({IMAGECLEF_MODEL}, F1)")
print("  ROCOv2 leaderboard: baseline 0.6264 | CSIRO (best) 0.6413")
print("=" * 66, flush=True)
for name, n, f1 in summary:
    print(f"  {name:22s} (n={n:>4}) : {f1:.4f}", flush=True)
