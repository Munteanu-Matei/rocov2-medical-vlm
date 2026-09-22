#!/usr/bin/env bash
# Install the two ImageCLEF-2025 factuality metrics: UMLS Concept F1 + AlignScore.
# Run once:  bash install_factuality_metrics.sh
#
# NOTE on UMLS Concept F1: ImageCLEF uses MedCAT, whose UMLS model packs require a
# UMLS/UTS licence (free, but needs registration + approval). We use scispaCy's UMLS
# EntityLinker instead -- its knowledge base is freely downloadable. Absolute values will
# therefore NOT match the ImageCLEF leaderboard, but BOTH our models are scored with the
# SAME extractor, so the A-vs-B comparison is internally valid.
set -e
PY=/home/matei/miniconda3/envs/vlm/bin/python
PIP="$PY -m pip"

echo "=== 1/4 scispaCy + a scientific spaCy model ==="
$PIP install -q scispacy
$PIP install -q https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

echo "=== 2/4 AlignScore (from source; not on PyPI) ==="
cd /home/matei
[ -d AlignScore ] || git clone -q https://github.com/yuh-zha/AlignScore.git
$PIP install -q -e ./AlignScore
$PY -m spacy download en_core_web_sm -q || true   # AlignScore splits sentences with this

echo "=== 3/4 AlignScore checkpoint (RoBERTa-base, ~1.5 GB) ==="
mkdir -p /home/matei/alignscore_ckpt
CKPT=/home/matei/alignscore_ckpt/AlignScore-base.ckpt
[ -f "$CKPT" ] || curl -L --retry 5 --retry-delay 10 -o "$CKPT" \
  https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt

echo "=== 4/4 verify ==="
$PY - <<'PY'
import importlib
for m in ("scispacy", "spacy", "alignscore"):
    try:
        importlib.import_module(m); print(f"  {m}: OK")
    except Exception as e:
        print(f"  {m}: FAILED -> {e}")
import os
p = "/home/matei/alignscore_ckpt/AlignScore-base.ckpt"
print(f"  checkpoint: {'OK ' + str(round(os.path.getsize(p)/1e9,2)) + ' GB' if os.path.isfile(p) else 'MISSING'}")
PY
echo "done. The scispaCy UMLS knowledge base (~1 GB) downloads on FIRST USE of the scorer."
