#!/usr/bin/env python
"""CUI → Caption Matcher: label enrichment for ROCOv2.

Implements the rigorous design from matcher_design.md:
  1. Build alias dictionaries from UMLS Metathesaurus via scispaCy
  2. Token-level matching with lemmatisation, case-sensitive short aliases,
     inverted index, and negation detection (negspaCy / NegEx)
  3. Training enrichment — caption match AND model veto (sigmoid ≥ per-label τ)
  4. Test/valid enrichment — caption match only (no model involvement)

Usage:
  # Install dependencies (once):
  pip install spacy scispacy negspacy
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz

  # Run enrichment on training set (model-gated):
  python cui_caption_matcher.py --mode train

  # Run enrichment on valid/test set (caption-only):
  python cui_caption_matcher.py --mode valid

  # Quick smoke test on 200 images:
  python cui_caption_matcher.py --mode train --limit 200

Environment variables (same convention as roco_cui_classifier.py):
  ROCO_DIR    path to rocov2 data           (default: ./rocov2)
  OUT_DIR     path to model checkpoint dir  (default: ./cui_clf_ckpt)
"""

import os, sys, json, re, time, argparse, collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# ── config ──────────────────────────────────────────────────────────────────
ROCO_DIR = os.environ.get("ROCO_DIR", "./rocov2")
OUT_DIR  = os.environ.get("OUT_DIR", "./cui_clf_ckpt")
ENRICH_DIR = os.environ.get("ENRICH_DIR", "./enriched")

# ── short medical abbreviations: case-sensitive matching required ───────────
# These 2-char aliases are common in radiology but collide with English words
# when lowercased.  We match them CASE-SENSITIVELY on the raw token.
# Short single-token aliases are the single biggest source of false matches.
# "CT" is a legitimate UMLS alias for BOTH Computed Tomography (C0040405) and
# Carpal Tunnel Syndrome (C0007286); "as" for Aortic Valve Stenosis; "it" for
# Intrathecal Route.  Measured on the training captions, accepting them blindly
# injects ~17,000 spurious matches (ct alone fires on 20.8% of captions).
# Rule: a single-token alias of <=3 chars is kept ONLY if
#   (a) it is a letter+digit level code (l5, t12, c7) -- unambiguous, or
#   (b) the (ABBREV, CUI) pair is on this curated allowlist,
# and in both cases it is stored ONLY as a case-sensitive sentinel.
ABBREV_ALLOWLIST = {
    "CT":  {"C0040405"},   # X-Ray Computed Tomography
    "MR":  {"C0024485"},   # Magnetic Resonance Imaging
    "MRI": {"C0024485"},
    "US":  {"C0041618"},   # Ultrasonography
    "AP":  {"C1999039"},   # Anterior-Posterior
    "PA":  {"C1996865"},   # Postero-Anterior
    "PET": {"C0032743"},   # Positron-Emission Tomography
}
LEVEL_CODE = re.compile(r"^[a-z]\d{1,2}$")     # l5, t12, c7 -- vertebral levels
# Short aliases that are ordinary English words, not abbreviations.  These are
# safe to match case-insensitively as normal lemmas; the gate above would
# otherwise discard them and cost real recall on anatomy terms.
WORD_ALIAS_ALLOW = {
    "arm", "eye", "ear", "fat", "gut", "hip", "leg", "rib",
    "jaw", "lip", "toe", "gum", "sac", "air", "vein",
}
SHORT_ALIAS_MAXLEN = 3
K_MAX_CAP = 8          # longest alias worth sweeping for in a caption

# ── alias blocklist: aliases that fire too broadly ──────────────────────────
# These are valid UMLS atoms but too generic to provide concept-specific evidence.
# Matching "scan" for CT would also fire on ultrasound/MRI captions, etc.
ALIAS_BLOCKLIST = {
    # Single words that are too common / ambiguous in radiology text
    "scan", "study", "image", "imaging", "images", "view", "finding",
    "findings", "case", "procedure", "examination", "exam", "test",
    "normal", "disease", "syndrome", "disorder", "lesion", "lesions",
    "condition", "process", "treatment", "therapy", "surgery",
    "patient", "patients", "diagnosis", "change", "changes",
    "type", "form", "group", "level", "levels", "area", "areas",
    "region", "regions", "site", "sites", "part", "parts",
    "side", "structure", "structures", "body", "tissue", "tissues",
    "cell", "cells", "system", "agent", "other", "nos",
    "mass", "tube", "line", "ring", "band", "wall",
    "space", "block", "plane", "segment", "branch",
    "stage", "phase", "course", "state", "status",
    "left", "right", "upper", "lower", "small", "large",
    "open", "closed", "present", "absent", "single", "multiple",
    "male", "female", "adult", "child", "infant",
    "anterior", "posterior", "lateral", "medial", "proximal", "distal",
    "superior", "inferior", "central", "peripheral",
    "acute", "chronic", "primary", "secondary",
    "general", "local", "complete", "incomplete", "partial",
    "entire", "total", "major", "minor",
}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0 — NLP Pipeline Setup
# ═══════════════════════════════════════════════════════════════════════════

def build_nlp_pipeline():
    """Build spaCy pipeline with negation detection.

    Uses en_core_sci_sm (scispaCy's small biomedical model) for tokenization
    and lemmatization, plus negspaCy for NegEx-based negation detection.
    """
    import spacy
    from negspacy.negation import Negex
    # Import scispacy.linking to ensure the 'scispacy_linker' component is registered
    import scispacy.linking

    nlp = spacy.load("en_core_sci_sm", disable=["ner", "parser"])
    # Re-enable sentencizer (parser is disabled for speed, but we need
    # sentence boundaries for negation scope).
    nlp.add_pipe("sentencizer")
    # Add NegEx — it requires NER entities to check negation on.
    # We'll use it differently: we run NER on our alias-matched spans.
    # But we still add the component so we can access its negation cues.
    from negspacy.termsets import termset
    ts = termset("en_clinical")
    nlp.add_pipe(
        "negex",
        config={
            "neg_termset": ts.get_patterns(),    # clinical negation triggers
            "chunk_prefix": [],                  # we handle spans ourselves
        },
        last=True,
    )
    return nlp


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Alias Dictionary Construction (§3 of design)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_alias(s: str) -> str:
    """Normalize an alias string: lowercase, strip qualifiers, collapse spaces."""
    s = s.strip().lower()
    # Remove trailing parenthetical qualifiers: "kidney (organ)" → "kidney"
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_safe_alias(tokens: tuple, raw_text: str) -> bool:
    """Filter out unsafe aliases: too short, or in the blocklist."""
    # Single-character aliases match everywhere
    if len(tokens) == 1 and len(tokens[0]) <= 1:
        return False
    # Single-token aliases that are in the blocklist
    if len(tokens) == 1 and tokens[0] in ALIAS_BLOCKLIST:
        return False
    # Empty after normalization
    if not tokens or all(t == "" for t in tokens):
        return False
    return True


def build_alias_dictionary(label_space: list[str], nlp, cui_canonical: dict[str, str]):
    """Build the alias dictionary and inverted index from UMLS via scispaCy.

    Returns:
        alias_dict:  {CUI: set of alias tuples (lemmatized tokens)}
        inv_index:   {alias_tuple: set of CUIs}
        k_max:       length of the longest alias (in tokens)
    """
    print("loading UMLS knowledge base via scispaCy ...", flush=True)
    t0 = time.time()
    if "scispacy_linker" not in nlp.pipe_names:
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    kb = nlp.get_pipe("scispacy_linker").kb
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    alias_dict = {}   # CUI → set of tuples
    inv_index = {}    # tuple → set of CUIs
    k_max = 1
    skipped_no_cui = 0

    label_set = set(label_space)
    print(f"building alias sets for {len(label_set)} concepts ...", flush=True)

    for c in tqdm(label_space, desc="alias construction"):
        aliases_raw = set()

        # Get UMLS atoms from scispaCy's knowledge base
        try:
            entity = kb.cui_to_entity.get(c)
            if entity is not None:
                # entity.aliases is a list of alias strings
                for a in entity.aliases:
                    aliases_raw.add(a)
                # Also add the canonical name from UMLS KB
                if entity.canonical_name:
                    aliases_raw.add(entity.canonical_name)
        except Exception:
            skipped_no_cui += 1

        # Also add the canonical name from our cui_mapping.csv
        if c in cui_canonical:
            aliases_raw.add(cui_canonical[c])

        # Process each alias
        alias_set = set()
        for raw_alias in aliases_raw:
            a_norm = normalize_alias(raw_alias)
            if not a_norm:
                continue

            # Tokenize and lemmatize using spaCy
            doc = nlp(a_norm)
            a_tokens = tuple(tok.text.lower() for tok in doc)
            a_lemmas = tuple(tok.lemma_.lower() for tok in doc)

            if not is_safe_alias(a_tokens, a_norm):
                continue

            # Short single-token aliases: case-sensitive sentinel ONLY, and only
            # if allowlisted or a level code.  Adding the bare lemma here would
            # make them case-INSENSITIVE and defeat the whole protection.
            if (len(a_tokens) == 1 and len(a_tokens[0]) <= SHORT_ALIAS_MAXLEN
                    and a_tokens[0] not in WORD_ALIAS_ALLOW):
                up = a_tokens[0].upper()
                if LEVEL_CODE.match(a_tokens[0]) or c in ABBREV_ALLOWLIST.get(up, set()):
                    alias_set.add(("__CASE__", up))
                continue                      # never add the bare lemma form

            # Normal (>=4 char or multi-token) alias: lemma match, case-insensitive
            alias_set.add(a_lemmas)
            k_max = max(k_max, len(a_lemmas))

        alias_dict[c] = alias_set

        # Build inverted index
        for a in alias_set:
            inv_index.setdefault(a, set()).add(c)

    if skipped_no_cui:
        print(f"  {skipped_no_cui} CUIs not found in UMLS KB", flush=True)

    # Cap the n-gram sweep: UMLS contains 60-token descriptive atoms that can never
    # match a caption phrase, and the matcher costs O(k_max x sentence_len) per sentence.
    k_max = min(k_max, K_MAX_CAP)

    n_aliases = sum(len(v) for v in alias_dict.values())
    n_case = sum(1 for aliases in alias_dict.values()
                 for a in aliases if len(a) >= 2 and a[0] == "__CASE__")
    print(f"  {n_aliases} alias entries ({n_case} case-sensitive) | "
          f"k_max = {k_max} tokens | "
          f"inverted index: {len(inv_index)} unique patterns", flush=True)

    return alias_dict, inv_index, k_max


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Negation Detection (§5 of design)
# ═══════════════════════════════════════════════════════════════════════════

# NegEx-style negation cues for manual scope detection.
# We use these to build negation scopes over sentence tokens, since we do
# our own alias-span matching (not spaCy NER).

NEGATION_CUES_PRE = [
    # Multi-word cues first (greedy match)
    "no evidence of", "no signs of", "no sign of",
    "negative for", "absence of", "absent of",
    "free of", "ruled out", "rule out", "rules out",
    "no definite", "no significant", "no obvious",
    "no acute", "no focal", "no suspicious",
    "without evidence of", "with no",
    "no new", "no residual", "no recurrent",
    "does not demonstrate", "did not demonstrate",
    "does not show", "did not show",
    "fails to demonstrate", "failed to demonstrate",
    "cannot be identified", "could not be identified",
    "not identified", "not demonstrated", "not seen",
    "not visualized", "not appreciated", "not detected",
    "not present", "not evident",
    # Single-word cues
    "no", "not", "without", "denies", "denied",
    "neither", "never", "nor",
    "unremarkable",
]

NEGATION_CUES_POST = [
    "unlikely", "was ruled out", "has been ruled out",
    "were ruled out", "has been excluded", "was excluded",
    "were excluded", "is excluded",
]

# Scope terminators: negation scope ends at these tokens
SCOPE_TERMINATORS = {",", ";", ".", ":", "but", "however", "although", "though",
                     "except", "apart", "aside", "whereas"}


def find_negation_scopes(tokens_lower: list[str]) -> list[tuple[int, int]]:
    """Identify negation scopes within a sentence's token list.

    Uses NegEx-style rules:
    - Preceding cues: scope extends forward from the cue until a terminator
    - Post-positive cues: scope extends backward from the cue

    Returns list of (start_idx, end_idx) spans that are negated (inclusive).
    """
    n = len(tokens_lower)
    negated = [False] * n
    joined = " ".join(tokens_lower)

    # ── Forward (preceding) negation cues ───────────────────────────────
    for cue in NEGATION_CUES_PRE:
        cue_tokens = cue.split()
        cue_len = len(cue_tokens)

        for i in range(n - cue_len + 1):
            if tokens_lower[i : i + cue_len] == cue_tokens:
                # Scope starts after the cue
                scope_start = i + cue_len
                scope_end = n  # default: rest of sentence

                # Find scope terminator
                for j in range(scope_start, n):
                    if tokens_lower[j] in SCOPE_TERMINATORS:
                        scope_end = j
                        break

                # Mark tokens in scope as negated
                for j in range(scope_start, min(scope_end, n)):
                    negated[j] = True

    # ── Backward (post-positive) negation cues ──────────────────────────
    for cue in NEGATION_CUES_POST:
        cue_tokens = cue.split()
        cue_len = len(cue_tokens)

        for i in range(n - cue_len + 1):
            if tokens_lower[i : i + cue_len] == cue_tokens:
                # Scope extends backward from the cue
                scope_end = i
                scope_start = 0

                # Find scope terminator going backward
                for j in range(i - 1, -1, -1):
                    if tokens_lower[j] in SCOPE_TERMINATORS:
                        scope_start = j + 1
                        break

                for j in range(scope_start, scope_end):
                    negated[j] = True

    # Convert boolean mask to spans
    spans = []
    in_span = False
    start = 0
    for i in range(n):
        if negated[i] and not in_span:
            start = i
            in_span = True
        elif not negated[i] and in_span:
            spans.append((start, i))   # end is exclusive
            in_span = False
    if in_span:
        spans.append((start, n))

    return spans


def is_negated(match_start: int, match_end: int,
               neg_spans: list[tuple[int, int]]) -> bool:
    """Check if a match span overlaps with any negation scope.

    Conservative: any overlap counts as negated.
    match_end is exclusive (i.e., the span is [match_start, match_end)).
    """
    for ns, ne in neg_spans:
        # Overlap check: two intervals [a,b) and [c,d) overlap iff a < d and c < b
        if match_start < ne and ns < match_end:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Matching Function (§9.1 of design)
# ═══════════════════════════════════════════════════════════════════════════

def match_caption(caption: str, inv_index: dict, label_set: set,
                  k_max: int, nlp) -> set[str]:
    """Return set of CUIs with positive, non-negated mention in caption.

    Implements §9.1 of the design:
    - Tokenize + lemmatize the caption per sentence
    - Slide n-gram windows of length 1..k_max
    - Check lemmatized match in inverted index
    - Check case-sensitive match for short aliases
    - Reject matches inside negation scopes
    """
    doc = nlp(caption)
    matched_cuis = set()

    for sent in doc.sents:
        tokens = [tok for tok in sent]
        if not tokens:
            continue

        lemmas = [tok.lemma_.lower() for tok in tokens]
        raws   = [tok.text for tok in tokens]
        lowers = [tok.text.lower() for tok in tokens]
        n_tok  = len(tokens)

        # Identify negation scopes in this sentence
        neg_spans = find_negation_scopes(lowers)

        # Slide window of length 1..k_max over the sentence
        for ngram_len in range(1, min(k_max, n_tok) + 1):
            for i in range(n_tok - ngram_len + 1):
                ngram_lemma = tuple(lemmas[i : i + ngram_len])

                # Look up lemmatized n-gram in inverted index
                candidates = set()
                found = inv_index.get(ngram_lemma)
                if found:
                    candidates.update(found)

                # Case-sensitive match for single-token short aliases
                if ngram_len == 1:
                    raw_tok = raws[i]
                    # Check uppercase form (CT, MR, US, ...)
                    case_key = ("__CASE__", raw_tok.upper())
                    found_case = inv_index.get(case_key)
                    if found_case:
                        # Accept ONLY if the token is genuinely uppercase in the
                        # caption.  The old second clause (`raw_tok.upper() in
                        # CASE_SENSITIVE_ABBREVS`) was a loophole: it re-admitted
                        # lowercase "ct"/"as"/"it", which is precisely the
                        # case-sensitivity this branch exists to enforce.
                        # "L5".isupper() is True (digits are uncased), so vertebral
                        # level codes still match.
                        if raw_tok.isupper():
                            candidates.update(found_case)

                # Filter to label space and check negation
                for c in candidates:
                    if c not in label_set:
                        continue
                    # Check if this match span is negated
                    if not is_negated(i, i + ngram_len, neg_spans):
                        matched_cuis.add(c)

    return matched_cuis


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Model Scoring for Training Enrichment (§9.2 of design)
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_thresholds(device: str):
    """Load the trained CUINet model and per-label thresholds.

    Returns:
        model:       CUINet on device, in eval mode
        thresholds:  np.ndarray of shape [C] — per-label τ_c
        vocab:       list of CUI strings (the label space)
        idx:         dict CUI → int index
    """
    from roco_cui_classifier import CUINet
    from torchvision import transforms as T

    thresh_path = os.path.join(OUT_DIR, "thresholds.json")
    ckpt_path = os.path.join(OUT_DIR, "best.pt")

    th = json.load(open(thresh_path))
    vocab = th["vocab"]
    tau = np.asarray(th["per_label_tau"], dtype=np.float64)
    img_size = int(th.get("img_size", 224))
    C = len(vocab)

    assert tau.shape == (C,), f"threshold/vocab mismatch: {tau.shape} vs {C}"

    model = CUINet(C).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f"  loaded model from {ckpt_path} ({C} concepts)", flush=True)
    print(f"  loaded thresholds from {thresh_path}", flush=True)

    return model, tau, vocab, img_size


def score_images_batch(model, image_paths: list[str], img_size: int,
                       device: str, batch_size: int = 64) -> np.ndarray:
    """Run forward pass on a batch of images, return sigmoid scores [N, C].

    Uses the same eval transform as roco_cui_classifier.py.
    """
    from torchvision import transforms as T
    from PIL import Image

    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    eval_tf = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    all_scores = []
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (img_size, img_size))
            imgs.append(eval_tf(img))

        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            scores = torch.sigmoid(logits).float().cpu().numpy()
        all_scores.append(scores)

    return np.concatenate(all_scores, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Enrichment Functions (§9.2 of design)
# ═══════════════════════════════════════════════════════════════════════════

def enrich_train_labels(caption_cuis: set[str], current_labels: set[str],
                        model_scores: np.ndarray, thresholds: np.ndarray,
                        idx: dict[str, int]) -> set[str]:
    """Training enrichment: accept caption-matched CUI only if model agrees.

    §9.2 Step 2: c ∉ L_i AND match(c, t_i) = 1 AND model_score[c] ≥ τ_c
    """
    new_labels = set()
    for c in caption_cuis:
        if c in current_labels:
            continue               # already labeled
        ci = idx.get(c)
        if ci is None:
            continue               # not in model's vocabulary
        if model_scores[ci] >= thresholds[ci]:
            new_labels.add(c)      # model agrees → accept
    return current_labels | new_labels


def enrich_test_labels(caption_cuis: set[str],
                       current_labels: set[str]) -> set[str]:
    """Test enrichment: accept all non-negated caption matches, no model.

    §9.2 Step 3: c ∉ L_i AND match(c, t_i) = 1
    """
    return current_labels | caption_cuis


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def load_captions(split: str) -> dict[str, str]:
    """Load captions: {image_id: caption_text}."""
    df = pd.read_csv(os.path.join(ROCO_DIR, f"{split}_captions.csv"))
    return {str(r.ID): str(r.Caption) for r in df.itertuples()}


def load_concepts(split: str) -> dict[str, list[str]]:
    """Load concept labels: {image_id: [CUI, ...]}."""
    df = pd.read_csv(os.path.join(ROCO_DIR, f"{split}_concepts.csv"))
    return {str(r.ID): [c for c in str(r.CUIs).split(";") if c and c != "nan"]
            for r in df.itertuples()}


def load_cui_mapping() -> dict[str, str]:
    """Load CUI → canonical name mapping."""
    # NB: itertuples() renames "Canonical name" (it contains a space) to "_2",
    # so attribute access silently yields "" for every row.  Index by column name.
    df = pd.read_csv(os.path.join(ROCO_DIR, "cui_mapping.csv"))
    return {str(k): str(v) for k, v in
            df.set_index("CUI")["Canonical name"].to_dict().items()}


def save_enriched_concepts(image_concepts: dict[str, list[str]],
                           output_path: str):
    """Save enriched concept labels in the same CSV format as ROCOv2."""
    rows = []
    for img_id in sorted(image_concepts.keys()):
        cuis = image_concepts[img_id]
        rows.append({"ID": img_id, "CUIs": ";".join(cuis) if cuis else ""})
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  saved {len(rows)} images → {output_path}", flush=True)


def run_enrichment(mode: str, limit: int = 0, batch_size: int = 64,
                   save_alias_cache: bool = True, no_veto: bool = False):
    """Run the full enrichment pipeline.

    Args:
        mode:   "train" (model-gated) or "valid"/"test" (caption-only)
        limit:  process only this many images (0 = all); for smoke testing
        batch_size: batch size for model scoring
        save_alias_cache: whether to cache the alias dictionary to disk
    """
    t_start = time.time()

    # ── device ──────────────────────────────────────────────────────────
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    # ── load data ───────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Loading {mode} data ...", flush=True)
    print(f"{'='*60}", flush=True)

    captions = load_captions(mode)
    concepts = load_concepts(mode)
    cui_mapping = load_cui_mapping()
    print(f"  {len(captions)} captions, {len(concepts)} concept records", flush=True)

    # ── load label space + model (for train mode) ───────────────────────
    th = json.load(open(os.path.join(OUT_DIR, "thresholds.json")))
    vocab = th["vocab"]      # the 1571-concept label space
    idx = {c: i for i, c in enumerate(vocab)}
    label_set = set(vocab)

    model, thresholds, img_size = None, None, 224
    use_veto = (mode == "train") and not no_veto
    if use_veto:
        print("\nLoading trained model for training enrichment (model veto) ...",
              flush=True)
        model, thresholds, _, img_size = load_model_and_thresholds(device)

    # ── build NLP pipeline + alias dictionary ───────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Building NLP pipeline ...", flush=True)
    print(f"{'='*60}", flush=True)
    nlp = build_nlp_pipeline()

    alias_cache_path = os.path.join(ENRICH_DIR, "alias_cache.json")
    os.makedirs(ENRICH_DIR, exist_ok=True)

    # Try loading cached alias dictionary
    alias_dict, inv_index, k_max = None, None, 1
    if save_alias_cache and os.path.isfile(alias_cache_path):
        print(f"  loading cached alias dictionary from {alias_cache_path} ...",
              flush=True)
        try:
            cache = json.load(open(alias_cache_path))
            alias_dict = {c: {tuple(a) for a in aliases}
                          for c, aliases in cache["alias_dict"].items()}
            inv_index = {tuple(k): set(v) for k, v in cache["inv_index"]}
            k_max = cache["k_max"]
            print(f"  loaded {sum(len(v) for v in alias_dict.values())} aliases, "
                  f"k_max={k_max}", flush=True)
        except Exception as e:
            print(f"  cache load failed ({e}), rebuilding ...", flush=True)
            alias_dict = None

    if alias_dict is None:
        alias_dict, inv_index, k_max = build_alias_dictionary(
            vocab, nlp, cui_mapping
        )
        # Cache to disk
        if save_alias_cache:
            cache = {
                "alias_dict": {c: [list(a) for a in aliases]
                               for c, aliases in alias_dict.items()},
                "inv_index": [[list(k), list(v)]
                              for k, v in inv_index.items()],
                "k_max": k_max,
            }
            json.dump(cache, open(alias_cache_path, "w"))
            print(f"  cached alias dictionary → {alias_cache_path}", flush=True)

    # ── select images to process ────────────────────────────────────────
    # Only process images that have both a caption and a concept record
    img_ids = sorted(set(captions.keys()) & set(concepts.keys()))
    if limit:
        img_ids = img_ids[:limit]
    print(f"\nProcessing {len(img_ids)} images ({mode} enrichment) ...",
          flush=True)

    # ── Phase 1: caption matching (all modes) ───────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Phase 1: Caption → CUI matching ...", flush=True)
    print(f"{'='*60}", flush=True)

    caption_matches = {}     # image_id → set of matched CUIs
    match_stats = collections.Counter()

    for img_id in tqdm(img_ids, desc="matching captions"):
        caption = captions.get(img_id, "")
        if not caption or caption == "nan":
            caption_matches[img_id] = set()
            match_stats["empty_caption"] += 1
            continue

        matched = match_caption(caption, inv_index, label_set, k_max, nlp)
        caption_matches[img_id] = matched

        if matched:
            match_stats["images_with_match"] += 1
            match_stats["total_matches"] += len(matched)
            # How many are NEW (not in current labels)?
            current = set(concepts.get(img_id, []))
            new_from_caption = matched - current
            match_stats["total_new_matches"] += len(new_from_caption)
            if new_from_caption:
                match_stats["images_with_new_match"] += 1
        else:
            match_stats["images_without_match"] += 1

    print(f"\nCaption matching results:", flush=True)
    print(f"  images with ≥1 match:       {match_stats['images_with_match']}", flush=True)
    print(f"  images with ≥1 NEW match:   {match_stats['images_with_new_match']}", flush=True)
    print(f"  total caption matches:       {match_stats['total_matches']}", flush=True)
    print(f"  total NEW matches:           {match_stats['total_new_matches']}", flush=True)
    print(f"  images without any match:    {match_stats['images_without_match']}", flush=True)

    # ── Phase 2: enrichment ─────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    if use_veto:
        print("Phase 2: Training enrichment (model-gated) ...", flush=True)
    else:
        print(f"Phase 2: {mode.capitalize()} enrichment (caption-only) ...",
              flush=True)
    print(f"{'='*60}", flush=True)

    enriched_concepts = {}
    enrich_stats = collections.Counter()

    if use_veto:
        # ── Training enrichment: need model scores ──────────────────────
        # Collect all images that have NEW caption matches (need scoring)
        images_needing_scores = []
        for img_id in img_ids:
            current = set(concepts.get(img_id, []))
            new_matches = caption_matches.get(img_id, set()) - current
            if new_matches:
                img_path = os.path.join(ROCO_DIR, mode, f"{img_id}.jpg")
                if os.path.isfile(img_path):
                    images_needing_scores.append((img_id, img_path))

        print(f"  {len(images_needing_scores)} images need model scoring ...",
              flush=True)

        # Score in batches
        if images_needing_scores:
            score_ids = [x[0] for x in images_needing_scores]
            score_paths = [x[1] for x in images_needing_scores]

            print("  running model forward passes ...", flush=True)
            all_scores = score_images_batch(
                model, score_paths, img_size, device, batch_size
            )
            score_map = {sid: all_scores[i]
                         for i, sid in enumerate(score_ids)}
        else:
            score_map = {}

        # Apply enrichment
        for img_id in tqdm(img_ids, desc="enriching (train)"):
            current = set(concepts.get(img_id, []))
            caption_cuis = caption_matches.get(img_id, set())

            if img_id in score_map:
                enriched = enrich_train_labels(
                    caption_cuis, current, score_map[img_id], thresholds, idx
                )
            else:
                enriched = current  # no new matches → keep original

            n_added = len(enriched) - len(current)
            enrich_stats["total_added"] += n_added
            if n_added > 0:
                enrich_stats["images_enriched"] += 1
            enrich_stats["total_original"] += len(current)
            enrich_stats["total_final"] += len(enriched)

            enriched_concepts[img_id] = sorted(enriched)

    else:
        # ── Test/valid enrichment: caption-only, no model ───────────────
        for img_id in tqdm(img_ids, desc=f"enriching ({mode})"):
            current = set(concepts.get(img_id, []))
            caption_cuis = caption_matches.get(img_id, set())

            enriched = enrich_test_labels(caption_cuis, current)

            n_added = len(enriched) - len(current)
            enrich_stats["total_added"] += n_added
            if n_added > 0:
                enrich_stats["images_enriched"] += 1
            enrich_stats["total_original"] += len(current)
            enrich_stats["total_final"] += len(enriched)

            enriched_concepts[img_id] = sorted(enriched)

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\nEnrichment results ({mode}):", flush=True)
    print(f"  images enriched:           {enrich_stats['images_enriched']} / {len(img_ids)}", flush=True)
    print(f"  total labels added:        {enrich_stats['total_added']}", flush=True)
    print(f"  original total labels:     {enrich_stats['total_original']}", flush=True)
    print(f"  final total labels:        {enrich_stats['total_final']}", flush=True)
    if enrich_stats["total_original"] > 0:
        pct = 100 * enrich_stats["total_added"] / enrich_stats["total_original"]
        print(f"  label increase:            +{pct:.1f}%", flush=True)
    avg_orig = enrich_stats["total_original"] / max(1, len(img_ids))
    avg_final = enrich_stats["total_final"] / max(1, len(img_ids))
    print(f"  avg concepts/image:        {avg_orig:.2f} → {avg_final:.2f}", flush=True)

    # ── Save ────────────────────────────────────────────────────────────
    suffix = "_noveto" if (mode == "train" and no_veto) else ""
    out_path = os.path.join(ENRICH_DIR, f"{mode}_concepts_enriched{suffix}.csv")
    save_enriched_concepts(enriched_concepts, out_path)

    # Also save enrichment metadata
    meta = {
        "mode": mode,
        "n_images": len(img_ids),
        "n_enriched": int(enrich_stats["images_enriched"]),
        "n_labels_added": int(enrich_stats["total_added"]),
        "n_labels_original": int(enrich_stats["total_original"]),
        "n_labels_final": int(enrich_stats["total_final"]),
        "label_increase_pct": float(100 * enrich_stats["total_added"] /
                                    max(1, enrich_stats["total_original"])),
        "avg_concepts_original": float(avg_orig),
        "avg_concepts_final": float(avg_final),
        "model_checkpoint": os.path.join(OUT_DIR, "best.pt") if use_veto else None,
        "thresholds_file": os.path.join(OUT_DIR, "thresholds.json") if use_veto else None,
        "elapsed_seconds": time.time() - t_start,
    }
    meta_path = os.path.join(ENRICH_DIR, f"{mode}_enrichment_meta{suffix}.json")
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"  metadata → {meta_path}", flush=True)

    # ── Per-CUI enrichment analysis ─────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Per-CUI enrichment analysis (top 30 most-added concepts):", flush=True)
    print(f"{'='*60}", flush=True)

    cui_added_count = collections.Counter()
    for img_id in img_ids:
        original = set(concepts.get(img_id, []))
        final = set(enriched_concepts.get(img_id, []))
        for c in final - original:
            cui_added_count[c] += 1

    print(f"{'CUI':<12} {'Added':>6}  Canonical name", flush=True)
    print(f"{'-'*12} {'-'*6}  {'-'*40}", flush=True)
    for cui, count in cui_added_count.most_common(30):
        name = cui_mapping.get(cui, "?")[:40]
        print(f"{cui:<12} {count:>6}  {name}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    print("Done.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CUI → Caption Matcher: enrich ROCOv2 labels using captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enrich training labels (model-gated):
  python cui_caption_matcher.py --mode train

  # Enrich validation labels (caption-only):
  python cui_caption_matcher.py --mode valid

  # Smoke test on 200 images:
  python cui_caption_matcher.py --mode train --limit 200

  # Custom batch size for GPU scoring:
  python cui_caption_matcher.py --mode train --batch-size 128
""",
    )
    parser.add_argument(
        "--mode", choices=["train", "valid", "test"], default="train",
        help="Which split to enrich. 'train' uses model veto; "
             "'valid'/'test' use caption-only matching. (default: train)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only this many images (0 = all). For smoke testing.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for model forward passes (train mode). (default: 64)",
    )
    parser.add_argument(
        "--no-veto", action="store_true",
        help="Train mode: skip the CNN veto, use caption-only matching (same "
             "rule as valid/test). Writes to *_noveto.csv so the vetoed run "
             "is preserved. Needs no model and no image loading -> ~5 min.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Don't cache/load the alias dictionary to/from disk.",
    )
    args = parser.parse_args()

    run_enrichment(
        mode=args.mode,
        limit=args.limit,
        batch_size=args.batch_size,
        save_alias_cache=not args.no_cache,
        no_veto=args.no_veto,
    )


if __name__ == "__main__":
    main()
