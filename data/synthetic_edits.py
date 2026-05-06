"""Synthetic edit pair generation from unlabelled text corpora.

Applies lightweight rule-based and model-based perturbations to produce
(source, target, instruction) triples suitable for self-supervised training.

Usage:
    python data/synthetic_edits.py \
        --input /data/c4/train.txt \
        --output /data/c4_edit/train.jsonl \
        --n_samples 1000000
"""
import argparse
import json
import random
import re
import string
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Word-level utilities
# ---------------------------------------------------------------------------

FORMAL_TO_CASUAL = {
    "cannot": "can't", "will not": "won't", "do not": "don't",
    "does not": "doesn't", "is not": "isn't", "are not": "aren't",
    "was not": "wasn't", "were not": "weren't", "I am": "I'm",
    "you are": "you're", "he is": "he's", "she is": "she's",
    "it is": "it's", "we are": "we're", "they are": "they're",
    "I will": "I'll", "you will": "you'll", "we will": "we'll",
    "would not": "wouldn't", "could not": "couldn't", "should not": "shouldn't",
}
CASUAL_TO_FORMAL = {v: k for k, v in FORMAL_TO_CASUAL.items()}

# Small synonym table for word_substitution without requiring WordNet
SYNONYMS = {
    "said": ["stated", "remarked", "noted", "mentioned"],
    "big": ["large", "substantial", "considerable", "sizable"],
    "small": ["little", "tiny", "modest", "compact"],
    "good": ["excellent", "fine", "solid", "strong"],
    "bad": ["poor", "weak", "subpar", "inadequate"],
    "happy": ["pleased", "content", "glad", "satisfied"],
    "sad": ["unhappy", "somber", "downcast", "dejected"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "slow": ["gradual", "unhurried", "measured", "leisurely"],
    "important": ["significant", "crucial", "key", "critical"],
    "show": ["demonstrate", "reveal", "illustrate", "exhibit"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "get": ["obtain", "acquire", "receive", "gain"],
    "make": ["create", "produce", "build", "generate"],
    "think": ["believe", "consider", "feel", "suppose"],
}


def _split_sentences(text: str):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def _rejoin(sentences):
    return ' '.join(sentences)


# ---------------------------------------------------------------------------
# Edit type implementations
# ---------------------------------------------------------------------------

def word_substitution(text: str, rate: float = 0.15) -> Optional[str]:
    words = text.split()
    changed = False
    out = []
    for w in words:
        lower = w.lower().rstrip(string.punctuation)
        if lower in SYNONYMS and random.random() < rate:
            syn = random.choice(SYNONYMS[lower])
            # Preserve trailing punctuation
            punct = w[len(lower):] if w.lower().startswith(lower) else ''
            out.append(syn + punct)
            changed = True
        else:
            out.append(w)
    return ' '.join(out) if changed else None


def sentence_reorder(text: str) -> Optional[str]:
    sents = _split_sentences(text)
    if len(sents) < 2:
        return None
    # Swap one random adjacent pair
    i = random.randint(0, len(sents) - 2)
    sents[i], sents[i + 1] = sents[i + 1], sents[i]
    return _rejoin(sents)


def insertion(text: str) -> Optional[str]:
    sents = _split_sentences(text)
    if len(sents) < 2:
        return None
    # Insert a copy of a random sentence at another position
    src_idx = random.randint(0, len(sents) - 1)
    insert_idx = random.randint(0, len(sents))
    new_sent = sents[src_idx]
    sents.insert(insert_idx, new_sent)
    return _rejoin(sents)


def deletion(text: str) -> Optional[str]:
    sents = _split_sentences(text)
    if len(sents) < 2:
        return None
    del_idx = random.randint(0, len(sents) - 1)
    sents.pop(del_idx)
    return _rejoin(sents)


def formality_shift(text: str) -> Optional[str]:
    lower = text.lower()
    # Decide direction based on whether formal contractions are present
    if any(k in lower for k in CASUAL_TO_FORMAL):
        table = CASUAL_TO_FORMAL
        direction = "formal"
    else:
        table = FORMAL_TO_CASUAL
        direction = "casual"
    result = text
    changed = False
    for src, dst in table.items():
        if src in result:
            result = result.replace(src, dst, 1)
            changed = True
            break
    return result if changed else None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EDIT_TYPES = {
    "word_substitution": word_substitution,
    "sentence_reorder": sentence_reorder,
    "insertion": insertion,
    "deletion": deletion,
    "formality_shift": formality_shift,
}


def generate_edit_pair(text: str):
    """Try each edit type in random order; return first that produces a change."""
    types = list(EDIT_TYPES.keys())
    random.shuffle(types)
    for edit_type in types:
        result = EDIT_TYPES[edit_type](text)
        if result is not None and result != text:
            return {
                "source": text,
                "target": result,
                "instruction": edit_type,
            }
    return None


def generate_from_file(input_path: str, output_path: str,
                       n_samples: int = 1_000_000,
                       min_words: int = 10):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            if written >= n_samples:
                break
            text = line.strip()
            if len(text.split()) < min_words:
                continue
            pair = generate_edit_pair(text)
            if pair is not None:
                fout.write(json.dumps(pair) + '\n')
                written += 1
    print(f"Wrote {written} edit pairs to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n_samples', type=int, default=1_000_000)
    parser.add_argument('--min_words', type=int, default=10)
    args = parser.parse_args()
    generate_from_file(args.input, args.output, args.n_samples, args.min_words)
