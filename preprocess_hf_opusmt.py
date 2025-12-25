#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess zh-en JSONL using HuggingFace OPUS-MT tokenizer (Plan A):
- Use the tokenizer's native IDs:
    pad=tok.pad_token_id (e.g., 65000)
    unk=tok.unk_token_id (e.g., 1)
    eos=tok.eos_token_id (e.g., 0)
    bos is typically None for Marian/OPUS-MT; decoder start uses eos in training.

Pipeline:
1) Read jsonl (expects keys: "zh", "en", optional "index")
2) Clean text (Unicode NFKC, remove control chars, normalize punctuation, collapse whitespace; optional lowercasing for English)
3) Char-level length policy BEFORE tokenization: filter|truncate
4) Tokenize with HF AutoTokenizer:
   - source (zh): tok(zh, add_special_tokens=False)
   - target (en): tok(text_target=en, add_special_tokens=False)
5) Token-level length policy AFTER tokenization: filter|truncate (applied to content token ids, excluding eos)
6) Append eos_id to both zh_ids/en_ids
7) Write processed jsonl with fields:
   {"index": ..., "zh_tokens": [...], "en_tokens": [...], "zh_ids": [..., eos], "en_ids": [..., eos]}

Outputs:
- <output_dir>/<out_name>
- <output_dir>/preprocess_config.json
- <output_dir>/tokenizer_meta.json

Example:
python preprocess_hf_opusmt.py \
  --input /path/train_100k.jsonl \
  --output_dir /path/out_hf_opusmt \
  --out_name processed_train.jsonl \
  --hf_checkpoint Helsinki-NLP/opus-mt-zh-en \
  --local_files_only \
  --cache_dir /data/250010105/hf_cache \
  --max_char_len_zh 300 --max_char_len_en 500 --char_len_policy filter \
  --max_len_zh 80 --max_len_en 80 --len_policy filter
"""

import argparse
import json
import os
import re
import unicodedata
from typing import Iterable, Optional, Dict, Any
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


# -----------------------------
# JSONL IO
# -----------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_line(f, obj: Dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------
# Cleaning utilities
# -----------------------------
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_REPLACEMENT = re.compile(r"\uFFFD+")  # '�'


def _remove_control_chars_keep_whitespace(text: str) -> str:
    """Remove Unicode control chars but keep \\t \\n \\r as whitespace."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            if ch in ("\t", "\n", "\r"):
                out.append(ch)
            else:
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def clean_text(text: str, lang: str, lower_en: bool = True) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _remove_control_chars_keep_whitespace(text)
    text = _RE_REPLACEMENT.sub("", text)
    text = (text.replace("“", '"').replace("”", '"')
                .replace("’", "'")
                .replace("–", "-").replace("—", "-"))
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    if lang == "en" and lower_en:
        text = text.lower()
    return text


# -----------------------------
# Char-level length filter/truncate
# -----------------------------
def char_length_filter(text: str, max_len: int, policy: str) -> Optional[str]:
    """
    max_len<=0 => disabled
    policy: filter|truncate
    truncate: try to cut at word boundary if there is a space (mainly English)
    """
    if max_len <= 0:
        return text

    if policy == "filter":
        return text if len(text) <= max_len else None

    if policy == "truncate":
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        last_space = truncated.rfind(" ")
        if last_space != -1 and last_space > int(max_len * 0.5):
            truncated = truncated[:last_space].strip()
        else:
            truncated = truncated.strip()
        return truncated if truncated else None

    raise ValueError("char_len_policy must be filter|truncate")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Preprocess zh-en jsonl with OPUS-MT tokenizer (Plan A)")

    ap.add_argument("--input", required=True, help="Path to input *.jsonl (expects keys: zh, en, optional index)")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--out_name", default="processed.jsonl", help="Output jsonl file name")

    ap.add_argument("--hf_checkpoint", default="Helsinki-NLP/opus-mt-zh-en", help="HF checkpoint name or local path")
    ap.add_argument("--cache_dir", default="", help="HF cache dir (optional)")
    ap.add_argument("--local_files_only", action="store_true", help="Only use local HF files (offline)")

    # English lowercasing (default True to match your notebook)
    ap.add_argument("--lower_en", dest="lower_en", action="store_true", help="Lowercase English (default: true)")
    ap.add_argument("--no_lower_en", dest="lower_en", action="store_false", help="Do not lowercase English")
    ap.set_defaults(lower_en=True)

    # Char-level length (BEFORE tokenization)
    ap.add_argument("--max_char_len_zh", type=int, default=300, help="Char-level max length for zh BEFORE tokenization; 0 disables.")
    ap.add_argument("--max_char_len_en", type=int, default=500, help="Char-level max length for en BEFORE tokenization; 0 disables.")
    ap.add_argument("--char_len_policy", default="filter", choices=["filter", "truncate"],
                    help="Char-level policy BEFORE tokenization: filter|truncate")

    # Token-level length (AFTER tokenization; applied to content ids, excluding eos)
    ap.add_argument("--max_len_zh", type=int, default=80, help="Token-level max length for zh AFTER tokenization; 0 disables.")
    ap.add_argument("--max_len_en", type=int, default=80, help="Token-level max length for en AFTER tokenization; 0 disables.")
    ap.add_argument("--len_policy", default="filter", choices=["filter", "truncate"],
                    help="Token-level policy AFTER tokenization: filter|truncate")

    # Output size toggle
    ap.add_argument("--write_tokens", action="store_true",
                    help="Also write zh_tokens/en_tokens arrays (default: true). If omitted, still true for compatibility.")
    ap.add_argument("--no_write_tokens", dest="write_tokens", action="store_false",
                    help="Do not write zh_tokens/en_tokens (smaller files)")
    ap.set_defaults(write_tokens=True)

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("This script requires transformers. Install: pip install transformers") from e

    tok = AutoTokenizer.from_pretrained(
        args.hf_checkpoint,
        cache_dir=(args.cache_dir or None),
        local_files_only=args.local_files_only,
        use_fast=False,
    )

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    unk_id = tok.unk_token_id
    bos_id = getattr(tok, "bos_token_id", None)

    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id; Plan A requires eos_token_id.")

    # Vocab sizes (best-effort)
    try:
        src_vocab_size = len(tok.get_vocab())
    except Exception:
        src_vocab_size = getattr(tok, "vocab_size", None)

    tgt_vocab_size = None
    try:
        with tok.as_target_tokenizer():
            tgt_vocab_size = len(tok.get_vocab())
    except Exception:
        # fallback: may be same as src or unavailable
        tgt_vocab_size = src_vocab_size

    # Save config + tokenizer meta for training reproducibility
    cfg_path = os.path.join(args.output_dir, "preprocess_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    meta = {
        "hf_checkpoint": args.hf_checkpoint,
        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "eos_token_id": eos_id,
        "bos_token_id": bos_id,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "notes": "Plan A: native tokenizer ids; no BOS; decoder start uses eos_token_id in training.",
    }
    meta_path = os.path.join(args.output_dir, "tokenizer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Meta] hf_checkpoint={args.hf_checkpoint}")
    print(f"[Meta] pad={pad_id} unk={unk_id} eos={eos_id} bos={bos_id}")
    print(f"[Meta] src_vocab_size={src_vocab_size} tgt_vocab_size={tgt_vocab_size}")
    print(f"[Config] saved: {cfg_path}")
    print(f"[Config] saved: {meta_path}")

    def apply_len(ids, max_len: int) -> Optional[list]:
        if max_len and max_len > 0:
            if args.len_policy == "truncate":
                return ids[:max_len]
            elif args.len_policy == "filter":
                return None if len(ids) > max_len else ids
        return ids

    in_count = 0
    dropped_char = 0
    dropped_tok = 0
    kept = 0

    out_path = os.path.join(args.output_dir, args.out_name)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for ex in tqdm(read_jsonl(args.input), desc="Clean+CharLen+HF"):
            in_count += 1
            en = clean_text(ex.get("en", ""), lang="en", lower_en=args.lower_en)
            zh = clean_text(ex.get("zh", ""), lang="zh", lower_en=False)

            zh2 = char_length_filter(zh, args.max_char_len_zh, args.char_len_policy)
            en2 = char_length_filter(en, args.max_char_len_en, args.char_len_policy)
            if zh2 is None or en2 is None:
                dropped_char += 1
                continue

            # Tokenize: source(zh) / target(en)
            zh_enc = tok(zh2, add_special_tokens=False, padding=False, truncation=False)
            en_enc = tok(text_target=en2, add_special_tokens=False, padding=False, truncation=False)

            zh_ids_content = apply_len(zh_enc["input_ids"], args.max_len_zh)
            en_ids_content = apply_len(en_enc["input_ids"], args.max_len_en)
            if not zh_ids_content or not en_ids_content:
                dropped_tok += 1
                continue

            if zh_ids_content is None or en_ids_content is None:
                dropped_tok += 1
                continue

            # Convert ids -> tokens (optional)
            obj = {
                "index": ex.get("index", None),
                "zh_ids": zh_ids_content + [eos_id],
                "en_ids": en_ids_content + [eos_id],
            }

            if args.write_tokens:
                obj["zh_tokens"] = tok.convert_ids_to_tokens(zh_ids_content)
                try:
                    with tok.as_target_tokenizer():
                        obj["en_tokens"] = tok.convert_ids_to_tokens(en_ids_content)
                except Exception:
                    obj["en_tokens"] = tok.convert_ids_to_tokens(en_ids_content)

            write_jsonl_line(out_f, obj)
            kept += 1

    print(f"[Done] input={in_count} kept={kept} dropped_char={dropped_char} dropped_tok={dropped_tok}")
    print(f"[Save] processed -> {out_path}")


if __name__ == "__main__":
    main()
