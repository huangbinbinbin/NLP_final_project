#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RNN inference (Seq2Seq + Attention) aligned to your Transformer inference interface.

Goals:
1) HF mode: no vocab files needed; load AutoTokenizer from --hf_name_or_dir and ids from --tokenizer_meta.
2) Decode switch: --decode greedy|beam.
3) Output/metrics similar to transformer inference:
   - writes --out_path predictions (one line per sample)
   - prints [MODE]/[BLEU]/[LATENCY]/[MODEL]
   - optional --save_details jsonl (per-sample diagnostics)

Supports two modes:
A) HF-ids mode (recommended):
   Provide --tokenizer_meta and --hf_name_or_dir (repo id or local dir), optionally --local_files_only and --hf_cache_dir.
B) SPM/vocab mode (legacy):
   Provide --vocab_tgt (and optional --spm_tgt_model) to decode ids to text.

Expected data_jsonl rows contain:
  - zh_ids (list[int]) and en_ids (list[int])
  - optional raw zh/en strings, optional index
"""

import os
import json
import time
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# IO helpers
# -------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def load_tokenizer_meta(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    need = ["pad_token_id", "unk_token_id", "eos_token_id",
            "src_vocab_size", "tgt_vocab_size", "decoder_start_token_id"]
    for k in need:
        if k not in meta:
            raise ValueError(f"tokenizer_meta missing key: {k}")
    # bos_token_id may be null / missing
    if "bos_token_id" not in meta:
        meta["bos_token_id"] = None

    for k in need:
        meta[k] = int(meta[k])
    if meta["bos_token_id"] is not None:
        meta["bos_token_id"] = int(meta["bos_token_id"])
    return meta


# -------------------------
# Legacy vocab (optional)
# -------------------------
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def read_vocab_tokens(vocab_path: str) -> List[str]:
    itos = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n").split("\t")[0]
            itos.append(tok)
    if itos[:4] != SPECIAL_TOKENS:
        raise ValueError(f"Vocab special token order mismatch: {itos[:4]} != {SPECIAL_TOKENS}")
    return itos


def try_load_spm(model_path: str):
    if not model_path:
        return None
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return sp
    except Exception:
        return None


def ids_to_text_spm(ids: List[int], itos: List[str], sp=None,
                    pad_id: int = 0, bos_id: int = 2, eos_id: int = 3,
                    start_id: Optional[int] = None) -> str:
    """
    Legacy decode: id -> piece string -> detok.
    - start_id: if provided, skip it (useful if you switch start token).
    """
    pieces = []
    for j, i in enumerate(ids):
        i = int(i)
        if start_id is not None and i == int(start_id):
            continue
        if i == int(pad_id) or i == int(bos_id):
            continue
        if i == int(eos_id):
            break
        if 0 <= i < len(itos):
            pieces.append(itos[i])
        else:
            pieces.append("<unk>")
    if sp is not None:
        return sp.decode_pieces(pieces).strip()
    return "".join(pieces).replace("▁", " ").strip()


# -------------------------
# BLEU (same behavior as your transformer script)
# -------------------------
def _zh_to_char_spaced(s: str) -> str:
    s = s.strip().replace(" ", "")
    return " ".join(list(s))


def compute_bleu_sacrebleu(hyps: List[str], refs: List[str], tgt_lang: str) -> float:
    import sacrebleu
    if tgt_lang == "zh":
        hyps2 = [_zh_to_char_spaced(h) for h in hyps]
        refs2 = [_zh_to_char_spaced(r) for r in refs]
        bleu = sacrebleu.corpus_bleu(hyps2, [refs2], tokenize="none")
    else:
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a")
    return float(bleu.score)


def sentence_bleu_sacrebleu(hyp: str, ref: str) -> Optional[float]:
    try:
        import sacrebleu
        if hasattr(sacrebleu, "sentence_bleu"):
            s = sacrebleu.sentence_bleu(hyp, [ref])
            return float(s.score)
        return None
    except Exception:
        return None


def infer_tgt_lang(direction: str) -> str:
    return "en" if direction == "zh2en" else "zh"


# -------------------------
# Model (must match train)
# -------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size: int, alignment: str):
        super().__init__()
        self.alignment = alignment
        if alignment == "multiplicative":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif alignment == "additive":
            self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        elif alignment == "dot":
            pass
        else:
            raise ValueError("alignment must be dot|multiplicative|additive")

    def forward(self, dec_h, enc_out, src_mask, cov_prev=None, cov_beta: float = 0.0):
        if self.alignment == "dot":
            scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)  # [B,S]
        elif self.alignment == "multiplicative":
            q = self.W(dec_h)
            scores = torch.bmm(enc_out, q.unsqueeze(2)).squeeze(2)
        else:
            s_proj = self.W_s(enc_out)
            h_proj = self.W_h(dec_h).unsqueeze(1)
            energy = torch.tanh(s_proj + h_proj)
            scores = self.v(energy).squeeze(2)

        if cov_prev is not None and cov_beta > 0:
            scores = scores - cov_beta * cov_prev

        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn_w = torch.softmax(scores, dim=1)  # [B,S]
        ctx = torch.bmm(attn_w.unsqueeze(1), enc_out).squeeze(1)  # [B,H]
        return ctx, attn_w


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, rnn_type, dropout, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(int(vocab_size), int(emb_size), padding_idx=int(pad_id))
        self.ln_emb = nn.LayerNorm(int(emb_size))
        self.drop_emb = nn.Dropout(float(dropout))

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            int(emb_size), int(hidden_size), num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=False, batch_first=True
        )

        self.ln_rnn = nn.LayerNorm(int(hidden_size))
        self.drop_rnn = nn.Dropout(float(dropout))

    def forward(self, src):
        x = self.emb(src)
        x = self.drop_emb(self.ln_emb(x))
        out, h = self.rnn(x)
        out = self.drop_rnn(self.ln_rnn(out))
        return out, h


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, rnn_type, dropout, alignment, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(int(vocab_size), int(emb_size), padding_idx=int(pad_id))
        self.ln_emb = nn.LayerNorm(int(emb_size))
        self.drop_emb = nn.Dropout(float(dropout))

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            int(emb_size) + int(hidden_size), int(hidden_size), num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=False, batch_first=True
        )

        self.ln_rnn = nn.LayerNorm(int(hidden_size))
        self.drop_rnn = nn.Dropout(float(dropout))

        self.attn = Attention(int(hidden_size), alignment)

        self.out = nn.Sequential(
            nn.Linear(int(hidden_size) * 2, int(hidden_size)),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(vocab_size)),
        )

    def _top_hidden(self, h):
        if self.rnn_type == "lstm":
            h0, _c0 = h
            return h0[-1]
        return h[-1]

    def forward_step(self, y_prev, h, enc_out, src_mask, ctx_prev, cov_prev, cov_beta: float):
        emb = self.emb(y_prev)
        emb = self.drop_emb(self.ln_emb(emb))

        rnn_in = torch.cat([emb, ctx_prev], dim=-1)
        out, h2 = self.rnn(rnn_in.unsqueeze(1), h)
        out = out.squeeze(1)
        out = self.drop_rnn(self.ln_rnn(out))

        top_h = self._top_hidden(h2)
        ctx, attn_w = self.attn(top_h, enc_out, src_mask, cov_prev=cov_prev, cov_beta=cov_beta)
        cov2 = cov_prev + attn_w

        logits = self.out(torch.cat([out, ctx], dim=-1))
        return logits, h2, ctx, cov2


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_size, hidden_size, num_layers, rnn_type, dropout, alignment,
                 pad_id_src: int, pad_id_tgt: int):
        super().__init__()
        self.pad_id_src = int(pad_id_src)
        self.pad_id_tgt = int(pad_id_tgt)
        self.encoder = Encoder(src_vocab, emb_size, hidden_size, num_layers, rnn_type, dropout, pad_id=self.pad_id_src)
        self.decoder = Decoder(tgt_vocab, emb_size, hidden_size, num_layers, rnn_type, dropout, alignment, pad_id=self.pad_id_tgt)


# -------------------------
# HF decode helper (align to your transformer strip_and_decode_hf behavior)
# -------------------------
def strip_and_decode_hf(ids: List[int], tok, eos_id: int, pad_id: int, start_id: Optional[int] = None) -> str:
    out = []
    for j, i in enumerate(ids):
        i = int(i)

        # skip decoder_start if provided and appears as first token
        if start_id is not None and j == 0 and i == int(start_id):
            continue

        if i == int(pad_id):
            continue

        # for Marian, decoder_start may equal eos; only treat as terminator if not first position
        if i == int(eos_id):
            if j == 0 and start_id is not None and int(start_id) == int(eos_id):
                continue
            break

        out.append(i)

    return tok.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


# -------------------------
# Length normalization (same as transformer)
# -------------------------
def length_norm(score: float, length: int, alpha: float, mode: str) -> float:
    mode = (mode or "gnmt").lower()
    if mode == "none" or alpha <= 0:
        return score
    if mode == "gnmt":
        lp = ((5.0 + length) / 6.0) ** alpha
        return score / lp
    if mode == "length":
        lp = (max(1, length) ** alpha)
        return score / lp
    raise ValueError("len_norm must be gnmt|length|none")


# -------------------------
# Decoding (greedy / beam) with HF ids
# -------------------------
@torch.no_grad()
def greedy_decode_batch(model: Seq2Seq, src: torch.Tensor, src_mask: torch.Tensor,
                        max_len: int, min_len: int,
                        start_id: int, eos_id: int,
                        cov_beta: float,
                        ban_ids: Optional[List[int]] = None) -> Tuple[List[List[int]], List[int]]:
    """
    Returns:
      - hyps: list of generated token ids (EXCLUDING start token)
      - gen_lens: generated lengths INCLUDING eos if produced, excluding start (so aligns to hyps length)
    """
    model.eval()
    enc_out, enc_h = model.encoder(src)
    dec_h = enc_h

    B = src.size(0)
    S = enc_out.size(1)
    H = enc_out.size(2)

    ctx_prev = torch.zeros(B, H, device=src.device)
    cov_prev = torch.zeros(B, S, device=src.device)

    y_prev = torch.full((B,), int(start_id), dtype=torch.long, device=src.device)

    finished = torch.zeros(B, dtype=torch.bool, device=src.device)
    hyps: List[List[int]] = [[] for _ in range(B)]

    for step in range(int(max_len)):
        logits, dec_h, ctx_prev, cov_prev = model.decoder.forward_step(
            y_prev, dec_h, enc_out, src_mask, ctx_prev, cov_prev, float(cov_beta)
        )

        if ban_ids:
            logits[:, ban_ids] = float("-inf")
        if int(min_len) > 0 and step < int(min_len):
            logits[:, int(eos_id)] = float("-inf")

        y_next = torch.argmax(logits, dim=-1)  # [B]
        y_prev = y_next

        for i in range(B):
            if finished[i]:
                continue
            tid = int(y_next[i].item())
            hyps[i].append(tid)
            if tid == int(eos_id) and (step + 1) >= int(min_len):
                finished[i] = True

        if bool(finished.all().item()):
            break

    gen_lens = [len(h) for h in hyps]
    return hyps, gen_lens


@torch.no_grad()
def beam_search_decode_one(model: Seq2Seq, src1: torch.Tensor, mask1: torch.Tensor,
                          max_len: int, beam_size: int,
                          alpha: float, len_norm_mode: str,
                          min_len: int,
                          start_id: int, eos_id: int,
                          cov_beta: float,
                          ban_ids: Optional[List[int]] = None) -> Tuple[List[int], int]:
    """
    Returns best sequence ids INCLUDING start token as first element.
    gen_len includes the full seq length (including start).
    """
    model.eval()
    enc_out, enc_h = model.encoder(src1)
    dec_h0 = enc_h

    S = enc_out.size(1)
    H = enc_out.size(2)
    device = src1.device

    ctx0 = torch.zeros(1, H, device=device)
    cov0 = torch.zeros(1, S, device=device)

    beams = [([int(start_id)], dec_h0, ctx0, cov0, 0.0, False)]

    for step in range(int(max_len)):
        new_beams = []
        for seq, dec_h, ctx_prev, cov_prev, logp, ended in beams:
            if ended:
                new_beams.append((seq, dec_h, ctx_prev, cov_prev, logp, True))
                continue

            y_prev = torch.tensor([seq[-1]], device=device, dtype=torch.long)
            logits, dec_h2, ctx2, cov2 = model.decoder.forward_step(
                y_prev, dec_h, enc_out, mask1, ctx_prev, cov_prev, float(cov_beta)
            )
            logits = logits.squeeze(0)

            if ban_ids:
                logits[ban_ids] = float("-inf")
            if int(min_len) > 0 and step < int(min_len):
                logits[int(eos_id)] = float("-inf")

            log_probs = torch.log_softmax(logits, dim=-1)
            topk_lp, topk_id = torch.topk(log_probs, k=int(beam_size))

            for lp, tid in zip(topk_lp.tolist(), topk_id.tolist()):
                tid = int(tid)
                ended2 = (tid == int(eos_id))
                new_beams.append((seq + [tid], dec_h2, ctx2, cov2, logp + float(lp), ended2))

        def score_fn(b):
            seq, _h, _c, _cov, lp, _ended = b
            # normalize excluding the initial start token
            L = max(1, len(seq) - 1)
            return length_norm(lp, L, float(alpha), str(len_norm_mode))

        new_beams.sort(key=score_fn, reverse=True)
        beams = new_beams[: int(beam_size)]
        if all(b[5] for b in beams):
            break

    best = max(beams, key=lambda b: length_norm(b[4], max(1, len(b[0]) - 1), float(alpha), str(len_norm_mode)))
    best_ids = best[0]
    return best_ids, len(best_ids)


@torch.no_grad()
def beam_decode(model: Seq2Seq, src: torch.Tensor, src_mask: torch.Tensor,
                max_len: int, beam_size: int, alpha: float, len_norm_mode: str,
                min_len: int,
                start_id: int, eos_id: int,
                cov_beta: float,
                ban_ids: Optional[List[int]] = None) -> Tuple[List[List[int]], List[int]]:
    hyps = []
    gen_lens = []
    for i in range(src.size(0)):
        best_ids, L = beam_search_decode_one(
            model, src[i:i+1], src_mask[i:i+1],
            max_len=max_len, beam_size=beam_size,
            alpha=alpha, len_norm_mode=len_norm_mode,
            min_len=min_len,
            start_id=start_id, eos_id=eos_id,
            cov_beta=cov_beta,
            ban_ids=ban_ids,
        )
        # drop the leading start token for detok/output token count
        hyps.append(best_ids[1:])
        gen_lens.append(max(0, L - 1))
    return hyps, gen_lens


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_jsonl", required=True)

    ap.add_argument("--direction", default="zh2en", choices=["zh2en", "en2zh"])

    # HF ids mode
    ap.add_argument("--tokenizer_meta", default="", help="(HF ids mode) tokenizer_meta.json")
    ap.add_argument("--hf_name_or_dir", default="", help="HF tokenizer name or LOCAL dir")
    ap.add_argument("--hf_cache_dir", default="", help="Optional HF cache_dir (ACP recommended)")
    ap.add_argument("--local_files_only", action="store_true")

    # legacy SPM mode
    ap.add_argument("--vocab_tgt", default="", help="(SPM mode) target vocab txt (optional)")
    ap.add_argument("--spm_tgt_model", default="", help="(SPM mode) optional sentencepiece model for detok")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--len_norm", choices=["gnmt", "length", "none"], default="gnmt")

    ap.add_argument("--infer_batch_size", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--min_len", type=int, default=1)

    ap.add_argument("--long_src_len", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0)

    ap.add_argument("--cov_beta", type=float, default=None,
                    help="coverage strength; if not set, reads from ckpt config/args (default 0.0)")

    ap.add_argument("--out_path", default="pred.txt")

    ap.add_argument("--compute_bleu", action="store_true")
    ap.add_argument("--tgt_lang", default="", choices=["", "en", "zh"])

    ap.add_argument("--save_details", type=str, default="", help="Save per-sample details to jsonl (like transformer).")
    ap.add_argument("--save_max", type=int, default=0, help="Max lines to save (0=all).")

    args = ap.parse_args()

    device = torch.device(args.device)

    # ---------- choose mode ----------
    use_hf = bool(args.tokenizer_meta)
    if use_hf:
        if not args.hf_name_or_dir:
            raise ValueError("HF-ids mode requires --hf_name_or_dir (repo id or local dir).")
        meta = load_tokenizer_meta(args.tokenizer_meta)
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(
            args.hf_name_or_dir,
            use_fast=False,
            local_files_only=bool(args.local_files_only),
            cache_dir=(args.hf_cache_dir if args.hf_cache_dir else None),
        )

        pad_id_s = int(meta["pad_token_id"])
        pad_id_t = int(meta["pad_token_id"])
        eos_id = int(meta["eos_token_id"])
        start_id_meta = int(meta["decoder_start_token_id"])
        src_vocab_size_meta = int(meta["src_vocab_size"])
        tgt_vocab_size_meta = int(meta["tgt_vocab_size"])
    else:
        meta = None
        hf_tok = None
        if not args.vocab_tgt:
            raise ValueError("Need --tokenizer_meta for HF-ids mode OR --vocab_tgt for SPM mode.")
        itos_tgt = read_vocab_tokens(args.vocab_tgt)
        sp_tgt = try_load_spm(args.spm_tgt_model)
        # legacy specials fixed by vocab order
        pad_id_s = 0
        pad_id_t = 0
        eos_id = 3
        start_id_meta = 2  # BOS
        src_vocab_size_meta = None
        tgt_vocab_size_meta = len(itos_tgt)

    # ---------- load ckpt ----------
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # new aligned ckpt: {"model": state_dict, "config": run_config, ...}
    cfg = ckpt.get("config", None)
    state = ckpt.get("model", None)

    # legacy: {"model_state": ..., "args": ..., "src_vocab": ..., "tgt_vocab": ...}
    if cfg is None and "args" in ckpt:
        cfg = ckpt["args"]
    if state is None and "model_state" in ckpt:
        state = ckpt["model_state"]

    if cfg is None or state is None:
        raise ValueError("Checkpoint missing expected keys. Need (model, config) or legacy (model_state, args).")

    # vocab sizes: prefer ckpt config; fallback to meta; fallback to legacy fields
    src_vocab = int(cfg.get("src_vocab_size", ckpt.get("src_vocab", src_vocab_size_meta)))
    tgt_vocab = int(cfg.get("tgt_vocab_size", ckpt.get("tgt_vocab", tgt_vocab_size_meta)))

    if src_vocab is None or tgt_vocab is None:
        raise ValueError("Cannot infer vocab sizes. Ensure ckpt has src_vocab_size/tgt_vocab_size or provide tokenizer_meta.")

    emb_size = int(cfg["emb_size"])
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg["num_layers"])
    rnn_type = str(cfg.get("rnn_type", "lstm"))
    dropout = float(cfg.get("dropout", 0.0))
    alignment = str(cfg.get("alignment", "dot"))

    pad_id_src = int(cfg.get("pad_id_src", pad_id_s))
    pad_id_tgt = int(cfg.get("pad_id_tgt", pad_id_t))

    # decoder start id: prefer ckpt config; else meta/legacy BOS
    start_id = int(cfg.get("decoder_start_id", start_id_meta))

    # safety: start_id should not equal PAD (padding_idx freeze / semantics mismatch)
    if int(start_id) == int(pad_id_tgt):
        raise ValueError(f"BAD CONFIG: decoder_start_id == pad_id_tgt == {start_id}. Fix tokenizer_meta/config.")

    cov_beta = float(args.cov_beta) if args.cov_beta is not None else float(cfg.get("cov_beta", 0.0))

    model = Seq2Seq(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        emb_size=emb_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        alignment=alignment,
        pad_id_src=pad_id_src,
        pad_id_tgt=pad_id_tgt,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # ban generating PAD; ban generating START if it's not EOS (avoid loops)
    ban = {int(pad_id_tgt)}
    if int(start_id) != int(eos_id):
        ban.add(int(start_id))
    ban_ids = sorted(ban)

    # ---------- data ----------
    rows = read_jsonl(args.data_jsonl)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: int(args.max_samples)]

    def get_pair(r: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        zh = r["zh_ids"]
        en = r["en_ids"]
        return (zh, en) if args.direction == "zh2en" else (en, zh)

    if args.long_src_len and args.long_src_len > 0:
        rows = [r for r in rows if len(get_pair(r)[0]) >= int(args.long_src_len)]

    # ---------- decoding loop ----------
    preds: List[str] = []
    refs: List[str] = []
    out_tokens = 0
    t_decode = 0.0

    details_f = None
    saved = 0

    def write_detail(obj: Dict[str, Any]):
        nonlocal saved
        if details_f is None:
            return
        if args.save_max and args.save_max > 0 and saved >= args.save_max:
            return
        details_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        saved += 1

    if args.save_details:
        os.makedirs(os.path.dirname(args.save_details) or ".", exist_ok=True)
        details_f = open(args.save_details, "w", encoding="utf-8")

    try:
        bs = max(1, int(args.infer_batch_size))
        for st in range(0, len(rows), bs):
            batch = rows[st:st + bs]
            srcs, tgts = zip(*[get_pair(r) for r in batch])

            smax = max(len(s) for s in srcs)
            src = torch.full((len(srcs), smax), int(pad_id_src), dtype=torch.long, device=device)
            for i, s in enumerate(srcs):
                src[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            src_mask = (src != int(pad_id_src))

            t0 = time.time()
            if args.decode == "greedy":
                hyp_ids_list, gen_lens = greedy_decode_batch(
                    model, src, src_mask,
                    max_len=int(args.max_len),
                    min_len=int(args.min_len),
                    start_id=int(start_id),
                    eos_id=int(eos_id),
                    cov_beta=float(cov_beta),
                    ban_ids=ban_ids,
                )
            else:
                hyp_ids_list, gen_lens = beam_decode(
                    model, src, src_mask,
                    max_len=int(args.max_len),
                    beam_size=int(args.beam_size),
                    alpha=float(args.alpha),
                    len_norm_mode=str(args.len_norm),
                    min_len=int(args.min_len),
                    start_id=int(start_id),
                    eos_id=int(eos_id),
                    cov_beta=float(cov_beta),
                    ban_ids=ban_ids,
                )
            t1 = time.time()
            t_decode += (t1 - t0)

            for r, hyp_ids, gl, tgt_ids in zip(batch, hyp_ids_list, gen_lens, tgts):
                # detok pred
                if use_hf:
                    hyp_text = strip_and_decode_hf(hyp_ids, hf_tok, eos_id=eos_id, pad_id=pad_id_tgt, start_id=None)
                else:
                    hyp_text = ids_to_text_spm(hyp_ids, itos_tgt, sp=sp_tgt,
                                               pad_id=pad_id_tgt, bos_id=2, eos_id=eos_id, start_id=None)

                preds.append(hyp_text)

                # detok ref (optional)
                ref_text = ""
                if args.compute_bleu:
                    # prefer raw text if exists
                    if args.direction == "zh2en":
                        raw = r.get("en", "")
                    else:
                        raw = r.get("zh", "")
                    if isinstance(raw, str) and raw.strip():
                        ref_text = raw.strip()
                    else:
                        if use_hf:
                            ref_text = strip_and_decode_hf(tgt_ids, hf_tok, eos_id=eos_id, pad_id=pad_id_tgt, start_id=None)
                        else:
                            ref_text = ids_to_text_spm(tgt_ids, itos_tgt, sp=sp_tgt,
                                                       pad_id=pad_id_tgt, bos_id=2, eos_id=eos_id, start_id=None)
                    refs.append(ref_text)

                out_tokens += int(gl)

                # details
                if details_f is not None:
                    if args.direction == "zh2en":
                        src_ids = r.get("zh_ids", None)
                        raw_src = r.get("zh", None)
                    else:
                        src_ids = r.get("en_ids", None)
                        raw_src = r.get("en", None)

                    if isinstance(raw_src, str) and raw_src.strip():
                        src_text = raw_src.strip()
                    else:
                        if use_hf and isinstance(src_ids, list):
                            src_text = strip_and_decode_hf(src_ids, hf_tok, eos_id=eos_id, pad_id=pad_id_src, start_id=None)
                        else:
                            src_text = None

                    sent_bleu = sentence_bleu_sacrebleu(hyp_text, ref_text) if (args.compute_bleu and ref_text) else None

                    write_detail({
                        "mode": "scratch",
                        "index": r.get("index", None),
                        "src_len_tokens": len(src_ids) if isinstance(src_ids, list) else None,
                        "gen_len_tokens": int(gl),
                        "src": src_text,
                        "ref": ref_text if args.compute_bleu else None,
                        "hyp": hyp_text,
                        "sentence_bleu": sent_bleu,
                        "decode": args.decode,
                        "beam_size": int(args.beam_size) if args.decode == "beam" else 1,
                        "len_norm": str(args.len_norm) if args.decode == "beam" else "n/a",
                        "alpha": float(args.alpha) if args.decode == "beam" else 0.0,
                        "min_len": int(args.min_len),
                    })

        # ---------- write outputs ----------
        with open(args.out_path, "w", encoding="utf-8") as f:
            for s in preds:
                f.write(s + "\n")

        avg_ms = (t_decode / max(1, len(rows))) * 1000.0
        tok_per_sec = float(out_tokens / max(1e-9, t_decode))

        print("[MODE] scratch")
        if args.compute_bleu:
            tgt_lang = args.tgt_lang if args.tgt_lang else infer_tgt_lang(args.direction)
            try:
                bleu = compute_bleu_sacrebleu(preds, refs, tgt_lang=tgt_lang)
            except Exception as e:
                bleu = -1.0
                print("[WARN] BLEU failed:", repr(e))
            print(f"[BLEU] {bleu:.2f}")
        else:
            print("[BLEU] (skipped)")

        print(f"[LATENCY] avg_ms_per_sent={avg_ms:.2f} tok_per_sec={tok_per_sec:.1f}")
        print(f"[MODEL] param_count={count_parameters(model)}")
        print(f"[OK] wrote: {args.out_path}")

    finally:
        if details_f is not None:
            details_f.close()
            print(f"[DETAILS] saved to: {args.save_details}")


if __name__ == "__main__":
    main()
