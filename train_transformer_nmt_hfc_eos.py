#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import argparse
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


# -------------------------
# Common utils
# -------------------------



def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_vocab_from_txt(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing vocab file: {path}")
    itos: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tok = line.split("\t", 1)[0]
            itos.append(tok)
    if len(itos) < 4 or itos[:4] != SPECIAL_TOKENS:
        raise ValueError(f"Vocab file {path} must start with: {SPECIAL_TOKENS}, got: {itos[:4]}")
    stoi = {t: i for i, t in enumerate(itos)}
    return {
        "itos": itos,
        "stoi": stoi,
        "pad_id": stoi["<pad>"],
        "unk_id": stoi["<unk>"],
        "bos_id": stoi["<bos>"],
        "eos_id": stoi["<eos>"],
    }

def load_tokenizer_meta(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    need = ["pad_token_id", "unk_token_id", "eos_token_id", "src_vocab_size", "tgt_vocab_size", "decoder_start_token_id"]
    for k in need:
        if k not in meta:
            raise ValueError(f"tokenizer_meta missing key: {k}")
    # 强制成 int，避免 json 里出现字符串
    for k in need:
        meta[k] = int(meta[k])
    return meta




def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_peak_gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def save_checkpoint(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


# -------------------------
# Scratch datasets
# -------------------------
class NMTProcessedIdsDataset(Dataset):
    """
    processed_*.jsonl: each row has zh_ids, en_ids
    """
    def __init__(self, jsonl_path: str, max_samples: int = 0):
        rows = read_jsonl(jsonl_path)
        if max_samples and max_samples > 0:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return torch.tensor(r["zh_ids"], dtype=torch.long), torch.tensor(r["en_ids"], dtype=torch.long)


def collate_ids(batch, pad_id_src: int, pad_id_tgt: int):
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(x) for x in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgts], dtype=torch.long)
    max_src = int(src_lens.max().item())
    max_tgt = int(tgt_lens.max().item())

    src_pad = torch.full((len(batch), max_src), pad_id_src, dtype=torch.long)
    tgt_pad = torch.full((len(batch), max_tgt), pad_id_tgt, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : s.numel()] = s
        tgt_pad[i, : t.numel()] = t

    return src_pad, tgt_pad, src_lens, tgt_lens


# -------------------------
# Norms
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def make_norm(norm_type: str, dim: int, eps: float = 1e-5) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    raise ValueError("norm_type must be layernorm|rmsnorm")


# -------------------------
# Positional encodings / T5-relative bias (fixed)
# -------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


def t5_relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """
    Faithful to T5 official logic.
    relative_position = memory_position - query_position
    - If bidirectional: half buckets for negative, half for positive.
    - Else (causal): only allow memory <= query; positive distances treated as 0.
    """
    ret = 0
    n = -relative_position  # convert so "past" is positive distances in causal mode

    if bidirectional:
        half = num_buckets // 2
        ret = ret + (n < 0).to(torch.long) * half
        n = n.abs()
        num_buckets = half
    else:
        n = torch.clamp(n, min=0)

    max_exact = num_buckets // 2
    is_small = n < max_exact

    n_float = n.to(torch.float)
    val_if_large = max_exact + (
        torch.log(n_float / max_exact + 1e-6) /
        math.log(max_distance / max_exact)
    ) * (num_buckets - max_exact)
    val_if_large = val_if_large.to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    bucket = torch.where(is_small, n.to(torch.long), val_if_large)
    return ret + bucket


class T5RelativePositionBias(nn.Module):
    """
    Returns bias: [1, H, Q, K]
    """
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def forward(self, qlen: int, klen: int, device: torch.device, bidirectional: bool) -> torch.Tensor:
        q_pos = torch.arange(qlen, device=device)[:, None]
        k_pos = torch.arange(klen, device=device)[None, :]
        rel = k_pos - q_pos  # memory - query
        buckets = t5_relative_position_bucket(
            rel, bidirectional=bidirectional, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        bias = self.relative_attention_bias(buckets)      # [Q,K,H]
        return bias.permute(2, 0, 1).unsqueeze(0)         # [1,H,Q,K]


# -------------------------
# Transformer blocks (scratch)
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor], rel_bias: Optional[torch.Tensor]):
        B, Q, D = q.size()
        _, K, _ = k.size()
        q = self.q_proj(q).view(B, Q, self.h, self.dh).transpose(1, 2)  # [B,H,Q,dh]
        k = self.k_proj(k).view(B, K, self.h, self.dh).transpose(1, 2)  # [B,H,K,dh]
        v = self.v_proj(v).view(B, K, self.h, self.dh).transpose(1, 2)  # [B,H,K,dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B,H,Q,K]
        if rel_bias is not None:
            scores = scores + rel_bias  # broadcast over B

        if attn_mask is not None:
            # scores = scores.masked_fill(attn_mask == 0, -1e9)
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # [B,H,Q,dh]
        out = out.transpose(1, 2).contiguous().view(B, Q, D)
        out = self.o_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, norm_type, use_rel, rel_bias_mod: Optional[T5RelativePositionBias]):
        super().__init__()
        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)
        self.use_rel = use_rel
        self.rel_bias_mod = rel_bias_mod

    def forward(self, x, pad_mask):
        Q = x.size(1)
        rel = self.rel_bias_mod(Q, Q, x.device, bidirectional=True) if (self.use_rel and self.rel_bias_mod is not None) else None
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, pad_mask, rel))
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, norm_type, use_rel, rel_bias_mod: Optional[T5RelativePositionBias]):
        super().__init__()
        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)
        self.norm3 = make_norm(norm_type, d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)
        self.use_rel = use_rel
        self.rel_bias_mod = rel_bias_mod

    def forward(self, y, enc, self_mask, enc_mask):
        T = y.size(1)
        rel = self.rel_bias_mod(T, T, y.device, bidirectional=False) if (self.use_rel and self.rel_bias_mod is not None) else None
        h = self.norm1(y)
        y = y + self.drop(self.self_attn(h, h, h, self_mask, rel))
        h = self.norm2(y)
        y = y + self.drop(self.cross_attn(h, enc, enc, enc_mask, None))
        h = self.norm3(y)
        y = y + self.drop(self.ffn(h))
        return y


class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        pad_id_src,
        pad_id_tgt,
        num_layers,
        d_model,
        n_heads,
        d_ff,
        dropout,
        pos_type,
        norm_type,
        max_len=4096,
        tie_embeddings=False,
    ):
        super().__init__()
        self.pad_id_src = pad_id_src
        self.pad_id_tgt = pad_id_tgt
        self.pos_type = pos_type.lower()
        self.norm_type = norm_type.lower()

        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id_src)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id_tgt)
        self.drop = nn.Dropout(dropout)

        self.use_abs = (self.pos_type == "absolute")
        self.use_rel = (self.pos_type == "relative")

        self.abs_pos = SinusoidalPositionalEncoding(d_model, max_len=max_len) if self.use_abs else None
        self.rel_bias = T5RelativePositionBias(n_heads, num_buckets=32, max_distance=128) if self.use_rel else None

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, self.norm_type, self.use_rel, self.rel_bias)
            for _ in range(num_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, self.norm_type, self.use_rel, self.rel_bias)
            for _ in range(num_layers)
        ])
        self.enc_norm = make_norm(self.norm_type, d_model)
        self.dec_norm = make_norm(self.norm_type, d_model)
        self.out_proj = nn.Linear(d_model, tgt_vocab, bias=False)
        if tie_embeddings:
            self.out_proj.weight = self.tgt_emb.weight

    def make_src_pad_mask(self, src_ids):
        return (src_ids != self.pad_id_src).unsqueeze(1).unsqueeze(1)  # [B,1,1,S]

    # def make_tgt_causal_mask(self, tgt_ids):
    #     B, T = tgt_ids.size()
    #     pad = (tgt_ids != self.pad_id_tgt).unsqueeze(1).unsqueeze(2)   # [B,1,1,T]
    #     causal = torch.tril(torch.ones((T, T), device=tgt_ids.device, dtype=torch.long)).unsqueeze(0).unsqueeze(1)
    #     return pad * causal  
    # def make_tgt_causal_mask(self, tgt_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     Return a boolean mask of shape [B, 1, T, T].
    #     True = allowed, False = masked.
    #     Masks:
    #     - future positions (causal)
    #     - key padding positions
    #     - query padding positions (optional but recommended)
    #     """
    #     B, T = tgt_ids.size()
    #     device = tgt_ids.device

    #     # [B, 1, 1, T]  key mask
    #     k_mask = (tgt_ids != self.pad_id_tgt).unsqueeze(1).unsqueeze(2)
    #     k_mask[..., 0] = True
    #     # [B, 1, T, 1]  query mask
    #     q_mask = (tgt_ids != self.pad_id_tgt).unsqueeze(1).unsqueeze(3)

    #     # [1, 1, T, T] causal mask
    #     causal = torch.ones((T, T), device=device, dtype=torch.bool).tril().unsqueeze(0).unsqueeze(0)

    #     # [B, 1, T, T]
    #     return causal & k_mask 

    def make_tgt_causal_mask(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean mask of shape [B, 1, T, T].
        True = allowed, False = masked.
        Masks:
        - future positions (causal)
        - key padding positions
        NOTE: do NOT apply query-padding mask here, otherwise fully-masked rows -> NaN softmax.
        """
        B, T = tgt_ids.size()
        device = tgt_ids.device

        # [1, 1, T, T] causal mask
        causal = torch.ones((T, T), device=device, dtype=torch.bool).tril().unsqueeze(0).unsqueeze(0)

        # [B, 1, 1, T] key padding mask
        k_mask = (tgt_ids != self.pad_id_tgt).unsqueeze(1).unsqueeze(2)

        # Critical fix for Marian-style decoder_start_id == pad_id:
        # ensure key position 0 is always visible so t=0 row is not all-False
        if T > 0:
            k_mask[..., 0] = True

        return causal & k_mask
                                    

    def encode(self, src_ids):
        mask = self.make_src_pad_mask(src_ids)
        x = self.src_emb(src_ids) * math.sqrt(self.src_emb.embedding_dim)
        x = self.drop(x)
        if self.abs_pos is not None:
            x = self.abs_pos(x)
        for layer in self.enc_layers:
            x = layer(x, mask)
        x = self.enc_norm(x)
        return x, mask

    def decode(self, tgt_ids, enc, enc_mask):
        self_mask = self.make_tgt_causal_mask(tgt_ids)
        y = self.tgt_emb(tgt_ids) * math.sqrt(self.tgt_emb.embedding_dim)
        y = self.drop(y)
        if self.abs_pos is not None:
            y = self.abs_pos(y)
        for layer in self.dec_layers:
            y = layer(y, enc, self_mask, enc_mask)
        y = self.dec_norm(y)
        return y

    def forward(self, src_ids, tgt_inp):
        enc, enc_mask = self.encode(src_ids)
        dec = self.decode(tgt_inp, enc, enc_mask)
        return self.out_proj(dec)


# -------------------------
# Loss / eval
# -------------------------
def label_smoothed_nll_loss(logits, targets, pad_id, eps):
    lprobs = F.log_softmax(logits, dim=-1)
    nll = -lprobs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -lprobs.mean(dim=-1)
    mask = (targets != pad_id).float()
    denom = mask.sum().clamp_min(1.0)
    nll = (nll * mask).sum() / denom
    smooth = (smooth * mask).sum() / denom
    return (1.0 - eps) * nll + eps * smooth

###########################
# @torch.no_grad()
# def eval_dev_loss_scratch(model, loader, pad_id_tgt, device, label_smoothing):
#     model.eval()
#     losses = []
#     for src, tgt, _, _ in loader:
#         src = src.to(device)
#         tgt = tgt.to(device)
#         logits = model(src, tgt[:, :-1])
#         loss = label_smoothed_nll_loss(logits, tgt[:, 1:], pad_id_tgt, label_smoothing)
#         losses.append(float(loss.item()))
#     return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def eval_dev_loss_scratch(model, loader, pad_id_tgt, device, label_smoothing, no_bos: bool, decoder_start_id: int):
    model.eval()
    losses = []
    for src, tgt, _, _ in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        if no_bos:
            start = torch.full((tgt.size(0), 1), int(decoder_start_id), dtype=tgt.dtype, device=tgt.device)
            tgt_inp = torch.cat([start, tgt[:, :-1]], dim=1)
            tgt_out = tgt
        else:
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

        logits = model(src, tgt_inp)
        loss = label_smoothed_nll_loss(logits, tgt_out, pad_id_tgt, label_smoothing)
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))

###############################

# -------------------------
# T5 dataset (raw jsonl: zh/en)
# -------------------------
class T5RawJsonlDataset(Dataset):
    """
    raw jsonl rows: {"zh": "...", "en": "...", ...}
    """
    def __init__(self, jsonl_path: str, tokenizer, max_src_len: int, max_tgt_len: int, max_samples: int = 0):
        rows = read_jsonl(jsonl_path)
        if max_samples and max_samples > 0:
            rows = rows[:max_samples]
        self.rows = rows
        self.tok = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        zh = r["zh"]
        en = r["en"]
        inp = "translate Chinese to English: " + zh
        enc = self.tok(inp, truncation=True, max_length=self.max_src_len, padding=False)
        dec = self.tok(en, truncation=True, max_length=self.max_tgt_len, padding=False)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": dec["input_ids"],
        }


def t5_collate_fn(batch, pad_token_id: int, label_pad_id: int = -100):
    max_in = max(len(x["input_ids"]) for x in batch)
    max_lab = max(len(x["labels"]) for x in batch)
    input_ids = torch.full((len(batch), max_in), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_in), dtype=torch.long)
    labels = torch.full((len(batch), max_lab), label_pad_id, dtype=torch.long)

    for i, ex in enumerate(batch):
        inp = torch.tensor(ex["input_ids"], dtype=torch.long)
        am = torch.tensor(ex["attention_mask"], dtype=torch.long)
        lab = torch.tensor(ex["labels"], dtype=torch.long)
        input_ids[i, : inp.numel()] = inp
        attention_mask[i, : am.numel()] = am
        labels[i, : lab.numel()] = lab

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # scratch uses processed ids
    # ap.add_argument("--train_jsonl", required=True, help="processed_train.jsonl (ids)")
    # ap.add_argument("--dev_jsonl", required=True, help="processed_dev.jsonl (ids)")
    ap.add_argument("--train_jsonl", default="", help="(scratch) processed_train.jsonl (ids)")
    ap.add_argument("--dev_jsonl", default="", help="(scratch) processed_dev.jsonl (ids)")
    # T5 uses raw text jsonl
    ap.add_argument("--t5_train_jsonl", default="", help="raw train_*.jsonl with zh/en (recommended)")
    ap.add_argument("--t5_dev_jsonl", default="", help="raw valid.jsonl with zh/en (recommended)")
########
    ap.add_argument("--vocab_zh", default="", help="(SPM mode) vocab_zh.txt")
    ap.add_argument("--vocab_en", default="", help="(SPM mode) vocab_en.txt")
    ap.add_argument("--tokenizer_meta", default="", help="tokenizer_meta.json from preprocessing (HF ids mode).")
    ap.add_argument(
    "--no_bos",
    action="store_true",
    help="HF/Marian-style ids: targets have NO BOS. Build decoder input as [decoder_start_id] + tgt[:-1], and set tgt_out=tgt."
    )
##########    
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_name", default="best.pt")

    ap.add_argument("--model_type", choices=["scratch", "t5"], default="scratch")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_n", type=int, default=2, help="print first N samples/batches")


    # scratch model scales + ablations
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tie_embeddings", action="store_true")
    ap.add_argument("--pos_type", choices=["absolute", "relative"], default="absolute")
    ap.add_argument("--norm_type", choices=["layernorm", "rmsnorm"], default="layernorm")

    # T5
    ap.add_argument("--t5_name", default="t5-base")
    ap.add_argument("--t5_max_src_len", type=int, default=128)
    ap.add_argument("--t5_max_tgt_len", type=int, default=128)

    # training hypers
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)          # scratch lr
    ap.add_argument("--t5_lr", type=float, default=5e-5)       # t5 lr
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=800)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.01)

    ap.add_argument("--grad_accum_steps", type=int, default=1)

    # low-resource knobs
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_dev_samples", type=int, default=0)

    # eval + early stop
    ap.add_argument("--eval_every", type=int, default=500)       # in optimizer steps
    ap.add_argument("--early_stop_patience", type=int, default=10)

    ap.add_argument("--min_delta", type=float, default=1e-4, help="Minimum improvement to count as progress (early-stop / plateau).")
    ap.add_argument("--plateau_patience", type=int, default=3, help="ReduceLROnPlateau patience (in eval rounds).")
    ap.add_argument("--plateau_factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    ap.add_argument("--min_lr", type=float, default=1e-7, help="Minimum LR for ReduceLROnPlateau.")

    # misc
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    def dprint(*a, **k):
        if args.debug:
            print(*a, **k)


    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = torch.device(args.device)
###########
    # decoder_start_id = None

    # if args.tokenizer_meta:
    #     meta = load_tokenizer_meta(args.tokenizer_meta)

    #     # HF ids 模式：src/tgt vocab 同大小（你这里 65001）
    #     src_vocab_size = meta["src_vocab_size"]
    #     tgt_vocab_size = meta["tgt_vocab_size"]

    #     vsrc = {
    #         "pad_id": meta["pad_token_id"],
    #         "unk_id": meta["unk_token_id"],
    #         "eos_id": meta["eos_token_id"],
    #         "bos_id": None,
    #     }
    #     vtgt = dict(vsrc)

    #     decoder_start_id = meta["decoder_start_token_id"]

    #     # 强一致：你的数据“无 BOS”，训练必须走 no_bos 分支
    #     if not args.no_bos:
    #         print("[warn] tokenizer_meta provided but --no_bos not set; forcing no_bos=True for consistency.")
    #         args.no_bos = True

    # else:
    #     if not args.vocab_zh or not args.vocab_en:
    #         raise ValueError("Need --vocab_zh/--vocab_en for SPM mode, or provide --tokenizer_meta for HF-ids mode.")

    #     vsrc = load_vocab_from_txt(args.vocab_zh)
    #     vtgt = load_vocab_from_txt(args.vocab_en)

    #     src_vocab_size = len(vsrc["itos"])
    #     tgt_vocab_size = len(vtgt["itos"])
    #     decoder_start_id = vtgt["bos_id"]  # SPM 一般有 <bos>

###########

    # run_config = vars(args).copy()
    # # run_config.update({
    # #     "src_vocab_size": len(vsrc["itos"]),
    # #     "tgt_vocab_size": len(vtgt["itos"]),
    # #     "pad_id_src": vsrc["pad_id"],
    # #     "pad_id_tgt": vtgt["pad_id"],
    # # })
    # run_config.update({
    # "src_vocab_size": int(src_vocab_size),
    # "tgt_vocab_size": int(tgt_vocab_size),
    # "pad_id_src": int(vsrc["pad_id"]),
    # "pad_id_tgt": int(vtgt["pad_id"]),
    # "decoder_start_id": int(decoder_start_id),
    # })
    run_config = vars(args).copy()  # 先只保存参数本身（通用）
    t_start = time.time()


##################



    # ------------------ SCRATCH ------------------
    if args.model_type == "scratch":

        if not args.train_jsonl or not args.dev_jsonl:
            raise ValueError("scratch mode requires --train_jsonl and --dev_jsonl")
        
        train_ds = NMTProcessedIdsDataset(args.train_jsonl, max_samples=args.max_train_samples)
        dev_ds = NMTProcessedIdsDataset(args.dev_jsonl, max_samples=args.max_dev_samples)


        

#####################3
        decoder_start_id = None

        if args.tokenizer_meta:
            meta = load_tokenizer_meta(args.tokenizer_meta)

            # HF ids 模式：src/tgt vocab 同大小（你这里 65001）
            src_vocab_size = meta["src_vocab_size"]
            tgt_vocab_size = meta["tgt_vocab_size"]

            vsrc = {
                "pad_id": meta["pad_token_id"],
                "unk_id": meta["unk_token_id"],
                "eos_id": meta["eos_token_id"],
                "bos_id": None,
            }
            vtgt = dict(vsrc)

            decoder_start_id = meta["decoder_start_token_id"]
            ##################

            # 强一致：你的数据“无 BOS”，训练必须走 no_bos 分支
            if not args.no_bos:
                print("[warn] tokenizer_meta provided but --no_bos not set; forcing no_bos=True for consistency.")
                args.no_bos = True

            # 硬保证：start 不能是 pad（否则 padding_idx 冻住）
            assert decoder_start_id != int(vtgt["pad_id"]), "decoder_start_id == pad_id will freeze start embedding (padding_idx)."




            ###################

            # 强一致：你的数据“无 BOS”，训练必须走 no_bos 分支
            if not args.no_bos:
                print("[warn] tokenizer_meta provided but --no_bos not set; forcing no_bos=True for consistency.")
                args.no_bos = True

        else:
            if not args.vocab_zh or not args.vocab_en:
                raise ValueError("Need --vocab_zh/--vocab_en for SPM mode, or provide --tokenizer_meta for HF-ids mode.")

            vsrc = load_vocab_from_txt(args.vocab_zh)
            vtgt = load_vocab_from_txt(args.vocab_en)

            src_vocab_size = len(vsrc["itos"])
            tgt_vocab_size = len(vtgt["itos"])
            decoder_start_id = vtgt["bos_id"]  # SPM 一般有 <bos>
        
        if args.debug:
            dprint("\n[DBG][data-samples]")
            for i in range(min(args.debug_n, len(train_ds))):
                zh_i, en_i = train_ds[i]
                zh_list = zh_i.tolist()
                en_list = en_i.tolist()
                dprint(f"  sample#{i} zh_len={len(zh_list)} en_len={len(en_list)}")
                dprint(f"    zh_head={zh_list[:8]} ... zh_tail={zh_list[-8:]}")
                dprint(f"    en_head={en_list[:8]} ... en_tail={en_list[-8:]}")
                # 检查 eos 是否存在（很多数据必须以 eos 结尾）
                dprint(f"    en_has_eos={int(vtgt['eos_id']) in set(en_list)}  en_last_is_eos={len(en_list)>0 and en_list[-1]==int(vtgt['eos_id'])}")
                # 检查是否出现 bos（如果 no_bos=True，出现 bos 就很可疑）
                if vtgt.get("bos_id", None) is not None:
                    dprint(f"    en_has_bos={int(vtgt['bos_id']) in set(en_list)}")



        dprint("\n[DBG][special-ids]")
        dprint(f"  mode=HF_meta? {bool(args.tokenizer_meta)}  no_bos={bool(args.no_bos)}")
        dprint(f"  src_vocab_size={src_vocab_size}  tgt_vocab_size={tgt_vocab_size}")
        dprint(f"  vsrc.pad={vsrc['pad_id']} vsrc.unk={vsrc['unk_id']} vsrc.eos={vsrc['eos_id']} vsrc.bos={vsrc.get('bos_id', None)}")
        dprint(f"  vtgt.pad={vtgt['pad_id']} vtgt.unk={vtgt['unk_id']} vtgt.eos={vtgt['eos_id']} vtgt.bos={vtgt.get('bos_id', None)}")
        dprint(f"  decoder_start_id={decoder_start_id}")

        # 强提示（不是 assert，先看输出）
        if args.no_bos:
            if int(decoder_start_id) == int(vtgt["pad_id"]):
                dprint("[DBG][WARN] decoder_start_id == pad_id_tgt  <-- HIGH RISK (padding_idx + mask may break decoding/training)")
            # if int(decoder_start_id) == int(vtgt["eos_id"]):
            #     dprint("[DBG][WARN] decoder_start_id == eos_id  <-- wrong")
            if int(decoder_start_id) == int(vtgt["eos_id"]):
                dprint("[DBG][INFO] decoder_start_id == eos_id (common for Marian-style no_bos training)")


        
        run_config.update({
        "src_vocab_size": int(src_vocab_size),
        "tgt_vocab_size": int(tgt_vocab_size),
        "pad_id_src": int(vsrc["pad_id"]),
        "pad_id_tgt": int(vtgt["pad_id"]),
        "decoder_start_id": int(decoder_start_id),
            })

#####################

        with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)



        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_ids(b, vsrc["pad_id"], vtgt["pad_id"]),
            pin_memory=True,
            drop_last=True,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_ids(b, vsrc["pad_id"], vtgt["pad_id"]),
            pin_memory=True,
            drop_last=False,
        )

        model = TransformerNMT(
            ###################
            # src_vocab=len(vsrc["itos"]),
            # tgt_vocab=len(vtgt["itos"]),
            # pad_id_src=vsrc["pad_id"],
            # pad_id_tgt=vtgt["pad_id"],
            src_vocab=src_vocab_size,
            tgt_vocab=tgt_vocab_size,
            pad_id_src=vsrc["pad_id"],
            pad_id_tgt=vtgt["pad_id"],

            ###################


            num_layers=args.num_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            pos_type=args.pos_type,
            norm_type=args.norm_type,
            tie_embeddings=args.tie_embeddings,
        ).to(device)

        dprint("\n[DBG][param-check]")
        dprint(f"  tie_embeddings={bool(args.tie_embeddings)} pos_type={args.pos_type} norm_type={args.norm_type}")
        dprint(f"  model_param_count={count_parameters(model)}")
        dprint(f"  src_emb={model.src_emb.weight.shape} tgt_emb={model.tgt_emb.weight.shape} out_proj={model.out_proj.weight.shape}")
        if args.pos_type == "relative":
            dprint(f"  rel_bias_table={model.rel_bias.relative_attention_bias.weight.shape} (num_buckets x num_heads)")


        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16))

        plateau_sched = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            threshold=float(args.min_delta),
            threshold_mode="abs",
            cooldown=0,
            min_lr=float(args.min_lr),
        )

        best_dev = float("inf")
        bad_rounds = 0
        opt_step = 0
        step_time_acc = 0.0
        step_count_acc = 0

        train_iter = iter(train_loader)
        sanity_done = False


        while opt_step < args.max_steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            t0 = time.time()

            # warmup 外置：每个 optimizer step 只设置一次（scratch 用 args.lr）
            if args.warmup_steps and opt_step < args.warmup_steps:
                lr_now = args.lr * float(opt_step + 1) / float(max(1, args.warmup_steps))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            for _ in range(args.grad_accum_steps):
                try:
                    src, tgt, _, _ = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    src, tgt, _, _ = next(train_iter)

                src = src.to(device)
                tgt = tgt.to(device)
    #################
                # tgt_inp = tgt[:, :-1]
                # tgt_out = tgt[:, 1:]

                if not sanity_done:
                    
                    # id range checks (must be before embedding lookup)
                    assert int(src.min()) >= 0 and int(tgt.min()) >= 0, "negative token id found"

                    assert int(src.max()) < int(src_vocab_size), \
                        f"src id out of range: max={int(src.max())}, vocab={src_vocab_size}"
                    assert int(tgt.max()) < int(tgt_vocab_size), \
                        f"tgt id out of range: max={int(tgt.max())}, vocab={tgt_vocab_size}"

                    # pad id sanity
                    assert 0 <= int(vsrc["pad_id"]) < int(src_vocab_size)
                    assert 0 <= int(vtgt["pad_id"]) < int(tgt_vocab_size)

                    # decoder_start sanity only matters when no_bos=True
                    if args.no_bos:
                        assert decoder_start_id is not None, "no_bos=True but decoder_start_id is None"
                        assert 0 <= int(decoder_start_id) < int(tgt_vocab_size), "decoder_start_id out of vocab range"
                        # assert int(decoder_start_id) != int(vtgt["pad_id"]), "decoder_start_id must not be pad_id"
                    # sanity_done = True

                    




                if args.no_bos:
                    start = torch.full((tgt.size(0), 1), int(decoder_start_id), dtype=tgt.dtype, device=tgt.device)
                    tgt_inp = torch.cat([start, tgt[:, :-1]], dim=1)
                    tgt_out = tgt
                else:
                    tgt_inp = tgt[:, :-1]
                    tgt_out = tgt[:, 1:]

                debug_once = args.debug and (not sanity_done)

                if debug_once:
                    dprint("\n[DBG][batch-shift-check]")
                    dprint(f"  src.shape={tuple(src.shape)} tgt.shape={tuple(tgt.shape)}")
                    dprint(f"  src.min/max={int(src.min())}/{int(src.max())}  tgt.min/max={int(tgt.min())}/{int(tgt.max())}")
                    dprint(f"  tgt_pad_id={int(vtgt['pad_id'])} tgt_eos_id={int(vtgt['eos_id'])} decoder_start_id={decoder_start_id}")

                    dprint(f"  tgt_inp.shape={tuple(tgt_inp.shape)} tgt_out.shape={tuple(tgt_out.shape)}")
                    dprint(f"  tgt_inp[0,:12]={tgt_inp[0,:12].tolist()}")
                    dprint(f"  tgt_out[0,:12]={tgt_out[0,:12].tolist()}")
                    dprint(f"  tgt_inp_first_token={int(tgt_inp[0,0])}")

                    if args.no_bos and int(tgt_inp[0,0]) == int(vtgt["pad_id"]):
                        dprint("[DBG][WARN] tgt_inp first token is PAD (decoder_start==pad). With q_mask this can cause all-masked query rows.")

                
######################
                
                with torch.cuda.amp.autocast(enabled=bool(args.fp16)):

                    
                    if debug_once:
                        self_mask = model.make_tgt_causal_mask(tgt_inp)   # 注意：直接用 tgt_inp，不要再用 tgt_inp_dbg
                        dprint("\n[DBG][mask-check]")
                        dprint(f"  self_mask.dtype={self_mask.dtype} shape={tuple(self_mask.shape)}  true_ratio={self_mask.float().mean().item():.4f}")

                        row_ok = self_mask.any(dim=-1)  # [B,1,T]
                        bad = (~row_ok).any().item()
                        dprint(f"  any_query_row_all_false={bool(bad)}")
                        if bad:
                            idx_bad = torch.nonzero(~row_ok, as_tuple=False)[0].tolist()
                            dprint(f"  first_bad_row_index={idx_bad}  (means that query position is fully masked)")




                    logits = model(src, tgt_inp)
                    if debug_once:
                        dprint("\n[DBG][nan-check]")
                        dprint(f"  logits.isfinite={bool(torch.isfinite(logits).all().item())}  logits.min/max={float(logits.min()):.3f}/{float(logits.max()):.3f}")
                    loss = label_smoothed_nll_loss(logits, tgt_out, vtgt["pad_id"], args.label_smoothing)
                    if debug_once:
                        dprint(f"  loss={float(loss.item()):.6f}  loss.isfinite={bool(torch.isfinite(loss).item())}")
                    loss = loss / float(args.grad_accum_steps)

                scaler.scale(loss).backward()
                if debug_once:
                    dprint("\n[DBG][grad-check-emb]")
                    g = model.tgt_emb.weight.grad
                    if g is None:
                        dprint("  tgt_emb.grad=None (unexpected if backward succeeded)")
                    else:
                        pad_id = int(vtgt["pad_id"])
                        dprint(f"  grad_norm_tgt_emb={float(g.norm().item()):.6f}")
                        dprint(f"  grad_norm_row(pad_id={pad_id})={float(g[pad_id].norm().item()):.6f}")
                        if args.no_bos:
                            sid = int(decoder_start_id)
                            dprint(f"  grad_norm_row(decoder_start_id={sid})={float(g[sid].norm().item()):.6f}")
                            if sid == pad_id:
                                dprint("  [DBG][INFO] decoder_start_id is pad_id -> this row grad is expected to be ~0 due to padding_idx")
                # 所有 debug 都跑完了，再置位
                if debug_once:
                    sanity_done = True

            grad_norm = None
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))

            scaler.step(optimizer)
            scaler.update()

            t1 = time.time()
            step_time_acc += (t1 - t0)
            step_count_acc += 1
            opt_step += 1

            if opt_step % 50 == 0:
                avg_step = step_time_acc / max(1, step_count_acc)
                gn = "NA" if grad_norm is None else f"{grad_norm:.2f}"
                # print(f"[scratch opt_step {opt_step}] loss={loss.item()*args.grad_accum_steps:.4f} lr={optimizer.param_groups[0]['lr']:.2e} avg_step={avg_step*1000:.1f}ms")
                print(f"[scratch opt_step {opt_step}] loss={loss.item()*args.grad_accum_steps:.4f} lr={optimizer.param_groups[0]['lr']:.2e} grad_norm={gn} avg_step={avg_step*1000:.1f}ms")

            if opt_step % args.eval_every == 0:
                ##################
                # dev_loss = eval_dev_loss_scratch(model, dev_loader, vtgt["pad_id"], device, args.label_smoothing)
                dev_loss = eval_dev_loss_scratch(
                    model, dev_loader, vtgt["pad_id"], device, args.label_smoothing,
                    args.no_bos, int(decoder_start_id)
                )
                #####################
                print(f"[scratch eval @ {opt_step}] dev_loss={dev_loss:.4f} best={best_dev:.4f}")

                improved = dev_loss < (best_dev - float(args.min_delta))

                # Plateau LR step（warmup 结束后才允许）
                if (not args.warmup_steps) or (opt_step >= args.warmup_steps):
                    prev_lr = float(optimizer.param_groups[0]["lr"])
                    plateau_sched.step(dev_loss)
                    new_lr = float(optimizer.param_groups[0]["lr"])
                    if new_lr < prev_lr - 1e-12:
                        print(f"[plateau-lr] lr {prev_lr:.2e} -> {new_lr:.2e}")

                if improved:
                    best_dev = dev_loss
                    bad_rounds = 0
                    ckpt_path = os.path.join(args.output_dir, args.save_name)
                    save_checkpoint(ckpt_path, {
                        "mode": "scratch",
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": opt_step,
                        "best_metric": best_dev,
                        "config": run_config,
                    })
                    print(f"[save] best -> {ckpt_path}")
                else:
                    bad_rounds += 1
                    print(f"[early-stop] rounds={bad_rounds}/{args.early_stop_patience}")
                    if bad_rounds >= args.early_stop_patience:
                        print("[early-stop] triggered.")
                        break

        total_time = time.time() - t_start
        metrics = {
            "mode": "scratch",
            "best_dev_loss": best_dev,
            "total_time_sec": total_time,
            "avg_step_time_sec": (step_time_acc / max(1, step_count_acc)),
            "steps_per_sec": (step_count_acc / max(1e-9, step_time_acc)),
            "peak_gpu_mem_mb": get_peak_gpu_mem_mb(),
            "param_count": count_parameters(model),
            "final_opt_step": opt_step,
        }
        with open(os.path.join(args.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("[metrics]", json.dumps(metrics, ensure_ascii=False, indent=2))

    # ------------------ T5 ------------------
    else:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except Exception as e:
            raise RuntimeError("T5 requires `transformers` (and typically `sentencepiece`).") from e

        if not args.t5_train_jsonl or not args.t5_dev_jsonl:
            raise ValueError("For --model_type t5, please provide --t5_train_jsonl and --t5_dev_jsonl (raw zh/en jsonl).")

        tok = AutoTokenizer.from_pretrained(args.t5_name, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.t5_name).to(device)
        run_config.update({
                "t5_tokenizer_pad_id": int(tok.pad_token_id) if tok.pad_token_id is not None else None
            })
        with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(run_config, f, ensure_ascii=False, indent=2)


        train_ds = T5RawJsonlDataset(
            args.t5_train_jsonl, tok, args.t5_max_src_len, args.t5_max_tgt_len, max_samples=args.max_train_samples
        )
        dev_ds = T5RawJsonlDataset(
            args.t5_dev_jsonl, tok, args.t5_max_src_len, args.t5_max_tgt_len, max_samples=args.max_dev_samples
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: t5_collate_fn(b, pad_token_id=tok.pad_token_id),
            pin_memory=True,
            drop_last=True,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: t5_collate_fn(b, pad_token_id=tok.pad_token_id),
            pin_memory=True,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.t5_lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16))

        plateau_sched = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            threshold=float(args.min_delta),
            threshold_mode="abs",
            cooldown=0,
            min_lr=float(args.min_lr),
        )

        best_dev = float("inf")
        bad_rounds = 0
        opt_step = 0
        step_time_acc = 0.0
        step_count_acc = 0

        train_iter = iter(train_loader)

        while opt_step < args.max_steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            t0 = time.time()

            # warmup 外置：每个 optimizer step 只设置一次（T5 用 args.t5_lr）
            if args.warmup_steps and opt_step < args.warmup_steps:
                lr_now = args.t5_lr * float(opt_step + 1) / float(max(1, args.warmup_steps))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            for _ in range(args.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                for k in batch:
                    batch[k] = batch[k].to(device)

                with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                    out = model(**batch)
                    loss = out.loss / float(args.grad_accum_steps)

                scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            t1 = time.time()
            step_time_acc += (t1 - t0)
            step_count_acc += 1
            opt_step += 1

            if opt_step % 50 == 0:
                avg_step = step_time_acc / max(1, step_count_acc)
                print(f"[t5 opt_step {opt_step}] loss={loss.item()*args.grad_accum_steps:.4f} lr={optimizer.param_groups[0]['lr']:.2e} avg_step={avg_step*1000:.1f}ms")

            if opt_step % args.eval_every == 0:
                model.eval()
                dev_losses = []
                with torch.no_grad():
                    for b in dev_loader:
                        for k in b:
                            b[k] = b[k].to(device)
                        with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                            o = model(**b)
                            dev_losses.append(float(o.loss.item()))
                dev_loss = float(sum(dev_losses) / max(1, len(dev_losses)))
                print(f"[t5 eval @ {opt_step}] dev_loss={dev_loss:.4f} best={best_dev:.4f}")

                improved = dev_loss < (best_dev - float(args.min_delta))

                # Plateau LR step（warmup 结束后才允许）
                if opt_step >= args.warmup_steps:
                    prev_lr = float(optimizer.param_groups[0]["lr"])
                    plateau_sched.step(dev_loss)
                    new_lr = float(optimizer.param_groups[0]["lr"])
                    if new_lr < prev_lr - 1e-12:
                        print(f"[plateau-lr] lr {prev_lr:.2e} -> {new_lr:.2e}")

                if improved:
                    best_dev = dev_loss
                    bad_rounds = 0
                    ckpt_path = os.path.join(args.output_dir, args.save_name)
                    save_checkpoint(ckpt_path, {
                        "mode": "t5",
                        "t5_name": args.t5_name,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": opt_step,
                        "best_metric": best_dev,
                        "config": run_config,
                    })
                    print(f"[save] best -> {ckpt_path}")
                else:
                    bad_rounds += 1
                    print(f"[early-stop] rounds={bad_rounds}/{args.early_stop_patience}")
                    if bad_rounds >= args.early_stop_patience:
                        print("[early-stop] triggered.")
                        break

        total_time = time.time() - t_start
        metrics = {
            "mode": "t5",
            "t5_name": args.t5_name,
            "best_dev_loss": best_dev,
            "total_time_sec": total_time,
            "avg_step_time_sec": (step_time_acc / max(1, step_count_acc)),
            "steps_per_sec": (step_count_acc / max(1e-9, step_time_acc)),
            "peak_gpu_mem_mb": get_peak_gpu_mem_mb(),
            "param_count": count_parameters(model),
            "final_opt_step": opt_step,
        }
        with open(os.path.join(args.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("[metrics]", json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
