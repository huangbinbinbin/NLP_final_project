#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# -------------------------
# Common utils (align with transformer style)
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def get_peak_gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def save_checkpoint(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def load_tokenizer_meta(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    need = [
        "pad_token_id", "unk_token_id", "eos_token_id",
        "src_vocab_size", "tgt_vocab_size",
        "decoder_start_token_id"
    ]
    for k in need:
        if k not in meta:
            raise ValueError(f"tokenizer_meta missing key: {k}")

    # bos_token_id may be null
    if "bos_token_id" not in meta:
        meta["bos_token_id"] = None

    # 强制 int（bos_token_id 允许 None）
    for k in need:
        meta[k] = int(meta[k])
    if meta["bos_token_id"] is not None:
        meta["bos_token_id"] = int(meta["bos_token_id"])
    return meta


# label smoothing (same style as transformer)
def label_smoothed_nll_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, eps: float) -> torch.Tensor:
    """
    logits: [N, V] or [B,T,V]
    targets: [N] or [B,T]
    """
    if logits.dim() == 3:
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)

    lprobs = F.log_softmax(logits, dim=-1)
    nll = -lprobs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -lprobs.mean(dim=-1)

    mask = (targets != pad_id).float()
    denom = mask.sum().clamp_min(1.0)
    nll = (nll * mask).sum() / denom
    smooth = (smooth * mask).sum() / denom
    return (1.0 - eps) * nll + eps * smooth


# -------------------------
# Dataset / Collate (HF ids compatible)
# -------------------------
class NMTProcessedIdsDataset(Dataset):
    """
    processed_*.jsonl: each row has zh_ids, en_ids
    """
    def __init__(self, jsonl_path: str, direction: str = "zh2en", max_samples: int = 0):
        rows = read_jsonl(jsonl_path)
        if max_samples and max_samples > 0:
            rows = rows[:max_samples]
        self.rows = rows
        self.direction = direction

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        zh = r["zh_ids"]
        en = r["en_ids"]
        if self.direction == "zh2en":
            return torch.tensor(zh, dtype=torch.long), torch.tensor(en, dtype=torch.long)
        elif self.direction == "en2zh":
            return torch.tensor(en, dtype=torch.long), torch.tensor(zh, dtype=torch.long)
        else:
            raise ValueError("direction must be zh2en|en2zh")


def collate_ids(
    batch,
    pad_id_src: int,
    pad_id_tgt: int,
    eos_id_tgt: int,
    max_src_len: int = 0,
    max_tgt_len: int = 0,
):
    srcs, tgts = zip(*batch)

    def trunc(seq: torch.Tensor, max_len: int, eos_id: int) -> torch.Tensor:
        if max_len and seq.numel() > max_len:
            # keep EOS at end
            seq = torch.cat([seq[: max_len - 1], torch.tensor([eos_id], dtype=seq.dtype)], dim=0)
        return seq

    srcs = [trunc(s, max_src_len, eos_id_tgt) for s in srcs]  # src 不一定有 eos，但不影响；只用于长度截断
    tgts = [trunc(t, max_tgt_len, eos_id_tgt) for t in tgts]

    src_lens = torch.tensor([len(x) for x in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgts], dtype=torch.long)

    max_src = int(src_lens.max().item())
    max_tgt = int(tgt_lens.max().item())

    src_pad = torch.full((len(batch), max_src), pad_id_src, dtype=torch.long)
    tgt_pad = torch.full((len(batch), max_tgt), pad_id_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : s.numel()] = s
        tgt_pad[i, : t.numel()] = t

    src_mask = (src_pad != pad_id_src)  # [B,S] True=real token
    return src_pad, tgt_pad, src_mask, src_lens, tgt_lens


# -------------------------
# Model (same as your RNN, but pad_id paramized)
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

    def forward(
        self,
        dec_h: torch.Tensor,        # [B,H]
        enc_out: torch.Tensor,      # [B,S,H]
        src_mask: torch.Tensor,     # [B,S] bool
        cov_prev: Optional[torch.Tensor] = None,  # [B,S]
        cov_beta: float = 0.0
    ):
        if self.alignment == "dot":
            scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2)  # [B,S]
        elif self.alignment == "multiplicative":
            q = self.W(dec_h)  # [B,H]
            scores = torch.bmm(enc_out, q.unsqueeze(2)).squeeze(2)
        else:
            s_proj = self.W_s(enc_out)                 # [B,S,H]
            h_proj = self.W_h(dec_h).unsqueeze(1)      # [B,1,H]
            energy = torch.tanh(s_proj + h_proj)       # [B,S,H]
            scores = self.v(energy).squeeze(2)         # [B,S]

        if cov_prev is not None and cov_beta > 0:
            scores = scores - cov_beta * cov_prev

        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn_w = torch.softmax(scores, dim=1)  # [B,S]
        ctx = torch.bmm(attn_w.unsqueeze(1), enc_out).squeeze(1)  # [B,H]
        return ctx, attn_w


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int,
                 num_layers: int, rnn_type: str, dropout: float, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.ln_emb = nn.LayerNorm(emb_size)
        self.drop_emb = nn.Dropout(dropout)

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True
        )

        self.ln_rnn = nn.LayerNorm(hidden_size)
        self.drop_rnn = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        x = self.emb(src)
        x = self.drop_emb(self.ln_emb(x))
        out, h = self.rnn(x)
        out = self.drop_rnn(self.ln_rnn(out))
        return out, h


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int,
                 num_layers: int, rnn_type: str, dropout: float, alignment: str, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.ln_emb = nn.LayerNorm(emb_size)
        self.drop_emb = nn.Dropout(dropout)

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            input_size=emb_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True
        )

        self.ln_rnn = nn.LayerNorm(hidden_size)
        self.drop_rnn = nn.Dropout(dropout)

        self.attn = Attention(hidden_size, alignment=alignment)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )

    def _top_hidden(self, h):
        if self.rnn_type == "lstm":
            h0, _c0 = h
            return h0[-1]  # [B,H]
        return h[-1]

    def forward_step(
        self,
        y_prev: torch.Tensor,      # [B]
        h,
        enc_out: torch.Tensor,     # [B,S,H]
        src_mask: torch.Tensor,    # [B,S]
        ctx_prev: torch.Tensor,    # [B,H]
        cov_prev: torch.Tensor,    # [B,S]
        cov_beta: float
    ):
        emb = self.drop_emb(self.ln_emb(self.emb(y_prev)))       # [B,E]
        rnn_in = torch.cat([emb, ctx_prev], dim=-1)              # [B,E+H]
        out, h_new = self.rnn(rnn_in.unsqueeze(1), h)            # [B,1,H]
        out = out.squeeze(1)
        out = self.drop_rnn(self.ln_rnn(out))

        top_h = self._top_hidden(h_new)
        ctx, attn_w = self.attn(top_h, enc_out, src_mask, cov_prev=cov_prev, cov_beta=cov_beta)
        cov_new = cov_prev + attn_w

        logits = self.out(torch.cat([out, ctx], dim=-1))         # [B,V]
        return logits, h_new, ctx, cov_new


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, emb_size: int, hidden_size: int,
                 num_layers: int, rnn_type: str, dropout: float, alignment: str,
                 pad_id_src: int, pad_id_tgt: int):
        super().__init__()
        self.pad_id_src = int(pad_id_src)
        self.pad_id_tgt = int(pad_id_tgt)
        self.encoder = Encoder(src_vocab, emb_size, hidden_size, num_layers, rnn_type, dropout, pad_id=self.pad_id_src)
        self.decoder = Decoder(tgt_vocab, emb_size, hidden_size, num_layers, rnn_type, dropout, alignment, pad_id=self.pad_id_tgt)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        training_policy: str,
        cov_beta: float = 0.0,
        no_bos: bool = False,
        decoder_start_id: int = 0,
    ) -> torch.Tensor:
        """
        HF no_bos:
          - tgt contains NO BOS, usually ends with EOS.
          - we predict tgt[0..T-1] given y_prev starting from decoder_start_id.
          - returns logits: [B, T, V]
        BOS mode:
          - tgt includes BOS/EOS
          - we predict next tokens for t=1..T-1
          - returns logits: [B, T-1, V]
        """
        enc_out, enc_h = self.encoder(src)
        dec_h = enc_h

        B, T = tgt.shape
        S = enc_out.size(1)
        H = enc_out.size(2)

        ctx_prev = torch.zeros(B, H, device=src.device)
        cov_prev = torch.zeros(B, S, device=src.device)

        logits_all: List[torch.Tensor] = []

        if no_bos:
            y_prev = torch.full((B,), int(decoder_start_id), dtype=tgt.dtype, device=tgt.device)
            for t in range(0, T):
                logits, dec_h, ctx_prev, cov_prev = self.decoder.forward_step(
                    y_prev, dec_h, enc_out, src_mask, ctx_prev, cov_prev, cov_beta
                )
                logits_all.append(logits.unsqueeze(1))
                if training_policy == "teacher_forcing":
                    y_prev = tgt[:, t]
                elif training_policy == "free_running":
                    y_prev = torch.argmax(logits, dim=-1)
                else:
                    raise ValueError("training_policy must be teacher_forcing|free_running")
            return torch.cat(logits_all, dim=1)  # [B,T,V]
        else:
            y_prev = tgt[:, 0]  # BOS
            for t in range(1, T):
                logits, dec_h, ctx_prev, cov_prev = self.decoder.forward_step(
                    y_prev, dec_h, enc_out, src_mask, ctx_prev, cov_prev, cov_beta
                )
                logits_all.append(logits.unsqueeze(1))
                if training_policy == "teacher_forcing":
                    y_prev = tgt[:, t]
                elif training_policy == "free_running":
                    y_prev = torch.argmax(logits, dim=-1)
                else:
                    raise ValueError("training_policy must be teacher_forcing|free_running")
            return torch.cat(logits_all, dim=1)  # [B,T-1,V]


# -------------------------
# Train / Eval (step-based, align transformer)
# -------------------------
@torch.no_grad()
def eval_dev_loss(
    model: Seq2Seq,
    loader: DataLoader,
    device: torch.device,
    pad_id_tgt: int,
    label_smoothing: float,
    cov_beta: float,
    no_bos: bool,
    decoder_start_id: int,
) -> float:
    model.eval()
    losses: List[float] = []
    for src, tgt, src_mask, *_ in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)

        logits = model(
            src, tgt, src_mask,
            training_policy="teacher_forcing",
            cov_beta=cov_beta,
            no_bos=no_bos,
            decoder_start_id=decoder_start_id
        )

        if no_bos:
            gold = tgt
        else:
            gold = tgt[:, 1:]

        loss = label_smoothed_nll_loss(logits, gold, pad_id=int(pad_id_tgt), eps=float(label_smoothing))
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


# -------------------------
# Optional inference (for results_rnn.json parity)
# -------------------------
def _ids_to_text(tok, ids: List[int]) -> str:
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


@torch.no_grad()
def greedy_decode_batch(
    model: Seq2Seq,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    min_len: int,
    eos_id: int,
    decoder_start_id: int,
    cov_beta: float,
) -> Tuple[List[List[int]], int]:
    """
    Batched greedy decode. Returns (generated_ids_without_start, total_generated_tokens).
    """
    model.eval()
    device = src.device

    enc_out, enc_h = model.encoder(src)
    dec_h = enc_h

    B = src.size(0)
    S = enc_out.size(1)
    H = enc_out.size(2)

    ctx_prev = torch.zeros(B, H, device=device)
    cov_prev = torch.zeros(B, S, device=device)

    y_prev = torch.full((B,), int(decoder_start_id), dtype=torch.long, device=device)

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    hyps: List[List[int]] = [[] for _ in range(B)]
    total_tok = 0

    for t in range(max_len):
        logits, dec_h, ctx_prev, cov_prev = model.decoder.forward_step(
            y_prev, dec_h, enc_out, src_mask, ctx_prev, cov_prev, cov_beta
        )
        next_id = torch.argmax(logits, dim=-1)  # [B]
        # enforce min_len (disallow eos early)
        if t < int(min_len):
            next_id = torch.where(next_id == int(eos_id), torch.full_like(next_id, int(decoder_start_id)), next_id)

        y_prev = next_id

        for i in range(B):
            if not finished[i]:
                hyps[i].append(int(next_id[i].item()))
                total_tok += 1
                if int(next_id[i].item()) == int(eos_id) and (t + 1) >= int(min_len):
                    finished[i] = True

        if bool(finished.all().item()):
            break

    return hyps, total_tok


def compute_bleu_sacrebleu(preds: List[str], refs: List[str]) -> float:
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        return float(bleu.score)
    except Exception:
        return -1.0


@torch.no_grad()
def run_inference_bleu_latency(
    model: Seq2Seq,
    jsonl_path: str,
    tok,
    device: torch.device,
    pad_id_src: int,
    pad_id_tgt: int,
    eos_id_tgt: int,
    decoder_start_id: int,
    max_src_len: int,
    max_tgt_len: int,
    infer_batch_size: int,
    max_len: int,
    min_len: int,
    cov_beta: float,
    max_samples: int = 0,
) -> Dict[str, float]:
    ds = NMTProcessedIdsDataset(jsonl_path, direction="zh2en", max_samples=max_samples)
    ld = DataLoader(
        ds,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_ids(
            b, pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt, eos_id_tgt=eos_id_tgt,
            max_src_len=max_src_len, max_tgt_len=max_tgt_len
        ),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    preds: List[str] = []
    refs: List[str] = []

    t0 = time.time()
    total_gen_tok = 0
    total_sent = 0

    for src, tgt, src_mask, *_ in ld:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)

        hyp_ids, gen_tok = greedy_decode_batch(
            model, src, src_mask,
            max_len=max_len, min_len=min_len,
            eos_id=eos_id_tgt,
            decoder_start_id=decoder_start_id,
            cov_beta=cov_beta,
        )
        total_gen_tok += int(gen_tok)
        total_sent += int(src.size(0))

        for i in range(src.size(0)):
            # pred: greedy output ids already exclude start; decode skips </s>/<pad>
            pred = _ids_to_text(tok, hyp_ids[i])
            # ref: target ids; decode skip specials
            ref = _ids_to_text(tok, tgt[i].tolist())
            preds.append(pred)
            refs.append(ref)

    dt = time.time() - t0
    bleu = compute_bleu_sacrebleu(preds, refs)

    avg_ms_per_sent = (dt / max(1, total_sent)) * 1000.0
    tok_per_sec = float(total_gen_tok / max(1e-9, dt))

    return {"bleu": float(bleu), "avg_ms_per_sent": float(avg_ms_per_sent), "tok_per_sec": float(tok_per_sec)}


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train_jsonl", required=True, help="processed_train.jsonl (ids)")
    ap.add_argument("--dev_jsonl", required=True, help="processed_valid.jsonl (ids)")
    ap.add_argument("--test_jsonl", default="", help="processed_test.jsonl (ids)")
    ap.add_argument("--direction", default="zh2en", choices=["zh2en", "en2zh"])

    # HF meta (Plan A)
    ap.add_argument("--tokenizer_meta", required=True, help="tokenizer_meta.json from preprocessing (HF ids mode).")
    ap.add_argument("--hf_name_or_dir", default="", help="e.g., Helsinki-NLP/opus-mt-zh-en (for BLEU detok if do_infer)")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--no_bos", action="store_true", help="targets have NO BOS; decoder input starts from decoder_start_id")

    # output
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_name", default="best.pt")

    # model
    ap.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--alignment", default="dot", choices=["dot", "multiplicative", "additive"])
    ap.add_argument("--training_policy", default="teacher_forcing", choices=["teacher_forcing", "free_running"])

    ap.add_argument("--emb_size", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)  # teacher requirement
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--cov_beta", type=float, default=0.0)

    # training hypers (step-based, align transformer)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--grad_accum_steps", type=int, default=1)

    # low-resource knobs
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_dev_samples", type=int, default=0)

    # eval + early stop (eval rounds)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    # plateau LR
    ap.add_argument("--plateau_patience", type=int, default=3)
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-7)

    # misc
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_n", type=int, default=2)

    # inference parity (optional)
    ap.add_argument("--do_infer", action="store_true", help="after training, run greedy inference on dev/test to fill results_rnn.json")
    ap.add_argument("--infer_batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--min_len", type=int, default=0)
    ap.add_argument("--max_src_len", type=int, default=0)
    ap.add_argument("--max_tgt_len", type=int, default=0)
    ap.add_argument("--infer_max_samples", type=int, default=0)

    args = ap.parse_args()

    if args.num_layers != 2:
        raise ValueError("Teacher requirement: encoder/decoder must be 2-layer unidirectional. Use --num_layers 2.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = torch.device(args.device)

    meta = load_tokenizer_meta(args.tokenizer_meta)
    pad_id_src = int(meta["pad_token_id"])
    pad_id_tgt = int(meta["pad_token_id"])
    eos_id_tgt = int(meta["eos_token_id"])
    unk_id = int(meta["unk_token_id"])
    decoder_start_id = int(meta["decoder_start_token_id"])
    src_vocab_size = int(meta["src_vocab_size"])
    tgt_vocab_size = int(meta["tgt_vocab_size"])
    bos_id = meta.get("bos_token_id", None)

    # 强一致：bos_token_id 为 None -> 默认 no_bos=True
    if bos_id is None and not args.no_bos:
        print("[warn] tokenizer_meta indicates no BOS (bos_token_id=null). Forcing --no_bos for consistency.")
        args.no_bos = True

    # 重要：decoder_start 不能是 pad，否则 embedding padding_idx 冻住（你 transformer 已经踩过这个坑）
    if int(decoder_start_id) == int(pad_id_tgt):
        raise ValueError("decoder_start_id == pad_id_tgt. This will freeze start embedding (padding_idx). Fix tokenizer_meta or choose different start.")

    # ---------------- config.json (align transformer) ----------------
    run_config = vars(args).copy()
    run_config.update({
        "model_type": "rnn",
        "src_vocab_size": int(src_vocab_size),
        "tgt_vocab_size": int(tgt_vocab_size),
        "pad_id_src": int(pad_id_src),
        "pad_id_tgt": int(pad_id_tgt),
        "unk_id": int(unk_id),
        "eos_id": int(eos_id_tgt),
        "bos_id": (int(bos_id) if bos_id is not None else None),
        "decoder_start_id": int(decoder_start_id),
    })
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # ---------------- data ----------------
    train_ds = NMTProcessedIdsDataset(args.train_jsonl, direction=args.direction, max_samples=args.max_train_samples)
    dev_ds = NMTProcessedIdsDataset(args.dev_jsonl, direction=args.direction, max_samples=args.max_dev_samples)

    if args.debug:
        print("\n[DBG][special-ids]")
        print(f"  pad_id={pad_id_tgt} eos_id={eos_id_tgt} unk_id={unk_id} bos_id={bos_id} decoder_start_id={decoder_start_id}")
        for i in range(min(args.debug_n, len(train_ds))):
            s, t = train_ds[i]
            print(f"[DBG] sample#{i} src_len={s.numel()} tgt_len={t.numel()} tgt_head={t[:8].tolist()} tgt_tail={t[-8:].tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_ids(
            b, pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt, eos_id_tgt=eos_id_tgt,
            max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len
        ),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_ids(
            b, pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt, eos_id_tgt=eos_id_tgt,
            max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len
        ),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ---------------- model ----------------
    model = Seq2Seq(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        alignment=args.alignment,
        pad_id_src=pad_id_src,
        pad_id_tgt=pad_id_tgt,
    ).to(device)

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
    t_start = time.time()

    train_iter = iter(train_loader)

    print("[MODE] rnn")
    print(f"[MODEL] param_count={count_parameters(model)}")
    print(f"[Device] {device.type}")

    while opt_step < int(args.max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        # linear warmup
        if int(args.warmup_steps) > 0 and opt_step < int(args.warmup_steps):
            lr_now = float(args.lr) * float(opt_step + 1) / float(max(1, int(args.warmup_steps)))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        last_loss_val = None
        for _ in range(int(args.grad_accum_steps)):
            try:
                src, tgt, src_mask, *_ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                src, tgt, src_mask, *_ = next(train_iter)

            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)

            # id range sanity (first time only if debug)
            if args.debug and opt_step == 0:
                assert int(src.min()) >= 0 and int(tgt.min()) >= 0
                assert int(src.max()) < int(src_vocab_size), f"src id out of range: max={int(src.max())}, vocab={src_vocab_size}"
                assert int(tgt.max()) < int(tgt_vocab_size), f"tgt id out of range: max={int(tgt.max())}, vocab={tgt_vocab_size}"

            with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                logits = model(
                    src, tgt, src_mask,
                    training_policy=args.training_policy,
                    cov_beta=float(args.cov_beta),
                    no_bos=bool(args.no_bos),
                    decoder_start_id=int(decoder_start_id),
                )
                gold = tgt if args.no_bos else tgt[:, 1:]
                loss = label_smoothed_nll_loss(logits, gold, pad_id=int(pad_id_tgt), eps=float(args.label_smoothing))
                loss = loss / float(args.grad_accum_steps)

            scaler.scale(loss).backward()
            last_loss_val = float(loss.item() * float(args.grad_accum_steps))

        grad_norm = None
        if float(args.grad_clip) > 0:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)))

        scaler.step(optimizer)
        scaler.update()

        t1 = time.time()
        step_time_acc += (t1 - t0)
        step_count_acc += 1
        opt_step += 1

        if opt_step % 50 == 0:
            avg_step = step_time_acc / max(1, step_count_acc)
            gn = "NA" if grad_norm is None else f"{grad_norm:.2f}"
            print(f"[rnn opt_step {opt_step}] loss={last_loss_val:.4f} lr={optimizer.param_groups[0]['lr']:.2e} grad_norm={gn} avg_step={avg_step*1000:.1f}ms")

        if opt_step % int(args.eval_every) == 0:
            dev_loss = eval_dev_loss(
                model, dev_loader, device,
                pad_id_tgt=int(pad_id_tgt),
                label_smoothing=float(args.label_smoothing),
                cov_beta=float(args.cov_beta),
                no_bos=bool(args.no_bos),
                decoder_start_id=int(decoder_start_id),
            )
            print(f"[rnn eval @ {opt_step}] dev_loss={dev_loss:.4f} best={best_dev:.4f}")

            improved = dev_loss < (best_dev - float(args.min_delta))

            # Plateau LR (after warmup)
            if (int(args.warmup_steps) == 0) or (opt_step >= int(args.warmup_steps)):
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
                    "mode": "rnn",
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
                if bad_rounds >= int(args.early_stop_patience):
                    print("[early-stop] triggered.")
                    break

    total_time = time.time() - t_start
    metrics = {
        "mode": "rnn",
        "best_dev_loss": float(best_dev),
        "total_time_sec": float(total_time),
        "avg_step_time_sec": float(step_time_acc / max(1, step_count_acc)),
        "steps_per_sec": float(step_count_acc / max(1e-9, step_time_acc)),
        "peak_gpu_mem_mb": float(get_peak_gpu_mem_mb()),
        "param_count": int(count_parameters(model)),
        "final_opt_step": int(opt_step),
    }
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[metrics]", json.dumps(metrics, ensure_ascii=False, indent=2))

    # ---------------- optional inference for results_rnn.json parity ----------------
    # defaults for parity if not do_infer
    dev_bleu = -1.0
    dev_latency_ms = -1.0
    dev_tok_per_sec = -1.0
    test_bleu = -1.0

    infer_cfg = {
        "decode": "greedy",
        "beam_size": 5,
        "alpha": 0.6,
        "len_norm": "gnmt",
        "max_len": int(args.max_len),
        "infer_batch_size": int(args.infer_batch_size),
        "t5_length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "max_len_ratio": 0.0,
        "max_len_extra": 0,
        "min_len": int(args.min_len),
    }

    if args.do_infer:
        if not args.hf_name_or_dir:
            raise ValueError("--do_infer requires --hf_name_or_dir for detokenization BLEU.")
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise RuntimeError("Inference BLEU requires transformers (and usually sentencepiece).") from e

        # load best ckpt
        ckpt_path = os.path.join(args.output_dir, args.save_name)
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        model.eval()

        tok = AutoTokenizer.from_pretrained(args.hf_name_or_dir, use_fast=False, local_files_only=bool(args.local_files_only))

        dev_inf = run_inference_bleu_latency(
            model=model,
            jsonl_path=args.dev_jsonl,
            tok=tok,
            device=device,
            pad_id_src=pad_id_src,
            pad_id_tgt=pad_id_tgt,
            eos_id_tgt=eos_id_tgt,
            decoder_start_id=decoder_start_id,
            max_src_len=int(args.max_src_len),
            max_tgt_len=int(args.max_tgt_len),
            infer_batch_size=int(args.infer_batch_size),
            max_len=int(args.max_len),
            min_len=int(args.min_len),
            cov_beta=float(args.cov_beta),
            max_samples=int(args.infer_max_samples),
        )
        dev_bleu = float(dev_inf["bleu"])
        dev_latency_ms = float(dev_inf["avg_ms_per_sent"])
        dev_tok_per_sec = float(dev_inf["tok_per_sec"])
        print(f"[BLEU] {dev_bleu:.2f}")
        print(f"[LATENCY] avg_ms_per_sent={dev_latency_ms:.2f} tok_per_sec={dev_tok_per_sec:.1f}")

        if args.test_jsonl:
            test_inf = run_inference_bleu_latency(
                model=model,
                jsonl_path=args.test_jsonl,
                tok=tok,
                device=device,
                pad_id_src=pad_id_src,
                pad_id_tgt=pad_id_tgt,
                eos_id_tgt=eos_id_tgt,
                decoder_start_id=decoder_start_id,
                max_src_len=int(args.max_src_len),
                max_tgt_len=int(args.max_tgt_len),
                infer_batch_size=int(args.infer_batch_size),
                max_len=int(args.max_len),
                min_len=int(args.min_len),
                cov_beta=float(args.cov_beta),
                max_samples=int(args.infer_max_samples),
            )
            test_bleu = float(test_inf["bleu"])

    # results_rnn.json (single record, transformer-like)
    rec = {
        "exp": f"rnn_rnn={args.rnn_type}_align={args.alignment}_policy={args.training_policy}",
        "type": "single",
        "model_type": "scratch",
        "pos_type": "",
        "norm_type": "",
        "train_batch_size": int(args.batch_size),
        "train_lr": float(args.lr),
        "dev_bleu": float(dev_bleu),
        "dev_long_bleu": -1.0,
        "dev_latency_ms": float(dev_latency_ms),
        "dev_tok_per_sec": float(dev_tok_per_sec),
        "train_metrics": metrics,
        "infer_config": infer_cfg,
        "test_bleu": float(test_bleu) if args.test_jsonl else -1.0,
    }
    with open(os.path.join(args.output_dir, "results_rnn.json"), "w", encoding="utf-8") as f:
        json.dump([rec], f, ensure_ascii=False, indent=2)

    print("\n[Done]")
    print("  config.json        ->", os.path.join(args.output_dir, "config.json"))
    print("  train_metrics.json ->", os.path.join(args.output_dir, "train_metrics.json"))
    print("  results_rnn.json   ->", os.path.join(args.output_dir, "results_rnn.json"))


if __name__ == "__main__":
    main()
