#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch

try:
    from train_transformer_nmt_hfc_eos import TransformerNMT, load_vocab_from_txt, read_jsonl, count_parameters
except Exception:
    from train_transformer_nmt import TransformerNMT, load_vocab_from_txt, read_jsonl, count_parameters




# -------------------------
# BLEU (robust warning)
# -------------------------

def load_tokenizer_meta(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    need = ["pad_token_id", "unk_token_id", "eos_token_id", "src_vocab_size", "tgt_vocab_size", "decoder_start_token_id"]
    for k in need:
        if k not in meta:
            raise ValueError(f"tokenizer_meta missing key: {k}")
    for k in need:
        meta[k] = int(meta[k])
    return meta


def compute_bleu_sacrebleu(preds: List[str], refs: List[str]) -> float:
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        return float(bleu.score)
    except ImportError:
        print("Warning: sacrebleu not installed. BLEU will be 0.0. Install via: pip install sacrebleu")
        return 0.0
    except Exception as e:
        print(f"Warning: BLEU calculation failed: {e}. BLEU will be 0.0.")
        return 0.0


def sentence_bleu_sacrebleu(hyp: str, ref: str) -> Optional[float]:
    """
    Sentence-level BLEU is optional diagnostic only.
    Some sacrebleu versions may not provide sentence_bleu.
    """
    try:
        import sacrebleu
        if hasattr(sacrebleu, "sentence_bleu"):
            s = sacrebleu.sentence_bleu(hyp, [ref])
            return float(s.score)
        return None
    except (ImportError, AttributeError):
        return None
    except Exception:
        return None


# -------------------------
# Scratch text conversion
# -------------------------
# def ids_to_text(ids: List[int], vocab: Dict[str, Any]) -> str:
#     bos, eos, pad = vocab["bos_id"], vocab["eos_id"], vocab["pad_id"]
#     out = []
#     for i in ids:
#         if i in (bos, pad):
#             continue
#         if i == eos:
#             break
#         if 0 <= i < len(vocab["itos"]):
#             out.append(vocab["itos"][i])
#         else:
#             out.append("<unk>")
#     return " ".join(out)

import re

# def ids_to_text(ids: List[int], vocab: Dict[str, Any]) -> str:
#     bos, eos, pad = vocab["bos_id"], vocab["eos_id"], vocab["pad_id"]
#     toks = []
#     for i in ids:
#         if i in (bos, pad):
#             continue
#         if i == eos:
#             break
#         if 0 <= i < len(vocab["itos"]):
#             toks.append(vocab["itos"][i])
#         else:
#             toks.append("<unk>")

#     # ---- detokenize heuristics ----
#     # SentencePiece-like: tokens contain '▁' to mark word start
#     if any("▁" in t for t in toks):
#         s = "".join(toks).replace("▁", " ").strip()
#     else:
#         # BPE-like fallback (if you used @@)
#         s = " ".join(toks).replace("@@ ", "")
#     s = re.sub(r"\s+([.,!?;:%\)\]\}，。！？；：％）】】》’”])", r"\1", s)
#     s = re.sub(r"([(\[\{（【《“‘])\s+", r"\1", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     s = re.sub(r"\s+'", "'", s)
#     s = re.sub(r'\s+"', '"', s)
#     return s.strip()
# def ids_to_text(ids, vocab, sp=None):
#     bos, eos, pad = vocab["bos_id"], vocab["eos_id"], vocab["pad_id"]

#     pieces = []
#     for i in ids:
#         if i in (bos, pad):
#             continue
#         if i == eos:
#             break
#         pieces.append(vocab["itos"][i] if 0 <= i < len(vocab["itos"]) else "<unk>")
#     if sp is not None:
#         return sp.decode_pieces(pieces).strip()
#     # SPM detok: join pieces then replace ▁ with space
#     s = "".join(pieces).replace("▁", " ").strip()

#     # minimal cleanup (EN)  
#     s = re.sub(r"\s+([.,!?;:%\)\]\}])", r"\1", s)
#     s = re.sub(r"([(\[\{])\s+", r"\1", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     s = re.sub(r"\s+'", "'", s)
#     s = re.sub(r'\s+"', '"', s)
#     return s

def ids_to_text(ids, vocab, sp=None, start_id: Optional[int] = None):
    """
    Convert token ids -> detokenized string.
    - start_id: the decoder start token id (HF/Marian) or BOS id (SPM).
      If provided, we will skip it in output.
    """
    eos = int(vocab["eos_id"])
    pad = int(vocab["pad_id"])

    # fallback: if start_id not given, use vocab bos_id if exists
    if start_id is None:
        # some HF vocab dict you built may not have bos_id; so use get
        start_id = vocab.get("bos_id", None)
        if start_id is not None:
            start_id = int(start_id)

    pieces = []
    for i in ids:
        i = int(i)

        # skip decoder start token (HF decoder_start_id or SPM BOS)
        if start_id is not None and i == start_id:
            continue

        # skip padding
        if i == pad:
            continue

        # stop at EOS
        if i == eos:
            break

        if 0 <= i < len(vocab["itos"]):
            pieces.append(vocab["itos"][i])
        else:
            pieces.append("<unk>")

    # detok
    if sp is not None:
        return sp.decode_pieces(pieces).strip()

    s = "".join(pieces).replace("▁", " ").strip()
    s = re.sub(r"\s+([.,!?;:%\)\]\}])", r"\1", s)
    s = re.sub(r"([(\[\{])\s+", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+'", "'", s)
    s = re.sub(r'\s+"', '"', s)
    return s



# def strip_and_decode_hf(ids: List[int], tok, eos_id: int, pad_id: int, start_id: Optional[int] = None) -> str:
#     out = []
#     for j, i in enumerate(ids):
#         i = int(i)
#         if start_id is not None and j == 0 and i == int(start_id):
#             continue
#         if i == int(pad_id):
#             continue
#         if i == int(eos_id):
#             break
#         out.append(i)
#     return tok.decode(out, skip_special_tokens=True).strip()

def strip_and_decode_hf(ids: List[int], tok, eos_id: int, pad_id: int, start_id: Optional[int] = None) -> str:
    out = []
    for j, i in enumerate(ids):
        i = int(i)

        # 跳过首位 decoder_start（如果提供）
        if start_id is not None and j == 0 and i == int(start_id):
            continue

        # 跳过 padding
        if i == int(pad_id):
            continue

        # 关键修复：如果序列第一个 token 就是 eos（Marian 常见当 decoder_start），不要 break，直接跳过
        if i == int(eos_id):
            if j == 0:
                continue
            break

        out.append(i)

    return tok.decode(out, skip_special_tokens=True).strip()



def detok_spm_pieces(pieces: List[str], sp=None) -> str:
    if sp is not None:
        # spm 按 piece 字符串解码（不依赖你的 vocab id 是否等于 spm id）
        return sp.decode_pieces(pieces).strip()

    s = "".join(pieces).replace("▁", " ").strip()
    s = re.sub(r"\s+([.,!?;:%\)\]\}])", r"\1", s)
    s = re.sub(r"([(\[\{])\s+", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+'", "'", s)
    s = re.sub(r'\s+"', '"', s)
    return s


def normalize_ref_text(ref: str, sp=None) -> str:
    """
    兼容三种 ref 形态：
    1) 自然文本：直接 strip
    2) SPM pieces 串：可能含 ▁，可能空格分隔
    3) BPE(@@) 形态
    """
    if not isinstance(ref, str):
        return str(ref)

    t = ref.strip()

    # looks like sentencepiece pieces (very common)
    if "▁" in t:
        # 如果有空格，按空格拆 pieces；否则按字符/子串拼接意义不大，直接当成一整串处理
        pieces = t.split() if (" " in t) else [t]
        return detok_spm_pieces(pieces, sp=sp)

    # looks like BPE with @@
    if "@@" in t:
        t2 = t.replace("@@ ", "").replace("@@", "")
        t2 = re.sub(r"\s+", " ", t2).strip()
        return t2

    # otherwise treat as already detokenized natural sentence
    return t


# -------------------------
# Length normalization
# -------------------------
def length_norm(score: float, length: int, alpha: float, mode: str) -> float:
    mode = mode.lower()
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
# Scratch decoding
# -------------------------

#########################


@torch.no_grad()
def greedy_decode_scratch_single(model, src_ids, start_id, eos_id, max_len, ban_ids=None, min_len: int = 1):
    enc, enc_mask = model.encode(src_ids.unsqueeze(0))
    ys = torch.tensor([[int(start_id)]], dtype=torch.long, device=src_ids.device)
    gen_len = 1
    for step in range(max_len - 1):
        dec = model.decode(ys, enc, enc_mask)
        logits = model.out_proj(dec[:, -1, :])  # [1,V]

        if ban_ids:
            logits[:, ban_ids] = float("-inf")
        if min_len is not None and int(min_len) > 0 and step < int(min_len):
            logits[:, int(eos_id)] = float("-inf")

        next_id = int(torch.argmax(logits, dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=src_ids.device)], dim=1)
        gen_len += 1
        if next_id == eos_id:
            break
    return ys.squeeze(0).tolist(), gen_len


############################

@torch.no_grad()
def greedy_decode_scratch_batch(model: TransformerNMT, src_list: List[torch.Tensor], start_id: int, eos_id: int, max_len: int,ban_ids: Optional[List[int]] = None
                                ,min_len: int = 1) -> Tuple[List[List[int]], List[int]]:
    device = src_list[0].device
    pad_id_src = model.pad_id_src

    B = len(src_list)
    lens = torch.tensor([int(s.numel()) for s in src_list], device=device)
    max_src = int(lens.max().item())
    src = torch.full((B, max_src), pad_id_src, dtype=torch.long, device=device)
    for i, s in enumerate(src_list):
        src[i, : s.numel()] = s

    # 编码所有源序列
    enc, enc_mask = model.encode(src)

    # ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    ys = torch.full((B, 1), int(start_id), dtype=torch.long, device=device)
    done = torch.zeros((B,), dtype=torch.bool, device=device)
    gen_lens = torch.ones((B,), dtype=torch.long, device=device)  # include BOS

    for step in range(max_len - 1):
        if bool(done.all().item()):
            break

        active = ~done
        if not bool(active.any().item()):
            break

        # 只对活跃序列解码
        active_indices = torch.where(active)[0]
        if len(active_indices) == 0:
            break
            
        # 获取活跃序列的编码
        active_enc = enc[active_indices]
        # 调整enc_mask以匹配活跃序列
        if enc_mask is not None:
            # enc_mask形状为 [B, 1, 1, S]，选择活跃的batch
            active_enc_mask = enc_mask[active_indices]
        else:
            active_enc_mask = None

        # 解码活跃序列
        dec = model.decode(ys[active_indices], active_enc, active_enc_mask)
        logits = model.out_proj(dec[:, -1, :])  # [B_active, V]



        if ban_ids:
            logits[:, ban_ids] = float("-inf")


        if min_len is not None and int(min_len) > 0 and step < int(min_len):
            logits[:, int(eos_id)] = float("-inf")

        next_ids_active = torch.argmax(logits, dim=-1)  # [B_active]

        # 为所有序列添加新列
        new_col = torch.full((B, 1), eos_id, dtype=torch.long, device=device)
        ys = torch.cat([ys, new_col], dim=1)
        
        # 更新活跃序列的输出
        ys[active_indices, -1] = next_ids_active

        # 更新长度和完成标志
        gen_lens[active_indices] = gen_lens[active_indices] + 1
        newly_done = (next_ids_active == eos_id)
        done[active_indices] = done[active_indices] | newly_done

    # 转换为列表
    seqs = [ys[i].tolist() for i in range(B)]
    lens_out = [int(gen_lens[i].item()) for i in range(B)]
    
    return seqs, lens_out



###########

@torch.no_grad()
def beam_search_decode_scratch_single(model: TransformerNMT, src_ids: torch.Tensor, start_id: int, eos_id: int,
                                     max_len: int, beam_size: int, alpha: float, len_norm_mode: str,ban_ids: Optional[List[int]] = None
                                     ,min_len: int = 1,) -> Tuple[List[int], int]:
    """
    Note: batch beam search is intentionally not implemented (complex & bug-prone).
    This is a design choice; we run per-sentence beam for correctness.
    """
    enc, enc_mask = model.encode(src_ids.unsqueeze(0))
    device = src_ids.device
    # beams: List[Tuple[torch.Tensor, float, bool]] = [(torch.tensor([[start_id]], device=device), 0.0, False)]
    beams: List[Tuple[torch.Tensor, float, bool]] = [
    (torch.tensor([[int(start_id)]], dtype=torch.long, device=device), 0.0, False)]
    for step  in range(max_len - 1):
        new_beams: List[Tuple[torch.Tensor, float, bool]] = []
        for seq, score, done in beams:
            if done:
                new_beams.append((seq, score, done))
                continue

            dec = model.decode(seq, enc, enc_mask)
            logits = model.out_proj(dec[:, -1, :])
            logp = torch.log_softmax(logits, dim=-1).squeeze(0) 

            if ban_ids:
                logp[ban_ids] = float("-inf")


            if min_len is not None and int(min_len) > 0 and step < int(min_len):
                logp[int(eos_id)] = float("-inf")
            # logp = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk = torch.topk(logp, k=beam_size)

            for lp, wid in zip(topk.values.tolist(), topk.indices.tolist()):
                seq2 = torch.cat([seq, torch.tensor([[wid]], dtype=torch.long,device=device)], dim=1)
                new_beams.append((seq2, score + lp, wid == eos_id))

        new_beams.sort(key=lambda x: length_norm(x[1], x[0].size(1), alpha, len_norm_mode), reverse=True)
        beams = new_beams[:beam_size]
        if all(d for _, _, d in beams):
            break

    best = max(beams, key=lambda x: length_norm(x[1], x[0].size(1), alpha, len_norm_mode))
    best_ids = best[0].squeeze(0).tolist()
    # gen_len = up to eos inclusive if present else full length
    gen_len = len(best_ids)
    return best_ids, gen_len


# -------------------------
# Length filtering
# -------------------------
def estimate_zh_token_len_from_text(zh: str) -> int:
    return int(len(zh) * 1.5) + 1


def get_src_token_len(row: Dict[str, Any], mode: str, t5_tok=None, t5_prompt: str = "translate Chinese to English: ") -> int:
    if "zh_ids" in row and isinstance(row["zh_ids"], list):
        return len(row["zh_ids"])
    if mode == "t5" and t5_tok is not None and "zh" in row and isinstance(row["zh"], str):
        inp = (t5_prompt or "") + row["zh"]
        ids = t5_tok(inp, truncation=False, padding=False).get("input_ids", [])
        return len(ids)
    if "zh" in row and isinstance(row["zh"], str):
        return estimate_zh_token_len_from_text(row["zh"])
    return 0


def maybe_get_src_text(row: Dict[str, Any], vsrc: Dict[str, Any]) -> Optional[str]:
    if "zh" in row and isinstance(row["zh"], str):
        return row["zh"]
    if "zh_ids" in row and isinstance(row["zh_ids"], list):
        try:
            # return ids_to_text(row["zh_ids"], vsrc)
            src_start_id = vsrc.get("bos_id", None)
            return ids_to_text(row["zh_ids"], vsrc, start_id=src_start_id)
        except Exception:
            return None
    return None


# -------------------------
# T5 loading (avoid double download)
# -------------------------
def load_t5_tokenizer_and_model(t5_id_or_dir: str,
                               ckpt_state: Dict[str, torch.Tensor],
                               device: torch.device,
                               local_files_only: bool = False):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tok = AutoTokenizer.from_pretrained(t5_id_or_dir, use_fast=False, local_files_only=local_files_only)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            t5_id_or_dir,
            state_dict=ckpt_state,
            local_files_only=local_files_only,
            # low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            t5_id_or_dir,
            local_files_only=local_files_only,
        )
        model.load_state_dict(ckpt_state, strict=True)

    model.to(device).eval()
    return tok, model


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_jsonl", required=True)
    ###################
    # ap.add_argument("--vocab_zh", required=True)
    # ap.add_argument("--vocab_en", required=True)
    ap.add_argument("--vocab_zh", default="", help="(SPM mode) vocab_zh.txt")
    ap.add_argument("--vocab_en", default="", help="(SPM mode) vocab_en.txt")

    ap.add_argument("--tokenizer_meta", default="", help="(HF ids mode) tokenizer_meta.json")
    ap.add_argument("--hf_name_or_dir", default="Helsinki-NLP/opus-mt-zh-en", help="HF tokenizer name or local dir")
    ap.add_argument("--hf_cache_dir", default="", help="Optional cache_dir for HF tokenizer")

    ###################

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--len_norm", choices=["gnmt", "length", "none"], default="gnmt")
    ap.add_argument("--max_len", type=int, default=128)

    ap.add_argument("--infer_batch_size", type=int, default=1, help="Batch size for inference (effective for scratch greedy and T5).")

    ap.add_argument("--long_src_len", type=int, default=0, help="Filter by source token length >= threshold.")
    ap.add_argument("--max_samples", type=int, default=0)

    ap.add_argument("--t5_length_penalty", type=float, default=1.0)
    ap.add_argument("--t5_prompt", type=str, default="")
    ap.add_argument("--t5_model_dir", type=str, default="")
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--save_details", type=str, default="", help="Save per-sample details to jsonl.")
    ap.add_argument("--save_max", type=int, default=0, help="Max lines to save (0=all).")
    ap.add_argument("--spm_en_model", type=str, default="", help="Optional: sentencepiece .model for exact EN detok.")
    ap.add_argument("--min_len", type=int, default=1, help="Minimum generated length (excluding start token) before EOS is allowed.")



    args = ap.parse_args()
    ###########
    # vsrc = load_vocab_from_txt(args.vocab_zh)
    # vtgt = load_vocab_from_txt(args.vocab_en)
    hf_tok = None
    meta = None
    vsrc = None
    vtgt = None

    if args.tokenizer_meta:
        meta = load_tokenizer_meta(args.tokenizer_meta)
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(
            args.hf_name_or_dir,
            use_fast=False,
            local_files_only=bool(args.local_files_only),
            cache_dir=(args.hf_cache_dir if args.hf_cache_dir else None),
        )
    else:
        if (not args.vocab_zh) or (not args.vocab_en):
            raise ValueError("Need --vocab_zh/--vocab_en for SPM mode, or provide --tokenizer_meta for HF-ids mode.")
        vsrc = load_vocab_from_txt(args.vocab_zh)
        vtgt = load_vocab_from_txt(args.vocab_en)

    ########

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sp_en = None
    if args.spm_en_model:
        import sentencepiece as spm
        sp_en = spm.SentencePieceProcessor(model_file=args.spm_en_model)
    mode = ckpt.get("mode", "scratch")
    cfg = ckpt.get("config", {})

    device = torch.device(args.device)
    rows = read_jsonl(args.data_jsonl)
    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]

    preds: List[str] = []
    refs: List[str] = []
    t_decode = 0.0
    out_tokens = 0

    details_f = None
    saved = 0

    try:
        if args.save_details:
            os.makedirs(os.path.dirname(args.save_details) or ".", exist_ok=True)
            details_f = open(args.save_details, "w", encoding="utf-8")

        def write_detail(obj: Dict[str, Any]):
            nonlocal saved
            if details_f is None:
                return
            if args.save_max and args.save_max > 0 and saved >= args.save_max:
                return
            details_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            saved += 1

        # ------------------ SCRATCH ------------------
        if mode == "scratch":
            # 取解码起始 token：优先用 ckpt config，其次 meta
            # 1) ids: 优先来自 tokenizer_meta（HF ids 模式），避免 cfg 老旧/写错
            if meta is not None:
                eos_id = int(meta["eos_token_id"])
                pad_id_t = int(meta["pad_token_id"])
                pad_id_s = int(meta["pad_token_id"])  # 你的 HF meta 是同一套 pad

                # decoder_start_id：优先用 ckpt.config 里保存的（新 ckpt 正确），否则用 meta
                start_id = int(cfg.get("decoder_start_id", meta["decoder_start_token_id"]))

                # 强一致校验：start 不能是 pad（否则 padding_idx 冻住 / mask 风险）
                if start_id == pad_id_t:
                    # 这里不要“瞎 override”，直接报错更安全：说明 ckpt/config 被写坏了
                    raise ValueError(f"BAD CKPT: decoder_start_id == pad_id_tgt == {start_id}. "
                                    f"Expected decoder_start_id={int(meta['decoder_start_token_id'])} (often eos=0).")
            else:
                # SPM 模式：按 vocab 的 bos/eos/pad
                start_id = int(vtgt["bos_id"])
                eos_id   = int(vtgt["eos_id"])
                pad_id_t = int(vtgt["pad_id"])
                pad_id_s = int(vsrc["pad_id"])



            
            ban = {int(pad_id_t)}
            if int(start_id) != int(eos_id):
                ban.add(int(start_id))
            ban_ids = sorted(ban)



            def decode_tgt_ids(ids: List[int]) -> str:
                if meta is not None:
                    # HF/Marian：跳过 decoder_start_id；遇到 eos 停；过滤 pad
                    return strip_and_decode_hf(ids, hf_tok, eos_id=eos_id, pad_id=pad_id_t, start_id=start_id)
                return ids_to_text(ids, vtgt, sp=sp_en, start_id=start_id)

            def decode_src_ids(ids: List[int]) -> str:
                if meta is not None:
                    # 源端一般没有 decoder_start_id，所以 start_id=None
                    return strip_and_decode_hf(ids, hf_tok, eos_id=eos_id, pad_id=pad_id_s, start_id=None)
                src_start_id = vsrc.get("bos_id", None)
                return ids_to_text(ids, vsrc, start_id=src_start_id)




            model = TransformerNMT(
                src_vocab=cfg["src_vocab_size"], tgt_vocab=cfg["tgt_vocab_size"],
                pad_id_src=cfg["pad_id_src"], pad_id_tgt=cfg["pad_id_tgt"],
                num_layers=cfg["num_layers"], d_model=cfg["d_model"], n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
                dropout=cfg["dropout"], pos_type=cfg["pos_type"], norm_type=cfg["norm_type"],
                tie_embeddings=bool(cfg.get("tie_embeddings", False)),
            )
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device).eval()

            if args.long_src_len and args.long_src_len > 0:
                rows = [r for r in rows if get_src_token_len(r, mode="scratch") >= args.long_src_len]

            bs = max(1, int(args.infer_batch_size))
            idx = 0
            while idx < len(rows):
                batch_rows = rows[idx: idx + bs]
                idx += bs

                src_list = []
                ref_list = []
                for r in batch_rows:
                    if "zh_ids" not in r:
                        raise ValueError("Scratch inference expects data_jsonl rows to have 'zh_ids'.")
                    src_list.append(torch.tensor(r["zh_ids"], dtype=torch.long, device=device))
                    # ref_list.append(r["en"] if ("en" in r and isinstance(r["en"], str)) else ids_to_text(r["en_ids"], vtgt, sp=sp_en))
                    raw_ref = ""
                    if "en" in r and isinstance(r["en"], str):
                        raw_ref = r["en"]
                    elif "en_ids" in r and isinstance(r["en_ids"], list):
                        # raw_ref = ids_to_text(r["en_ids"], vtgt, sp=sp_en)
                        # raw_ref = ids_to_text(r["en_ids"], vtgt, sp=sp_en, start_id=start_id)
                        raw_ref = decode_tgt_ids(r["en_ids"])

                    ref_list.append(normalize_ref_text(raw_ref, sp=sp_en))

                t0 = time.time()
                if args.decode == "greedy" and bs > 1:
                    hyp_ids_list, gen_lens = greedy_decode_scratch_batch(model, src_list, start_id, eos_id, args.max_len, ban_ids=ban_ids,min_len=args.min_len)
                else:
                    hyp_ids_list = []
                    gen_lens = []
                    for s in src_list:
                        if args.decode == "greedy":
                            ids, gl = greedy_decode_scratch_single(model, s, start_id, eos_id, args.max_len, ban_ids=ban_ids,min_len=args.min_len)
                        else:
                            ids, gl = beam_search_decode_scratch_single(
                                model, s, start_id, eos_id,
                                args.max_len, args.beam_size, args.alpha, args.len_norm, ban_ids=ban_ids, min_len=args.min_len
                            )
                        hyp_ids_list.append(ids)
                        gen_lens.append(gl)
                t1 = time.time()
                batch_time = (t1 - t0)
                t_decode += batch_time

                for r, hyp_ids, ref_text, gl in zip(batch_rows, hyp_ids_list, ref_list, gen_lens):
                    # hyp_text = ids_to_text(hyp_ids, vtgt, sp=sp_en, start_id=start_id)
                    hyp_text = decode_tgt_ids(hyp_ids) 
                    preds.append(hyp_text)
                    refs.append(ref_text)

                    # use actual generated length, not padded eos tail
                    out_tokens += max(0, gl - 1)  # exclude BOS
                    # t_decode += (t1 - t0) / max(1, len(batch_rows))
                    # t_decode += (t1 - t0)
                    ##################

                    if meta is not None:
                        src_text = decode_src_ids(r["zh_ids"]) if ("zh_ids" in r and isinstance(r["zh_ids"], list)) else r.get("zh", None)
                    else:
                        src_text = maybe_get_src_text(r, vsrc)

                    sent_bleu = sentence_bleu_sacrebleu(hyp_text, ref_text)
                    write_detail({
                        "mode": "scratch",
                        "index": r.get("index", None),
                        "src_len_tokens": get_src_token_len(r, mode="scratch"),
                        "gen_len_tokens": gl,
                        "src": src_text,
                        "ref": ref_text,
                        "hyp": hyp_text,
                        "sentence_bleu": sent_bleu,
                        "decode": args.decode,
                        "beam_size": args.beam_size if args.decode == "beam" else 1,
                        "len_norm": args.len_norm if args.decode == "beam" else "n/a",
                        "alpha": args.alpha if args.decode == "beam" else 0.0,
                    })


####################
            bleu = compute_bleu_sacrebleu(preds, refs)
            avg_ms = (t_decode / max(1, len(rows))) * 1000.0
            tok_per_sec = out_tokens / max(1e-9, t_decode)

            print("[MODE] scratch")
            print(f"[BLEU] {bleu:.2f}")
            print(f"[LATENCY] avg_ms_per_sent={avg_ms:.2f} tok_per_sec={tok_per_sec:.1f}")
            print(f"[MODEL] param_count={count_parameters(model)}")

        # ------------------ T5 ------------------
        else:
            t5_name = ckpt.get("t5_name", cfg.get("t5_name", "t5-base"))
            t5_id_or_dir = args.t5_model_dir if args.t5_model_dir else t5_name

            default_prompt = cfg.get("t5_prompt", "translate Chinese to English: ")
            t5_prompt = args.t5_prompt if args.t5_prompt else default_prompt

            tok, model = load_t5_tokenizer_and_model(
                t5_id_or_dir=t5_id_or_dir,
                ckpt_state=ckpt["model"],
                device=device,
                local_files_only=bool(args.local_files_only),
            )

            if args.long_src_len and args.long_src_len > 0:
                rows = [r for r in rows if get_src_token_len(r, mode="t5", t5_tok=tok, t5_prompt=t5_prompt) >= args.long_src_len]

            bs = max(1, int(args.infer_batch_size))
            idx = 0
            while idx < len(rows):
                batch_rows = rows[idx: idx + bs]
                idx += bs

                inp_texts = []
                ref_texts = []
                for r in batch_rows:
                    if "zh" not in r or "en" not in r:
                        raise ValueError("T5 evaluation expects raw jsonl with 'zh' and 'en' fields.")
                    inp_texts.append((t5_prompt or "") + r["zh"])
                    ref_texts.append(r["en"])

                enc = tok(
                    inp_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(cfg.get("t5_max_src_len", 128)),
                ).to(device)

                t0 = time.time()
                gen = model.generate(
                    **enc,
                    max_length=args.max_len,
                    num_beams=(args.beam_size if args.decode == "beam" else 1),
                    length_penalty=args.t5_length_penalty,
                    early_stopping=True,
                )
                t1 = time.time()
                batch_time = (t1 - t0)
                t_decode += batch_time
                out_tokens += int(gen.size(1)) * len(batch_rows)
                hyps = tok.batch_decode(gen, skip_special_tokens=True)
                for r, hyp_text, ref_text in zip(batch_rows, hyps, ref_texts):
                    hyp_text = hyp_text.strip()
                    preds.append(hyp_text)
                    refs.append(ref_text)

                    # t_decode += (t1 - t0) / max(1, len(batch_rows))
                    # t_decode += (t1 - t0)
                    # out_tokens += int(gen.size(1))
                    # out_tokens += int(gen.size(1)) * len(batch_rows)

                    sent_bleu = sentence_bleu_sacrebleu(hyp_text, ref_text)
                    write_detail({
                        "mode": "t5",
                        "index": r.get("index", None),
                        "src_len_tokens": get_src_token_len(r, mode="t5", t5_tok=tok, t5_prompt=t5_prompt),
                        "gen_len_tokens": int(gen.size(1)),
                        "src": r.get("zh", None),
                        "ref": ref_text,
                        "hyp": hyp_text,
                        "sentence_bleu": sent_bleu,
                        "t5_prompt": t5_prompt,
                        "decode": args.decode,
                        "beam_size": args.beam_size if args.decode == "beam" else 1,
                        "t5_length_penalty": args.t5_length_penalty,
                    })

            bleu = compute_bleu_sacrebleu(preds, refs)
            avg_ms = (t_decode / max(1, len(rows))) * 1000.0
            tok_per_sec = out_tokens / max(1e-9, t_decode)

            print(f"[MODE] t5 ({t5_name})")
            print(f"[BLEU] {bleu:.2f}")
            print(f"[LATENCY] avg_ms_per_sent={avg_ms:.2f} tok_per_sec={tok_per_sec:.1f}")
            print(f"[MODEL] param_count={count_parameters(model)}")

    finally:
        if details_f is not None:
            details_f.close()
            print(f"[DETAILS] saved to: {args.save_details}")


if __name__ == "__main__":
    main()
