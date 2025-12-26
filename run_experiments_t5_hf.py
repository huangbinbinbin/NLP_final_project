#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import subprocess
from typing import Dict, Any, List

RE_BLEU = re.compile(r"\[BLEU\]\s*([0-9]+(?:\.[0-9]+)?)")
RE_LAT  = re.compile(r"\[LATENCY\]\s*avg_ms_per_sent=([0-9]+(?:\.[0-9]+)?)\s+tok_per_sec=([0-9]+(?:\.[0-9]+)?)")

def run_train(cmd: List[str]) -> None:
    print("\n[CMD][TRAIN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_infer_and_parse(cmd: List[str]) -> Dict[str, float]:
    print("\n[CMD][INFER] " + " ".join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError("Inference failed.\n\n" + out)

    m = RE_BLEU.search(out)
    if not m:
        raise RuntimeError("Cannot parse [BLEU] from output.\n\n" + out)
    bleu = float(m.group(1))

    m = RE_LAT.search(out)
    if not m:
        avg_ms, tokps = -1.0, -1.0
    else:
        avg_ms, tokps = float(m.group(1)), float(m.group(2))

    return {"bleu": bleu, "avg_ms_per_sent": avg_ms, "tok_per_sec": tokps}

def load_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser("Run T5/MT5 finetune + greedy/beam inference + write result_t5.json")

    # scripts
    ap.add_argument("--train_py", default="/data/250010105/NLP_final_project/train_transformer_nmt_hfc_eos.py")
    ap.add_argument("--infer_py", default="/data/250010105/NLP_final_project/inference_transformer_hf_eos.py")

    # data (RAW zh/en jsonl)
    ap.add_argument("--t5_train_jsonl", required=True)
    ap.add_argument("--t5_dev_jsonl", required=True)
    ap.add_argument("--t5_test_jsonl", required=True)

    # model / output
    ap.add_argument("--t5_name", required=True, help="HF name or local dir, e.g. /data/.../mt5-small")
    ap.add_argument("--t5_model_dir", default="", help="optional override for inference loading; default uses --t5_name")
    ap.add_argument("--t5_prompt", default="translate Chinese to English: ")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_name", default="best.pt")

    # IMPORTANT: inference_transformer_hf_eos.py 参数校验需要这两个（即便是 T5 模式）
    ap.add_argument("--tokenizer_meta", default="/data/250010105/NLP_final_project/out_hf_opusmt_train/tokenizer_meta.json")
    ap.add_argument("--hf_name_or_dir", default="/data/250010105/hf_models/opus-mt-zh-en")

    # train hypers
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--t5_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=800)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--t5_max_src_len", type=int, default=80)
    ap.add_argument("--t5_max_tgt_len", type=int, default=80)

    # infer config
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--long_src_len", type=int, default=50)

    ap.add_argument("--infer_bs_greedy", type=int, default=32)
    ap.add_argument("--infer_bs_beam", type=int, default=1)
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--t5_length_penalty", type=float, default=1.0)
    ap.add_argument("--local_files_only", action="store_true", help="inference only")

    # skip training if ckpt already exists
    ap.add_argument("--skip_train", action="store_true", help="Skip training if best.pt already exists")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, args.save_name)

    # 1) TRAIN (optional)
    if not args.skip_train or (not os.path.isfile(ckpt_path)):
        train_cmd = [
            "python", args.train_py,
            "--model_type", "t5",
            "--t5_train_jsonl", args.t5_train_jsonl,
            "--t5_dev_jsonl", args.t5_dev_jsonl,
            "--t5_name", args.t5_name,
            "--t5_max_src_len", str(args.t5_max_src_len),
            "--t5_max_tgt_len", str(args.t5_max_tgt_len),
            "--output_dir", args.output_dir,
            "--save_name", args.save_name,
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--t5_lr", str(args.t5_lr),
            "--warmup_steps", str(args.warmup_steps),
            "--max_steps", str(args.max_steps),
            "--eval_every", str(args.eval_every),
            "--early_stop_patience", str(args.early_stop_patience),
            "--min_delta", str(args.min_delta),
        ]
        if args.fp16:
            train_cmd.append("--fp16")
        run_train(train_cmd)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    train_metrics = load_json_if_exists(os.path.join(args.output_dir, "train_metrics.json"))

    # 2) INFER helper
    t5_model_dir = args.t5_model_dir if args.t5_model_dir else args.t5_name

    def infer_one(split_name: str, data_jsonl: str, decode: str, long_len: int) -> Dict[str, float]:
        details_path = os.path.join(args.output_dir, f"details_{split_name}_{decode}_long{long_len}.jsonl")
        cmd = [
            "python", args.infer_py,
            "--ckpt", ckpt_path,
            "--data_jsonl", data_jsonl,

            # satisfy tokenizer/vocab guard
            "--tokenizer_meta", args.tokenizer_meta,
            "--hf_name_or_dir", args.hf_name_or_dir,

            "--device", args.device,
            "--decode", decode,
            "--infer_batch_size", str(args.infer_bs_greedy if decode == "greedy" else args.infer_bs_beam),
            "--max_len", str(args.max_len),
            "--min_len", str(args.min_len),

            "--t5_model_dir", t5_model_dir,
            "--t5_prompt", args.t5_prompt,
            "--t5_length_penalty", str(args.t5_length_penalty),

            "--save_details", details_path,
        ]
        if args.local_files_only:
            cmd.append("--local_files_only")
        if decode == "beam":
            cmd += ["--beam_size", str(args.beam_size)]
        if long_len and long_len > 0:
            cmd += ["--long_src_len", str(long_len)]
        return run_infer_and_parse(cmd)

    def build_record(decode: str) -> Dict[str, Any]:
        dev_all  = infer_one("dev",  args.t5_dev_jsonl,  decode, long_len=0)
        dev_long = infer_one("dev",  args.t5_dev_jsonl,  decode, long_len=args.long_src_len)
        test_all  = infer_one("test", args.t5_test_jsonl, decode, long_len=0)
        test_long = infer_one("test", args.t5_test_jsonl, decode, long_len=args.long_src_len)

        infer_bs = args.infer_bs_greedy if decode == "greedy" else args.infer_bs_beam
        beam_sz  = 1 if decode == "greedy" else args.beam_size

        alpha = 0.0 if decode == "greedy" else 0.6
        len_norm = "none" if decode == "greedy" else "gnmt"

        exp = f"ft_t5=mt5-small_decode={decode}"
        if decode == "beam":
            exp += f"_bs{beam_sz}_lp{args.t5_length_penalty}"

        return {
            "exp": exp,
            "type": "finetune",
            "model_type": "t5",
            "train_batch_size": args.batch_size,
            "train_lr": args.t5_lr,

            "dev_bleu": dev_all["bleu"],
            "dev_long_bleu": dev_long["bleu"],
            "dev_latency_ms": dev_all["avg_ms_per_sent"],
            "dev_tok_per_sec": dev_all["tok_per_sec"],

            "train_metrics": train_metrics,

            "infer_config": {
                "decode": decode,
                "beam_size": beam_sz,
                "alpha": alpha,
                "len_norm": len_norm,
                "max_len": args.max_len,
                "min_len": args.min_len,
                "infer_batch_size": infer_bs,
                "t5_length_penalty": args.t5_length_penalty,
                "t5_model_dir": t5_model_dir,
                "t5_prompt": args.t5_prompt,
                "local_files_only": bool(args.local_files_only),
                "hf_name_or_dir": args.hf_name_or_dir,
                "tokenizer_meta": args.tokenizer_meta,
            },

            "test_bleu": test_all["bleu"],
            "test_long_bleu": test_long["bleu"],
        }

    records: List[Dict[str, Any]] = [build_record("greedy"), build_record("beam")]

    out_json = os.path.join(args.output_dir, "result_t5.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n[OK] wrote:", out_json)

if __name__ == "__main__":
    main()
