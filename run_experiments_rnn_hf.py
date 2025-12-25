#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_experiments_rnn_hf.py

Run experiments for RNN (Seq2Seq + Attention) in HF-ids mode, aligned to the Transformer experiment workflow.

Default expected scripts in the SAME directory (override with --train_py/--infer_py):
  - train_rnn_nmt_hf.py
  - inference_rnn_hf_aligned.py

Design:
1) One decode mode per run: --decode greedy OR --decode beam (so you can run in two ACP jobs).
2) Training outputs are stored under: <output_root>/train/<exp>/
   Inference outputs are stored under: <output_root>/<decode>/<exp>/
   This allows beam job to reuse checkpoints from greedy job via --skip_train_if_ckpt_exists.
3) Results JSON (Transformer-like):
   <output_root>/<decode>/results_transformer.json
   <output_root>/<decode>/failures_transformer.json (if any)

Robustness:
- The runner auto-detects whether the train/infer script supports certain CLI flags (e.g., --valid_jsonl vs --dev_jsonl),
  and only passes supported flags to avoid "unrecognized arguments" crashes.
"""

import os
import re
import json
import argparse
import subprocess
import time
from typing import List, Dict, Any, Tuple, Optional

FAILURES: List[Dict[str, Any]] = []
RUN_MAX_RETRIES = 2
RUN_RETRY_SLEEP = 5.0


# --------------------------
# subprocess helpers
# --------------------------
def _make_env(hf_cache_dir: str, local_files_only: bool) -> Dict[str, str]:
    env = os.environ.copy()
    if hf_cache_dir:
        # Common HF env vars; safe even if your scripts also set them externally.
        env.setdefault("HF_HOME", hf_cache_dir)
        env.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
        env.setdefault("HF_DATASETS_CACHE", hf_cache_dir)
    if local_files_only:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def run_retry(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    max_retries: Optional[int] = None,
    retry_sleep: Optional[float] = None,
):
    if max_retries is None:
        max_retries = RUN_MAX_RETRIES
    if retry_sleep is None:
        retry_sleep = RUN_RETRY_SLEEP

    for attempt in range(int(max_retries) + 1):
        print(f"\n[RUN attempt {attempt+1}/{int(max_retries)+1}] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=env)
            return
        except subprocess.CalledProcessError as e:
            FAILURES.append({"kind": "run", "cmd": " ".join(cmd), "returncode": int(e.returncode)})
            print(f"[ERROR] Command failed with exit code {e.returncode}")
            if attempt < int(max_retries):
                print(f"[RETRY] waiting {float(retry_sleep):.1f}s...")
                time.sleep(float(retry_sleep))
            else:
                raise


def read_out(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    max_retries: Optional[int] = None,
    retry_sleep: Optional[float] = None,
    log_trunc: int = 2000,
) -> Optional[str]:
    """Run a command and capture stdout+stderr. Returns None on failure."""
    if max_retries is None:
        max_retries = RUN_MAX_RETRIES
    if retry_sleep is None:
        retry_sleep = RUN_RETRY_SLEEP

    for attempt in range(int(max_retries) + 1):
        print(f"\n[RUN attempt {attempt+1}/{int(max_retries)+1}] {' '.join(cmd)}")
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env)
            if log_trunc and log_trunc > 0:
                print(out[:log_trunc] + "..." if len(out) > log_trunc else out)
            else:
                print(out)
            return out
        except subprocess.CalledProcessError as e:
            full = (e.output or "")
            snippet = full[:log_trunc] if (log_trunc and log_trunc > 0) else full
            FAILURES.append({
                "kind": "read_out",
                "cmd": " ".join(cmd),
                "returncode": int(e.returncode),
                "output": snippet,
            })
            print(f"[ERROR] Command failed with exit code {e.returncode}")
            if e.output:
                if log_trunc and log_trunc > 0:
                    print("[OUTPUT]\n" + e.output[:log_trunc])
                else:
                    print("[OUTPUT]\n" + e.output)
            if attempt < int(max_retries):
                print(f"[RETRY] waiting {float(retry_sleep):.1f}s...")
                time.sleep(float(retry_sleep))
            else:
                return None
    return None


# --------------------------
# lightweight "does this script support this flag?" detection
# --------------------------
_SUPPORT_CACHE: Dict[Tuple[str, str], bool] = {}


def supports_flag(py_path: str, flag: str) -> bool:
    """
    Best-effort check: looks for add_argument("--flag") in the script text.
    This avoids passing unsupported args to older versions of your scripts.
    """
    key = (py_path, flag)
    if key in _SUPPORT_CACHE:
        return _SUPPORT_CACHE[key]

    ok = False
    try:
        txt = open(py_path, "r", encoding="utf-8").read()
        # allow single/double quotes; whitespace tolerant
        pat = r'add_argument\(\s*["\']' + re.escape(flag) + r'["\']'
        ok = re.search(pat, txt) is not None
    except Exception:
        ok = False

    _SUPPORT_CACHE[key] = ok
    return ok


def choose_flag(py_path: str, preferred: str, fallback: str) -> str:
    """Pick preferred if supported else fallback (if supported) else preferred."""
    if supports_flag(py_path, preferred):
        return preferred
    if supports_flag(py_path, fallback):
        return fallback
    return preferred


# --------------------------
# output parsers (align with inference_rnn_hf_aligned.py prints)
# --------------------------
def parse_bleu(out: str) -> float:
    # Accept: "[BLEU] 3.26" / "[BLEU] -1.00" / "[BLEU] (skipped)"
    if re.search(r"\[BLEU\].*skipped", out, flags=re.IGNORECASE):
        return -1.0
    m = re.search(r"\[BLEU\]\s*(-?[0-9]+(?:\.[0-9]+)?)", out)
    if not m:
        raise RuntimeError("Cannot parse BLEU.")
    return float(m.group(1))


def parse_latency(out: str) -> Dict[str, float]:
    m = re.search(
        r"\[LATENCY\]\s*avg_ms_per_sent=([0-9]+(?:\.[0-9]+)?|inf|nan)\s+tok_per_sec=([0-9]+(?:\.[0-9]+)?|inf|nan)",
        out,
        flags=re.IGNORECASE,
    )
    if not m:
        return {"avg_ms_per_sent": -1.0, "tok_per_sec": -1.0}

    def _to_float(x: str) -> float:
        x = x.lower()
        if x == "inf":
            return float("inf")
        if x == "nan":
            return float("nan")
        return float(x)

    return {"avg_ms_per_sent": _to_float(m.group(1)), "tok_per_sec": _to_float(m.group(2))}


def load_train_metrics(out_dir: str) -> Dict[str, Any]:
    p = os.path.join(out_dir, "train_metrics.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_int_list(csv: str, name: str) -> List[int]:
    xs: List[int] = []
    for t in csv.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            xs.append(int(t))
        except ValueError as e:
            raise ValueError(f"Invalid integer in --{name}: {t}") from e
    return xs


def _parse_float_list(csv: str, name: str) -> List[float]:
    xs: List[float] = []
    for t in csv.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            xs.append(float(t))
        except ValueError as e:
            raise ValueError(f"Invalid float in --{name}: {t}") from e
    return xs


def _parse_str_list(csv: str) -> List[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


# --------------------------
# inference wrapper
# --------------------------
def infer_rnn_hf(
    infer_py: str,
    ckpt: str,
    data_jsonl: str,
    direction: str,
    tokenizer_meta: str,
    hf_name_or_dir: str,
    device: str,
    local_files_only: bool,
    hf_cache_dir: str,
    env: Optional[Dict[str, str]],

    decode: str = "greedy",
    beam_size: int = 5,
    alpha: float = 0.6,
    len_norm: str = "gnmt",
    max_len: int = 80,
    min_len: int = 0,
    infer_batch_size: int = 64,
    long_src_len: int = 0,

    cov_beta: Optional[float] = None,

    out_path: str = "pred.txt",
    save_details: str = "",
    save_max: int = 0,

    # retry controls
    max_retries: int = 2,
    retry_sleep: float = 5.0,
    log_trunc: int = 2000,

    no_bos: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(ckpt):
        print(f"[WARNING] ckpt not found: {ckpt}")
        return {"bleu": -1.0, "avg_ms_per_sent": -1.0, "tok_per_sec": -1.0}

    data_flag = choose_flag(infer_py, "--data_jsonl", "--dev_jsonl")

    cmd = [
        "python", infer_py,
        "--ckpt", ckpt,
        data_flag, data_jsonl,
        "--direction", direction,
        "--tokenizer_meta", tokenizer_meta,
        "--hf_name_or_dir", hf_name_or_dir,
        "--device", device,
        "--decode", decode,
        "--beam_size", str(int(beam_size)),
        "--alpha", str(float(alpha)),
        "--len_norm", str(len_norm),
        "--max_len", str(int(max_len)),
        "--min_len", str(int(min_len)),
        "--infer_batch_size", str(int(infer_batch_size)),
        "--out_path", out_path,
    ]

    # Optional flags only if the inference script supports them
    if supports_flag(infer_py, "--compute_bleu"):
        cmd.append("--compute_bleu")
    if local_files_only and supports_flag(infer_py, "--local_files_only"):
        cmd.append("--local_files_only")
    if hf_cache_dir and supports_flag(infer_py, "--hf_cache_dir"):
        cmd += ["--hf_cache_dir", hf_cache_dir]
    if long_src_len and long_src_len > 0 and supports_flag(infer_py, "--long_src_len"):
        cmd += ["--long_src_len", str(int(long_src_len))]
    if cov_beta is not None and supports_flag(infer_py, "--cov_beta"):
        cmd += ["--cov_beta", str(float(cov_beta))]
    if no_bos and supports_flag(infer_py, "--no_bos"):
        cmd.append("--no_bos")
    if save_details and supports_flag(infer_py, "--save_details"):
        cmd += ["--save_details", save_details]
        if save_max and save_max > 0 and supports_flag(infer_py, "--save_max"):
            cmd += ["--save_max", str(int(save_max))]

    out = read_out(cmd, env=env, max_retries=max_retries, retry_sleep=retry_sleep, log_trunc=log_trunc)
    if out is None:
        print("[WARNING] Inference failed, returning default values")
        return {"bleu": -1.0, "avg_ms_per_sent": -1.0, "tok_per_sec": -1.0}

    try:
        bleu = parse_bleu(out)
        latency = parse_latency(out)
        return {"bleu": bleu, **latency}
    except Exception as e:
        print(f"[WARNING] Failed to parse inference output: {e}")
        return {"bleu": -1.0, "avg_ms_per_sent": -1.0, "tok_per_sec": -1.0}


# --------------------------
# main
# --------------------------
def main():
    ap = argparse.ArgumentParser()

    # processed ids
    ap.add_argument("--train_jsonl", required=True, help="processed_train.jsonl (ids)")
    ap.add_argument("--dev_jsonl", required=True, help="processed_valid.jsonl (ids)")
    ap.add_argument("--test_jsonl", default="", help="processed_test.jsonl (ids)")

    ap.add_argument("--direction", default="zh2en", choices=["zh2en", "en2zh"])

    # HF meta
    ap.add_argument("--tokenizer_meta", required=True, help="tokenizer_meta.json (HF ids)")
    ap.add_argument("--hf_name_or_dir", required=True, help="HF tokenizer repo id or local dir")
    ap.add_argument("--hf_cache_dir", default="", help="Optional cache dir (ACP recommended)")
    ap.add_argument("--local_files_only", action="store_true")

    # scripts
    ap.add_argument("--train_py", default="", help="Training script path (default: train_rnn_nmt_hf.py in this dir)")
    ap.add_argument("--infer_py", default="", help="Inference script path (default: inference_rnn_hf_aligned.py in this dir)")

    # output
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--device", default="cuda")

    # common training controls
    ap.add_argument("--max_steps", type=int, default=2000000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    ap.add_argument("--plateau_patience", type=int, default=3)
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-7)

    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_accum_steps", type=int, default=1)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--no_bos", action="store_true")

    # model baseline
    ap.add_argument("--base_rnn_type", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--base_alignment", default="dot", choices=["dot", "multiplicative", "additive"])
    ap.add_argument("--training_policy", default="teacher_forcing", choices=["teacher_forcing", "free_running"])

    ap.add_argument("--base_emb_size", type=int, default=256)
    ap.add_argument("--base_hidden_size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--cov_beta", type=float, default=0.0)

    # baseline train hypers
    ap.add_argument("--abl_batch_size", type=int, default=64)
    ap.add_argument("--abl_lr", type=float, default=1e-3)
    ap.add_argument("--lowres_batch_size", type=int, default=64)
    ap.add_argument("--lowres_lr", type=float, default=1e-3)

    # experiment grids
    ap.add_argument("--abl_alignments", default="dot,multiplicative,additive")
    ap.add_argument("--abl_rnn_types", default="lstm")  # e.g., "lstm,gru"
    ap.add_argument("--batch_sizes", default="64,128")
    ap.add_argument("--lrs", default="1e-3,5e-4")
    ap.add_argument("--scales", default="small,base,large")
    ap.add_argument("--low_resource_sizes", default="1000")

    # inference knobs
    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--len_norm", choices=["gnmt", "length", "none"], default="gnmt")
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--min_len", type=int, default=0)
    ap.add_argument("--infer_batch_size", type=int, default=0)  # 0 means auto (greedy->64, beam->1)
    ap.add_argument("--long_src_len", type=int, default=50)

    # retry knobs
    ap.add_argument("--infer_max_retries", type=int, default=2)
    ap.add_argument("--infer_retry_sleep", type=float, default=5.0)
    ap.add_argument("--log_trunc", type=int, default=2000)

    # debug
    ap.add_argument("--save_details", action="store_true", help="Save per-sample details jsonl for dev/test (debug).")
    ap.add_argument("--save_max", type=int, default=200)

    ap.add_argument("--skip_train_if_ckpt_exists", action="store_true",
                    help="If set, skip training when <output_root>/train/<exp>/best.pt already exists.")

    args = ap.parse_args()

    # resolve script paths relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_py = args.train_py.strip() or os.path.join(script_dir, "train_rnn_nmt_hf.py")
    infer_py = args.infer_py.strip() or os.path.join(script_dir, "inference_rnn_hf_aligned.py")

    if not os.path.isfile(train_py):
        raise FileNotFoundError(f"Cannot find train script: {train_py}")
    if not os.path.isfile(infer_py):
        raise FileNotFoundError(f"Cannot find inference script: {infer_py}")

    env = _make_env(args.hf_cache_dir, bool(args.local_files_only))

    # Train outputs shared across decode runs
    train_root = os.path.join(args.output_root, "train")
    os.makedirs(train_root, exist_ok=True)

    # Decode-specific outputs
    decode_root = os.path.join(args.output_root, str(args.decode))
    os.makedirs(decode_root, exist_ok=True)

    # auto infer batch size
    auto_infer_bs = int(args.infer_batch_size)
    if auto_infer_bs <= 0:
        auto_infer_bs = 1 if args.decode == "beam" else 64

    infer_common = dict(
        direction=args.direction,
        tokenizer_meta=args.tokenizer_meta,
        hf_name_or_dir=args.hf_name_or_dir,
        device=args.device,
        local_files_only=bool(args.local_files_only),
        hf_cache_dir=args.hf_cache_dir,
        env=env,
        decode=args.decode,
        beam_size=args.beam_size,
        alpha=args.alpha,
        len_norm=args.len_norm,
        max_len=args.max_len,
        min_len=args.min_len,
        infer_batch_size=auto_infer_bs,
        cov_beta=None,  # leave to inference script / ckpt unless overridden
        max_retries=args.infer_max_retries,
        retry_sleep=args.infer_retry_sleep,
        log_trunc=args.log_trunc,
        no_bos=bool(args.no_bos),
    )

    def infer_config_record() -> Dict[str, Any]:
        return {
            "decode": args.decode,
            "beam_size": int(args.beam_size),
            "alpha": float(args.alpha),
            "len_norm": str(args.len_norm),
            "max_len": int(args.max_len),
            "min_len": int(args.min_len),
            "infer_batch_size": int(auto_infer_bs),
            "hf_name_or_dir": args.hf_name_or_dir,
            "local_files_only": bool(args.local_files_only),
            "hf_cache_dir": args.hf_cache_dir if args.hf_cache_dir else "",
        }

    results: List[Dict[str, Any]] = []

    def save():
        with open(os.path.join(decode_root, "results_transformer.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        if FAILURES:
            with open(os.path.join(decode_root, "failures_transformer.json"), "w", encoding="utf-8") as f:
                json.dump(FAILURES, f, ensure_ascii=False, indent=2)

    # scale mapping
    def scale_cfg(scale: str) -> Tuple[int, int]:
        scale = scale.lower()
        if scale == "small":
            return (128, 256)
        if scale == "base":
            return (int(args.base_emb_size), int(args.base_hidden_size))
        if scale == "large":
            return (512, 1024)
        raise ValueError("scale must be small|base|large")

    # Build train cmd with optional flags only if supported by train script
    def build_train_cmd(
        out_dir: str,
        rnn_type: str,
        alignment: str,
        emb_size: int,
        hidden_size: int,
        bs: int,
        lr: float,
        max_train_samples: int = 0,
    ) -> List[str]:
        dev_flag = choose_flag(train_py, "--dev_jsonl", "--valid_jsonl")
        cmd = [
            "python", train_py,
            "--train_jsonl", args.train_jsonl,
            dev_flag, args.dev_jsonl,
            "--direction", args.direction,
            "--tokenizer_meta", args.tokenizer_meta,
            "--output_dir", out_dir,
            "--save_name", "best.pt",
            "--rnn_type", rnn_type,
            "--alignment", alignment,
            "--training_policy", args.training_policy,
            "--emb_size", str(int(emb_size)),
            "--hidden_size", str(int(hidden_size)),
            "--num_layers", "2",
            "--dropout", str(float(args.dropout)),
            "--batch_size", str(int(bs)),
            "--lr", str(float(lr)),
            "--seed", str(int(args.seed)),
            "--device", args.device,
        ]

        # Optional data / model args
        if args.test_jsonl and supports_flag(train_py, "--test_jsonl"):
            cmd += ["--test_jsonl", args.test_jsonl]
        if supports_flag(train_py, "--hf_name_or_dir"):
            cmd += ["--hf_name_or_dir", args.hf_name_or_dir]
        if args.local_files_only and supports_flag(train_py, "--local_files_only"):
            cmd.append("--local_files_only")
        if args.no_bos and supports_flag(train_py, "--no_bos"):
            cmd.append("--no_bos")
        if args.fp16 and supports_flag(train_py, "--fp16"):
            cmd.append("--fp16")

        # Optional training controls (only if supported)
        opt_pairs = [
            ("--cov_beta", str(float(args.cov_beta))),
            ("--weight_decay", str(float(args.weight_decay))),
            ("--warmup_steps", str(int(args.warmup_steps))),
            ("--max_steps", str(int(args.max_steps))),
            ("--grad_clip", str(float(args.grad_clip))),
            ("--label_smoothing", str(float(args.label_smoothing))),
            ("--grad_accum_steps", str(int(args.grad_accum_steps))),
            ("--eval_every", str(int(args.eval_every))),
            ("--early_stop_patience", str(int(args.early_stop_patience))),
            ("--min_delta", str(float(args.min_delta))),
            ("--plateau_patience", str(int(args.plateau_patience))),
            ("--plateau_factor", str(float(args.plateau_factor))),
            ("--min_lr", str(float(args.min_lr))),
            ("--num_workers", str(int(args.num_workers))),
        ]
        for flag, val in opt_pairs:
            if supports_flag(train_py, flag):
                cmd += [flag, val]

        if max_train_samples and max_train_samples > 0 and supports_flag(train_py, "--max_train_samples"):
            cmd += ["--max_train_samples", str(int(max_train_samples))]

        return cmd

    # ----------------------------
    # 1) Ablation: alignments (and optional rnn_type)
    # ----------------------------
    abl_alignments = _parse_str_list(args.abl_alignments)
    abl_rnn_types = _parse_str_list(args.abl_rnn_types)

    for rnn_type in abl_rnn_types:
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"Invalid rnn_type in --abl_rnn_types: {rnn_type}")
        for alignment in abl_alignments:
            if alignment not in ("dot", "multiplicative", "additive"):
                raise ValueError(f"Invalid alignment in --abl_alignments: {alignment}")

            exp = f"abl_rnn_type={rnn_type}_align={alignment}"
            train_dir = os.path.join(train_root, exp)
            decode_dir = os.path.join(decode_root, exp)
            ckpt = os.path.join(train_dir, "best.pt")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(decode_dir, exist_ok=True)

            if not (args.skip_train_if_ckpt_exists and os.path.isfile(ckpt)):
                run_retry(build_train_cmd(
                    out_dir=train_dir,
                    rnn_type=rnn_type,
                    alignment=alignment,
                    emb_size=int(args.base_emb_size),
                    hidden_size=int(args.base_hidden_size),
                    bs=int(args.abl_batch_size),
                    lr=float(args.abl_lr),
                ), env=env)
            else:
                print(f"[SKIP] ckpt exists, skip training: {ckpt}")

            dev_all = infer_rnn_hf(
                infer_py=infer_py,
                ckpt=ckpt,
                data_jsonl=args.dev_jsonl,
                long_src_len=0,
                out_path=os.path.join(decode_dir, f"pred_dev_all_{args.decode}.txt"),
                save_details=(os.path.join(decode_dir, f"details_dev_all_{args.decode}.jsonl") if args.save_details else ""),
                save_max=int(args.save_max),
                **infer_common,
            )
            dev_long = infer_rnn_hf(
                infer_py=infer_py,
                ckpt=ckpt,
                data_jsonl=args.dev_jsonl,
                long_src_len=int(args.long_src_len),
                out_path=os.path.join(decode_dir, f"pred_dev_long_{args.decode}.txt"),
                save_details=(os.path.join(decode_dir, f"details_dev_long_{args.decode}.jsonl") if args.save_details else ""),
                save_max=int(args.save_max),
                **infer_common,
            )

            rec: Dict[str, Any] = {
                "exp": exp,
                "type": "ablation",
                "model_type": "scratch",
                "pos_type": "",
                "norm_type": "",
                "rnn_type": rnn_type,
                "alignment": alignment,
                "training_policy": args.training_policy,
                "emb_size": int(args.base_emb_size),
                "hidden_size": int(args.base_hidden_size),
                "train_batch_size": int(args.abl_batch_size),
                "train_lr": float(args.abl_lr),
                "train_dir": train_dir,
                "decode_dir": decode_dir,
                "dev_bleu": float(dev_all["bleu"]),
                "dev_long_bleu": float(dev_long["bleu"]),
                "dev_latency_ms": float(dev_all["avg_ms_per_sent"]),
                "dev_tok_per_sec": float(dev_all["tok_per_sec"]),
                "train_metrics": load_train_metrics(train_dir),
                "infer_config": infer_config_record(),
            }

            if args.test_jsonl:
                test_all = infer_rnn_hf(
                    infer_py=infer_py,
                    ckpt=ckpt,
                    data_jsonl=args.test_jsonl,
                    long_src_len=0,
                    out_path=os.path.join(decode_dir, f"pred_test_all_{args.decode}.txt"),
                    save_details=(os.path.join(decode_dir, f"details_test_all_{args.decode}.jsonl") if args.save_details else ""),
                    save_max=int(args.save_max),
                    **infer_common,
                )
                test_long = infer_rnn_hf(
                    infer_py=infer_py,
                    ckpt=ckpt,
                    data_jsonl=args.test_jsonl,
                    long_src_len=int(args.long_src_len),
                    out_path=os.path.join(decode_dir, f"pred_test_long_{args.decode}.txt"),
                    save_details=(os.path.join(decode_dir, f"details_test_long_{args.decode}.jsonl") if args.save_details else ""),
                    save_max=int(args.save_max),
                    **infer_common,
                )
                rec["test_bleu"] = float(test_all["bleu"])
                rec["test_long_bleu"] = float(test_long["bleu"])

            results.append(rec)
            save()

    # ----------------------------
    # 2) Sensitivity: (scale, batch_size, lr)
    # ----------------------------
    bs_list = _parse_int_list(args.batch_sizes, "batch_sizes")
    lr_list = _parse_float_list(args.lrs, "lrs")
    scale_list = _parse_str_list(args.scales)

    for scale in scale_list:
        emb_size, hidden_size = scale_cfg(scale)
        for bs in bs_list:
            for lr in lr_list:
                exp = f"sens_rnn_scale={scale}_bs={bs}_lr={lr}"
                train_dir = os.path.join(train_root, exp)
                decode_dir = os.path.join(decode_root, exp)
                ckpt = os.path.join(train_dir, "best.pt")
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(decode_dir, exist_ok=True)

                if not (args.skip_train_if_ckpt_exists and os.path.isfile(ckpt)):
                    run_retry(build_train_cmd(
                        out_dir=train_dir,
                        rnn_type=args.base_rnn_type,
                        alignment=args.base_alignment,
                        emb_size=int(emb_size),
                        hidden_size=int(hidden_size),
                        bs=int(bs),
                        lr=float(lr),
                    ), env=env)
                else:
                    print(f"[SKIP] ckpt exists, skip training: {ckpt}")

                dev_all = infer_rnn_hf(
                    infer_py=infer_py,
                    ckpt=ckpt,
                    data_jsonl=args.dev_jsonl,
                    long_src_len=0,
                    out_path=os.path.join(decode_dir, f"pred_dev_all_{args.decode}.txt"),
                    save_details="",
                    save_max=int(args.save_max),
                    **infer_common,
                )
                dev_long = infer_rnn_hf(
                    infer_py=infer_py,
                    ckpt=ckpt,
                    data_jsonl=args.dev_jsonl,
                    long_src_len=int(args.long_src_len),
                    out_path=os.path.join(decode_dir, f"pred_dev_long_{args.decode}.txt"),
                    save_details="",
                    save_max=int(args.save_max),
                    **infer_common,
                )

                results.append({
                    "exp": exp,
                    "type": "sensitivity",
                    "model_type": "scratch",
                    "pos_type": "",
                    "norm_type": "",
                    "scale": scale,
                    "rnn_type": args.base_rnn_type,
                    "alignment": args.base_alignment,
                    "training_policy": args.training_policy,
                    "emb_size": int(emb_size),
                    "hidden_size": int(hidden_size),
                    "batch_size": int(bs),
                    "lr": float(lr),
                    "train_dir": train_dir,
                    "decode_dir": decode_dir,
                    "dev_bleu": float(dev_all["bleu"]),
                    "dev_long_bleu": float(dev_long["bleu"]),
                    "train_metrics": load_train_metrics(train_dir),
                    "infer_config": infer_config_record(),
                })
                save()

    # ----------------------------
    # 3) Low-resource: subset size
    # ----------------------------
    low_sizes = _parse_int_list(args.low_resource_sizes, "low_resource_sizes")
    for n_int in low_sizes:
        exp = f"lowres_rnn_n={n_int}"
        train_dir = os.path.join(train_root, exp)
        decode_dir = os.path.join(decode_root, exp)
        ckpt = os.path.join(train_dir, "best.pt")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(decode_dir, exist_ok=True)

        if not (args.skip_train_if_ckpt_exists and os.path.isfile(ckpt)):
            run_retry(build_train_cmd(
                out_dir=train_dir,
                rnn_type=args.base_rnn_type,
                alignment=args.base_alignment,
                emb_size=int(args.base_emb_size),
                hidden_size=int(args.base_hidden_size),
                bs=int(args.lowres_batch_size),
                lr=float(args.lowres_lr),
                max_train_samples=int(n_int),
            ), env=env)
        else:
            print(f"[SKIP] ckpt exists, skip training: {ckpt}")

        dev_all = infer_rnn_hf(
            infer_py=infer_py,
            ckpt=ckpt,
            data_jsonl=args.dev_jsonl,
            long_src_len=0,
            out_path=os.path.join(decode_dir, f"pred_dev_all_{args.decode}.txt"),
            save_details="",
            save_max=int(args.save_max),
            **infer_common,
        )
        dev_long = infer_rnn_hf(
            infer_py=infer_py,
            ckpt=ckpt,
            data_jsonl=args.dev_jsonl,
            long_src_len=int(args.long_src_len),
            out_path=os.path.join(decode_dir, f"pred_dev_long_{args.decode}.txt"),
            save_details="",
            save_max=int(args.save_max),
            **infer_common,
        )

        results.append({
            "exp": exp,
            "type": "low_resource",
            "model_type": "scratch",
            "pos_type": "",
            "norm_type": "",
            "train_subset_n": int(n_int),
            "train_batch_size": int(args.lowres_batch_size),
            "train_lr": float(args.lowres_lr),
            "train_dir": train_dir,
            "decode_dir": decode_dir,
            "dev_bleu": float(dev_all["bleu"]),
            "dev_long_bleu": float(dev_long["bleu"]),
            "train_metrics": load_train_metrics(train_dir),
            "infer_config": infer_config_record(),
        })
        save()

    print("\n[Done] results saved:", os.path.join(decode_root, "results_transformer.json"))
    if FAILURES:
        print("[WARN] failures saved:", os.path.join(decode_root, "failures_transformer.json"))


if __name__ == "__main__":
    main()
