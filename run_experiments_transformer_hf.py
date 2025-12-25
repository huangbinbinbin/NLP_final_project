#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import argparse
import subprocess
from typing import List, Dict, Any, Tuple, Optional, Set

FAILURES: List[Dict[str, Any]] = []
RUN_MAX_RETRIES = 2
RUN_RETRY_SLEEP = 5.0


# --------------------------
# Robust subprocess helpers
# --------------------------
def run_retry(cmd: List[str], max_retries=None, retry_sleep=None):
    if max_retries is None:
        max_retries = RUN_MAX_RETRIES
    if retry_sleep is None:
        retry_sleep = RUN_RETRY_SLEEP

    for attempt in range(int(max_retries) + 1):
        print(f"\n[RUN attempt {attempt+1}/{int(max_retries)+1}] " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            FAILURES.append({"kind": "run", "cmd": " ".join(cmd), "returncode": int(e.returncode)})
            print(f"[ERROR] Command failed with exit code {e.returncode}")
            if attempt < int(max_retries):
                print(f"[RETRY] waiting {float(retry_sleep):.1f}s...")
                time.sleep(float(retry_sleep))
            else:
                raise


def read_out(cmd: List[str], max_retries=None, retry_sleep=None) -> Optional[str]:
    """Run a command and capture stdout+stderr. Returns None on failure."""
    if max_retries is None:
        max_retries = RUN_MAX_RETRIES
    if retry_sleep is None:
        retry_sleep = RUN_RETRY_SLEEP

    for attempt in range(int(max_retries) + 1):
        print(f"\n[RUN attempt {attempt+1}/{int(max_retries)+1}] " + " ".join(cmd))
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            print(out[:800] + ("..." if len(out) > 800 else ""))
            return out
        except subprocess.CalledProcessError as e:
            FAILURES.append({
                "kind": "read_out",
                "cmd": " ".join(cmd),
                "returncode": int(e.returncode),
                "output": (e.output[:800] if e.output else ""),
            })
            print(f"[ERROR] Command failed with exit code {e.returncode}")
            if e.output:
                print("[OUTPUT]\n" + e.output[:800])
            if attempt < int(max_retries):
                print(f"[RETRY] waiting {float(retry_sleep):.1f}s...")
                time.sleep(float(retry_sleep))
            else:
                return None
    return None


run = run_retry


# --------------------------
# Help/args compatibility layer
# --------------------------
_FLAG_RE = re.compile(r"(?<!\w)(--[a-zA-Z0-9][a-zA-Z0-9_-]*)")

def get_supported_flags(py_script: str) -> Set[str]:
    """
    Parse `python script.py -h` output and extract all flags like --xxx.
    If -h fails, return empty set (then we will pass minimal args).
    """
    out = read_out([sys.executable, py_script, "-h"], max_retries=0)
    if out is None:
        return set()
    return set(_FLAG_RE.findall(out))


def add_arg(cmd: List[str], supported: Set[str], name: str, value: Optional[str] = None):
    """Add --flag value only if supported by the target script."""
    if name not in supported:
        return
    cmd.append(name)
    if value is not None:
        cmd.append(str(value))


def add_flag(cmd: List[str], supported: Set[str], name: str, enabled: bool):
    """Add boolean flag only if supported and enabled."""
    if enabled and (name in supported):
        cmd.append(name)


# --------------------------
# Metrics parsing
# --------------------------
def parse_bleu(out: str) -> float:
    # Primary: [BLEU] 12.34
    m = re.search(r"\[BLEU\]\s*([0-9]+(?:\.[0-9]+)?)", out)
    if m:
        return float(m.group(1))

    # Fallback patterns:
    m2 = re.search(r"\bBLEU\b[^0-9]*([0-9]+(?:\.[0-9]+)?)", out, flags=re.IGNORECASE)
    if m2:
        return float(m2.group(1))

    raise RuntimeError("Cannot parse BLEU from inference output.")


def parse_latency(out: str) -> Dict[str, float]:
    m = re.search(
        r"\[LATENCY\]\s*avg_ms_per_sent=([0-9]+(?:\.[0-9]+)?)\s+tok_per_sec=([0-9]+(?:\.[0-9]+)?)",
        out,
    )
    if not m:
        return {"avg_ms_per_sent": -1.0, "tok_per_sec": -1.0}
    return {"avg_ms_per_sent": float(m.group(1)), "tok_per_sec": float(m.group(2))}


def load_train_metrics(out_dir: str) -> Dict[str, Any]:
    p = os.path.join(out_dir, "train_metrics.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# Low-resource subset fallback (when --max_train_samples not supported)
# --------------------------
def write_subset_jsonl(in_path: str, out_path: str, n: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kept = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                fout.write(line)
                kept += 1
                if kept >= n:
                    break
    print(f"[LOWRES] wrote subset: {out_path} (n={kept})")


# --------------------------
# Inference wrapper (HF tokenizer version)
# --------------------------
def infer(
    infer_py: str,
    ckpt: str,
    data_jsonl: str,
    tokenizer_meta: str,
    hf_name_or_dir: str,
    device: str,
    local_files_only: bool,
    decode: str = "greedy",
    beam_size: int = 5,
    alpha: float = 0.6,
    len_norm: str = "gnmt",
    max_len: int = 80,
    min_len: int = 0,
    infer_batch_size: int = 32,
    long_src_len: int = 0,
    t5_length_penalty: float = 1.0,
    max_retries: int = 2,
    retry_sleep: float = 5.0,
) -> Dict[str, Any]:

    sup = get_supported_flags(infer_py)

    cmd = [sys.executable, infer_py]
    add_arg(cmd, sup, "--ckpt", ckpt)
    add_arg(cmd, sup, "--data_jsonl", data_jsonl)
    add_arg(cmd, sup, "--tokenizer_meta", tokenizer_meta)
    add_arg(cmd, sup, "--hf_name_or_dir", hf_name_or_dir)
    add_flag(cmd, sup, "--local_files_only", local_files_only)

    add_arg(cmd, sup, "--device", device)
    add_arg(cmd, sup, "--decode", decode)
    add_arg(cmd, sup, "--beam_size", str(beam_size))
    add_arg(cmd, sup, "--alpha", str(alpha))
    add_arg(cmd, sup, "--len_norm", len_norm)
    add_arg(cmd, sup, "--max_len", str(max_len))
    add_arg(cmd, sup, "--min_len", str(min_len))
    add_arg(cmd, sup, "--infer_batch_size", str(infer_batch_size))
    add_arg(cmd, sup, "--t5_length_penalty", str(t5_length_penalty))

    if long_src_len and long_src_len > 0:
        add_arg(cmd, sup, "--long_src_len", str(long_src_len))

    out = read_out(cmd, max_retries=max_retries, retry_sleep=retry_sleep)
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


def main():
    ap = argparse.ArgumentParser()

    # processed ids jsonl (HF tokenizer already applied)
    ap.add_argument("--train_jsonl", required=True, help="processed_train.jsonl (ids)")
    ap.add_argument("--dev_jsonl", required=True, help="processed_valid.jsonl (ids)")
    ap.add_argument("--test_jsonl", default="", help="processed_test.jsonl (ids)")

    # HF tokenizer meta & name (for decode / detok / special ids)
    ap.add_argument("--tokenizer_meta", required=True, help="tokenizer_meta.json")
    ap.add_argument("--hf_name_or_dir", required=True, help="e.g., Helsinki-NLP/opus-mt-zh-en")
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--output_root", required=True)
    ap.add_argument("--device", default="cuda")

    # training schedule
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    # model base
    ap.add_argument("--base_layers", type=int, default=4)
    ap.add_argument("--base_d_model", type=int, default=256)
    ap.add_argument("--base_heads", type=int, default=4)
    ap.add_argument("--base_d_ff", type=int, default=1024)

    # optimization knobs (baseline)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=800)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.01)
    ap.add_argument("--no_bos", action="store_true")
    ap.add_argument("--grad_accum_steps", type=int, default=1)

    # sensitivity sweep knobs
    ap.add_argument("--batch_sizes", default="32,64")
    ap.add_argument("--lrs", default="3e-4,1e-4")
    ap.add_argument("--scales", default="small,base,large")

    ap.add_argument("--long_src_len", type=int, default=50)
    ap.add_argument("--low_resource_sizes", default="1000")

    # fixed-baseline training knobs for ablation/lowres
    ap.add_argument("--abl_batch_size", type=int, default=32)
    ap.add_argument("--abl_lr", type=float, default=3e-4)
    ap.add_argument("--lowres_batch_size", type=int, default=32)
    ap.add_argument("--lowres_lr", type=float, default=3e-4)

    # decode / inference knobs
    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--len_norm", choices=["gnmt", "length", "none"], default="gnmt")
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--min_len", type=int, default=0)
    ap.add_argument("--infer_batch_size", type=int, default=0, help="0=auto: greedy->32, beam->1")

    ap.add_argument("--t5_length_penalty", type=float, default=1.0)

    # script paths (NEW)
    ap.add_argument(
        "--train_py",
        default="/data/250010105/NLP_final_project/train_transformer_nmt_hfc_eos.py",
        help="training script path",
    )
    ap.add_argument(
        "--infer_py",
        default="/data/250010105/NLP_final_project/inference_transformer_hf_eos.py",
        help="inference script path",
    )

    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    # auto infer batch size
    auto_infer_bs = args.infer_batch_size
    if auto_infer_bs <= 0:
        auto_infer_bs = 1 if args.decode == "beam" else 32

    infer_common = dict(
        decode=args.decode,
        beam_size=args.beam_size,
        alpha=args.alpha,
        len_norm=args.len_norm,
        max_len=args.max_len,
        min_len=args.min_len,
        infer_batch_size=auto_infer_bs,
        t5_length_penalty=args.t5_length_penalty,
    )

    def infer_config_record() -> Dict[str, Any]:
        return {
            "decode": args.decode,
            "beam_size": args.beam_size,
            "alpha": args.alpha,
            "len_norm": args.len_norm,
            "max_len": args.max_len,
            "min_len": args.min_len,
            "infer_batch_size": auto_infer_bs,
            "t5_length_penalty": args.t5_length_penalty,
            "hf_name_or_dir": args.hf_name_or_dir,
            "local_files_only": bool(args.local_files_only),
        }

    results: List[Dict[str, Any]] = []

    def save():
        with open(os.path.join(args.output_root, "results_transformer.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Cache supported flags for training script once
    train_sup = get_supported_flags(args.train_py)

    def build_train_cmd(
        out_dir: str,
        pos_type: str,
        norm_type: str,
        batch_size: int,
        lr: float,
        train_jsonl: str,
        max_train_samples: Optional[int] = None,
    ) -> List[str]:
        cmd = [sys.executable, args.train_py]

        add_arg(cmd, train_sup, "--model_type", "scratch")
        add_arg(cmd, train_sup, "--train_jsonl", train_jsonl)
        add_arg(cmd, train_sup, "--dev_jsonl", args.dev_jsonl)
        add_arg(cmd, train_sup, "--tokenizer_meta", args.tokenizer_meta)

        add_flag(cmd, train_sup, "--no_bos", args.no_bos)

        add_arg(cmd, train_sup, "--output_dir", out_dir)
        add_arg(cmd, train_sup, "--save_name", "best.pt")

        add_arg(cmd, train_sup, "--num_layers", str(args.base_layers))
        add_arg(cmd, train_sup, "--d_model", str(args.base_d_model))
        add_arg(cmd, train_sup, "--n_heads", str(args.base_heads))
        add_arg(cmd, train_sup, "--d_ff", str(args.base_d_ff))

        add_arg(cmd, train_sup, "--dropout", str(args.dropout))
        add_arg(cmd, train_sup, "--pos_type", pos_type)
        add_arg(cmd, train_sup, "--norm_type", norm_type)

        add_arg(cmd, train_sup, "--batch_size", str(batch_size))
        add_arg(cmd, train_sup, "--lr", str(lr))

        add_arg(cmd, train_sup, "--warmup_steps", str(args.warmup_steps))
        add_arg(cmd, train_sup, "--grad_clip", str(args.grad_clip))
        add_arg(cmd, train_sup, "--label_smoothing", str(args.label_smoothing))
        add_arg(cmd, train_sup, "--grad_accum_steps", str(args.grad_accum_steps))

        add_arg(cmd, train_sup, "--max_steps", str(args.max_steps))
        add_arg(cmd, train_sup, "--eval_every", str(args.eval_every))
        add_arg(cmd, train_sup, "--early_stop_patience", str(args.patience))
        add_arg(cmd, train_sup, "--min_delta", str(args.min_delta))
        add_arg(cmd, train_sup, "--device", args.device)

        # low-resource: prefer native flag, else caller will pass subset file
        if max_train_samples is not None:
            add_arg(cmd, train_sup, "--max_train_samples", str(max_train_samples))

        return cmd

    # 1) ablation grid
    ablations = [
        ("absolute", "layernorm"),
        ("absolute", "rmsnorm"),
        ("relative", "layernorm"),
        ("relative", "rmsnorm"),
    ]

    for pos_type, norm_type in ablations:
        exp = f"abl_scratch_pos={pos_type}_norm={norm_type}"
        out_dir = os.path.join(args.output_root, exp)
        ckpt = os.path.join(out_dir, "best.pt")

        cmd = build_train_cmd(
            out_dir=out_dir,
            pos_type=pos_type,
            norm_type=norm_type,
            batch_size=args.abl_batch_size,
            lr=args.abl_lr,
            train_jsonl=args.train_jsonl,
        )
        run(cmd)

        dev_all = infer(
            args.infer_py, ckpt, args.dev_jsonl,
            args.tokenizer_meta, args.hf_name_or_dir,
            args.device, args.local_files_only,
            long_src_len=0, **infer_common
        )
        dev_long = infer(
            args.infer_py, ckpt, args.dev_jsonl,
            args.tokenizer_meta, args.hf_name_or_dir,
            args.device, args.local_files_only,
            long_src_len=args.long_src_len, **infer_common
        )

        rec: Dict[str, Any] = {
            "exp": exp,
            "type": "ablation",
            "model_type": "scratch",
            "pos_type": pos_type,
            "norm_type": norm_type,
            "train_batch_size": args.abl_batch_size,
            "train_lr": args.abl_lr,
            "dev_bleu": dev_all["bleu"],
            "dev_long_bleu": dev_long["bleu"],
            "dev_latency_ms": dev_all["avg_ms_per_sent"],
            "dev_tok_per_sec": dev_all["tok_per_sec"],
            "train_metrics": load_train_metrics(out_dir),
            "infer_config": infer_config_record(),
        }

        if args.test_jsonl:
            test_all = infer(
                args.infer_py, ckpt, args.test_jsonl,
                args.tokenizer_meta, args.hf_name_or_dir,
                args.device, args.local_files_only,
                long_src_len=0, **infer_common
            )
            test_long = infer(
                args.infer_py, ckpt, args.test_jsonl,
                args.tokenizer_meta, args.hf_name_or_dir,
                args.device, args.local_files_only,
                long_src_len=args.long_src_len, **infer_common
            )
            rec["test_bleu"] = test_all["bleu"]
            rec["test_long_bleu"] = test_long["bleu"]

        results.append(rec)
        save()

    # 2) sensitivity (scratch)
    bs_list = [x.strip() for x in args.batch_sizes.split(",") if x.strip()]
    lr_list = [x.strip() for x in args.lrs.split(",") if x.strip()]
    scale_list = [x.strip() for x in args.scales.split(",") if x.strip()]

    def scale_cfg(scale: str) -> Tuple[int, int, int, int]:
        if scale == "small":
            return (2, 128, 4, 512)
        if scale == "base":
            return (args.base_layers, args.base_d_model, args.base_heads, args.base_d_ff)
        if scale == "large":
            return (6, 512, 8, 2048)
        raise ValueError("scale must be small|base|large")

    for scale in scale_list:
        L, D, H, FF = scale_cfg(scale)
        for bs in bs_list:
            for lr in lr_list:
                exp = f"sens_scratch_scale={scale}_bs={bs}_lr={lr}"
                out_dir = os.path.join(args.output_root, exp)
                ckpt = os.path.join(out_dir, "best.pt")

                # Build cmd, but override scaled model sizes if supported
                cmd = build_train_cmd(
                    out_dir=out_dir,
                    pos_type="absolute",
                    norm_type="layernorm",
                    batch_size=int(bs),
                    lr=float(lr),
                    train_jsonl=args.train_jsonl,
                )
                # replace model size flags if exist
                if "--num_layers" in train_sup:
                    # find index and replace value
                    for i in range(len(cmd) - 1):
                        if cmd[i] == "--num_layers":
                            cmd[i + 1] = str(L)
                        if cmd[i] == "--d_model":
                            cmd[i + 1] = str(D)
                        if cmd[i] == "--n_heads":
                            cmd[i + 1] = str(H)
                        if cmd[i] == "--d_ff":
                            cmd[i + 1] = str(FF)

                run(cmd)

                dev_all = infer(
                    args.infer_py, ckpt, args.dev_jsonl,
                    args.tokenizer_meta, args.hf_name_or_dir,
                    args.device, args.local_files_only,
                    long_src_len=0, **infer_common
                )
                dev_long = infer(
                    args.infer_py, ckpt, args.dev_jsonl,
                    args.tokenizer_meta, args.hf_name_or_dir,
                    args.device, args.local_files_only,
                    long_src_len=args.long_src_len, **infer_common
                )

                results.append({
                    "exp": exp,
                    "type": "sensitivity",
                    "model_type": "scratch",
                    "scale": scale,
                    "batch_size": int(bs),
                    "lr": float(lr),
                    "num_layers": L,
                    "d_model": D,
                    "n_heads": H,
                    "d_ff": FF,
                    "dev_bleu": dev_all["bleu"],
                    "dev_long_bleu": dev_long["bleu"],
                    "train_metrics": load_train_metrics(out_dir),
                    "infer_config": infer_config_record(),
                })
                save()

    # 3) low-resource (scratch)
    low_sizes = [x.strip() for x in args.low_resource_sizes.split(",") if x.strip()]
    for n in low_sizes:
        n_int = int(n)
        exp = f"lowres_scratch_n={n_int}"
        out_dir = os.path.join(args.output_root, exp)
        ckpt = os.path.join(out_dir, "best.pt")

        # If training script supports --max_train_samples, use it.
        # Otherwise, create a subset jsonl file under output dir.
        train_jsonl = args.train_jsonl
        max_train_samples = None
        if "--max_train_samples" in train_sup:
            max_train_samples = n_int
        else:
            subset_path = os.path.join(out_dir, f"subset_train_n={n_int}.jsonl")
            write_subset_jsonl(args.train_jsonl, subset_path, n_int)
            train_jsonl = subset_path

        cmd = build_train_cmd(
            out_dir=out_dir,
            pos_type="absolute",
            norm_type="layernorm",
            batch_size=args.lowres_batch_size,
            lr=args.lowres_lr,
            train_jsonl=train_jsonl,
            max_train_samples=max_train_samples,
        )
        run(cmd)

        dev_all = infer(
            args.infer_py, ckpt, args.dev_jsonl,
            args.tokenizer_meta, args.hf_name_or_dir,
            args.device, args.local_files_only,
            long_src_len=0, **infer_common
        )
        dev_long = infer(
            args.infer_py, ckpt, args.dev_jsonl,
            args.tokenizer_meta, args.hf_name_or_dir,
            args.device, args.local_files_only,
            long_src_len=args.long_src_len, **infer_common
        )

        results.append({
            "exp": exp,
            "type": "low_resource",
            "model_type": "scratch",
            "train_subset_n": n_int,
            "train_batch_size": args.lowres_batch_size,
            "train_lr": args.lowres_lr,
            "dev_bleu": dev_all["bleu"],
            "dev_long_bleu": dev_long["bleu"],
            "train_metrics": load_train_metrics(out_dir),
            "infer_config": infer_config_record(),
        })
        save()

    print("\n[Done] results saved:", os.path.join(args.output_root, "results_transformer.json"))


if __name__ == "__main__":
    main()
