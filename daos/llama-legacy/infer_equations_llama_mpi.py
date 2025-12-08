#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-node, multi-XPU MPI pipeline for equation inference using llama.cpp.
ROW-SHARDED VERSION (optimal scaling)

EACH RANK:
    - Binds to one Intel XPU via ONEAPI_DEVICE_SELECTOR
    - Iterates through all Parquet files
    - Processes ONLY rows i where (i % world_size == world_rank)
    - Writes outputs + checkpoints in per-rank files (no conflict)

This yields perfect load balancing for massive Parquet files.
"""

# =============================================================================
# MPI + XPU binding
# =============================================================================
from mpi4py import MPI
import os, socket, subprocess, json, re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from jsonschema import Draft7Validator

from math_prompt import PPROMPT_TEMPLATE_v2 as PROMPT_TEMPLATE, PURE_MATH_JSON_SCHEMA

comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

# Node-local rank (0..11)
node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
local_rank = node_comm.Get_rank()
hostname = socket.gethostname()

# One XPU per rank
os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{local_rank}"
print(
    f"[WORLD {world_rank}/{world_size} | NODE {hostname} | LOCAL_RANK={local_rank}] "
    f"Using ONEAPI_DEVICE_SELECTOR={os.environ['ONEAPI_DEVICE_SELECTOR']}",
    flush=True,
)

# =============================================================================
# Helper functions
# =============================================================================

def katex_hygiene(s: str) -> str:
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    return s.strip().rstrip(",")

def build_prompt(latex_raw: str) -> str:
    return PROMPT_TEMPLATE.format(
        schema=json.dumps(PURE_MATH_JSON_SCHEMA, ensure_ascii=False),
        latex_raw=latex_raw,
    )

def parse_strict_json(s: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}\s*$", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON found")
    obj = json.loads(m.group(0))
    Draft7Validator(PURE_MATH_JSON_SCHEMA).validate(obj)
    return obj

def load_parquets(src_dir: Path):
    return sorted([p for p in src_dir.iterdir() if p.suffix == ".parquet"])

# ---------------- llama.cpp invocation ----------------

def llama_cpp_infer(model_path: str, prompt: str, ctx: int = 2048, ngl: int = 80) -> str:
    cmd = [
        "/lus/flare/projects/FoundEpidem/siebenschuh/gpt-oss-120b-intel-max-gpu/scripts/llama.cpp/build/bin/llama-cli",
        "-m", model_path,
        "-p", prompt, # TODO
        "-c", str(ctx),
        "-ngl", str(ngl),
        "-n", "500",
        "-no-cnv",
        "--simple-io"
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return out
    except subprocess.CalledProcessError as e:
        # Attach full llama-cli output so we can see exactly why it failed
        msg = (
            f"llama-cli failed with return code {e.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{e.output}"
        )
        raise RuntimeError(msg)

# =============================================================================
# Row-sharded processing
# =============================================================================

def process_file_row_sharded(pq_path: Path, dst_dir: Path, model: str, ctx: int, ngl: int):
    df = pd.read_parquet(pq_path)

    # optional filtering
    if "LLM_prompt" in df.columns:
        df = df[df["LLM_prompt"].isin({"LLM", "API"})]

    total_rows = len(df)
    print(f"[RANK {world_rank}] {pq_path.name}: {total_rows} rows", flush=True)

    # Per-rank output file
    out_path = dst_dir / f"{pq_path.stem}__rank{world_rank:04d}.parquet"
    ckpt_path = dst_dir / "checkpoints" / f"{pq_path.stem}__rank{world_rank:04d}.ckpt.json"

    # Only rank 0 creates directories
    if world_rank == 0:
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Read checkpoint (last processed row)
    if ckpt_path.exists():
        try:
            start_offset = int(json.loads(ckpt_path.read_text()).get("row_offset", 0))
        except Exception:
            start_offset = 0
    else:
        start_offset = 0

    print(f"[RANK {world_rank}] Resume offset = {start_offset}", flush=True)

    # Row-sharding: each rank handles rows i where i % world_size == world_rank
    my_rows = list(range(world_rank + start_offset * world_size, total_rows, world_size))

    buffer = []
    flush_every = 200

    count = 0
    for global_row_index in my_rows:
        row = df.iloc[global_row_index]

        paper_id = row.get("paper_id")
        eq_id = row.get("equation_id")

        latex_raw = str(row.get("content_resolved", "") or "")
        latex_clean = katex_hygiene(latex_raw)
        prompt = build_prompt(latex_clean)

        try:
            raw_output = llama_cpp_infer(model, prompt, ctx=ctx, ngl=ngl)
            try:
                parsed = parse_strict_json(raw_output)
            except Exception:
                parsed = None
        except Exception as e:
            print(f"[RANK {world_rank}] ERROR @ row {global_row_index}: {e}", flush=True)
            continue

        rec = {
            "paper_id": paper_id,
            "equation_id": eq_id,
            "latex_raw": latex_raw,
            "latex_clean": latex_clean,
            "llm_raw_output": raw_output,
            "global_row": global_row_index,
        }

        if parsed:
            analysis = parsed["analysis"]
            equivs = parsed["equivalents"]
            rec.update({
                "math_keywords": json.dumps(analysis["math_keywords"], ensure_ascii=False),
                "math_sentence": analysis["math_sentence"],
                "katex": analysis["katex"],
                "equiv_form_1": json.dumps(equivs["equiv_form_1"], ensure_ascii=False),
                "equiv_form_2": json.dumps(equivs["equiv_form_2"], ensure_ascii=False),
                "output_json": json.dumps(parsed, ensure_ascii=False),
            })

        buffer.append(rec)
        count += 1

        if count % flush_every == 0:
            _flush(out_path, buffer)
            buffer.clear()
            _write_rank_ckpt(ckpt_path, count // flush_every)
            print(f"[RANK {world_rank}] Flushed {count} rows", flush=True)

    # Final flush
    _flush(out_path, buffer)
    _write_rank_ckpt(ckpt_path, len(my_rows))
    print(f"[RANK {world_rank}] DONE {pq_path.name}: processed {len(my_rows)} rows", flush=True)


# ---- small helpers used above ----

def _flush(out_path: Path, records: List[Dict[str, Any]]):
    if not records:
        return
    df_new = pd.DataFrame(records)
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        pd.concat([df_old, df_new], ignore_index=True).to_parquet(out_path, index=False)
    else:
        df_new.to_parquet(out_path, index=False)

def _write_rank_ckpt(ckpt: Path, offset: int):
    ckpt.write_text(json.dumps({"row_offset": offset}, ensure_ascii=False))

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ngl", type=int, default=80)
    args = ap.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    files = load_parquets(src_dir)
    print(f"[RANK {world_rank}] Found {len(files)} parquet files", flush=True)

    # Each rank processes *all files* (row-sharded)
    for pq in files:
        process_file_row_sharded(pq, dst_dir, args.model, args.ctx, args.ngl)

if __name__ == "__main__":
    main()

