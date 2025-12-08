#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-node, multi-XPU MPI pipeline for equation inference using llama-server.
ROW-SHARDED VERSION (optimal scaling)

EACH MPI RANK:
    - Has a node-local rank (0..11) via MPI.COMM_TYPE_SHARED
    - Talks to a llama-server bound to ONE Intel XPU:
          port = BASE_PORT + local_rank
      where BASE_PORT defaults to 18080 (must match the PBS script).
    - Iterates through all Parquet files
    - Processes ONLY rows i where (i % world_size == world_rank)
    - Writes outputs + checkpoints in per-rank files (no conflict)

This yields good load balancing over massive Parquet files while keeping
the 120B model loaded and "warm" inside llama-server.
"""

# =============================================================================
# MPI + imports
# =============================================================================
from mpi4py import MPI
import os
import time
import socket
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests
from jsonschema import Draft7Validator

from math_prompt import PROMPT_TEMPLATE_v2 as PROMPT_TEMPLATE
from math_prompt import PURE_MATH_JSON_SCHEMA

# -----------------------------------------------------------------------------
# MPI topology
# -----------------------------------------------------------------------------
comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

# Node-local communicator: ranks on the same node share COMM_TYPE_SHARED
node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
local_rank = node_comm.Get_rank()
hostname = socket.gethostname()

print(
    f"[WORLD {world_rank}/{world_size} | NODE {hostname} | LOCAL_RANK={local_rank}] "
    f"Client rank started.",
    flush=True,
)

# -----------------------------------------------------------------------------
# Server configuration (MUST match PBS script)
# -----------------------------------------------------------------------------
# Base port for llama-server on each node. PBS script uses BASE_PORT=18080.
SERVER_BASE_PORT = int(os.environ.get("LLAMA_SERVER_BASE_PORT", "18080"))

# Max tokens to generate per request (completions side)
MAX_GENERATION_TOKENS = int(os.environ.get("LLAMA_SERVER_MAX_TOKENS", "500"))

# HTTP timeout (seconds)
SERVER_TIMEOUT = float(os.environ.get("LLAMA_SERVER_TIMEOUT", "600"))

# =============================================================================
# Helper functions
# =============================================================================

def katex_hygiene(s: str) -> str:
    """Lightweight KaTeX hygiene."""
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    return s.strip().rstrip(",")


def build_prompt(latex_raw: str) -> str:
    """Fill your pure-math JSON prompt template."""
    return PROMPT_TEMPLATE.format(
        schema=json.dumps(PURE_MATH_JSON_SCHEMA, ensure_ascii=False),
        latex_raw=latex_raw,
    )


def parse_strict_json(s: str) -> Dict[str, Any]:
    """
    Extract the final JSON object from the model's message *content* and
    validate it against PURE_MATH_JSON_SCHEMA.
    """
    m = re.search(r"\{.*\}\s*$", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON found in model output")
    obj = json.loads(m.group(0))
    Draft7Validator(PURE_MATH_JSON_SCHEMA).validate(obj)
    return obj


def load_parquets(src_dir: Path) -> List[Path]:
    """List Parquet files in a directory, sorted."""
    return sorted([p for p in src_dir.iterdir() if p.suffix == ".parquet"])


# =============================================================================
# llama-server invocation
# =============================================================================
def llama_server_infer(prompt: str, max_tokens: int = MAX_GENERATION_TOKENS) -> str:
    """
    Call the node-local llama-server for this rank, with retries for transient
    'Loading model' (HTTP 503) responses.
    """
    port = SERVER_BASE_PORT + local_rank
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    payload = {
        "model": "gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
    }

    max_attempts = 60          # e.g. up to 10 minutes if sleep=10s
    sleep_seconds = 10.0

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, timeout=SERVER_TIMEOUT)
        except Exception as e:
            # Network / connection error: backoff and retry
            if attempt == max_attempts:
                raise RuntimeError(
                    f"[RANK {world_rank}] HTTP error talking to llama-server at {url} "
                    f"after {max_attempts} attempts: {e}"
                ) from e
            print(
                f"[RANK {world_rank}] transient HTTP error talking to llama-server at {url} "
                f"(attempt {attempt}/{max_attempts}): {e}; sleeping {sleep_seconds}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
            continue

        # Special handling for 503 "Loading model"
        if resp.status_code == 503:
            try:
                data = resp.json()
                msg = data.get("error", {}).get("message", "")
            except Exception:
                msg = ""
            if "Loading model" in msg:
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"[RANK {world_rank}] llama-server stuck in 'Loading model' on {url} "
                        f"after {max_attempts} attempts."
                    )
                print(
                    f"[RANK {world_rank}] llama-server on {url} still loading model "
                    f"(attempt {attempt}/{max_attempts}); sleeping {sleep_seconds}s",
                    flush=True,
                )
                time.sleep(sleep_seconds)
                continue

        # Other non-200: treat as hard error
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(
                f"[RANK {world_rank}] llama-server returned HTTP {resp.status_code}: {err}"
            )

        # Parse normal successful response
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"[RANK {world_rank}] Failed to parse JSON response from llama-server: {resp.text}"
            ) from e

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"[RANK {world_rank}] Unexpected /v1/chat/completions schema: {data}"
            ) from e

        return content

    # Defensive; should be unreachable
    raise RuntimeError(
        f"[RANK {world_rank}] Exhausted retries in llama_server_infer without success."
    )


# =============================================================================
# Row-sharded processing
# =============================================================================

def process_file_row_sharded(
    pq_path: Path,
    dst_dir: Path,
    model_path: str,  # kept for CLI compatibility; unused with llama-server
    ctx: int,
    ngl: int,
) -> None:
    """
    Process a single Parquet file with row-wise sharding:

        my_rows = { i | 0 <= i < N, i % world_size == world_rank }

    `ctx` is used only to approximate a character budget for skipping too-long
    prompts; the *actual* context is set in llama-server.
    """
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

    # Read checkpoint (last processed "block index" for this rank)
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

    # Approximate char budget from ctx (very coarse; we prefer skipping to OOM)
    max_prompt_chars = int(ctx * 4)  # ~4 chars/token

    buffer: List[Dict[str, Any]] = []
    # Smaller flush interval for easier debugging; bump once throughput is validated.
    flush_every = 20

    count = 0
    for global_row_index in my_rows:
        row = df.iloc[global_row_index]

        paper_id = row.get("paper_id")
        eq_id = row.get("equation_id")

        latex_raw = str(row.get("content_resolved", "") or "")
        latex_clean = katex_hygiene(latex_raw)
        prompt = build_prompt(latex_clean)

        # Aggressively skip over-long prompts to keep memory bounded & throughput high
        if len(prompt) > max_prompt_chars:
            print(
                f"[RANK {world_rank}] SKIP row {global_row_index}: "
                f"prompt too long (len={len(prompt)} > {max_prompt_chars})",
                flush=True,
            )
            continue

        try:
            content = llama_server_infer(prompt, max_tokens=MAX_GENERATION_TOKENS)
            try:
                parsed = parse_strict_json(content)
            except Exception:
                parsed = None
        except Exception as e:
            print(f"[RANK {world_rank}] ERROR @ row {global_row_index}: {e}", flush=True)
            continue

        rec: Dict[str, Any] = {
            "paper_id": paper_id,
            "equation_id": eq_id,
            "latex_raw": latex_raw,
            "latex_clean": latex_clean,
            "llm_raw_output": content,
            "global_row": global_row_index,
        }

        if parsed:
            analysis = parsed["analysis"]
            equivs = parsed["equivalents"]
            rec.update(
                {
                    "math_keywords": json.dumps(
                        analysis["math_keywords"], ensure_ascii=False
                    ),
                    "math_sentence": analysis["math_sentence"],
                    "katex": analysis["katex"],
                    "equiv_form_1": json.dumps(
                        equivs["equiv_form_1"], ensure_ascii=False
                    ),
                    "equiv_form_2": json.dumps(
                        equivs["equiv_form_2"], ensure_ascii=False
                    ),
                    "output_json": json.dumps(parsed, ensure_ascii=False),
                }
            )

        buffer.append(rec)
        count += 1

        if count % flush_every == 0:
            _flush(out_path, buffer)
            buffer.clear()
            # Keep checkpoint in units of "chunks flushed"
            _write_rank_ckpt(ckpt_path, count // flush_every)
            print(f"[RANK {world_rank}] Flushed {count} rows", flush=True)

    # Final flush
    _flush(out_path, buffer)
    _write_rank_ckpt(ckpt_path, len(my_rows))
    print(
        f"[RANK {world_rank}] DONE {pq_path.name}: processed {len(my_rows)} rows",
        flush=True,
    )


# ---- small helpers used above ----

def _flush(out_path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    df_new = pd.DataFrame(records)
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        pd.concat([df_old, df_new], ignore_index=True).to_parquet(
            out_path, index=False
        )
    else:
        df_new.to_parquet(out_path, index=False)


def _write_rank_ckpt(ckpt: Path, offset: int) -> None:
    ckpt.write_text(json.dumps({"row_offset": offset}, ensure_ascii=False))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--model", required=True)  # kept for interface symmetry (unused)
    ap.add_argument("--ctx", type=int, default=1024)  # used only for char budget
    ap.add_argument("--ngl", type=int, default=80)    # unused with llama-server
    args = ap.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    files = load_parquets(src_dir)
    print(
        f"[RANK {world_rank}] Found {len(files)} parquet files; "
        f"SERVER_BASE_PORT={SERVER_BASE_PORT}, local_rank={local_rank}",
        flush=True,
    )

    # Each rank processes *all files* (row-sharded)
    for pq in files:
        process_file_row_sharded(pq, dst_dir, args.model, args.ctx, args.ngl)


if __name__ == "__main__":
    main()

