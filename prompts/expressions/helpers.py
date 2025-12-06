import re
import json
import pandas as pd
from pathlib import Path
from typing import Any, List, Dict
from jsonschema import Draft7Validator

from math_prompt import PROMPT_TEMPLATE, PURE_MATH_JSON_SCHEMA

def katex_hygiene(s: str) -> str:
    """Minimal cleanup for better model output (we still prompt with cleaned string)."""
    s = re.sub(r"\\label\{[^}]*\}", "", s)  # remove \label{...}
    s = s.strip().rstrip(",")
    return s


def build_messages(latex_raw: str) -> List[Dict[str, str]]:
    """Construct the chat messages for OpenAI-compatible endpoint."""
    prompt = PROMPT_TEMPLATE.format(
        schema=json.dumps(PURE_MATH_JSON_SCHEMA, ensure_ascii=False),
        latex_raw=latex_raw,
    )
    return [
        {"role": "system", "content": "You are precise and always return strict JSON only."},
        {"role": "user", "content": prompt},
    ]

# - - - - - - -
# Output
# - - - - - - - 

def parse_strict_json(s: str) -> Dict[str, Any]:
    """Extract and validate the final JSON object from model content."""
    m = re.search(r"\{.*\}\s*$", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in response")
    obj = json.loads(m.group(0))
    Draft7Validator(PURE_MATH_JSON_SCHEMA).validate(obj)
    return obj


def load_parquets(src_dir: Path) -> List[Path]:
    return sorted([p for p in src_dir.iterdir() if p.suffix == ".parquet"])


def read_checkpoint(ckpt_path: Path) -> int:
    if not ckpt_path.exists():
        return 0
    try:
        return int(json.loads(ckpt_path.read_text()).get("row_index", 0))
    except Exception:
        return 0


def write_checkpoint(ckpt_path: Path, idx: int) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps({"row_index": idx}, ensure_ascii=False))

def flush_records(out_path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    out_df = pd.DataFrame(records)
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        pd.concat([prev, out_df], ignore_index=True).to_parquet(out_path, index=False)
    else:
        out_df.to_parquet(out_path, index=False)
