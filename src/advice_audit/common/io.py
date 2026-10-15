"""Append-only Parquet logs: each write is a new part file; readers concatenate parts."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import ROOT


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonify(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


def append_parquet(dir_: Path, rows: Iterable[dict], prefix: str = "part") -> Path | None:
    """Write rows as a NEW part file in dir_ (never rewrites existing parts)."""
    rows = list(rows)
    if not rows:
        return None
    dir_.mkdir(parents=True, exist_ok=True)
    n = len(list(dir_.glob(f"{prefix}-*.parquet")))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = dir_ / f"{prefix}-{n:05d}-{stamp}.parquet"
    df = pd.DataFrame([{k: _jsonify(v) for k, v in r.items()} for r in rows])
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)
    return path


def read_parquet_dir(dir_: Path, pattern: str = "*.parquet") -> pd.DataFrame:
    parts = sorted(Path(dir_).glob(pattern)) if Path(dir_).exists() else []
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        h = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        return (h + ("-dirty" if dirty else "")) if h else None
    except Exception:  # noqa: BLE001
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_question(title: str, body: str) -> None:
    """Append an open question to QUESTIONS.md (spec §0.6), deduplicated by title."""
    from .config import QUESTIONS_MD

    text = QUESTIONS_MD.read_text(encoding="utf-8") if QUESTIONS_MD.exists() else "# QUESTIONS.md\n"
    header = f"## {title}"
    if header in text:
        return
    QUESTIONS_MD.write_text(text.rstrip() + f"\n\n---\n\n{header}\n\n{body.strip()}\n\n> RESEARCHER REPLY:\n", encoding="utf-8")
