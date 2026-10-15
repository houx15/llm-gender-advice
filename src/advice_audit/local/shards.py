"""Prompt shards for Slurm array jobs (spec §6.3): plan -> shard_<k>.parquet; each shard
writes outputs/local/<model>/<stage>/shard_<k>/ and a DONE marker; reruns skip DONE."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ..common.config import LOCAL_DIR
from ..common.io import utcnow


def stage_dir(model_key: str, stage: str) -> Path:
    return LOCAL_DIR / model_key / stage


def plan_dir(model_key: str, stage: str) -> Path:
    return stage_dir(model_key, stage) / "plan"


def shard_out(model_key: str, stage: str, k: int) -> Path:
    return stage_dir(model_key, stage) / f"shard_{k:04d}"


def write_shards(df: pd.DataFrame, model_key: str, stage: str, shard_size: int) -> list[Path]:
    d = plan_dir(model_key, stage)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    paths = []
    for k, start in enumerate(range(0, len(df), shard_size)):
        p = d / f"shard_{k:04d}.parquet"
        df.iloc[start : start + shard_size].to_parquet(p, index=False)
        paths.append(p)
    (d / "plan.json").write_text(json.dumps({"n_rows": len(df), "n_shards": len(paths), "shard_size": shard_size,
                                             "created_at": utcnow()}, indent=2))
    return paths


def n_shards(model_key: str, stage: str) -> int:
    return json.loads((plan_dir(model_key, stage) / "plan.json").read_text())["n_shards"]


def load_shard(model_key: str, stage: str, k: int) -> pd.DataFrame:
    return pd.read_parquet(plan_dir(model_key, stage) / f"shard_{k:04d}.parquet")


def is_done(model_key: str, stage: str, k: int) -> bool:
    return (shard_out(model_key, stage, k) / "DONE").exists()


def mark_done(model_key: str, stage: str, k: int, info: dict | None = None) -> None:
    out = shard_out(model_key, stage, k)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").write_text(json.dumps({"finished_at": utcnow(), **(info or {})}, indent=2))


def run_shard(model_key: str, stage: str, k: int, fn) -> bool:
    """Run fn(df, out_dir) unless the shard is DONE. Returns True if work was done."""
    if is_done(model_key, stage, k):
        print(f"[shard] {model_key}/{stage}/shard_{k:04d} already DONE — skipping")
        return False
    out = shard_out(model_key, stage, k)
    out.mkdir(parents=True, exist_ok=True)
    info = fn(load_shard(model_key, stage, k), out) or {}
    mark_done(model_key, stage, k, info)
    return True


def shard_dirs(model_key: str, stage: str, require_done: bool = True) -> list[Path]:
    d = stage_dir(model_key, stage)
    dirs = sorted(p for p in d.glob("shard_*") if p.is_dir())
    return [p for p in dirs if (p / "DONE").exists()] if require_done else dirs
