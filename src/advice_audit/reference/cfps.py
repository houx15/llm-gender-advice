"""CFPS benchmark loader — INTERFACE ONLY (spec §7, deferred).

Disabled until config/paths.yaml `cfps_dir` is set. CFPS files are license-restricted:
never copy them into the repo or outputs; write only aggregated tables. Variable names
come from config/cfps_variables.yaml, which the researcher fills.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.config import CONFIG_DIR, is_todo, load_paths, load_yaml


class CFPSDisabled(RuntimeError):
    pass


def cfps_dir() -> Path:
    d = load_paths().get("cfps_dir")
    if not d or is_todo(d):
        raise CFPSDisabled("CFPS is disabled: set cfps_dir in config/paths.yaml")
    return Path(d)


def variables() -> dict:
    v = load_yaml(CONFIG_DIR / "cfps_variables.yaml")
    todo = [k for k, x in _flatten(v).items() if is_todo(x)]
    if todo:
        raise CFPSDisabled(f"config/cfps_variables.yaml has TODO entries: {todo}")
    return v


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def sex_by_subject_track() -> pd.DataFrame:
    """Weighted shares of 文/理 (or 物理/历史) track by sex. Returns columns: sex, track, share, n."""
    cfps_dir(), variables()
    raise NotImplementedError("fill config/cfps_variables.yaml, then implement the read for your CFPS wave")


def sex_by_major() -> pd.DataFrame:
    """Weighted shares of field of study by sex, mapped to taxonomy ids. Columns: sex, category_id, share, n."""
    cfps_dir(), variables()
    raise NotImplementedError("fill config/cfps_variables.yaml, then implement the read for your CFPS wave")
