"""Deterministic IDs (spec §0.5, §4.3)."""
from __future__ import annotations

import hashlib
import json
from typing import Any

CALL_ID_FIELDS = (
    "domain", "mode", "profile_id", "cell", "gender_cond", "name",
    "paraphrase_id", "sample_idx", "model_key", "prompt_version",
)


def sha256(obj: Any) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def call_id(**fields: Any) -> str:
    """SHA-256 of the design fields in spec §4.3 (fixed order, JSON-encoded)."""
    missing = [f for f in CALL_ID_FIELDS if f not in fields]
    if missing:
        raise ValueError(f"call_id missing fields: {missing}")
    return sha256([fields[f] for f in CALL_ID_FIELDS])


def stable_int(*parts: Any, mod: int | None = None) -> int:
    v = int(sha256(list(parts))[:12], 16)
    return v % mod if mod else v
