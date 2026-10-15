"""Config loading (pydantic v2) and repo paths.

This module must stay importable on offline compute nodes: no network, and no
imports of optional extras ([api], [analysis]) at module level.
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("ADVICE_AUDIT_ROOT", Path(__file__).resolve().parents[3]))

try:  # .env is never committed; harmless if absent
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = Path(os.environ.get("ADVICE_AUDIT_OUTPUTS", ROOT / "outputs"))
RUNS_DIR = OUTPUTS_DIR / "runs"
PLANS_DIR = OUTPUTS_DIR / "plans"
LOCAL_DIR = OUTPUTS_DIR / "local"
REFERENCE_RAW = DATA_DIR / "reference" / "raw"
REFERENCE_BUILT = DATA_DIR / "reference" / "built"
QUESTIONS_MD = ROOT / "QUESTIONS.md"

GENDER_CONDS = ("f_implicit", "f_explicit", "m_implicit", "m_explicit", "baseline")
DOMAINS = ("majors", "jobs")
MODES = ("open", "fixed")
TODO = "TODO"


def is_todo(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().upper().startswith(TODO))


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- API models
class ProviderCfg(BaseModel):
    order: list[str] = Field(default_factory=list)
    allow_fallbacks: bool = False


class ApiModelCfg(BaseModel):
    key: str
    slug: str
    canonical_slug: str | None = None
    provider: ProviderCfg = ProviderCfg()
    reasoning: dict[str, Any] = Field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int = 1200
    price_per_mtok: dict[str, float] = Field(default_factory=dict)
    peak_multiplier: float = 1.0

    def prices(self, peak: bool) -> tuple[float, float]:
        m = self.peak_multiplier if peak else 1.0
        return self.price_per_mtok.get("input", 0.0) * m, self.price_per_mtok.get("output", 0.0) * m


class ApiModelsCfg(BaseModel):
    defaults: dict[str, Any] = Field(default_factory=dict)
    models: list[ApiModelCfg]
    coding: dict[str, str] = Field(default_factory=dict)

    def get(self, key: str) -> ApiModelCfg:
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"unknown API model key {key!r}; known: {[m.key for m in self.models]}")

    @property
    def est_input_tokens(self) -> int:
        return int(self.defaults.get("est_input_tokens", 500))

    @property
    def est_output_tokens(self) -> int:
        return int(self.defaults.get("est_output_tokens", 600))


@lru_cache(maxsize=None)
def load_api_models() -> ApiModelsCfg:
    raw = load_yaml(CONFIG_DIR / "api_models.yaml")
    mt = raw.get("defaults", {}).get("max_tokens")
    if mt:
        for m in raw["models"]:
            m.setdefault("max_tokens", mt)
    return ApiModelsCfg(**raw)


# ---------------------------------------------------------------- budget
class BudgetCfg(BaseModel):
    total_cap_usd: float
    reserve_usd: float
    phase_caps: dict[str, float]
    assume_peak_pricing: bool = True
    estimate_safety_factor: float = 1.15
    kill_switch: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=None)
def load_budget() -> BudgetCfg:
    return BudgetCfg(**load_yaml(CONFIG_DIR / "budget.yaml"))


# ---------------------------------------------------------------- experiments
class ExperimentCfg(BaseModel):
    name: str
    phase: str
    prompt_version: str = "v1"
    requires_approval: str | None = None
    models: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=lambda: list(DOMAINS))
    modes: list[str] = Field(default_factory=lambda: list(MODES))
    profiles: dict[str, Any] = Field(default_factory=lambda: {"majors": "all", "jobs": "all"})
    cells: list[str] = Field(default_factory=lambda: ["A", "B", "C"])
    gender_conds: list[str] = Field(default_factory=lambda: list(GENDER_CONDS))
    samples: int = 10


def load_experiment(name: str) -> ExperimentCfg:
    path = Path(name)
    if not path.exists():
        path = CONFIG_DIR / "experiments" / f"{name}.yaml"
    return ExperimentCfg(**load_yaml(path))


# ---------------------------------------------------------------- materials
@lru_cache(maxsize=None)
def load_cells() -> dict[str, dict]:
    return load_yaml(CONFIG_DIR / "cells.yaml")["cells"]


@lru_cache(maxsize=None)
def load_names() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "names.yaml")


@lru_cache(maxsize=None)
def load_profiles(domain: str, country: str) -> dict:
    stem = {"majors": "majors", "jobs": "jobs"}[domain]
    return load_yaml(CONFIG_DIR / "profiles" / f"{stem}_{country}.yaml")


@lru_cache(maxsize=None)
def load_fixed_fields() -> dict:
    return load_yaml(CONFIG_DIR / "fixed_fields.yaml")


def profile_ids(domain: str) -> list[str]:
    return [p["id"] for p in load_profiles(domain, "us")["profiles"]]


@lru_cache(maxsize=None)
def load_taxonomy_config(domain: str) -> list[dict]:
    fname = {"majors": "majors.csv", "jobs": "jobs.csv"}[domain]
    with open(CONFIG_DIR / "taxonomy" / fname, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    code_col = "cip_codes" if domain == "majors" else "soc_codes"
    for r in rows:
        r["codes"] = [c.strip() for c in r.pop(code_col).split(";") if c.strip()]
        r["stem"] = int(r["stem"])
    return rows


# ---------------------------------------------------------------- local / cluster
class LocalModelCfg(BaseModel):
    key: str
    repo_id: str
    revision: str
    dtype: str = "bfloat16"
    role: str = "main"
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    quantization: str | None = None  # Step-0-only models (>32B); excluded from representation analysis

    @property
    def ready(self) -> bool:
        return not (is_todo(self.repo_id) or is_todo(self.revision))


class LocalModelsCfg(BaseModel):
    models: list[LocalModelCfg]
    generation: dict[str, Any] = Field(default_factory=dict)
    extraction: dict[str, Any] = Field(default_factory=dict)
    readout: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str) -> LocalModelCfg:
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"unknown local model key {key!r}; known: {[m.key for m in self.models]}")


@lru_cache(maxsize=None)
def load_local_models() -> LocalModelsCfg:
    return LocalModelsCfg(**load_yaml(CONFIG_DIR / "local_models.yaml"))


@lru_cache(maxsize=None)
def load_paths() -> dict:
    return load_yaml(CONFIG_DIR / "paths.yaml")


# ---------------------------------------------------------------- approvals
def is_approved(gate: str, questions_md: Path | None = None) -> bool:
    """Phases A4, B3, B4 need 'APPROVED: <gate>' written by the researcher (spec §10)."""
    path = questions_md or QUESTIONS_MD
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().lstrip(">").strip()
        if s.upper().startswith(f"APPROVED: {gate.upper()}"):
            return True
    return False


def require_approval(gate: str | None) -> None:
    if gate and not is_approved(gate):
        raise SystemExit(
            f"Phase {gate} requires explicit researcher approval: add a line "
            f"'APPROVED: {gate}' to QUESTIONS.md (spec §10)."
        )
