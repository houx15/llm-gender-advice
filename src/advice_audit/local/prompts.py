"""Prompt sets and shard planning for every Part B stage (`advice-audit local plan`).

Runs on the login node or a laptop (no GPU). Prompts are plain user texts plus char spans
of the name / cue / label; chat templating happens on the compute node with the model's
own tokenizer, so plans are model-agnostic apart from the output path.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..common.config import CONFIG_DIR, load_experiment, load_local_models, load_yaml, require_approval
from ..common.ids import sha256
from ..common.logs import get_logger
from ..materials.design import expand_design
from ..materials.taxonomy import load_taxonomy
from .shards import stage_dir, write_shards

log = get_logger("advice_audit.local.plan")
STAGES = ("behavior", "readout", "extract", "concepts", "patching", "steering", "erasure")
GATES = {"extract": "B3", "concepts": "B3", "patching": "B4", "steering": "B4", "erasure": "B4"}


def interp_cfg() -> dict:
    return load_yaml(CONFIG_DIR / "experiments" / "local_interp.yaml")


_CELLS: list[str] | None = None  # optional override (e.g. ["A", "B"] before CN anchors exist)


def _design(samples: int | None = None, name: str = "local_behavior") -> pd.DataFrame:
    exp = load_experiment(name)
    if samples:
        exp.samples = samples
    if _CELLS:
        exp.cells = list(_CELLS)
    d = expand_design(exp, strict=True)
    d["prompt_id"] = [sha256([r.domain, r.mode, r.profile_id, r.cell, r.gender_cond, r.name_idx, r.paraphrase_id,
                              r.sample_idx, r.prompt_version]) for r in d.itertuples()]
    return d


def behavior_set() -> pd.DataFrame:
    return _design()


def extract_set() -> pd.DataFrame:
    """Unique (name, paraphrase) per profile × cell × mode × domain × condition."""
    d = _design(samples=interp_cfg()["extract"]["samples"])
    return d.drop_duplicates(["domain", "mode", "profile_id", "cell", "gender_cond", "name_idx", "paraphrase_id"]).reset_index(drop=True)


def concept_set() -> pd.DataFrame:
    cfg = interp_cfg()["concepts"]["templates"]
    rows = []
    for domain in ("majors", "jobs"):
        tax = load_taxonomy(domain)
        for lang in ("en", "zh"):
            for ti, tpl in enumerate(cfg[lang][domain]):
                for r in tax.itertuples():
                    label = r.label_en if lang == "en" else r.label_zh
                    text = tpl.format(label=label)
                    s = text.index(label)
                    rows.append({"prompt_id": sha256(["concept", domain, lang, ti, r.id]), "domain": domain,
                                 "language": lang, "template_id": ti, "category_id": r.id, "prompt": text,
                                 "label_span": json.dumps([s, s + len(label)]), "name_span": "null", "cue_span": "null"})
    return pd.DataFrame(rows)


def sample_strata(df: pd.DataFrame, by: list[str], per: int, seed: int = 0) -> pd.DataFrame:
    """Up to `per` random rows per stratum (deterministic given seed)."""
    rng = np.random.default_rng(seed)
    d = df.iloc[rng.permutation(len(df))]
    return d[d.groupby(by).cumcount() < per].sort_values(by).reset_index(drop=True)


def patching_set(extract: pd.DataFrame, n_pairs: int, seed: int = 0) -> pd.DataFrame:
    """Matched female/male pairs (same everything except the gender of name and cue)."""
    key = ["domain", "mode", "profile_id", "cell", "name_idx", "paraphrase_id"]
    rows = []
    for cue in ("implicit", "explicit"):
        f = extract[extract["gender_cond"] == f"f_{cue}"]
        m = extract[extract["gender_cond"] == f"m_{cue}"]
        mm = f.merge(m, on=key, suffixes=("_f", "_m"))
        mm["cue"] = cue
        rows.append(mm)
    pairs = pd.concat(rows, ignore_index=True)
    by = ["domain", "mode", "cell", "cue"]  # balance strata
    per = max(1, n_pairs // max(pairs.groupby(by).ngroups, 1))
    pick = sample_strata(pairs, by, per, seed)
    pick["pair_id"] = [sha256([a, b]) for a, b in zip(pick["prompt_id_f"], pick["prompt_id_m"])]
    return pick


def steering_set(design: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    base = design[design["gender_cond"] == "baseline"].drop_duplicates(["domain", "mode", "profile_id", "cell", "paraphrase_id"])
    by = ["domain", "mode", "cell"]
    return sample_strata(base, by, max(1, n // max(base.groupby(by).ngroups, 1)), seed)


def erasure_fit_profiles(fit_fraction: float, seed: int = 0) -> list[str]:
    """Profiles used to FIT LEACE; the rest are evaluated. Depends only on profile ids."""
    from ..common.config import profile_ids

    profiles = sorted(profile_ids("majors") + profile_ids("jobs"))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(profiles, size=int(round(len(profiles) * fit_fraction)), replace=False).tolist())


def erasure_set(design: pd.DataFrame, fit_fraction: float, seed: int = 0) -> pd.DataFrame:
    """Held-out profiles (not used to fit LEACE) — all conditions."""
    fit = set(erasure_fit_profiles(fit_fraction, seed))
    return design[~design["profile_id"].isin(fit)].reset_index(drop=True).assign(erasure_split="eval")


def storage_estimate_gb(n_prompts: int, n_layers: int, hidden: int, n_positions: int, bytes_per: int = 2) -> float:
    return n_prompts * n_layers * hidden * n_positions * bytes_per / 1e9


def plan_stage(model_key: str, stage: str, n_layers: int | None = None, hidden: int | None = None,
               cells: list[str] | None = None, limit: int | None = None) -> dict:
    """Write prompt shards for a stage. Returns a summary dict.
    `cells` restricts the design (dry runs before CN anchors are filled); `limit` keeps the
    first N prompts (end-to-end smoke tests)."""
    global _CELLS
    _CELLS = cells
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage}; choose from {STAGES}")
    require_approval(GATES.get(stage))
    load_local_models().get(model_key)  # validates key
    cfg = interp_cfg()
    if stage == "behavior":
        df, size = behavior_set(), 250
    elif stage == "readout":
        df, size = behavior_set(), 250  # same prompts as Step 0 -> 1:1 validation
    elif stage == "extract":
        df, size = extract_set(), cfg["extract"]["shard_size"]
    elif stage == "concepts":
        df, size = concept_set(), cfg["concepts"]["shard_size"]
    elif stage == "patching":
        df, size = patching_set(extract_set(), cfg["patching"]["n_pairs"]), cfg["patching"]["shard_size"]
    elif stage == "steering":
        df, size = steering_set(extract_set(), cfg["steering"]["n_prompts"]), cfg["steering"]["shard_size"]
    else:
        df, size = erasure_set(extract_set(), cfg["erasure"]["fit_fraction"]), cfg["erasure"]["shard_size"]
    if limit:
        df = df.head(limit).reset_index(drop=True)
    paths = write_shards(df, model_key, stage, size)
    summary = {"model": model_key, "stage": stage, "n_prompts": int(len(df)), "n_shards": len(paths),
               "plan_dir": str(stage_dir(model_key, stage) / "plan")}
    if stage in ("extract", "concepts"):
        npos = len(load_local_models().extraction.get("positions", [])) if stage == "extract" else 2
        if n_layers and hidden:
            gb = storage_estimate_gb(len(df), n_layers + 1, hidden, npos)
            summary["storage_gb_est"] = round(gb, 2)
            warn = float(load_local_models().extraction.get("warn_storage_gb", 50))
            if gb > warn:
                log.warning(f"projected activation storage {gb:.1f} GB > {warn} GB — reduce layers/positions in local_models.yaml")
        else:
            summary["storage_note"] = ("pass --n-layers/--hidden (or cache the model first) for a storage estimate; "
                                       "~1.3 MB/prompt for a 64-layer, 5120-d model at 2 positions (spec §6.7)")
    log.info(json.dumps(summary))
    return summary
