"""Experimental design expansion shared by Part A (API plans) and Part B (local prompts).

Name and paraphrase assignment (spec §3.5 "balanced across conditions"): for a given
(domain, mode, profile, cell), sample s gets
    name_idx      = s mod 3
    paraphrase_id = (s + s // 3 + h) mod 3,   h = hash(domain, mode, profile, cell) mod 3
which is a Latin square over the first 9 samples (every name x paraphrase pair once) and
is IDENTICAL across gender conditions, so female/male/baseline prompts are matched on
paraphrase and name index (same surname) sample by sample.
"""
from __future__ import annotations

import json

import pandas as pd

from ..common.config import ExperimentCfg, load_cells, load_profiles, profile_ids
from ..common.ids import call_id, sha256, stable_int
from .render import n_names, render_prompt

N_PARAPHRASES = 3


def assign(domain: str, mode: str, profile_id: str, cell: str, sample_idx: int, k_names: int = 3) -> tuple[int, int]:
    h = stable_int(domain, mode, profile_id, cell, mod=N_PARAPHRASES)
    name_idx = sample_idx % k_names
    para = (sample_idx + sample_idx // k_names + h) % N_PARAPHRASES
    return name_idx, para


def profile_orientation(domain: str, profile_id: str) -> tuple[str, str]:
    """(orientation, level) of a profile; jobs use the graduate's major track."""
    cfg = load_profiles(domain, "us")
    p = next(p for p in cfg["profiles"] if p["id"] == profile_id)
    if domain == "majors":
        return p["orientation"], p["level"]
    return cfg["majors"][p["major"]]["orientation"], p["level"]


def selected_profiles(exp: ExperimentCfg, domain: str) -> list[str]:
    sel = exp.profiles.get(domain, "all")
    allp = profile_ids(domain)
    if sel == "all":
        return allp
    unknown = set(sel) - set(allp)
    if unknown:
        raise ValueError(f"unknown profiles for {domain}: {unknown}")
    return list(sel)


def expand_design(exp: ExperimentCfg, strict: bool = True) -> pd.DataFrame:
    """One row per prompt (no model dimension)."""
    rows = []
    cells = load_cells()
    for domain in exp.domains:
        for mode in exp.modes:
            for pid in selected_profiles(exp, domain):
                orientation, level = profile_orientation(domain, pid)
                for cell in exp.cells:
                    k = n_names(cell)
                    for s in range(exp.samples):
                        name_idx, para = assign(domain, mode, pid, cell, s, k)
                        for gc in exp.gender_conds:
                            rp = render_prompt(domain, mode, pid, cell, gc, name_idx, para, strict=strict)
                            rows.append(
                                {
                                    "experiment": exp.name,
                                    "phase": exp.phase,
                                    "domain": domain,
                                    "mode": mode,
                                    "profile_id": pid,
                                    "orientation": orientation,
                                    "level": level,
                                    "cell": cell,
                                    "language": cells[cell]["language"],
                                    "country": cells[cell]["country"],
                                    "gender_cond": gc,
                                    "female": None if gc == "baseline" else int(gc.startswith("f")),
                                    "explicit": None if gc == "baseline" else int(gc.endswith("explicit")),
                                    "name": rp.name or "",
                                    "name_idx": name_idx,
                                    "paraphrase_id": para,
                                    "sample_idx": s,
                                    "prompt_version": exp.prompt_version,
                                    "fixed_field": rp.fixed_field or "",
                                    "prompt": rp.text,
                                    "prompt_hash": sha256(rp.text),
                                    "name_span": json.dumps(rp.name_span),
                                    "cue_span": json.dumps(rp.cue_span),
                                }
                            )
    df = pd.DataFrame(rows)
    df["female"] = df["female"].astype("Int64")
    df["explicit"] = df["explicit"].astype("Int64")
    return df


def attach_models(design: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
    out = []
    for mk in model_keys:
        d = design.copy()
        d["model_key"] = mk
        d["call_id"] = [
            call_id(
                domain=r.domain, mode=r.mode, profile_id=r.profile_id, cell=r.cell,
                gender_cond=r.gender_cond, name=r.name, paraphrase_id=int(r.paraphrase_id),
                sample_idx=int(r.sample_idx), model_key=mk, prompt_version=r.prompt_version,
            )
            for r in d.itertuples()
        ]
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    assert df["call_id"].is_unique, "call_id collision"
    return df
