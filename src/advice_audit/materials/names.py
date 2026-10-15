"""Name validation and the model manipulation check (spec §3.2).

Each API model is asked "What is the likely gender of a person named X?" 10 times per
name: cells A and B in English (Anglo and romanized Chinese names), cell C in Chinese
(the same Chinese names in characters). Names below the agreement threshold are flagged
in QUESTIONS.md.
"""
from __future__ import annotations

import re

import pandas as pd

from ..common.config import load_cells, load_names
from ..common.ids import sha256
from ..common.io import append_question
from .render import full_name, n_names, render_name_check


def validate_names() -> list[str]:
    problems = []
    cfg = load_names()
    for cell in load_cells():
        if cell not in cfg:
            problems.append(f"names.yaml: missing cell {cell}")
            continue
        try:
            k = n_names(cell)
        except AssertionError as e:
            problems.append(str(e))
            continue
        if k != 3:
            problems.append(f"names.yaml cell {cell}: spec asks for 3 names per gender, found {k}")
        if len(set(cfg[cell]["female"]) & set(cfg[cell]["male"])):
            problems.append(f"names.yaml cell {cell}: a given name appears for both genders")
    # cells B and C must be the same people
    if "B" in cfg and "C" in cfg and len(cfg["B"]["female"]) != len(cfg["C"]["female"]):
        problems.append("names.yaml: cells B and C must list the same people (romanized vs characters)")
    return problems


def name_check_plan(model_keys: list[str], samples: int | None = None) -> pd.DataFrame:
    cfg = load_names()
    samples = samples or int(cfg.get("name_check", {}).get("samples", 10))
    rows = []
    for cell in load_cells():
        for g in ("f", "m"):
            for idx in range(n_names(cell)):
                name = full_name(cell, g, idx)
                prompt = render_name_check(cell, name)
                for mk in model_keys:
                    for s in range(samples):
                        rows.append({
                            "experiment": "name_check", "phase": "materials", "task": "name_check",
                            "check_cell": cell, "cell": cell, "name": name, "name_idx": idx,
                            "gender_cond": f"{g}_implicit", "female": int(g == "f"), "sample_idx": s,
                            "model_key": mk, "prompt": prompt, "prompt_hash": sha256(prompt),
                            "call_id": sha256(["name_check", cell, name, s, mk, "v1"]),
                        })
    return pd.DataFrame(rows)


_F = re.compile(r"\b(female|woman|girl)\b|女", re.I)
_M = re.compile(r"\b(male|man|boy)\b|男", re.I)


def classify_answer(text: str) -> str:
    t = (text or "").strip()
    f, m = bool(_F.search(t)), bool(_M.search(t))
    if "unsure" in t.lower() or "不确定" in t:
        return "unsure"
    if f and not m:
        return "female"
    if m and not f:
        return "male"
    return "unsure"


def name_check_postprocess(text: str) -> dict:
    return {"answer": classify_answer(text), "refusal_flag": False}


def summarize_name_check(logs: pd.DataFrame, threshold: float | None = None, write_questions: bool = True) -> pd.DataFrame:
    threshold = threshold or float(load_names().get("name_check", {}).get("threshold", 0.90))
    df = logs[logs["status"] == "ok"].copy()
    df["answer"] = df["response"].map(classify_answer)
    df["intended"] = df["female"].map({1: "female", 0: "male"})
    df["agree"] = (df["answer"] == df["intended"]).astype(float)
    by_model = df.groupby(["check_cell", "name", "intended", "model_key"])["agree"].mean().unstack("model_key")
    summ = by_model.copy()
    summ["min_agreement"] = by_model.min(axis=1)
    summ["mean_agreement"] = by_model.mean(axis=1)
    summ["flag"] = summ["min_agreement"] < threshold
    summ = summ.reset_index()
    if write_questions and summ["flag"].any():
        flagged = summ[summ["flag"]]
        body = "Names below the {:.0%} agreement threshold in the model manipulation check:\n\n".format(threshold)
        body += "\n".join(
            f"- cell {r.check_cell}: **{r.name}** (intended {r.intended}) — min agreement {r.min_agreement:.0%}, mean {r.mean_agreement:.0%}"
            for r in flagged.itertuples()
        )
        body += "\n\nReplace these names in config/names.yaml (keep female[i]/male[i] surname-aligned) or confirm keeping them."
        append_question("Name manipulation check: flagged names", body)
    return summ
