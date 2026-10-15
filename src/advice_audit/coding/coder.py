"""LLM coding of free-text answers (spec §4.4). Two passes, each its own logged, budgeted job:

1. taxonomy  — ranked recommendations -> one taxonomy category each (or 'other')
2. discursive — warnings, hedges, ambition, gender/work-life/leadership mentions

Coder: Claude Haiku 5.5, except when coding Haiku's own answers (DeepSeek V4.1 Flash).
Works on Part A runs and on Part B generations saved in the same log schema.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..api.parse import parse_json_object
from ..common.config import PROMPTS_DIR, RUNS_DIR, load_api_models
from ..common.ids import sha256
from ..common.io import read_parquet_dir
from ..common.logs import get_logger
from ..materials.taxonomy import OTHER, load_taxonomy

log = get_logger("advice_audit.coding")

CODER_VERSION = "c1"
PASSES = ("taxonomy", "discursive")
EST_TOKENS = {"taxonomy": (1900, 350), "discursive": (900, 120)}
UNIT = {"majors": "majors", "jobs": "jobs"}
USER_KIND = {"majors": "high-school student", "jobs": "new university graduate"}

_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR / "coder")), undefined=StrictUndefined)


def coder_for(evaluated_model_key: str) -> str:
    c = load_api_models().coding
    default, fallback = c.get("default_coder", "claude_haiku_55"), c.get("self_coding_fallback", "deepseek_v41_flash")
    return fallback if evaluated_model_key == default else default


def source_logs(run: str) -> tuple[str, pd.DataFrame]:
    """Accept a run id under outputs/runs or a directory of Part-A-schema parquet logs."""
    from ..api.runner import load_run_logs

    p = Path(run)
    if p.exists():
        # a directory of Part-A-schema logs, e.g. outputs/local/<model>/behavior (shard_*/logs/*.parquet)
        parts = sorted(p.glob("**/logs/*.parquet")) or sorted(p.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            raise SystemExit(f"no logs under {p}")
        run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns else p.name
        return run_id, df
    df = load_run_logs(run)
    if df.empty:
        raise SystemExit(f"no logs for run {run}")
    return run, df


def build_coding_plan(src: pd.DataFrame, source_run_id: str, pass_name: str) -> pd.DataFrame:
    ok = src[(src["status"] == "ok") & src["response"].fillna("").str.len().gt(0)]
    rows = []
    for r in ok.itertuples():
        domain = r.domain
        if pass_name == "taxonomy":
            tax = load_taxonomy(domain, with_attributes=False)
            prompt = _env.get_template("taxonomy_map.j2").render(
                unit_plural=UNIT[domain], max_items=5, categories=tax.to_dict("records"), response=r.response
            )
        else:
            prompt = _env.get_template("discursive.j2").render(
                user_kind=USER_KIND[domain], unit_plural=UNIT[domain], response=r.response
            )
        coder = coder_for(r.model_key)
        rows.append({
            "task": pass_name, "phase": "coding", "experiment": f"code_{pass_name}",
            "source_run_id": source_run_id, "source_call_id": r.call_id, "domain": domain,
            "model_key": coder, "coder_key": coder, "prompt": prompt, "prompt_hash": sha256(prompt),
            "call_id": sha256(["code", pass_name, r.call_id, coder, CODER_VERSION]),
        })
    return pd.DataFrame(rows)


def _postprocess(pass_name: str):
    def fn(text: str) -> dict:
        obj = parse_json_object(text)
        if obj is None:
            return {"parse_ok": False, "coded_json": None, "refusal_flag": False}
        if pass_name == "taxonomy" and not isinstance(obj.get("recommendations"), list):
            return {"parse_ok": False, "coded_json": json.dumps(obj, ensure_ascii=False), "refusal_flag": False}
        return {"parse_ok": True, "coded_json": json.dumps(obj, ensure_ascii=False), "refusal_flag": False}
    return fn


def coding_dir(source_run_id: str, pass_name: str) -> Path:
    return RUNS_DIR / "coding" / source_run_id / pass_name


def run_coding(run: str, passes: tuple[str, ...] = PASSES, dry_run: bool = False, max_calls: int | None = None,
               allow_reserve: bool = False, concurrency: int = 8) -> dict:
    from ..api.runner import execute_plan

    source_run_id, src = source_logs(run)
    out = {}
    for pass_name in passes:
        plan = build_coding_plan(src, source_run_id, pass_name)
        if plan.empty:
            log.warning(f"{source_run_id}: nothing to code for {pass_name}")
            continue
        est_in, est_out = EST_TOKENS[pass_name]
        for coder, g in plan.groupby("coder_key"):
            rid = f"{source_run_id}__{pass_name}__{coder}"
            out[rid] = asyncio.run(execute_plan(
                g.reset_index(drop=True), coder, rid, "coding", run_dir=coding_dir(source_run_id, pass_name) / coder,
                dry_run=dry_run, max_calls=max_calls, allow_reserve=allow_reserve, concurrency=concurrency,
                json_mode=True, max_tokens=800, est_input_tokens=est_in, est_output_tokens=est_out,
                postprocess=_postprocess(pass_name),
            ))
    return out


# ------------------------------------------------------------------ results
def _coding_logs(source_run_id: str, pass_name: str) -> pd.DataFrame:
    d = coding_dir(source_run_id, pass_name)
    frames = [read_parquet_dir(p) for p in d.glob("*/logs")] if d.exists() else []
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("request_ts").drop_duplicates("source_call_id", keep="last")
    return df[df["parse_ok"].fillna(False).astype(bool)]


def taxonomy_codes(source_run_id: str, domain_of: dict[str, str] | None = None) -> pd.DataFrame:
    """Long table: source_call_id, rank, raw_text, category_id, confidence (invalid ids -> other)."""
    df = _coding_logs(source_run_id, "taxonomy")
    rows = []
    valid = {d: set(load_taxonomy(d, with_attributes=False)["id"]) for d in ("majors", "jobs")}
    for r in df.itertuples():
        obj = json.loads(r.coded_json)
        recs = sorted(obj.get("recommendations", []), key=lambda x: x.get("rank", 99))[:5]
        for i, rec in enumerate(recs, start=1):
            cid = str(rec.get("category_id", OTHER)).strip()
            if cid not in valid[r.domain]:
                cid = OTHER
            rows.append({"source_call_id": r.source_call_id, "rank": i, "raw_text": rec.get("raw_text", ""),
                         "category_id": cid, "confidence": rec.get("confidence"), "coder_key": r.coder_key})
    return pd.DataFrame(rows, columns=["source_call_id", "rank", "raw_text", "category_id", "confidence", "coder_key"])


DISCURSIVE_FIELDS = ("warning_unsolicited", "warning_quote", "hedge_count", "ambition_encouragement",
                     "mentions_gender", "mentions_worklife_or_family", "leadership_mention")


def discursive_codes(source_run_id: str) -> pd.DataFrame:
    df = _coding_logs(source_run_id, "discursive")
    rows = []
    for r in df.itertuples():
        obj = json.loads(r.coded_json)
        row = {"source_call_id": r.source_call_id, "coder_key": r.coder_key}
        for f in DISCURSIVE_FIELDS:
            v = obj.get(f)
            if f in ("hedge_count", "ambition_encouragement"):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = float("nan")
            elif f != "warning_quote":
                v = float(bool(v)) if v is not None else float("nan")
            row[f] = v
        rows.append(row)
    return pd.DataFrame(rows)


def coding_summary(source_run_id: str, n_source_ok: int) -> dict:
    out = {}
    for p in PASSES:
        d = coding_dir(source_run_id, p)
        frames = [read_parquet_dir(x) for x in d.glob("*/logs")] if d.exists() else []
        frames = [f for f in frames if not f.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        n_ok = int(df[df["parse_ok"].fillna(False).astype(bool)]["source_call_id"].nunique()) if not df.empty else 0
        out[p] = {"coded_ok": n_ok, "source_ok": n_source_ok, "rate": (n_ok / n_source_ok) if n_source_ok else None}
    return out
