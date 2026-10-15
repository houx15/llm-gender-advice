"""Fill taxonomy reference attributes (spec §3.6) from cached public files.

jobs  : female_share_us (BLS CPS Table 11, 2025), median_earnings (BLS OEWS May 2025),
        prestige_isei (ISEI-08 via SOC->ISCO-08), people_things (O*NET 31.0 interests)
majors: female_share_us (IPEDS 2023-24 bachelor's completions), median_earnings (College
        Scorecard field of study), prestige_isei and people_things averaged over the
        occupations linked by the NCES CIP-SOC crosswalk (postsecondary-teacher SOCs
        25-1xxx dropped unless they are the only link)
Values that cannot be computed stay empty and are listed in QUESTIONS.md.
"""
from __future__ import annotations

import pandas as pd

from ..common.config import REFERENCE_BUILT
from ..common.io import append_question, utcnow, write_json
from ..common.logs import get_logger
from ..materials.taxonomy import load_taxonomy

log = get_logger("advice_audit.reference")


def _safe(fn, *a, problems: list[str], what: str):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        problems.append(f"{what}: {type(e).__name__}: {e}")
        return None


def build_jobs(problems: list[str]) -> pd.DataFrame:
    from . import bls, isei, onet

    tax = load_taxonomy("jobs", with_attributes=False)
    shares, p = _safe(bls.cps_female_share_by_job, problems=problems, what="CPS") or ({}, [])
    problems.extend(p)
    oews = _safe(bls.oews_national, problems=problems, what="OEWS")
    rows = []
    for r in tax.itertuples():
        socs = list(r.codes)
        rows.append({
            "id": r.id,
            "female_share_us": shares.get(r.id),
            "median_earnings": bls.oews_median(socs, oews) if oews is not None else None,
            "prestige_isei": _safe(isei.isei_for_socs, socs, problems=problems, what=f"ISEI {r.id}"),
            "people_things": _safe(onet.people_things, socs, problems=problems, what=f"O*NET {r.id}"),
            "linked_socs": ";".join(socs),
        })
    return pd.DataFrame(rows)


def build_majors(problems: list[str]) -> pd.DataFrame:
    from . import crosswalks, ipeds, isei, onet, scorecard

    tax = load_taxonomy("majors", with_attributes=False)
    rows = []
    for r in tax.itertuples():
        cips = list(r.codes)
        socs = _safe(crosswalks.socs_for_cips, cips, problems=problems, what=f"CIP-SOC {r.id}") or []
        non_teach = [s for s in socs if not s.startswith("25-1")]
        socs_used = non_teach or socs
        fs = _safe(ipeds.female_share, cips, problems=problems, what=f"IPEDS {r.id}")
        rows.append({
            "id": r.id,
            "female_share_us": fs[0] if fs else None,
            "ipeds_n_degrees": fs[1] if fs else None,
            "ipeds_levels": fs[2] if fs else None,
            "median_earnings": _safe(scorecard.median_earnings, cips, problems=problems, what=f"Scorecard {r.id}"),
            "prestige_isei": _safe(isei.isei_for_socs, socs_used, problems=problems, what=f"ISEI {r.id}") if socs_used else None,
            "people_things": _safe(onet.people_things, socs_used, problems=problems, what=f"O*NET {r.id}") if socs_used else None,
            "linked_socs": ";".join(socs_used),
        })
    return pd.DataFrame(rows)


def build_all() -> dict[str, pd.DataFrame]:
    REFERENCE_BUILT.mkdir(parents=True, exist_ok=True)
    out, prov = {}, {"built_at": utcnow(), "missing": {}}
    for domain, fn in (("jobs", build_jobs), ("majors", build_majors)):
        problems: list[str] = []
        df = fn(problems)
        df.to_csv(REFERENCE_BUILT / f"taxonomy_{domain}.csv", index=False)
        out[domain] = df
        attrs = ["female_share_us", "median_earnings", "prestige_isei", "people_things"]
        missing = {a: df.loc[df[a].isna(), "id"].tolist() for a in attrs}
        missing = {a: v for a, v in missing.items() if v}
        prov["missing"][domain] = missing
        prov.setdefault("problems", {})[domain] = sorted(set(problems))[:200]
        log.info(f"{domain}: {len(df)} categories; missing: { {a: len(v) for a, v in missing.items()} }")
        if missing or problems:
            body = "\n".join(f"- `{a}` missing for: {', '.join(v)}" for a, v in missing.items())
            if problems:
                body += "\n\nLoader problems (first 20):\n" + "\n".join(f"- {p}" for p in sorted(set(problems))[:20])
            append_question(f"Reference attributes incomplete: {domain}", body + "\n\nValues were left empty (never invented).")
    write_json(REFERENCE_BUILT / "provenance.json", prov)
    return out
