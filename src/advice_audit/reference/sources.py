"""Public reference files (spec §6.1 cache_reference_data). Online side only.

bls.gov is Akamai-blocked from some networks; BLS files fall back to Wayback copies of
the same official files. Any file that cannot be fetched is written to QUESTIONS.md.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..common.config import REFERENCE_RAW
from ..common.io import append_question

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
WAYBACK = "https://web.archive.org/web/{ts}id_/{url}"

SOURCES = {
    # name: (url, local filename, wayback fallback?)
    "cps_table11": ("https://www.bls.gov/cps/cpsaat11.htm", "cpsaat11.html", True),
    "oews_national": ("https://www.bls.gov/oes/special-requests/oesm25nat.zip", "oesm25nat.zip", True),
    "onet_db": ("https://www.onetcenter.org/dl_files/database/db_31_0_excel.zip", "onet/db_31_0_excel.zip", False),
    "ipeds_completions": ("https://nces.ed.gov/ipeds/datacenter/data/C2024_A.zip", "C2024_A.zip", False),
    "cip_soc_crosswalk": ("https://nces.ed.gov/ipeds/cipcode/Files/CIP2020_SOC2018_Crosswalk.xlsx", "CIP2020_SOC2018_Crosswalk.xlsx", False),
    "isco_soc_crosswalk": ("https://www.bls.gov/soc/ISCO_SOC_Crosswalk.xls", "ISCO_SOC_Crosswalk.xls", True),
    "soc_2010_2018": ("https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx", "soc_2010_to_2018_crosswalk.xlsx", True),
    "isei08": ("http://www.harryganzeboom.nl/isco08/isco08_with_isei.pdf", "isco08_with_isei.pdf", False),
    "scorecard_fos": ("SCORECARD", "scorecard_fos.zip", False),  # resolved from the data page
    "wikitext_ppl_en": ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet",
                        "heldout/wikitext2_test.parquet", False),
}
SCORECARD_PAGE = "https://collegescorecard.ed.gov/data/"


def path(name: str) -> Path:
    return REFERENCE_RAW / SOURCES[name][1]


def _get(url: str) -> bytes:
    import httpx

    with httpx.Client(headers=UA, follow_redirects=True, timeout=600) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content


def _scorecard_url() -> str:
    html = _get(SCORECARD_PAGE).decode("utf-8", "ignore")
    links = sorted(set(re.findall(r"https?://[^\"']*Most-Recent-Cohorts-Field-of-Study[^\"']*\.zip", html)))
    if not links:
        raise RuntimeError("no Field-of-Study zip link on the Scorecard data page")
    return links[-1]


def fetch(name: str, force: bool = False) -> Path | None:
    url, fname, wayback = SOURCES[name]
    dest = REFERENCE_RAW / fname
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    try:
        if url == "SCORECARD":
            url = _scorecard_url()
        dest.write_bytes(_get(url))
        return dest
    except Exception as e:  # noqa: BLE001
        errors.append(f"{url}: {type(e).__name__}: {e}")
    if wayback:
        for ts in ("2026", "2025", "2024"):
            try:
                data = _get(WAYBACK.format(ts=ts, url=url))
                if data[:15].lower().startswith((b"<!doctype html", b"<html")) and not fname.endswith((".html", ".htm")):
                    raise RuntimeError("wayback returned an HTML page")
                dest.write_bytes(data)
                return dest
            except Exception as e:  # noqa: BLE001
                errors.append(f"wayback {ts}: {type(e).__name__}: {e}")
    append_question(
        f"Reference download blocked: {name}",
        f"Could not download `{name}` to `{dest}`.\n\n" + "\n".join(f"- {e}" for e in errors)
        + "\n\nPlease download it manually to that path (or provide an alternative source).",
    )
    return None


def fetch_all(force: bool = False) -> dict[str, Path | None]:
    return {n: fetch(n, force=force) for n in SOURCES}
