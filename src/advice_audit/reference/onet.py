"""O*NET 31.0 occupational interests (RIASEC) -> people–things score per SOC.

people_things = [(2S + E + A) - (2R + I + C)] / 8, Prediger's (1982) things–people
dimension with the sign flipped so that POSITIVE = people-oriented (cf. Su, Rounds &
Armstrong 2009). O*NET-SOC detail codes (xx-xxxx.yy) are averaged to 6-digit SOC.
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache

import pandas as pd

from .sources import path

RIASEC = {"Realistic": "R", "Investigative": "I", "Artistic": "A", "Social": "S", "Enterprising": "E", "Conventional": "C"}


@lru_cache(maxsize=1)
def interests() -> pd.DataFrame:
    zf = zipfile.ZipFile(path("onet_db"))
    member = next(n for n in zf.namelist() if n.endswith("Career Interest Types.xlsx") or n.endswith("/Interests.xlsx"))
    df = pd.read_excel(io.BytesIO(zf.read(member)))
    df = df[(df["Scale ID"] == "OI") & df["Element Name"].isin(RIASEC)]
    df["soc"] = df["O*NET-SOC Code"].astype(str).str[:7]
    wide = df.pivot_table(index="soc", columns="Element Name", values="Data Value", aggfunc="mean").rename(columns=RIASEC)
    wide["people_things"] = ((2 * wide["S"] + wide["E"] + wide["A"]) - (2 * wide["R"] + wide["I"] + wide["C"])) / 8.0
    return wide.reset_index()


def people_things(socs: list[str]) -> float | None:
    t = interests()
    sub = t[t["soc"].isin(socs)]
    return float(sub["people_things"].mean()) if not sub.empty else None
