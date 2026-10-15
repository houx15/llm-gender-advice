"""Rendering (spec §9): female/male versions differ ONLY in the name and the explicit cue;
CN/US versions share orientation and relative standing."""
import difflib

import pytest

from advice_audit.common.config import load_cells, load_profiles, profile_ids
from advice_audit.materials.design import profile_orientation
from advice_audit.materials.render import (
    CN_MAX, CUES, MaterialsTodoError, allocate_cn_scores, full_name, render_prompt, render_profile,
)


def _diff_tokens(a: str, b: str) -> tuple[list[str], list[str]]:
    sm = difflib.SequenceMatcher(a=list(a), b=list(b), autojunk=False)
    da, db = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            da.append(a[i1:i2])
            db.append(b[j1:j2])
    return da, db


@pytest.mark.parametrize("domain", ["majors", "jobs"])
@pytest.mark.parametrize("mode", ["open", "fixed"])
def test_female_male_differ_only_in_name_and_cue(domain, mode):
    for cell, c in load_cells().items():
        for pid in profile_ids(domain):
            for idx in range(3):
                for para in range(3):
                    for cue in ("implicit", "explicit"):
                        f = render_prompt(domain, mode, pid, cell, f"f_{cue}", idx, para, strict=False)
                        m = render_prompt(domain, mode, pid, cell, f"m_{cue}", idx, para, strict=False)
                        # remove name and cue from both; remainder must be identical
                        def strip(rp):
                            t = rp.text
                            spans = sorted([s for s in (rp.name_span, rp.cue_span) if s], reverse=True)
                            for s, e in spans:
                                t = t[:s] + "<X>" + t[e:]
                            return t
                        assert strip(f) == strip(m), (cell, pid, cue)
                        assert f.text[f.name_span[0]:f.name_span[1]] == full_name(cell, "f", idx)
                        assert m.text[m.name_span[0]:m.name_span[1]] == full_name(cell, "m", idx)
                        if cue == "explicit":
                            word_f = CUES[(c["language"], domain, "f")][1]
                            assert f.text[f.cue_span[0]:f.cue_span[1]] == word_f
                        else:
                            assert f.cue_span is None


def test_baseline_has_no_identity():
    for cell in load_cells():
        b = render_prompt("majors", "open", "maj_sci_high_r1", cell, "baseline", 0, 0, strict=False)
        f = render_prompt("majors", "open", "maj_sci_high_r1", cell, "f_implicit", 0, 0, strict=False)
        assert b.name is None and b.name_span is None
        assert f.text.endswith(b.text[-50:])
        assert full_name(cell, "f", 0) not in b.text


def test_same_surname_within_cell():
    for cell in load_cells():
        for i in range(3):
            f, m = full_name(cell, "f", i), full_name(cell, "m", i)
            n = load_profiles  # noqa: F841
            assert (f.split()[-1] == m.split()[-1]) if " " in f else (f[0] == m[0])


@pytest.mark.parametrize("domain", ["majors", "jobs"])
def test_cn_us_profiles_parallel(domain):
    us, cn = load_profiles(domain, "us"), load_profiles(domain, "cn")
    assert [p["id"] for p in us["profiles"]] == [p["id"] for p in cn["profiles"]]
    for pu, pc in zip(us["profiles"], cn["profiles"]):
        assert pu["level"] == pc["level"]
        if domain == "majors":
            assert pu["orientation"] == pc["orientation"] and pu["replicate"] == pc["replicate"]
        else:
            assert pu["major"] == pc["major"]
            assert us["majors"][pu["major"]]["orientation"] == cn["majors"][pc["major"]]["orientation"]
    for pid in profile_ids(domain):
        assert profile_orientation(domain, pid)[0] in ("science", "humanities", "balanced")


def test_us_sat_sections_sum_to_level_total():
    for pid in profile_ids("majors"):
        render_profile("majors", "us", pid, strict=True)  # asserts internally


def test_strict_mode_refuses_cn_todo():
    cfg = load_profiles("majors", "cn")
    if str(cfg["province"]).startswith("TODO"):
        with pytest.raises(MaterialsTodoError):
            render_profile("majors", "cn", "maj_sci_high_r1", strict=True)
        assert "[TODO:" in render_profile("majors", "cn", "maj_sci_high_r1", strict=False)


@pytest.mark.parametrize("total", [520, 600, 660, 690])
def test_allocate_cn_scores(total):
    cfg = load_profiles("majors", "cn")
    for ori, reps in cfg["subject_weights"].items():
        for r, w in reps.items():
            s = allocate_cn_scores(total, w)
            assert sum(s.values()) == total
            assert all(0 <= s[k] <= CN_MAX[k] for k in s)


def test_rendering_deterministic():
    a = render_prompt("jobs", "fixed", "job_cs_high", "C", "f_explicit", 1, 2)
    b = render_prompt("jobs", "fixed", "job_cs_high", "C", "f_explicit", 1, 2)
    assert a == b
