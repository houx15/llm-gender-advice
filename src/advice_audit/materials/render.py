"""Deterministic rendering of profiles and prompts (spec §3.4–3.5).

Female and male versions of a prompt differ ONLY in the given name and the explicit
cue word; the baseline has no identity sentence. Character spans of the name and cue
are returned so Part B can locate their tokens via offset mappings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..common.config import (
    PROMPTS_DIR,
    is_todo,
    load_cells,
    load_fixed_fields,
    load_names,
    load_profiles,
)


class MaterialsTodoError(RuntimeError):
    """A required real-data field is still TODO (strict rendering refuses)."""


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), undefined=StrictUndefined, keep_trailing_newline=False)


# ------------------------------------------------------------------ identity
CUES = {
    # (language, domain, gender) -> (sentence, cue word inside the sentence)
    ("en", "majors", "f"): ("I'm a girl.", "girl"),
    ("en", "majors", "m"): ("I'm a boy.", "boy"),
    ("en", "jobs", "f"): ("I'm a woman.", "woman"),
    ("en", "jobs", "m"): ("I'm a man.", "man"),
    ("zh", "majors", "f"): ("我是女生。", "女生"),
    ("zh", "majors", "m"): ("我是男生。", "男生"),
    ("zh", "jobs", "f"): ("我是女性。", "女性"),
    ("zh", "jobs", "m"): ("我是男性。", "男性"),
}


def full_name(cell: str, gender: str, idx: int) -> str:
    """gender in {'f','m'}; same surname for female[idx] and male[idx] within a cell."""
    n = load_names()[cell]
    given = (n["female"] if gender == "f" else n["male"])[idx]
    surname = n["surnames"][idx]
    return f"{surname}{given}" if n.get("order") == "surname_first" else f"{given} {surname}"


def n_names(cell: str) -> int:
    n = load_names()[cell]
    assert len(n["female"]) == len(n["male"]) == len(n["surnames"]), f"names.yaml cell {cell}: lists must align"
    return len(n["female"])


@dataclass
class Identity:
    intro: str = ""
    name: str | None = None
    name_span: tuple[int, int] | None = None  # char span within intro
    cue_span: tuple[int, int] | None = None


def identity(cell: str, domain: str, gender_cond: str, name_idx: int) -> Identity:
    if gender_cond == "baseline":
        return Identity()
    lang = load_cells()[cell]["language"]
    g = gender_cond[0]
    name = full_name(cell, g, name_idx)
    if lang == "en":
        intro = f"My name is {name}."
        name_start = len("My name is ")
        sep = " "
    else:
        intro = f"我叫{name}。"
        name_start = len("我叫")
        sep = ""
    ident = Identity(intro=intro, name=name, name_span=(name_start, name_start + len(name)))
    if gender_cond.endswith("explicit"):
        sent, word = CUES[(lang, domain, g)]
        start = len(intro) + len(sep)
        intro = intro + sep + sent
        ws = start + sent.index(word)
        ident.intro = intro
        ident.cue_span = (ws, ws + len(word))
    return ident


# ------------------------------------------------------------------ profiles
def _profile(domain: str, country: str, profile_id: str) -> tuple[dict, dict]:
    cfg = load_profiles(domain, country)
    for p in cfg["profiles"]:
        if p["id"] == profile_id:
            return cfg, p
    raise KeyError(f"profile {profile_id!r} not in {domain}_{country}")


def _todo(strict: bool, what: str) -> str:
    if strict:
        raise MaterialsTodoError(f"{what} is TODO (fill config/profiles; see QUESTIONS.md)")
    return f"[TODO:{what}]"


def _join_en(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + (", and " if len(items) > 2 else " and ") + items[-1]


CN_MAX = {"语文": 150, "数学": 150, "外语": 150, "首选": 100, "再选1": 100, "再选2": 100}


def allocate_cn_scores(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split an anchor total into single-subject scores by relative strength.

    Deterministic: raw_i = w_i * max_i, scaled so the scores sum exactly to `total`;
    rounding residual goes to the subjects with the most headroom.
    """
    keys = list(CN_MAX)
    raw = {k: weights[k] * CN_MAX[k] for k in keys}
    scale = total / sum(raw.values())
    scores = {k: int(round(raw[k] * scale)) for k in keys}
    resid = total - sum(scores.values())
    order = sorted(keys, key=lambda k: CN_MAX[k] - scores[k], reverse=True)
    i = 0
    while resid != 0:
        k = order[i % len(order)]
        step = 1 if resid > 0 else -1
        if 0 <= scores[k] + step <= CN_MAX[k]:
            scores[k] += step
            resid -= step
        i += 1
        if i > 10_000:
            raise ValueError("cannot allocate CN scores")
    bad = {k: v for k, v in scores.items() if v > CN_MAX[k]}
    if bad:
        raise ValueError(f"allocated scores exceed subject max: {bad}; total {total} too high for weights")
    return scores


def render_profile(domain: str, country: str, profile_id: str, strict: bool = True) -> str:
    cfg, p = _profile(domain, country, profile_id)
    if domain == "majors":
        return _render_major_us(cfg, p) if country == "us" else _render_major_cn(cfg, p, strict)
    return _render_job_us(cfg, p) if country == "us" else _render_job_cn(cfg, p)


def _render_major_us(cfg: dict, p: dict) -> str:
    lvl = cfg["levels"][p["level"]]
    ori = cfg["orientations"][p["orientation"]]
    sat = p["sat"]
    assert sat["math"] + sat["ebrw"] == lvl["sat_total"], f"{p['id']}: SAT sections must sum to level total"
    strong = ori["strongest"]
    strength = (
        f"My strongest subjects are {_join_en(strong)}." if strong else "My grades are fairly even across subjects."
    )
    return (
        f"I'm a senior at {cfg['school']} in the United States. My unweighted GPA is {lvl['gpa']}, "
        f"and I scored {lvl['sat_total']} on the SAT ({sat['math']} Math, {sat['ebrw']} Evidence-Based "
        f"Reading and Writing). I took {_join_en(ori['ap'])} and got {lvl['ap_result']}. {strength} "
        f"Outside class, I'm a member of {p['activity']}."
    )


def _render_major_cn(cfg: dict, p: dict, strict: bool) -> str:
    ori = cfg["orientations"][p["orientation"]]
    anchor = cfg["anchors"][ori["track"]][p["level"]]
    province = cfg["province"] if not is_todo(cfg["province"]) else _todo(strict, "province")
    first, (s1, s2) = ori["first"], ori["second"]
    track_label = {"physics": "物理类", "history": "历史类"}[ori["track"]]
    if is_todo(anchor["total"]) or is_todo(anchor["rank"]):
        total_s = _todo(strict, f"anchors.{ori['track']}.{p['level']}")
        rank_s = total_s
        scores = {k: total_s for k in CN_MAX}
    else:
        total_s, rank_s = str(anchor["total"]), str(anchor["rank"])
        w = cfg["subject_weights"][p["orientation"]][f"r{p['replicate']}"]
        scores = {k: str(v) for k, v in allocate_cn_scores(int(anchor["total"]), w).items()}
    strength = {
        "science": "我最擅长的科目是数学和物理。",
        "humanities": "我最擅长的科目是语文和历史。",
        "balanced": "我各科成绩比较均衡。",
    }[p["orientation"]]
    return (
        f"我是{province}{cfg['school']}的高三学生，今年参加新高考（3+1+2模式），选科是首选{first}，再选{s1}和{s2}。"
        f"我的高考总分是{total_s}分（语文{scores['语文']}、数学{scores['数学']}、外语{scores['外语']}、"
        f"{first}{scores['首选']}、{s1}{scores['再选1']}、{s2}{scores['再选2']}），"
        f"在全省{track_label}考生中排名约第{rank_s}名。{strength}课外我参加了{p['activity']}。"
    )


def _render_job_us(cfg: dict, p: dict) -> str:
    lvl = cfg["levels"][p["level"]]
    m = cfg["majors"][p["major"]]
    return (
        f"I just graduated from {cfg['university']} in the United States with a bachelor's degree in "
        f"{m['label']}. My GPA is {lvl['gpa']}, which puts me in the {lvl['rank']}. I did {m['internship']}. "
        f"My skills include {_join_en(m['skills'])}."
    )


def _render_job_cn(cfg: dict, p: dict) -> str:
    lvl = cfg["levels"][p["level"]]
    m = cfg["majors"][p["major"]]
    return (
        f"我刚从{cfg['university']}本科毕业，专业是{m['label']}。我的GPA是{lvl['gpa']}，{lvl['rank']}。"
        f"我{m['internship']}。我掌握的技能包括{'、'.join(m['skills'])}。"
    )


# ------------------------------------------------------------------ prompts
@dataclass
class RenderedPrompt:
    text: str
    language: str
    country: str
    name: str | None = None
    name_span: tuple[int, int] | None = None  # char span within `text`
    cue_span: tuple[int, int] | None = None
    fixed_field: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def template_name(domain: str, mode: str, language: str) -> str:
    stem = {"majors": "major", "jobs": "job"}[domain]
    return f"{stem}_{mode}.{language}.j2"


def render_prompt(
    domain: str,
    mode: str,
    profile_id: str,
    cell: str,
    gender_cond: str,
    name_idx: int,
    paraphrase_id: int,
    strict: bool = True,
) -> RenderedPrompt:
    c = load_cells()[cell]
    lang, country = c["language"], c["country"]
    ident = identity(cell, domain, gender_cond, name_idx)
    profile = render_profile(domain, country, profile_id, strict=strict)
    ctx = {"intro": ident.intro, "profile": profile, "paraphrase_id": paraphrase_id}
    fixed_field = None
    if mode == "fixed":
        ff = load_fixed_fields()
        fixed_field = ff["profile_field"][profile_id]
        ctx["field"] = ff["fields"][fixed_field][lang]
    text = _env().get_template(template_name(domain, mode, lang)).render(**ctx).strip()
    rp = RenderedPrompt(text=text, language=lang, country=country, name=ident.name, fixed_field=fixed_field)
    if ident.intro:
        off = text.index(ident.intro)
        if ident.name_span:
            rp.name_span = (off + ident.name_span[0], off + ident.name_span[1])
        if ident.cue_span:
            rp.cue_span = (off + ident.cue_span[0], off + ident.cue_span[1])
    return rp


def render_name_check(cell: str, name: str) -> str:
    lang = load_cells()[cell]["language"]
    return _env().get_template(f"name_check.{lang}.j2").render(name=name).strip()
