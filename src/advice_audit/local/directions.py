"""g (gender), s (status), p (people–things) directions per layer (spec §6.5).

g: mean(female) − mean(male) over matched design prompts, implicit and explicit cues
   separately (and pooled), at the `last` (readout) and `name_last` positions; the probe
   weight vector is saved as an alternative.
s: concept prompts naming taxonomy options; per layer, OLS of activations on
   [prestige_z, female_share_z] (+ domain, language, template fixed effects); s = the
   prestige coefficient vector. Held-out-category check: the s-projection should correlate
   with prestige but ~0 with female share (reported, and asserted to a tolerance with a
   warning).
p: same with people_things_z, controlling prestige_z.
Reports cos(g,s), cos(g,p), cos(s,p) per layer with bootstrap CIs over prompt sets,
cross-lingual cos(g_EN, g_ZH), cell-B alignment (cos(g_B,g_A) vs cos(g_B,g_C)), and
correlations of concept projections on g with real-world attributes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..common.io import write_json
from ..common.logs import get_logger
from ..materials.taxonomy import load_taxonomy
from .shards import stage_dir

log = get_logger("advice_audit.local.directions")
NEAR_ZERO_CORR = 0.2


def mean_diff(acts: torch.Tensor, mask_f: np.ndarray, mask_m: np.ndarray) -> torch.Tensor:
    """[n, L, d] -> [L, d]"""
    return acts[torch.as_tensor(mask_f)].float().mean(0) - acts[torch.as_tensor(mask_m)].float().mean(0)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine for [L, d] tensors -> [L]."""
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def g_directions(acts, valid, idx, meta) -> dict[str, torch.Tensor]:
    out = {}
    gc = idx["gender_cond"]
    for pos in ("last", "name_last"):
        if pos not in meta["positions"]:
            continue
        pi = meta["positions"].index(pos)
        a, v = acts[:, pi], valid[:, pi].numpy()
        for cue in ("implicit", "explicit", "all"):
            sel = gc.str.endswith(cue) if cue != "all" else gc.ne("baseline")
            f = (gc.str.startswith("f_") & sel).to_numpy() & v
            m = (gc.str.startswith("m_") & sel).to_numpy() & v
            if f.sum() and m.sum():
                out[f"g_{cue}_{pos}"] = mean_diff(a, f, m)
        for lab, cells in (("en", ["A", "B"]), ("zh", ["C"]), ("A", ["A"]), ("B", ["B"]), ("C", ["C"])):
            c = idx["cell"].isin(cells).to_numpy()
            f, m = gc.str.startswith("f_").to_numpy() & v & c, gc.str.startswith("m_").to_numpy() & v & c
            if f.sum() and m.sum():
                out[f"g_{lab}_{pos}"] = mean_diff(a, f, m)
    return out


def _concept_design(cidx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in ("majors", "jobs"):
        t = load_taxonomy(domain)[["id", "female_share_us", "prestige_isei", "people_things", "median_earnings"]].copy()
        for c in ("female_share_us", "prestige_isei", "people_things", "median_earnings"):
            t[c + "_z"] = (t[c] - t[c].mean()) / t[c].std()
        t["domain"] = domain
        rows.append(t)
    attrs = pd.concat(rows)
    return cidx.merge(attrs, left_on=["domain", "category_id"], right_on=["domain", "id"], how="left")


def _nuisance(d: pd.DataFrame) -> np.ndarray:
    dummies = pd.get_dummies(d[["domain", "language"]].astype(str) .assign(t=d["template_id"].astype(str)), drop_first=True)
    return np.column_stack([np.ones(len(d)), dummies.to_numpy(dtype=float)])


def regress_direction(X: torch.Tensor, d: pd.DataFrame, target: str, control: str) -> torch.Tensor:
    """X [n, L, d]; returns coefficient on target_z controlling control_z: [L, d]."""
    Z = np.column_stack([_nuisance(d), d[target + "_z"].to_numpy(), d[control + "_z"].to_numpy()])
    Zt = torch.as_tensor(Z, dtype=torch.float32)
    n, L, D = X.shape
    Y = X.float().reshape(n, L * D)
    beta = torch.linalg.lstsq(Zt, Y).solution  # [k, L*D]
    return beta[-2].reshape(L, D)


def held_out_check(X, d: pd.DataFrame, target: str, control: str, folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Fit on train categories, project held-out prompts; corr with target and control."""
    cats = np.array(sorted(d["category_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(cats)
    L = X.shape[1]
    proj = np.zeros((len(d), L))
    for part in np.array_split(cats, folds):
        te = d["category_id"].isin(part).to_numpy()
        vec = regress_direction(X[torch.as_tensor(~te)], d[~te], target, control)
        unit = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        proj[te] = torch.einsum("nld,ld->nl", X[torch.as_tensor(te)].float(), unit).numpy()
    rows = []
    for l in range(L):
        rows.append({"layer_idx": l,
                     f"corr_{target}": float(np.corrcoef(proj[:, l], d[target])[0, 1]),
                     f"corr_{control}": float(np.corrcoef(proj[:, l], d[control])[0, 1])})
    return pd.DataFrame(rows)


def run_directions(model_key: str, acts, valid, idx, meta, cfg: dict) -> dict:
    from .extract import load_activations
    from .probes import probe_weights

    out_dir = stage_dir(model_key, "directions")
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = meta["layers"]
    dirs = g_directions(acts, valid, idx, meta)
    if "last" in meta["positions"]:
        pi = meta["positions"].index("last")
        dirs["resid_norm_last"] = acts[:, pi].float().norm(dim=-1).mean(0)  # [L]
        try:
            dirs["g_probe_last"] = probe_weights(acts, valid, idx, meta, "last")
        except Exception as e:  # noqa: BLE001
            log.warning(f"probe-weight direction failed: {e}")

    metrics: dict = {"layers": layers}
    try:
        cacts, cvalid, cidx, cmeta = load_activations(model_key, "concepts")
        cpos = "label_last" if "label_last" in cmeta["positions"] else "last"
        cpi = cmeta["positions"].index(cpos)
        cd = _concept_design(cidx)
        ok = cd[["prestige_isei_z", "female_share_us_z", "people_things_z"]].notna().all(axis=1).to_numpy() & cvalid[:, cpi].numpy()
        X, cd = cacts[torch.as_tensor(ok), cpi], cd[ok].reset_index(drop=True)
        dirs["s"] = regress_direction(X, cd, "prestige_isei", "female_share_us")
        dirs["p"] = regress_direction(X, cd, "people_things", "prestige_isei")
        hs = held_out_check(X, cd, "prestige_isei", "female_share_us", cfg["directions"].get("cv_folds", 5))
        hp = held_out_check(X, cd, "people_things", "prestige_isei", cfg["directions"].get("cv_folds", 5))
        hs.to_csv(out_dir / "s_heldout.csv", index=False)
        hp.to_csv(out_dir / "p_heldout.csv", index=False)
        worst = float(hs["corr_female_share_us"].abs().max())
        metrics["s_heldout_max_abs_corr_female_share"] = worst
        if worst > NEAR_ZERO_CORR:
            log.warning(f"s-projection correlates with female share up to |r|={worst:.2f} on held-out categories "
                        f"(> {NEAR_ZERO_CORR}); status direction may be confounded — see s_heldout.csv")
        # concept projections on g (geometry of culture)
        g = dirs.get("g_implicit_last", dirs.get("g_all_last"))
        if g is not None:
            unit = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj = torch.einsum("nld,ld->nl", X.float(), unit).numpy()
            cd_proj = cd.assign(**{f"proj_g_L{layers[l]}": proj[:, l] for l in range(len(layers))})
            cat = cd_proj.groupby(["domain", "category_id"]).mean(numeric_only=True).reset_index()
            rows = []
            for l in range(len(layers)):
                c = f"proj_g_L{layers[l]}"
                rows.append({"layer": layers[l], **{f"corr_{a}": float(cat[[c, a]].corr().iloc[0, 1])
                                                     for a in ("female_share_us", "prestige_isei", "median_earnings", "people_things")}})
            pd.DataFrame(rows).to_csv(out_dir / "concept_projections_on_g.csv", index=False)
    except SystemExit as e:
        log.warning(f"concept activations unavailable ({e}); s and p directions skipped")

    torch.save({k: v.cpu() for k, v in dirs.items()}, out_dir / "directions.pt")
    geo = geometry(dirs, acts, valid, idx, meta, cfg["directions"].get("n_boot", 200), layers)
    geo.to_csv(out_dir / "geometry.csv", index=False)
    write_json(out_dir / "metrics.json", metrics)
    return dirs


def geometry(dirs, acts, valid, idx, meta, n_boot: int, layers: list[int]) -> pd.DataFrame:
    """Cosines per layer with bootstrap CIs (g resampled over design prompts)."""
    rng = np.random.default_rng(0)
    rows = {l: {"layer": l} for l in layers}

    def put(name, vals):
        for i, l in enumerate(layers):
            rows[l][name] = float(vals[i])

    g = dirs.get("g_implicit_last", dirs.get("g_all_last"))
    pairs = {"cos_g_s": ("g", "s"), "cos_g_p": ("g", "p"), "cos_s_p": ("s", "p")}
    for name, (a, b) in pairs.items():
        va = g if a == "g" else dirs.get(a)
        vb = dirs.get(b)
        if va is not None and vb is not None:
            put(name, cos(va, vb).numpy())
    for name, (a, b) in {"cos_gEN_gZH": ("g_en_last", "g_zh_last"), "cos_gB_gA": ("g_B_last", "g_A_last"),
                         "cos_gB_gC": ("g_B_last", "g_C_last"), "cos_gimp_gexp": ("g_implicit_last", "g_explicit_last")}.items():
        if a in dirs and b in dirs:
            put(name, cos(dirs[a], dirs[b]).numpy())
    # bootstrap g over prompts (implicit, last)
    if g is not None and "last" in meta["positions"] and n_boot:
        pi = meta["positions"].index("last")
        gc = idx["gender_cond"]
        f_ix = np.where((gc == "f_implicit").to_numpy() & valid[:, pi].numpy())[0]
        m_ix = np.where((gc == "m_implicit").to_numpy() & valid[:, pi].numpy())[0]
        boots = {k: [] for k in ("cos_g_s", "cos_g_p")}
        for _ in range(n_boot):
            fb, mb = rng.choice(f_ix, len(f_ix)), rng.choice(m_ix, len(m_ix))
            gb = acts[fb, pi].float().mean(0) - acts[mb, pi].float().mean(0)
            for k, other in (("cos_g_s", "s"), ("cos_g_p", "p")):
                if other in dirs:
                    boots[k].append(cos(gb, dirs[other]).numpy())
        for k, v in boots.items():
            if v:
                arr = np.stack(v)
                put(k + "_lo", np.percentile(arr, 2.5, axis=0))
                put(k + "_hi", np.percentile(arr, 97.5, axis=0))
    return pd.DataFrame(list(rows.values()))
