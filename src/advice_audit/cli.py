"""advice-audit CLI (spec §8)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

app = typer.Typer(help="Different or Lesser? — LLM gender bias in major and career advice.", no_args_is_help=True)
materials_app = typer.Typer(help="Materials: profiles, names, taxonomy.", no_args_is_help=True)
api_app = typer.Typer(help="Part A: API runs.", no_args_is_help=True)
local_app = typer.Typer(help="Part B: offline local-model pipeline (planning and reports; GPU work runs via Slurm).",
                        no_args_is_help=True)
reference_app = typer.Typer(help="Reference data (online side).", no_args_is_help=True)
models_app = typer.Typer(help="API model checks.", no_args_is_help=True)
app.add_typer(materials_app, name="materials")
app.add_typer(api_app, name="api")
app.add_typer(local_app, name="local")
app.add_typer(reference_app, name="reference")
app.add_typer(models_app, name="models")


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


# ------------------------------------------------------------------ materials
@materials_app.command("build")
def materials_build(strict: bool = typer.Option(False, help="fail on TODO fields (CN anchors, province)")):
    """Render all profiles/prompts, validate names and taxonomy, report TODOs."""
    from .common.config import OUTPUTS_DIR, load_cells, profile_ids
    from .materials.names import validate_names
    from .materials.render import MaterialsTodoError, render_prompt, render_profile
    from .materials.taxonomy import load_taxonomy, validate

    problems = validate_names() + validate("majors") + validate("jobs")
    todos = []
    out = OUTPUTS_DIR / "materials"
    out.mkdir(parents=True, exist_ok=True)
    for domain in ("majors", "jobs"):
        for country in ("us", "cn"):
            for pid in profile_ids(domain):
                try:
                    render_profile(domain, country, pid, strict=True)
                except MaterialsTodoError as e:
                    todos.append(f"{domain}/{country}/{pid}: {e}")
                    if strict:
                        raise
        lines = []
        for cell in load_cells():
            for mode in ("open", "fixed"):
                for gc in ("f_explicit", "m_explicit", "baseline"):
                    rp = render_prompt(domain, mode, profile_ids(domain)[0], cell, gc, 0, 0, strict=False)
                    lines.append(f"### {domain} | {mode} | cell {cell} | {gc}\n\n{rp.text}\n")
        (out / f"samples_{domain}.md").write_text("\n".join(lines), encoding="utf-8")
    for domain in ("majors", "jobs"):
        t = load_taxonomy(domain)
        miss = t[["female_share_us", "median_earnings", "prestige_isei", "people_things"]].isna().sum().to_dict()
        typer.echo(f"taxonomy {domain}: {len(t)} categories; missing attributes: {miss}")
    typer.echo(f"sample prompts -> {out}")
    for p in problems:
        typer.secho(f"PROBLEM: {p}", fg="red")
    if todos:
        typer.secho(f"{len(todos)} profiles still have TODO real-data fields (strict rendering refuses them):", fg="yellow")
        for t in sorted({t.split(': ', 1)[1] for t in todos}):
            typer.echo(f"  - {t}")


@materials_app.command("name-check")
def materials_name_check(models: str = typer.Option(None, help="comma list; default all API models"),
                         dry_run: bool = typer.Option(False), samples: int = typer.Option(None),
                         concurrency: int = typer.Option(8)):
    """API manipulation check of names (budgeted, phase 'materials')."""
    from .api.runner import execute_plan
    from .common.config import load_api_models
    from .materials.names import name_check_plan, name_check_postprocess

    keys = _csv(models) or [m.key for m in load_api_models().models]
    plan = name_check_plan(keys, samples)
    for mk, g in plan.groupby("model_key"):
        asyncio.run(execute_plan(g.reset_index(drop=True), mk, f"name_check__{mk}", "materials", dry_run=dry_run,
                                 concurrency=concurrency, max_tokens=20, est_input_tokens=40, est_output_tokens=5,
                                 postprocess=name_check_postprocess))
    if not dry_run:
        materials_name_summary()


@materials_app.command("name-summary")
def materials_name_summary():
    """Summarize the name check; flag names below threshold in QUESTIONS.md."""
    import pandas as pd

    from .api.runner import load_run_logs
    from .common.config import OUTPUTS_DIR, RUNS_DIR
    from .materials.names import summarize_name_check

    frames = [load_run_logs(p.name) for p in RUNS_DIR.glob("name_check__*") if p.is_dir()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise typer.Exit("no name-check logs yet")
    summ = summarize_name_check(pd.concat(frames, ignore_index=True))
    path = OUTPUTS_DIR / "materials" / "name_check_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(path, index=False)
    typer.echo(summ.to_string())
    typer.echo(f"-> {path}")


# ------------------------------------------------------------------ plan / budget / models
@app.command()
def plan(experiment: str = typer.Option(..., help="config/experiments/<name>.yaml")):
    """Expand an experiment into a call plan; print call counts and cost estimate."""
    from .api.plan import balance_report, build_plan, cost_table, write_plan
    from .common.config import load_experiment
    from .materials.render import MaterialsTodoError

    exp = load_experiment(experiment)
    try:
        df = build_plan(exp, strict=True)
    except MaterialsTodoError as e:
        raise typer.Exit(f"cannot build plan: {e}. Fill the TODOs or restrict `cells` in the experiment config.")
    path = write_plan(df, exp.name)
    typer.echo(f"plan {exp.name}: {len(df)} calls -> {path}")
    typer.echo(json.dumps(balance_report(df)["calls_per_domain_mode"], default=str))
    typer.echo(cost_table(df).to_string(index=False))
    typer.echo("(config prices; live endpoint prices are checked at run time)")


@app.command()
def budget():
    """Budget status from the live ledger."""
    from .api.budget import BudgetGuard

    typer.echo(BudgetGuard().status())


@models_app.command("verify")
def models_verify():
    """Check slugs, canonical (dated) slugs, pinned providers and live prices on OpenRouter."""
    import os

    from .api.client import OpenRouterClient
    from .common.config import load_api_models

    async def go():
        async with OpenRouterClient(os.environ.get("OPENROUTER_API_KEY", "")) as c:
            for m in load_api_models().models:
                r = await c.verify_model(m)
                status = "OK" if r["ok"] and not r["problems"] else "CHECK"
                typer.echo(f"[{status}] {m.key}: live={r.get('canonical_slug_live')} price={r.get('live_price')} "
                           f"config={m.price_per_mtok} problems={r['problems']}")

    asyncio.run(go())


# ------------------------------------------------------------------ api
@api_app.command("run")
def api_run(plan: str = typer.Option(..., help="plan name (outputs/plans/<name>.parquet) or path"),
            model: str = typer.Option(...), dry_run: bool = typer.Option(False), max_calls: int = typer.Option(None),
            allow_reserve: bool = typer.Option(False), concurrency: int = typer.Option(8),
            run_id: str = typer.Option(None, help="default: <plan>__<model>")):
    """Run a plan on one model (resumable; budget-guarded)."""
    from .api.plan import load_plan
    from .api.runner import execute_plan
    from .common.config import require_approval

    df = load_plan(plan)
    df = df[df["model_key"] == model].reset_index(drop=True)
    if df.empty:
        raise typer.Exit(f"plan has no rows for model {model}")
    phase = str(df["phase"].iloc[0])
    if phase == "main":
        require_approval("A4")
    name = Path(plan).stem
    asyncio.run(execute_plan(df, model, run_id or f"{name}__{model}", phase, dry_run=dry_run, max_calls=max_calls,
                             allow_reserve=allow_reserve, concurrency=concurrency))


# ------------------------------------------------------------------ coding / analysis
@app.command()
def code(run: str = typer.Option(..., help="run id or a directory of Part-A-schema logs (e.g. outputs/local/<m>/behavior)"),
         passes: str = typer.Option("taxonomy,discursive"), dry_run: bool = typer.Option(False),
         max_calls: int = typer.Option(None), allow_reserve: bool = typer.Option(False), concurrency: int = typer.Option(8)):
    """Taxonomy mapping + discursive coding (budgeted, phase 'coding')."""
    from .coding.coder import run_coding

    run_coding(run, tuple(_csv(passes)), dry_run=dry_run, max_calls=max_calls, allow_reserve=allow_reserve,
               concurrency=concurrency)


@app.command("code-export")
def code_export(run: str = typer.Option(...), frac: float = typer.Option(0.05)):
    """Export a stratified sample (model × cell × gender) for bilingual human coders."""
    from .coding.human import export_sample

    typer.echo(f"-> {export_sample(run, frac)}")


@app.command("code-kappa")
def code_kappa(run: str = typer.Option(...), coder1: Path = typer.Option(None), coder2: Path = typer.Option(None)):
    """Cohen's κ (human–human, human–LLM) once human codes are returned."""
    from .coding.human import kappa_report

    typer.echo(kappa_report(run, coder1, coder2).to_string(index=False))


@app.command()
def analyze(run: list[str] = typer.Option(..., help="run id(s); repeat --run for several models"),
            name: str = typer.Option(None), pilot: bool = typer.Option(False, help="add the A3 STOP banner"),
            n_boot: int = typer.Option(300)):
    """Outcomes, regressions (H1–H7), mirror-vs-amplify; markdown + PNG report."""
    from .analysis.report import build_report

    typer.echo(f"-> {build_report(run, name, pilot=pilot, n_boot=n_boot)}")


# ------------------------------------------------------------------ reference
@reference_app.command("fetch")
def reference_fetch(force: bool = typer.Option(False)):
    """Download public reference files (online). Blocked URLs -> QUESTIONS.md."""
    from .reference.sources import fetch_all

    for k, v in fetch_all(force).items():
        typer.echo(f"{'OK ' if v else 'MISSING'} {k}: {v}")


@reference_app.command("build")
def reference_build():
    """Fill taxonomy reference attributes -> data/reference/built/."""
    from .reference.build import build_all

    build_all()


# ------------------------------------------------------------------ local (Part B)
@local_app.command("plan")
def local_plan(model: str = typer.Option(...),
               stage: str = typer.Option(..., help="behavior|readout|extract|concepts|patching|steering|erasure"),
               cells: str = typer.Option(None, help="restrict cells, e.g. A,B (before CN anchors exist)"),
               limit: int = typer.Option(None, help="keep first N prompts (smoke tests)"),
               n_layers: int = typer.Option(None), hidden: int = typer.Option(None)):
    """Write prompt shards for a Slurm stage (no GPU needed)."""
    from .local.prompts import plan_stage

    typer.echo(json.dumps(plan_stage(model, stage, n_layers, hidden, _csv(cells), limit), indent=2))


@local_app.command("aggregate")
def local_aggregate(model: str = typer.Option(...), stage: str = typer.Option(..., help="patching|erasure")):
    """Aggregate shard outputs (CPU)."""
    if stage == "patching":
        from .local.patching import aggregate

        typer.echo(aggregate(model).head(20).to_string())
    elif stage == "erasure":
        from .local.erasure import aggregate as agg

        typer.echo(str(agg(model)["gaps"]))
    else:
        raise typer.Exit("stage must be patching or erasure")


@local_app.command("report")
def local_report(model: str = typer.Option(...)):
    """Part B report for one model."""
    from .analysis.local_report import build_local_report

    typer.echo(f"-> {build_local_report(model)}")


@local_app.command("status")
def local_status(model: str = typer.Option(...)):
    """Shard completion per stage."""
    from .local.prompts import STAGES
    from .local.shards import plan_dir, shard_dirs

    for st in STAGES:
        pd_ = plan_dir(model, st)
        if not (pd_ / "plan.json").exists():
            typer.echo(f"{st:<10} not planned")
            continue
        n = json.loads((pd_ / "plan.json").read_text())["n_shards"]
        typer.echo(f"{st:<10} {len(shard_dirs(model, st))}/{n} shards DONE")


if __name__ == "__main__":
    app()
