# advice-audit — Different or Lesser?

Pipelines for the project *Different or Lesser? Gender in LLM Advice on Majors and Careers, from
Behavior to Representation* (`docs/proposal.md`, spec `docs/SPEC.md`).

- **Part A**: API behavioral audit via OpenRouter. Runs anywhere with internet, budget-capped at $60.
- **Part B**: interpretability on open-weight models on a Slurm cluster whose **compute nodes are offline**.

Open questions and approvals live in **`QUESTIONS.md`**. Read it first: cell C (China) is blocked until
the gaokao score–rank anchors are filled.

## Setup (laptop / login node)

```bash
uv venv --python 3.12 && uv sync --extra api --extra analysis --extra local   # local = torch/transformers (CPU ok)
cp .env.example .env    # add OPENROUTER_API_KEY
uv run pytest -q        # 63 tests, CPU only (tiny random Llama for Part B code paths)
```

## Part A — API audit

```bash
advice-audit models verify                  # live check: dated slugs, pinned endpoints, prices (no spend)
advice-audit reference fetch && advice-audit reference build   # public reference data -> taxonomy attributes
advice-audit materials build                # renders sample prompts to outputs/materials/, lists TODOs
advice-audit materials name-check           # name manipulation check (all models, phase 'materials')

advice-audit plan --experiment pilot        # 1,200 calls; prints the estimate (~$0.43)
advice-audit api run --plan pilot --model deepseek_v41_flash --dry-run   # zero calls; budget check + sample prompts
advice-audit api run --plan pilot --model deepseek_v41_flash
advice-audit code --run pilot__deepseek_v41_flash          # taxonomy + discursive coding (phase 'coding')
advice-audit analyze --run pilot__deepseek_v41_flash --pilot   # report; STOP for review (A3)

# after 'APPROVED: A4' in QUESTIONS.md
advice-audit plan --experiment study1_main
for m in deepseek_v41_flash qwen38_flash claude_haiku_55 gpt54_mini gemini_38_flash_batch; do
  advice-audit api run --plan study1_main --model $m && advice-audit code --run study1_main__$m; done
advice-audit analyze --run study1_main__deepseek_v41_flash --run study1_main__qwen38_flash ... --name study1
advice-audit code-export --run study1_main__gpt54_mini     # 5% stratified sample for human coders
advice-audit code-kappa  --run study1_main__gpt54_mini
advice-audit budget
```

Runs are **resumable and idempotent**: a call is skipped when any run already logged a final outcome for
the same `call_id` and prompt hash. Logs are append-only Parquet in `outputs/runs/<run_id>/logs/`, with
`manifest.json` holding the git hash, config snapshot, live prices and timestamps. The budget guard refuses a run
whose estimate exceeds `min(phase cap, total cap − reserve)` minus spend. A kill switch stops it at
estimate +20% or when the cap is reached.

## Part B — offline Slurm pipeline

Online (login node):
```bash
bash scripts/online/setup_env.sh                       # shared env (config/paths.yaml: env_dir)
python scripts/online/cache_models.py --allow-unpinned # download pinned models to HF_HOME; pin printed commits
python scripts/online/cache_reference_data.py
advice-audit local plan --model small_iter --stage behavior   # and readout, extract, concepts, patching, steering, erasure
python scripts/online/verify_offline_ready.py --model small_iter   # MUST pass before submitting
```

Submit (chains with `--dependency=afterok`):
```bash
bash scripts/slurm/submit_pipeline.sh small_iter behavior   # Step 0: generation + readout validation
bash scripts/slurm/submit_pipeline.sh small_iter repr       # B3: extract + concepts -> probes/directions
bash scripts/slurm/submit_pipeline.sh small_iter interv     # B4: patching -> steering + erasure
```

Back online: `advice-audit code --run outputs/local/<model>/behavior` (Step 0 logs use the Part A schema),
`advice-audit analyze --run outputs/local/<model>/behavior`, and `advice-audit local report --model <model>`.
Use `advice-audit local status --model <model>` to see shard progress.

Every compute-node entry point (`python -m advice_audit.local.<stage>`) first calls
`offline_guard.activate()`. It asserts `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`,
`VLLM_NO_USAGE_STATS` and `HF_HOME`, and blocks all non-loopback sockets. Shards write a `DONE` marker and
reruns skip them.

Planning and dry runs before the China anchors exist: `advice-audit local plan ... --cells A,B [--limit N]`.

## Layout

```
config/            budget, API/local models, cluster paths, cells, names, profiles, taxonomy, experiments
prompts/           jinja2 templates (3 paraphrases each, EN/ZH), coder prompts, name check
src/advice_audit/
  common/          config, deterministic IDs, Parquet I/O, logging
  materials/       rendering (name/cue spans), design (Latin square), names check, taxonomy
  api/             OpenRouter client, budget guard, plan, runner, parse
  coding/          LLM coding (taxonomy, discursive), SCM dictionary counts, human export + kappa
  reference/       IPEDS, BLS CPS/OEWS, O*NET, Scorecard, ISEI, crosswalks, CFPS interface
  local/           offline guard, hooks, readout, generate, extract, probes, directions,
                   patching, steering, erasure, side effects, shards
  analysis/        outcomes, regressions (H1–H7), JSD, mirror-vs-amplify, reports (A and B)
scripts/online/    setup_env.sh, cache_models.py, cache_reference_data.py, verify_offline_ready.py
scripts/slurm/     common.sh, *.sbatch, submit_pipeline.sh
tests/             63 tests incl. an end-to-end Part B run on a tiny random model
```

`data/` and `outputs/` are git-ignored. CFPS data is never committed.
