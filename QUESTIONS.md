# QUESTIONS.md — open questions and decisions for the researcher

Ground rule (spec §0.6): anything unavailable or ambiguous lands here instead of being invented.
Reply under each item (`> RESEARCHER REPLY:`). Gated phases are approved by adding a line of the
form `APPROVED: A4`, `APPROVED: B3` or `APPROVED: B4` anywhere in this file (spec §10). The code checks for it.

Status 2026-10-14: pipeline implemented, 63 tests pass, **no API calls made, $0 spent**.

---

## 1. Blocking before the pilot (A3)

### 1.1 China gaokao anchors — blocks cell C (and therefore every plan)
`config/profiles/majors_cn.yaml` needs real values from one 3+1+2 province's 一分一段表:
`province`, `year`, and for **物理类** and **历史类** separately a `total` score and provincial `rank` at
roughly the top 2% (`high`) and top 30% (`mid`), plus `source`. Single-subject scores are then
derived deterministically from the total (`allocate_cn_scores`, orientation-specific weights; a design
choice, not data). Until these are filled, strict rendering refuses cell C, so `advice-audit plan`
refuses too.
Option if you want to start before the Chinese data is ready: temporarily set `cells: [A, B]` in
`config/experiments/pilot.yaml` (the pilot becomes 800 calls instead of 1,200).

> RESEARCHER REPLY:

### 1.2 Names (placeholders) — confirm or replace
`config/names.yaml` (female[i] and male[i] share surname[i]):
- A: Emily / Matthew Baker · Hannah / Jacob Walsh · Sarah / Ryan Miller
- B: Ting / Hao Zhang · Xuemei / Zhiqiang Wang · Jiayi / Junjie Li
- C: 张婷 / 张浩 · 王雪梅 / 王志强 · 李佳怡 / 李俊杰

Romanized Chinese names (cell B) are the main risk. Run `advice-audit materials name-check` (all 5
models × 36 names × 10 samples ≈ 1,800 tiny calls, < $0.05). Names below 90% agreement are
appended here automatically. Native-speaker validation is still needed.

> RESEARCHER REPLY:

### 1.3 DeepSeek provider: DeepInfra instead of first-party DeepSeek
The spec pins provider `DeepSeek`. In the hiring project this OpenRouter account's guardrails
blocked the first-party DeepSeek endpoint ("Paid model training violation (account settings)"),
so `config/api_models.yaml` pins **DeepInfra** (fp8, $0.14/$0.42 flat, verified live 2026-10-14)
with no fallbacks; the served provider is logged on every call. To use first-party DeepSeek, change
the account guardrail and set `order: ["DeepSeek"]`, price 0.15/0.60, `peak_multiplier: 2.0`.

> RESEARCHER REPLY:

---

## 2. Decisions taken without an explicit answer (please confirm)

1. **Budget caps.** Enforced as `min(phase cap − phase spent, total cap − reserve − total spent)`;
   the reserve needs `--allow-reserve`. Phase caps unchanged (pilot 3, main 45, coding 12) plus
   `materials: 1.00` for the name check. Note that the phase caps alone add up to the full $60.
2. **Paraphrase and name assignment.** Deterministic Latin square instead of "by hash":
   `name_idx = s mod 3`, `paraphrase = (s + s//3 + h) mod 3` with `h` = hash(domain, mode, profile, cell).
   This gives exact balance at 9 samples and is identical across gender conditions, so female, male and
   baseline prompts are matched sample by sample.
3. **Repo layout.** Built at the repo root (the spec says `advice-audit/`); the spec stays at `docs/SPEC.md`.
   Package name `advice_audit`, CLI `advice-audit`.
4. **Fixed-field assignment** (`config/fixed_fields.yaml`): science profiles r1→computing, r2→healthcare;
   humanities r1→law, r2→education; balanced r1→business/finance, r2→healthcare. Jobs: CS and
   mechanical engineering→computing, economics→business/finance, biology→healthcare,
   psychology→education, journalism→business/finance. `law` is unused for jobs.
   ME→computing and journalism→business/finance are the weakest fits.
5. **Off-track share.** Majors: science profiles are off-track on humanities-only categories and vice
   versa; balanced profiles get NaN. Jobs use the graduate's major track (CS/ME/biology = science,
   economics/psychology = balanced, journalism = humanities).
6. **Reasoning settings.** DeepSeek, Qwen and Haiku have reasoning disabled. GPT-5.4 Mini and Gemini 3.8 Flash
   use `effort: minimal` (lowest available). Identical across all runs.
7. **People–things score.** Prediger's dimension with the sign flipped so that positive means
   people-oriented: `[(2S + E + A) − (2R + I + C)] / 8`, from the O*NET 31.0 RIASEC "OI" scores.
8. **Steering scale.** `vector = α · unit(direction) · 0.05 · mean residual norm (last token, that layer)`,
   with α ∈ {−8 … 8}. Layers are selected automatically as the top 3 by gap closed on E[female share] in patching.

> RESEARCHER REPLY:

---

## 3. Budget notes

The main-run estimate at config prices with the 1.15 output safety factor is **$44.03 against the $45 phase cap**
(GPT-5.4 Mini alone is $25.06). Chinese answers probably use more output tokens than the 600 assumed,
and the pilot will measure that. If the estimate exceeds the cap, the budget guard refuses the GPT run.
Options:
(a) `openai/gpt-5.4-mini:batch` is live at $0.375/$2.25, which halves the GPT cost to about $11;
(b) raise `main` to about $48 and lower `coding`;
(c) use fewer samples for GPT.
`anthropic/claude-haiku-5.5:batch` ($0.05/$0.25) also exists.

> RESEARCHER REPLY:

---

## 4. Reference data (filled 2026-10-14; vintages and gaps)

Sources: BLS CPS Table 11, 2025 annual averages (bls.gov is blocked from this network, so a Wayback copy
of the official file is used); BLS OEWS May 2025 national (Wayback); O*NET 31.0; IPEDS completions
C2024_A (2023–24, bachelor's, first major); College Scorecard field of study (file of 06/10/2026);
ISEI-08 from Ganzeboom's `isco08_with_isei.pdf`; BLS ISCO-08 × SOC-2010 crosswalk plus the SOC 2010→2018
crosswalk; NCES CIP2020–SOC2018 crosswalk. Built values are in `data/reference/built/`, with provenance
in `provenance.json`.

Coding decisions:
- **data_scientist**: CPS has no data-scientist row. The nearest row, "Other mathematical science
  occupations", is used.
- **dentistry, pharmacy, rehabilitation_therapy** (US graduate degrees): fewer than 500 bachelor's completions, so the
  female share uses all award levels ≥ bachelor's. Scorecard earnings fall back to first-professional and
  master's credentials when there are no bachelor's programs (dentistry).
- **Majors' prestige and people–things scores** are averaged over occupations linked by the CIP–SOC crosswalk,
  with postsecondary-teacher SOCs (25-1xxx) dropped unless they are the only link.
- **physician**: mean of SOC 29-1215/1216/1229. **mental_health_counselor** adds the old O*NET code 21-1014.

Missing (left empty, never invented):
- `biological_technician`: female share (the CPS row has no value).
- `data_science`: median earnings (CIP 30.70 is new and absent from Scorecard).

China: national sex composition by major is sparse, so US shares serve as the common yardstick (as the
proposal states). China-specific benchmarks come later (CFPS, §7).

> RESEARCHER REPLY:

---

## 5. Coding resources

1. **Stereotype Content Model dictionaries** (Nicolas, Bai & Fiske 2021): place the English dictionary at
   `config/dictionaries/scm_en.csv` (columns `term,dimension[,direction]`) and a validated Chinese
   translation at `scm_zh.csv`. Until then warmth and competence counts are NaN. Nothing is invented.
2. **Bilingual human coders** for the 5% sample: `advice-audit code-export --run <run>`, then
   `advice-audit code-kappa --run <run>` once both coder CSVs are returned.

> RESEARCHER REPLY:

---

## 6. Part B (local models)

1. **Model choices** (`config/local_models.yaml`, all TODO): the Hugging Face repo id for Qwen3.8 27B open weights
   (OpenRouter lists `qwen/qwen3.8-27b`, but the HF repo id needs confirming), a Western open-weight model
   of about 8–32B, and a 7–9B dev model. Revisions get pinned after `scripts/online/cache_models.py --allow-unpinned`
   prints the resolved commits.
2. **Cluster paths** (`config/paths.yaml`): repo_dir, hf_home, env_dir, partition, account, module list, and
   the hostname patterns for login vs compute nodes.
3. **Chinese held-out text for perplexity**: `data/heldout/zh.txt` (paragraphs separated by blank lines) from a
   source you are licensed to use. English uses WikiText-2 (test split), which is already cached.

> RESEARCHER REPLY:

---

## 7. CFPS (deferred, spec §7)

Interface only (`src/advice_audit/reference/cfps.py`). Fill `config/cfps_variables.yaml` and set
`cfps_dir` in `config/paths.yaml`. CFPS files are never copied into the repo or outputs.

> RESEARCHER REPLY:

---

## 8. Analysis caveat

Standard errors are clustered by profile (12 clusters, or 4 in the pilot), which makes cluster-robust SEs optimistic.
A wild-cluster bootstrap is recommended as a robustness check before preregistration; it is not yet implemented.

> RESEARCHER REPLY:

---

## Approvals

Phases A4 (main API run), B3 (representations) and B4 (interventions) are gated.
Write `APPROVED: A4` (etc.) on its own line below when ready.

---

## Reference attributes incomplete: jobs

- `female_share_us` missing for: biological_technician

Loader problems (first 20):
- CPS: titles not found for biological_technician: ['Biological technicians']
- CPS: titles not found for drafter: ['Architectural and civil drafters']
- CPS: titles not found for physician: ['Emergency medicine physicians']
- CPS: titles not found for psychologist: ['Clinical and counseling psychologists', 'School psychologists']
- CPS: titles not found for social_worker: ['Mental health and substance abuse social workers']

Values were left empty (never invented).

> RESEARCHER REPLY:

---

## Reference attributes incomplete: majors

- `median_earnings` missing for: data_science

Values were left empty (never invented).

> RESEARCHER REPLY:
