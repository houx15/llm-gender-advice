# Different or Lesser? Gender in LLM Advice on Majors and Careers, from Behavior to Representation

Oct 14, 2026 · @Evie Hou

## Summary

This project asks whether LLMs that advise students and job seekers treat women as *different* or as *lesser*. Existing audits show that models recommend fewer STEM majors to girls. That finding conflates two cultural principles that sociology keeps apart: gender essentialism, which sorts men and women into different fields, and male primacy, which places men higher within them (Charles & Grusky 2004). Aligned models may have learned to suppress the second while keeping the first, often phrased as considerate personalization.

**Study 1** is a behavioral audit. Identical student and graduate profiles are presented to API models with the applicant's gender varied, in three cultural cells: English prompts in a US context with Anglo names, English prompts in a US context with Chinese-origin names, and Chinese prompts in a mainland-China context with Chinese names. Chinese profiles use gaokao scores and subject combinations; US profiles use GPA, SAT section scores, and AP coursework. Recommendations are scored on three dimensions: horizontal (which fields), vertical (what level within a field), and discursive (competence versus warmth language, encouragement, asymmetric warnings).

**Study 2** opens the box. On open-weight models that replicate the Study 1 behavior, run locally on an 80 GB GPU, we extract gender directions from the residual stream and ask how they relate to directions for status and for people versus things. We then test causality with activation patching, steering, and concept erasure: can the "different" and the "lesser" components be changed separately?

The contribution is a bridge between the sociology of gender segregation and mechanistic study of models: a behavioral test of whether LLMs reproduce essentialism without hierarchy, and a representational test of whether these two principles are encoded as separable structures. The API phase costs about $50 across five models, including coding of the free-text answers.

## At a glance

&#91;embedded content: project design · 2 studies, 3 outcome dimensions\]

Behavioral gaps and intervention effects are measured on the same three dimensions, so a change inside the model can be read directly as a change in advice.

## Motivation

Choosing a major and a first job are among the most consequential and least reversible decisions people make, and they are increasingly made in conversation with an LLM. Students ask chatbots which majors fit their scores. In China, AI tools for gaokao application choices (志愿填报) are now mainstream products. Graduates ask the same models which jobs to pursue.

Advice is where gender segregation is reproduced on the supply side. Unlike a hiring decision, advice is personalized by design, so a gendered pattern rarely looks like discrimination to the person receiving it. A girl told that nursing "suits her caring nature" hears encouragement, not steering.

The current evidence base answers a narrow question: does the model recommend STEM less often to girls? Three gaps remain:

- **Difference is not the same as hierarchy.** Steering women toward biology rather than physics is a different claim from steering them toward nursing rather than medicine. The first sorts by field; the second ranks by status. They have different causes, consequences, and remedies.
- **Language is not the same as culture.** Prior cross-lingual work varies the prompt language but not the national context or the user's ethnicity, so it cannot tell whether a model localizes gender norms or exports one set everywhere.
- **Behavior is not the same as mechanism.** Audits of closed models describe outputs but cannot say how gender is encoded or whether the pattern can be removed without damaging the advice. Open-weight models make this testable.

## Theoretical framework

The framework predicts that aligned LLMs will reproduce gender *difference* more than gender *hierarchy*, and that the difference will be expressed in benevolent, personalized language.

**1. Two principles of segregation.** Charles & Grusky (2004) distinguish *horizontal* segregation, rooted in gender essentialism (women suited to nurturing, people-oriented work; men to technical, thing-oriented work), from *vertical* segregation, rooted in male primacy (men suited to higher status, authority, and pay). The two are empirically separable. Charles & Bradley (2009) show that affluent, egalitarian societies can have weak vertical but strong horizontal segregation in fields of study, because essentialism survives the decline of overt hierarchy and is reframed as free choice. Our first prediction is that LLMs, trained on such societies' text and then aligned against overt hierarchy, follow the same pattern.

**2. Ambivalent sexism.** Glick & Fiske (1996) separate hostile sexism from benevolent sexism, which praises women for warmth and purity while confining them to supportive roles. Benevolent sexism is harder to recognize and more readily accepted by its targets. Alignment training penalizes hostile statements, so residual bias should appear mostly in benevolent form: praise for caring traits, gentle redirection, protective warnings.

**3. Stereotype content.** The Stereotype Content Model (Fiske et al. 2002) maps group perceptions on warmth and competence. Traditional stereotypes cast women as warm but less competent. This gives a direct linguistic test of devaluation: is advice to women framed in warmth terms where advice to men is framed in competence terms?

**4. Self-assessment and aspiration.** Correll (2001; 2004) shows that cultural beliefs about gender shape how students assess their own ability and which fields they aspire to, even at equal performance. An advisor that redirects equally able girls becomes a new link in this supply-side mechanism.

**5. The geometry of culture.** Kozlowski, Taddy & Evans (2019) show that cultural meanings, such as class and gender, are encoded as directions in word-embedding space, and that the relationships between these directions reveal cultural structure. We extend this to the contextual representations of LLMs: if "different but not lesser" holds behaviorally, the gender direction should align with a people–things direction but not with a status direction. Unlike static embeddings, LLMs also allow causal tests by intervening on these directions.

**6. Localization versus homogenization.** LLM values skew toward Western, educated, industrialized, rich, democratic populations (Atari et al. 2023). A model advising a student in Chinese about a Chinese context can either *localize*, adopting local gender norms, or *homogenize*, applying one set of norms everywhere. Recent cross-lingual evidence shows the pattern depends on the model–language pair, with some models amplifying and others suppressing gender gaps (Choi et al. 2026). Which happens is an open empirical question with normative stakes either way.

## Previous studies

The audits of LLM educational and career advice we found all measure horizontal steering; none separates it from vertical devaluation, and none combines language, national context, and name ethnicity.

**LLM advice on majors and careers**

| Study | Setting | Design | Gender finding |
| --- | --- | --- | --- |
| [Agarwal & Agarwal (2026)](https://neurips.cc/virtual/2026/162727) | India, Class 10 stream choice | 4 models (US, China, India), 540 matched profiles; gender, income, first-generation, region | Every model lowered girls' odds of a science-stream recommendation; models agreed on one stream for only 12.8% of profiles |
| [Dai, Ma & Wu (2026), *Acta Psychologica Sinica*](https://journal.psych.ac.cn/acps/EN/10.3724/SP.J.1041.2026.0399) | Chinese and English prompts | 6 models; empathy stereotype task; ranking of 16 majors and 16 professions for a male, female, or unspecified student | LLM stereotypes stronger than humans'; stronger in English and under female priming; high-empathy fields recommended to girls |
| [Shailya et al. (2025), IJCNLP-AACL Findings](https://arxiv.org/html/2509.04498v2) | University and master's programs, 40 nationalities | 3 open 7–8B models, 360 profiles; gender × class × nationality | Men steered to engineering and CS, women to social sciences; persisted when the profile stated a strong engineering background |
| [Zheng (2023)](https://arxiv.org/abs/2401.11699) | California 12th graders | GPT-3.5; gender, race, SES, score band; top-10 majors | Boys about 1.25× as likely as girls to receive STEM recommendations at the median |
| [Salinas et al. (2023)](https://arxiv.org/abs/2308.02053v2) | Job recommendations | ChatGPT and LLaMA; gender identity × nationality | Women steered toward secretarial roles |
| [GenAI-assisted recruitment (2026)](https://arxiv.org/pdf/2603.11736) | Italian young job seekers | ChatGPT-5, 24 profiles | Gendered job and industry suggestions; women described in emotional, empathetic terms |
| [Born, Curhan & Bowles (2026)](https://arxiv.org/abs/2610.14036) | Negotiation advice | 4 models | Lower salary targets for women; agendas steer women to flexibility and caregiving |

**Cross-lingual gender bias**

- [Zhao et al. (2024)](https://arxiv.org/abs/2403.00277) find gender bias in all six languages tested, including Chinese, with language-specific forms: appearance talk in women's dialogues was far more common in East Asian languages.
- [Choi et al. (2026)](https://arxiv.org/abs/2605.30804) benchmark model-generated gender gaps against human cross-national baselines in English, Korean, Japanese, and Chinese. English-centric models amplified gaps 2–5× in non-English prompts, while DeepSeek suppressed them in Chinese. The pattern depends on the model–language combination.

**Representations and interventions (methods we build on)**

- Static embeddings encode gender as a direction; removing it does not remove bias, which survives in clustering of other words (Bolukbasi et al. 2016; Gonen & Goldberg 2019).
- Causal mediation analysis locates the components through which gender information affects model outputs (Vig et al. 2020).
- High-level concepts are often linearly represented in LLM activations and can be read with probes and changed with steering vectors (Park et al. 2024; Turner et al. 2023; Rimsky et al. 2024; Zou et al. 2023).
- Linear concept erasure can remove a concept's linearly available information with minimal side effects (Ravfogel et al. 2020; Belrose et al. 2023).

**Gaps this project fills**

1. **No separation of difference from hierarchy.** All studies measure field steering; none fixes the field and measures level, or codes competence versus warmth framing.
2. **Language, context, and ethnicity are confounded.** Dai et al. vary language only; Choi et al. vary language and model but not context or names.
3. **No realistic Chinese profiles.** No study uses gaokao scores and subject combinations.
4. **Outcomes are crude.** Jaccard overlap, binary STEM shares, or empathy categories, not links to real sex composition, wages, or prestige that would show whether models mirror or amplify segregation.
5. **Explicit versus implicit gender cues are untested** in the advice setting.
6. **No link from behavior to mechanism.** Interpretability work on gender uses toy tasks such as pronoun resolution, not consequential advice.

## Research questions and hypotheses

**RQ1 (horizontal).** For identical profiles, do LLMs recommend different fields to women and men?

**RQ2 (vertical).** Within a field the user has chosen, do LLMs recommend lower-status, lower-paid, or less selective options to women?

**RQ3 (discursive).** Is advice to women framed more in warmth terms, more cautiously, or with more warnings than advice to men?

**RQ4 (culture).** Do these patterns differ between English–US and Chinese–China contexts, and is any difference driven by language and context or by the user's ethnicity?

**RQ5 (representation).** In open-weight models, how is inferred gender represented, and how does it relate to representations of status and of people versus things?

**RQ6 (intervention).** Can steering or erasing these representations change the horizontal and vertical components of advice separately?

| Hypothesis | Prediction | Study |
| --- | --- | --- |
| H1 Horizontal steering | Women receive recommendations with higher female share and more people-oriented content, net of stated interests | 1 |
| H2 Essentialism without hierarchy | The vertical gap (level within a fixed field) is smaller than the horizontal gap, and may be zero | 1 |
| H3 Benevolent framing | Advice to women contains more warmth language, more unsolicited warnings, and more hedging; advice to men contains more competence and ambition language | 1 |
| H4 Override of stated interest | Steering is strongest for balanced profiles, but persists for women whose profiles show a strong science track | 1 |
| H5 Explicit cue amplification | Stating gender explicitly increases gaps relative to name-only cues | 1 |
| H6 Localization vs homogenization | Competing: gaps are larger in the Chinese context (localization) or similar across contexts (homogenization); the Chinese-origin-name cell separates ethnicity from context | 1 |
| H7 Male default | Recommendations for gender-unspecified profiles are closer to the male than the female version | 1 |
| H8 Representational separability | The gender direction aligns more with a people–things direction than with a status direction | 2 |
| H9 Causal separability | Steering along the gender direction shifts horizontal outcomes; steering along a status direction shifts vertical outcomes, with limited cross-effects | 2 |
| H10 Residual leakage | Erasing the linear gender direction reduces but does not eliminate gaps, because gender remains recoverable from proxies | 2 |

## Study 1: Behavioral audit

Study 1 presents identical profiles to API models, varies only the user's gender cue, and scores the advice on horizontal, vertical, and discursive dimensions. Each profile exists in structurally parallel Chinese and US versions.

### Cultural cells

| Cell | Prompt language | Context stated | Names | Isolates |
| --- | --- | --- | --- | --- |
| A | English | United States | Anglo (e.g., Emily / Matthew) | US baseline |
| B | English | United States | Chinese-origin, romanized (e.g., Ting Zhang / Hao Zhang) | Ethnicity, holding language and context fixed (A vs B) |
| C | Chinese | Mainland China | Same names in characters (张婷 / 张浩) | Language and context, holding names fixed (B vs C) |

Surnames are held constant across genders within a cell. Given names are chosen to be strongly gendered and are validated by native-speaker raters and by a model manipulation check ("What is this person's likely gender?").

### Gender conditions

- **Implicit:** name only.
- **Explicit:** name plus "I'm a girl / boy" (我是女生 / 男生); for graduates, "I'm a woman / man."
- **Unspecified baseline:** no name and no gender, to test the male-default hypothesis (H7).

This gives five conditions per profile and cell: female-implicit, female-explicit, male-implicit, male-explicit, baseline.

### Domain 1: Choosing a major

Profiles describe a graduating high-school student. Twelve profiles cross three academic orientations with two score levels, with two replicate profiles per combination.

| Orientation | China (新高考 3+1+2) | United States |
| --- | --- | --- |
| Science track | 首选物理; 再选化学、生物; strong math and physics scores | Strong SAT Math; AP Calculus BC, AP Physics C, AP Chemistry |
| Humanities track | 首选历史; 再选政治、地理; strong Chinese and history scores | Strong SAT Reading & Writing; AP US History, AP English Literature, AP Psychology |
| Balanced | 首选物理; 再选政治、地理; near-equal scores across subjects | Balanced SAT sections; mixed AP courses (e.g., AP Calculus AB, AP Biology, AP US History) |

Score levels: *high* (Chinese total score and provincial rank in roughly the top 2%; US GPA ≈ 3.9 and SAT ≈ 1500) and *middle* (roughly the top 30%; GPA ≈ 3.4 and SAT ≈ 1250). Province, school type, and interests are held constant within each pair.

### Domain 2: Choosing a job

Profiles describe a new university graduate: major, university tier, GPA or class rank, one internship, and skills. Twelve profiles span six majors (computer science, mechanical engineering, economics, biology, psychology, journalism) at two academic-performance levels. Chinese profiles name a university tier (e.g., a 211 university); US profiles a state flagship university. No family information is included.

### Two prompt modes per domain

- **Open:** "Which majors (jobs) should I consider? List your top five, ranked, with one reason each." This measures horizontal steering.
- **Fixed-field:** the user names a broad field consistent with the profile (healthcare, computing, business and finance, law, education) and asks which specific options to pursue. Fixing the field isolates vertical steering: for example, clinical medicine versus nursing, software engineering versus IT support, investment analysis versus accounting clerk.

### Design size

12 profiles × 5 gender conditions × 3 cultural cells × 10 samples = 1,800 calls per domain and mode. Two domains × two modes give **7,200 calls per model**. Answers are free text, to match real use, with a ranked list requested.

### Measures

| Dimension | Measure | How |
| --- | --- | --- |
| Horizontal | Female share of recommended fields | Code recommendations to CIP (majors) or SOC (jobs); link to sex composition of degrees and employment |
| Horizontal | People–things orientation | O\*NET interest profiles (Realistic/Investigative vs Social), via the CIP–SOC crosswalk for majors |
| Horizontal | Departure from stated orientation | Share of recommendations outside the profile's academic track |
| Vertical | Expected earnings | Median earnings by field (College Scorecard by CIP; BLS OEWS by SOC) |
| Vertical | Occupational status | ISEI or SIOPS prestige scores via ISCO crosswalk |
| Vertical | Level and selectivity | Recommended degree level; selectivity of any named institution (US acceptance rate; China 985/211/Double First-Class) |
| Discursive | Warmth vs competence framing | Stereotype Content Model dictionaries (Nicolas, Bai & Fiske 2021), with a validated Chinese translation |
| Discursive | Warnings and hedging | LLM-coded: unsolicited warnings (difficulty, competition, male-dominated fields), hedges, discouragement |
| Discursive | Ambition and agency | LLM-coded: encouragement to aim high, leadership mentions, agentic vs communal wording |
| Discursive | Gender salience | Whether the response mentions gender, work–life balance, or family unprompted |

All rank-based measures are position-weighted (top recommendation weighted most). Free-text coding uses a coder model different from the evaluated model, with two human coders on a stratified 5% sample (Cohen's κ ≥ 0.7, bilingual coders for Chinese responses).

## Study 2: Representations and interventions

Study 2 asks whether the behavioral distinction between "different" and "lesser" corresponds to separable structures inside the model, and whether intervening on those structures changes advice. It runs on open-weight models on a local 80 GB GPU.

### Step 0: Model selection by behavioral replication

Interpretability findings only explain behavior that the same model exhibits. We first run a reduced Study 1 protocol (about 2,000 prompts) on three or four open-weight candidates served locally (e.g., with vLLM). Candidates: Qwen3.8 27B (open weights, same family as an API model in Study 1), one Western open-weight model of similar size, and one 7–9B model for fast iteration. Models that fit in bf16 (up to about 32B parameters) are preferred, because quantization adds noise to activations. Only models that reproduce at least one Study 1 gap proceed.

### Step 1: Where is gender encoded?

- Extract residual-stream activations at every layer for matched prompt pairs that differ only in the name or explicit cue.
- Train linear probes for inferred gender at the name tokens and at the final token before the answer, layer by layer.
- Test cross-lingual transfer: train probes on English prompts and evaluate on Chinese prompts, and the reverse. High transfer suggests a shared, language-independent gender representation.

### Step 2: How is gender structured relative to status and field type?

Following the geometry-of-culture logic, we estimate three directions per layer:

| Direction | Construction |
| --- | --- |
| Gender (g) | Mean difference between female and male versions of matched prompts |
| Status (s) | Contrast between high- and low-prestige options within the same field, chosen so that status is not confounded with female share (e.g., estimated by regressing activations on prestige while controlling for sex composition) |
| People–things (p) | Contrast between people-oriented and thing-oriented fields at matched prestige |

We then compare their geometry: cosine similarity of g with s and with p, and projections of about 100 major and occupation concepts onto each direction, correlated with real-world female share, earnings, and prestige. H8 predicts that g aligns with p more than with s. We also test whether g is the same direction in Chinese and English contexts, and whether cell B (Chinese-origin names in a US context) sits closer to cell A or cell C.

### Step 3: Where does gender affect the advice?

- Define a quantitative readout: the model's probability distribution over a fixed list of about 40 major or occupation categories at the answer position. From it we compute expected female share, expected prestige, and expected earnings.
- Use activation patching: copy activations from the male prompt into the female prompt at a given layer and position, and measure how much of the readout gap closes. This locates the layers and token positions that carry gender into the recommendation (cf. causal mediation analysis, Vig et al. 2020).

### Step 4: Interventions

| Intervention | Question |
| --- | --- |
| Steering along g (add or subtract α·g at selected layers) | Does advice for an unspecified profile shift toward the female or male pattern in a dose-dependent way? |
| Steering along s and along p separately | Can the vertical and horizontal components be changed independently (H9)? |
| Linear concept erasure of gender (e.g., LEACE) | Do female–male gaps disappear, or does gender leak back through other features (H10)? |
| Random directions of equal norm | Placebo control for all of the above |

**Side effects are measured, not assumed away.** For every intervention we report fluency (perplexity on held-out text), advice quality (blind pairwise ratings by a judge model and a human sample), refusal rate, and whether advice still tracks the profile's academic orientation. An intervention that erases gender gaps by also erasing sensitivity to the student's actual strengths is a failure, not a fix.

Free-text generations under each intervention are coded with the same Study 1 measures, so that representational effects are reported on the same scale as behavioral ones.

## Data sources, models, and cost

### Reference data

| Purpose | United States | China |
| --- | --- | --- |
| Sex composition of fields of study | NCES IPEDS completions by CIP code and sex | Open question: national data by major and sex are sparse; candidates are the China Education Statistics Yearbook and university admission reports. US shares serve as a common yardstick meanwhile |
| Sex composition of occupations | BLS CPS, employed persons by detailed occupation and sex | Population census occupation tables (coarse categories) |
| Earnings | College Scorecard earnings by field of study; BLS OEWS by SOC | Limited; report US-based measures and flag them |
| Occupational status | ISEI / SIOPS via ISCO crosswalk | Same international scales |
| People–things orientation | O\*NET interest profiles; NCES CIP–SOC crosswalk for majors | Same, applied to coded categories |
| Institutional selectivity | IPEDS acceptance rates | 985 / 211 / Double First-Class lists |
| Profile realism | SAT section score distributions; AP course lists | Provincial gaokao score–rank tables (一分一段表); 3+1+2 subject combinations |
| Human benchmark (subject choice) | AP exam participation by sex | China Family Panel Studies (CFPS) education modules, held by the researcher; to be added after the pipeline is built |

All Chinese free-text recommendations are coded into the same CIP and SOC categories as English ones, so both contexts are scored on one scale.

### API models (Study 1)

Two Chinese and three US models, with pinned, dated slugs on OpenRouter. Token assumptions: about 500 input and 600 output tokens per call; 7,200 calls per model. Prices are OpenRouter list prices as of 14 Oct 2026.

| Model | Origin | Price in / out ($ per 1M tokens) | Study 1 cost |
| --- | --- | --- | --- |
| [DeepSeek V4.1 Flash](https://openrouter.ai/deepseek/deepseek-v4.1-flash) | China | 0.15 / 0.60 (off-peak; peak ×2) | $3.13 |
| [Qwen3.8 Flash](https://openrouter.ai/qwen/qwen3.8-flash) | China | 0.15 / 0.47 | $2.57 |
| [Claude Haiku 5.5](https://openrouter.ai/anthropic/claude-haiku-5.5) | US | 0.10 / 0.50 | $2.52 |
| [GPT-5.4 Mini](https://openrouter.ai/openai/gpt-5.4-mini) | US | 0.75 / 4.50 | $22.14 |
| [Gemini 3.8 Flash, batch](https://openrouter.ai/google/gemini-3.8-flash:batch) | US | 0.375 / 1.875 | $9.45 |
| **Total, five models** |  |  | **$39.81** |

Coding free-text answers with Claude Haiku 5.5 as coder (and DeepSeek V4.1 Flash as coder for Haiku's own answers) adds about $1.60 per model, or about $8. With peak pricing and a 20% buffer for pilots and reruns, the API phase stays near $60.

### Local models (Study 2)

Runs on the researcher's 80 GB GPU at no API cost. Dense models up to about 32B parameters fit in bf16 with room for activation caching; 70B-class models need 8-bit or 4-bit quantization and are used only for behavioral checks, not representation analysis. Activation storage for about 5,000 prompts across all layers is the main disk requirement; we store residual streams at selected token positions only.

## Analysis plan

**Unit of analysis.** One response, summarized as position-weighted means of each outcome across its ranked recommendations.

**Core model (Study 1).** For response i from profile p, cell c, model m:

```latex
Y_{ipcm} = \beta_1 \mathrm{Female}_i + \beta_2 \mathrm{Explicit}_i + \beta_3 (\mathrm{Female}_i \times \mathrm{Explicit}_i) + \beta_4 (\mathrm{Female}_i \times C_c) + \alpha_p + \gamma_c + \delta_m + \varepsilon_{ipcm}
```

Profile fixed effects absorb everything about the student or graduate. β1 is the gender gap under name-only cues, β3 the added effect of explicit statement (H5), and β4 the difference in the gap across cultural cells (H6). Standard errors are clustered by profile. Models are estimated pooled and per LLM, and separately by domain and prompt mode.

**Key comparisons.**

- **H1, horizontal:** β1 on female share and people–things scores, open mode.
- **H2, difference without hierarchy:** standardized β1 on vertical outcomes in fixed-field mode compared with standardized β1 on horizontal outcomes in open mode. The hypothesis is supported if the vertical gap is significantly smaller, with an equivalence test against a smallest effect of interest of 0.1 SD.
- **H3, discursive:** β1 on warmth-minus-competence scores and on rates of warnings, hedges, and ambition cues.
- **H4:** Female × orientation interaction; steering for science-track girls is the critical test.
- **H6:** contrasts A vs B (ethnicity) and B vs C (language and context).
- **H7:** distance of the baseline response distribution from the female and male distributions (Jensen–Shannon divergence over coded categories, and differences in outcome means).
- **Mirror or amplify:** compare the female–male difference in recommended fields' female share with the real-world difference in female share between the fields men and women actually enter.

**Study 2 analyses.** Probe accuracy by layer with permutation baselines; cosine similarities between directions with bootstrap confidence intervals over prompt sets; patching effects as the share of the readout gap closed; dose–response curves for steering with random-direction placebo bands; pre- and post-erasure gaps with side-effect metrics reported alongside.

**Multiple testing and preregistration.** Hypotheses H1–H7 and their primary outcomes are preregistered on OSF before the main API run. Secondary outcomes are reported with false-discovery-rate correction. Study 2 is labeled exploratory except for H8–H10, which are preregistered after model selection in Step 0.

## Risks, timeline, and expected contributions

**Risks and mitigations**

| Risk | Mitigation |
| --- | --- |
| Chinese and US profiles are not truly equivalent | Align on relative standing (score percentile) and orientation, not raw scores; report within-cell gaps as the primary estimand, so cross-cell comparison relies on differences in gaps |
| Romanized Chinese given names are weakly gendered for English-language models | Native-speaker validation plus model manipulation check; drop names below a threshold |
| Free-text recommendations are hard to code consistently across languages | Fixed category list; bilingual human coders; report κ by language |
| Chinese reference data on sex composition by major are thin | Use US data as a common yardstick and state this; treat China-specific benchmarks as secondary |
| Open-weight models do not replicate API behavior | Step 0 screening; report null replications; Study 2 claims are limited to the models analyzed |
| Directions for status and field type are confounded with gender | Construct them on gender-balanced contrasts and verify low correlation with sex composition before use |
| Steering damages advice quality | Side-effect metrics reported with every intervention; random-direction placebos |
| "Personalization" defense: different advice may be appropriate | Profiles hold stated interests and abilities fixed, so any difference is attributable to the gender cue alone |

**Timeline (about six months).** Month 1: profiles, name validation, coding scheme. Month 2: pilot on DeepSeek, then preregistration. Month 3: Study 1 main runs and coding. Month 4: Study 1 analysis; Step 0 model screening. Months 5–6: Study 2 representation analysis and interventions; writing.

**Expected contributions**

1. **Conceptual.** Introduces the horizontal–vertical distinction from the sociology of segregation into LLM bias research, replacing the single question "is the model biased?" with "does it treat women as different, as lesser, or both?"
2. **Empirical.** The first audit of LLM major and career advice that separates field steering from status steering and benevolent framing, using realistic gaokao and US profiles, and that disentangles language, national context, and user ethnicity.
3. **Mechanistic.** A test of whether the cultural principles of essentialism and male primacy are separable inside a model, and whether they can be changed independently, linking the geometry-of-culture tradition to causal intervention.
4. **Policy.** Evidence on whether debiasing that targets overt hierarchy leaves essentialist steering intact, which matters for AI tools already used in college-application and career guidance.

**Possible venues.** *Sociological Science* or *Socius* for Study 1 framed as segregation research; FAccT, AIES, or an ACL venue for the combined behavioral and mechanistic paper.

## References

- Agarwal, H., & Agarwal, P. (2026). [Auditing demographic bias in LLM career advice for Indian students](https://neurips.cc/virtual/2026/162727). NeurIPS GlobalSouthAI.
- Atari, M., Xue, M. J., Park, P. S., Blasi, D., & Henrich, J. (2023). Which humans? PsyArXiv preprint.
- Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect linear concept erasure in closed form. NeurIPS.
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. NeurIPS.
- Born, N., Curhan, J. R., & Bowles, H. R. (2026). [AI advice reinforces traditional gender roles](https://arxiv.org/abs/2610.14036). arXiv:2610.14036.
- Charles, M., & Bradley, K. (2009). Indulging our gendered selves? Sex segregation by field of study in 44 countries. *American Journal of Sociology*, 114(4), 924–976.
- Charles, M., & Grusky, D. B. (2004). *Occupational Ghettos: The Worldwide Segregation of Women and Men*. Stanford University Press.
- Choi, J., Ahn, S., Zhang, T., & Jung, S. (2026). [Anchoring LLM gender bias to human baselines: A cross-lingual audit](https://arxiv.org/abs/2605.30804). arXiv:2605.30804.
- Correll, S. J. (2001). Gender and the career choice process: The role of biased self-assessments. *American Journal of Sociology*, 106(6), 1691–1730.
- Correll, S. J. (2004). Constraints into preferences: Gender, status, and emerging career aspirations. *American Sociological Review*, 69(1), 93–113.
- Dai, Y., Ma, X., & Wu, Z. (2026). [LLMs amplify gendered empathy stereotypes and influence major and career recommendations](https://journal.psych.ac.cn/acps/EN/10.3724/SP.J.1041.2026.0399). *Acta Psychologica Sinica*, 58(3), 399–415.
- Fiske, S. T., Cuddy, A. J. C., Glick, P., & Xu, J. (2002). A model of (often mixed) stereotype content. *Journal of Personality and Social Psychology*, 82(6), 878–902.
- [Gender bias in generative AI-assisted recruitment processes (2026)](https://arxiv.org/pdf/2603.11736). arXiv:2603.11736.
- Glick, P., & Fiske, S. T. (1996). The Ambivalent Sexism Inventory. *Journal of Personality and Social Psychology*, 70(3), 491–512.
- Gonen, H., & Goldberg, Y. (2019). Lipstick on a pig: Debiasing methods cover up systematic gender biases in word embeddings but do not remove them. NAACL.
- Kozlowski, A. C., Taddy, M., & Evans, J. A. (2019). The geometry of culture: Analyzing the meanings of class through word embeddings. *American Sociological Review*, 84(5), 905–949.
- Nicolas, G., Bai, X., & Fiske, S. T. (2021). Comprehensive stereotype content dictionaries using a semi-automated method. *European Journal of Social Psychology*, 51(1), 178–196.
- Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. ICML.
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. ACL.
- Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. (2024). Steering Llama 2 via contrastive activation addition. ACL.
- Salinas, A., Shah, P. V., Huang, Y., McCormack, R., & Morstatter, F. (2023). [The unequal opportunities of large language models: Revealing demographic bias through job recommendations](https://arxiv.org/abs/2308.02053v2). EAAMO.
- Shailya, K., Mishra, A. K., Krishnan, G. S., & Ravindran, B. (2025). [Where should I study? Biased language models decide!](https://arxiv.org/html/2509.04498v2) Findings of IJCNLP-AACL.
- Su, R., Rounds, J., & Armstrong, P. I. (2009). Men and things, women and people: A meta-analysis of sex differences in interests. *Psychological Bulletin*, 135(6), 859–884.
- Turner, A. M., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. arXiv:2308.10248.
- Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020). Investigating gender bias in language models using causal mediation analysis. NeurIPS.
- Zhao, J., Ding, Y., Jia, C., Wang, Y., & Qian, Z. (2024). [Gender bias in large language models across multiple languages](https://arxiv.org/abs/2403.00277). arXiv:2403.00277.
- Zheng, A. (2023). [Dissecting bias of ChatGPT in college major recommendations](https://arxiv.org/abs/2401.11699). arXiv:2401.11699.
- Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.

References without links were cited from memory and should be checked against the published versions before submission. Model prices are OpenRouter list prices as of 14 Oct 2026.

