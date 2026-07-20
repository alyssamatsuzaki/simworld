# RegWorld — Master Build Plan

**A policy world model of how a data-privacy regulation propagates through firms, consumers, and institutions — built as the maximal sixteen-tool stack from *A Practical Guide to the World-Modeling Research Stack*, Part XIX.**

- **Target executor:** Claude Code, one command.
- **Target runtime:** cloud- and cluster-portable — laptop → single cloud VM → Slurm/K8s. CPU-first, GPU-optional.
- **Core discipline:** the world is synthetic and known, so every downstream stage is graded against ground truth. Nothing "works" until it recovers a truth we planted.

Each numbered stage maps to one numbered step of Part XIX. The guide's closing warning — that a real project would start with stages 1–4 and stop — is honored in full in §17. You asked for the whole stack, so the whole stack is here, and each tool must justify its presence in a single sentence or be removed.

---

## Table of contents

- [§0 The one command, and the execution protocol](#0-the-one-command-and-the-execution-protocol)
- [§1 Project definition](#1-project-definition)
- [§2 The sixteen-tool map](#2-the-sixteen-tool-map)
- [§3 Design decisions, and where to flip them](#3-design-decisions-and-where-to-flip-them)
- [§4 Repository layout](#4-repository-layout)
- [§5 Environment, dependencies, and portability](#5-environment-dependencies-and-portability)
- [§6 Configuration](#6-configuration)
- [§7 The world: ground-truth DGP specification](#7-the-world-ground-truth-dgp-specification)
- [§8 Data contracts](#8-data-contracts)
- [§9 Interfaces](#9-interfaces)
- [§10 Execution: phases and stages](#10-execution-phases-and-stages)
- [§11 Evaluation suite](#11-evaluation-suite)
- [§12 Testing strategy](#12-testing-strategy)
- [§13 Reproducibility and seeds](#13-reproducibility-and-seeds)
- [§14 Compute profiles and wall-clock budgets](#14-compute-profiles-and-wall-clock-budgets)
- [§15 Failure policy, skip policy, run manifest](#15-failure-policy-skip-policy-run-manifest)
- [§16 Known hazards and guardrails](#16-known-hazards-and-guardrails)
- [§17 What this project deliberately does not do — the minimal path](#17-what-this-project-deliberately-does-not-do--the-minimal-path)
- [§18 Definition of done](#18-definition-of-done)
- [Appendix A — `pyproject.toml`](#appendix-a--pyprojecttoml)
- [Appendix B — `Makefile`](#appendix-b--makefile)
- [Appendix C — `docker/`](#appendix-c--docker)
- [Appendix D — GitHub Actions](#appendix-d--github-actions)
- [Appendix E — `CLAUDE.md`](#appendix-e--claudemd)
- [Appendix F — `PROGRESS.md` / `DEVIATIONS.md` templates](#appendix-f--progressmd--deviationsmd-templates)
- [Appendix G — File manifest by phase](#appendix-g--file-manifest-by-phase)
- [Appendix H — Command reference](#appendix-h--command-reference)
- [Provenance](#provenance)

---

## §0 The one command, and the execution protocol

### The command that launches you

From an empty directory containing only this file:

```bash
claude "Read PLAN.md end to end, then execute it. Work phase by phase (Phase 1 → Phase 7),
building the stages each phase contains in the order given in §10. At every phase gate, run the
gate's commands and confirm they exit 0 before advancing. After every stage, update PROGRESS.md
and git-commit with the message the stage specifies. Do not skip a gate. Do not stub a stage and
mark it done. Record every place the installed libraries differ from this plan in DEVIATIONS.md
and follow the library, not the plan. If a gate fails twice, stop and report what failed, what you
tried, and what you need from me. You are done when 'make lint', 'make typecheck', 'make test', and
'make smoke' all pass and the Definition of Done in §18 is satisfied."
```

Once the repository exists, the two one-command surfaces are:

```bash
make smoke                 # whole 17-stage pipeline at profile=smoke: CPU, < 6 min, writes a report
make all                   # the real run: profile=default, writes reports/FINDINGS.md + run_manifest.json
```

### Working rules (for the agent building this)

1. **Build in phase order, not in one pass.** The seven phases are in §10. Each phase ends in a **gate**: a named set of commands that must exit 0. Each stage inside a phase has the shape *Purpose → Files → Key decisions → Acceptance tests → Gate → Commit*. Do not start a stage until the previous stage's tests pass, and do not start a phase until the previous phase's gate is green.
2. **Reconnaissance before code.** Stage 0 (§10) exists because this plan was written against library APIs that move. Verify every version-sensitive API against what actually installed before writing code that depends on it. Where reality and this plan disagree, **reality wins**: record the divergence in `DEVIATIONS.md`, adapt the code, and never pin an ancient version to force the plan to be right.
3. **Write the test before or alongside the module.** Every gate is a shell command that must exit 0. `make lint && make typecheck && make test` must pass before every commit.
4. **Never stub to pass a gate.** A stage that cannot be built is marked `BLOCKED` in `PROGRESS.md` with a reason. A stage that is built but degraded (RLlib swapped for hand-rolled IPPO, say) is marked `DEGRADED` with the substitution recorded. Silent stubs make the entire run worthless.
5. **`PROGRESS.md` is the resume point.** If context compacts, a fresh session reads `PLAN.md §<current stage>` + `PROGRESS.md` and continues from the first unfinished stage. Keep it current or lose the run.
6. **Commit at every gate**, small and traceable, with the message the stage gives. Working tree clean before advancing.
7. **Never exceed the smoke budget.** `make smoke` must finish in **< 6 minutes on 4 CPU cores**. If a stage blows the budget, shrink the *profile*, never the science.
8. **Parallelize where independent.** Stages 1 and 2 touch disjoint files; Stage 5 (causal) and Stage 6–7 (emulator) are independent given Stages 1–4. Use subagents for independent file groups where it helps.
9. **Prefer boring code.** No clever abstractions. Every module under `src/` should be readable by someone who has never seen it, because in six months that person is you.
10. **Logging, not printing.** Python `logging` throughout, configured once in `regworld/logging_conf.py`. `print()` in `src/` is banned by lint (ruff `T20`).
11. **Run the fast tests constantly**, not only at gates: `uv run pytest -m "not slow" -q` should stay green while you work.
12. When something is genuinely underspecified, choose the simplest thing that satisfies the gate, write it in `DEVIATIONS.md`, and move on. Do not stall.

### What "done" looks like

`make all` runs to completion, writes `reports/FINDINGS.md` with every figure populated, and every claim in §1 is marked SUPPORTED, REFUTED, or INCONCLUSIVE with the evidence attached. A stage that failed is reported as failed, not hidden. The full checklist is §18.

---

## §1 Project definition

### The scientific question

A data-privacy regulation — the **Consumer Data Protection Act (CDPA)** — takes effect in eighteen months with a phased compliance schedule. A policy team needs **the distribution of plausible outcomes over the following twenty-four quarters**, not a point forecast:

- compliance rate (overall, and by firm size)
- market concentration (HHI)
- consumer behavior and trust
- **where the intervention backfires** — the regime where compliance rises *and* consumer welfare falls, because compliance has economies of scale, small firms exit, and concentration increases.

And the decision question: **which enforcement policy** — intensity × audit targeting × phase-in speed × small-firm subsidy — produces the best outcome distribution under honest uncertainty.

### The two-regime design (load-bearing — do not collapse it)

Behavioral parameters **θ** are shared across regimes; policies are not. So generalizing from the past regime to the future one is a genuine **policy-shift / transportability test** — which is exactly what a world model is *for*.

| Regime | What it is | Role |
|---|---|---|
| **P (past)** | An *analogous prior regulation*, already enacted, whose enforcement switched on **region by region at exogenously staggered quarters**. Twenty-four quarters of history exist; we **observe** quarters 1–12 noisily and partially (aggregates + a 20% firm panel + a consumer survey). | **Calibration data**, the **DiD natural experiment** (the staggered regional timing is exogenous by construction, which is what identifies the effect), and — quarters 13–24 — the **backtest** holdout. |
| **F (future)** | The CDPA. Different phase-in, and the policy levers are *ours to choose*. | **Forecasting target.** Never used for calibration. |

Calibrating on the *past* regulation is the scientifically correct setting: you cannot fit parameters to data from a regulation that has not happened yet. Regime P earns its keep three times over — as the calibration panel, the quasi-experiment, and the backtest — and this single structure is what makes every evaluation layer in §11 computable.

### Synthetic ground truth, and why it is the design and not a compromise

There is no real firm registry in the sandbox, and there would not be a *labeled* one in the real world either. So the project builds its own world first: a ground-truth data-generating process (the **DGP**, §7) with known parameters **θ\***, a known causal structure, a deliberately planted unobserved confounder, and the staggered historical rollout above. Everything downstream then has something to be graded against.

The guide says so twice. On calibration: *"Parameter recovery is the ABM equivalent of a unit test — generate synthetic data from known parameters, run your calibration, and confirm it finds them; a pipeline that fails on data it generated itself has nothing to say about reality."* And on causal inference: *"a world model can serve as the laboratory where identification strategies get stress-tested on data whose true causal structure you control, because you wrote it."*

Concretely, this buys four things a real-data project cannot have:

- **Parameter recovery is checkable** against θ\*.
- **Causal estimates are checkable** against real `do()` interventions we can actually run.
- **Backtesting is honest**, because the holdout is truly held out.
- The pipeline runs **end to end with zero external data and zero network access**, which is what makes one-command execution possible on any cluster.

> **Swap point.** `configs/data/real.yaml` + `regworld/data/ingest.py` define the adapter for real firm/consumer panels. Everything downstream reads the same Parquet schema (§8), so swapping in real data changes one config group and deletes the answer key — nothing else moves. The seam is documented in `docs/REAL_DATA.md`.

### The six claims the pipeline must produce or refute

The findings are methodological: what is demonstrated is that this pipeline recovers the truth when the truth is recoverable, and **fails legibly** when it is not.

| # | Claim | Where it is tested |
|---|---|---|
| **C1** | Bayesian calibration recovers the true behavioral parameters when the model is well specified, and fails *legibly* (a visibly biased peer coefficient `β_peer`) when supply-network capacity homophily is switched on. | Stage 4, `test_parameter_recovery.py` |
| **C2** | The observational estimate of the enforcement effect is confidently wrong when audit targeting correlates with unobserved firm capacity. The staggered-rollout DiD recovers the true effect; DoWhy's refuters catch the naive estimate. | Stage 5, `test_causal_recovers_known_effect.py` |
| **C3** | The graph-RSSM emulator reproduces the ABM's *distribution* of outcomes within tolerance at 10³–10⁴× the speed, and degrades honestly out of distribution. | Stages 6–7, §11 |
| **C4** | Of ~16 uncertain parameters, a small handful drive most outcome variance — which tells the client what to measure next. | Stage 14 |
| **C5** | **The headline finding.** Aggressive uniform enforcement maximizes compliance and backfires on market concentration: small firms exit, HHI rises. Phased, targeted enforcement buys nearly the same compliance for materially less concentration. Reported as a Pareto frontier with credible intervals across the parameter posterior — never as a point estimate, and never hard-coded (nobody writes `HHI += 0.1`; the effect is emergent). | Stage 11, Stage 15 |
| **C6** | Modeling the ten largest firms as strategic learners (MARL) either changes C5 or does not. Report which. | Stages 9–10 |

C5 is the deliverable the client asked for. C1 through C3 are what earn the right to state it. C4 is the actionable follow-up. C6 is an ablation most projects would skip — and the guide is right that skipping it is usually correct; it is here precisely so the plan can report a clean negative result honestly if that is what comes out.

### The leakage firewall (doubly enforced)

The truth lives in exactly two places, and both are walled off:

1. **The `dgp/` package is import-restricted.** Nothing downstream of Stage 1 may `import` from `regworld.dgp` except `regworld.evaluation`, which needs the answer key. Enforced by `tests/test_no_dgp_leakage.py`, which greps the source tree.
2. **The `oracle/` artifact tree is read-restricted.** `generate.py` writes `artifacts/data/observed/` (everything may read) and `artifacts/oracle/` (θ\*, Regime P q13–24, Regime F ground-truth trajectories, `do()` counterfactuals — **`regworld.evaluation` only**). Enforced by a stack-frame check in `data/store.py::read_oracle()` *and* by the same grep test.

Any convenience import of ground truth into calibration, training, or the emulator invalidates the entire evaluation section. This is not a suggestion, and neither test is optional.

---

## §2 The sixteen-tool map

Every row is auditable against Part XIX of the guide. If a row has no module, the build is incomplete. Stage numbers reference §10; phase numbers reference the seven-phase arc.

| # | Guide step | Tool(s) | Module | Script | Stage | Phase |
|---|---|---|---|---|---|---|
| 1 | Ingest raw material → Parquet | **pandas, Polars** (+ pyarrow, DuckDB) | `data/` | `generate_world.py`, `make_data.py` | 1 | 2 |
| 2 | Construct interaction structure | **NetworkX** | `graphs/build.py`, `graphs/analyze.py` | `build_graphs.py` | 2 | 2 |
| 3 | First agent-based simulation | **Mesa** (≥3.0) | `abm/` | `run_abm.py` | 3 | 3 |
| 3b | Tensorized / differentiable ABM | **AgentTorch → PyTorch fallback** | `abm/tensorized.py` | `run_abm.py --tensorized` | 3b | 3 |
| 4 | Calibrate what the data can't pin down | **NumPyro + PyMC** (+ ArviZ; SMC-ABC) | `calibration/` | `calibrate.py` | 4 | 4 |
| 5 | Interrogate causal assumptions | **DoWhy + EconML** (+ statsmodels/linearmodels DiD, causal-learn) | `causal/` | `causal_analysis.py`, `validate_simulator.py` | 5 | 4 |
| 6 | Learn a fast latent emulator | **PyTorch** (+ einops) | `models/rssm.py`, `training/` | `train_emulator.py` | 6 | 5 |
| 7 | Structure the emulator on the graph | **PyTorch Geometric** | `models/encoder.py`, `models/gnn.py`, `graphs/to_pyg.py` | — | 7 | 5 |
| 8 | Standard env interface | **Gymnasium** (≥1.0) | `environments/{abm_env,emulator_env}.py` | — | 8 | 3, 5 |
| 9 | Strategic multi-agent version | **PettingZoo** | `environments/marl_env.py` | — | 9 | 3 |
| 10 | Train a regulator policy | **SB3** (req.) → **TorchRL** (experiment) → **RLlib** (opt.) | `agents/`, `training/train_policy.py` | `train_rl.py`, `train_marl.py` | 10 | 6 |
| 11 | Scenario ensemble at scale | **Ray** (Core) | `ensemble/` | `run_ensemble.py` | 11 | 6 |
| 12 | Keep the sprawl coherent | **Hydra** (+ OmegaConf, Pydantic) | `configs/`, `types.py` | all | 12 | 1 |
| 13 | Record it all | **MLflow** (default) / **W&B** (opt.) | `tracking.py` | all | 13 | 1 |
| 14 | Close the loop on rigor | **SALib + Optuna** (+ BoTorch/Ax opt.) | `sensitivity/` | `sensitivity.py`, `optimize_policy.py` | 14 | 6 |
| 15 | Deliver the result | **Plotly + Streamlit** (+ Matplotlib) | `visualization/` | `make_figures.py`, `dashboard.py` | 15 | 7 |
| 16 | Make it an instrument, not a demo | **pytest + Docker + GitHub Actions** (+ ruff, mypy, pre-commit, uv, Make) | `tests/`, `docker/`, `.github/` | — | 16 | 1, 7 |

Two stages sit outside the sixteen because reality demands them: **Stage 0** (reconnaissance and setup) and **Stage 17** (the report — the reason the sixteen exist). Supporting cast used where the guide names it: **xarray + Zarr** for the `(policy, draw, seed, quarter, variable)` ensemble cube; **scikit-learn / SciPy / statsmodels** for metrics; **ArviZ** for every posterior.

---

## §3 Design decisions, and where to flip them

These are defaults. Each is a one-line config change, recorded so a reviewer can see the fork.

| Decision | Chosen | Why | Flip it |
|---|---|---|---|
| Bayesian engine | **NumPyro** (micro, exact-likelihood NUTS) + **SMC-ABC** (macro) | The firm decision rule is a discrete-choice model and the panel contains the decisions, so the micro-likelihood is tractable and JIT-compiled NUTS is fast; only the aggregate-only parameters need simulation-based inference. | `calibration=numpyro_bsl` for a single Bayesian-synthetic-likelihood pass; `calibration=smc_abc` for pure ABC |
| Calibration cross-check | **PyMC** re-implementation of the micro-model | Two independent implementations agreeing is a real reproducibility practice and the honest way both frameworks earn a place. In a real project you would pick one. | `calibration.crosscheck=false` |
| DGP variant (default) | **`confounded`** | The interesting case: capacity homophily on, capacity correlated with size. `wellspecified` is the unit-test world where recovery must succeed. | `dgp=wellspecified` / `dgp=misspecified` |
| Tracker | **MLflow**, local file backend | Runs with zero credentials and no network on any cluster — a prerequisite for one-command execution. | `tracking=wandb` (offline unless `WANDB_API_KEY` is set); `tracking=none` for tests |
| RL control group | **SB3 required** (PPO/SAC) | The guide calls SB3 the control group, not the experiment. Its number exists so the Dreamer agent has something to beat. | — |
| RL experiment | **TorchRL Dreamer** on imagined rollouts | The model-based acid test needs a planner that lives inside the emulator; a Dreamer-style agent is a nonstandard assembly of standard parts and TorchRL ships them. | `rl=sb3_ppo` alone if TorchRL will not install; hand-rolled loop fallback (Stage 10) |
| MARL | **RLlib optional**, IPPO fallback required | RLlib's API churn must not be able to block the build; the four-number gate does not depend on which library produced the policies. | `rl=rllib_marl` (`--extra rl`) vs the built-in IPPO |
| Sensitivity | **Morris screen → Sobol** on the emulator | Morris prunes 16 parameters to the ~8 that move anything; Sobol quantifies what remains. Sobol on the raw ABM is unaffordable — this is why the emulator exists. | `sensitivity=morris` only; `--extra opt` for BoTorch policy search |
| Framework | **PyTorch** | The guide's default; the emulator's training loop is exotic (alternating ABM collection / dynamics / policy) and PyTorch hands you the loop. | — |
| Device | **CPU-first, GPU-optional** | Cluster portability. Nothing in the required path assumes CUDA. | `device=cuda`, `--extra gpu` |
| Step resolution | **Quarterly** | Firm-level economic dynamics (adoption, exit cascades, HHI) read naturally on a quarterly clock, and the consumer survey is quarterly. 24 quarters ≈ 6 years. | `horizon_quarters` in `config.yaml` |

If any of these is wrong for your setting, it is a one-line change — say so and revise the plan, not the code around it.

---

## §4 Repository layout

Follows the guide's Part XIII layout exactly, extended only where the sixteen tools demand it. One package, `regworld`, imported everywhere.

```text
regworld/
├── .github/workflows/{ci.yml, docker.yml, nightly.yml}
├── configs/                         # Hydra: composable groups, CLI-overridable (§6)
├── src/regworld/
│   ├── types.py                     # Pydantic config models; validate_config(DictConfig) -> RegWorldConfig
│   ├── seeding.py  logging_conf.py  tracking.py
│   ├── dgp/                         # THE ANSWER KEY — import-restricted (evaluation only)
│   │   ├── world.py                 #   entity generation, true parameters θ*
│   │   ├── dynamics.py              #   thin wrapper binding θ* to the shared decision rules
│   │   ├── observation.py           #   measurement error, lags, missingness, sampling
│   │   └── history.py               #   Regime P: the analogous prior regulation, staggered rollout
│   ├── rules.py                     # PURE decision functions (§7.4), shared by dgp/ AND abm/
│   ├── data/       generate.py  ingest.py  schema.py  store.py  duck.py
│   ├── graphs/     build.py  analyze.py  to_pyg.py
│   ├── abm/                         # the ESTIMATED model (shares rules.py, uses estimated θ)
│   │   ├── model.py  agents.py  collect.py  policies.py
│   │   └── tensorized.py            #   Stage 3b: differentiable ABM (AgentTorch or pure torch)
│   ├── calibration/ summaries.py  micro_numpyro.py  micro_pymc.py  macro_smc.py  diagnostics.py
│   ├── causal/     graph.py  estimate.py  did.py  refute.py  discovery.py  ground_truth.py  gate.py
│   ├── models/     encoder.py  rssm.py  gnn.py  heads.py  world_model.py
│   ├── training/   datamodule.py  losses.py  train_emulator.py  train_policy.py  checkpoint.py
│   ├── environments/ abm_env.py  emulator_env.py  marl_env.py  wrappers.py
│   ├── agents/     scripted.py  sb3_agents.py  dreamer.py  marl.py
│   ├── evaluation/ predictive.py  distributional.py  calibration_curves.py  dtw.py
│   │                planning_utility.py  behavioral_fidelity.py  parameter_recovery.py
│   │                causal_eval.py  ood.py  backtest.py  ablations.py  report.py
│   ├── ensemble/   scenarios.py  ray_ensemble.py  cube.py
│   ├── sensitivity/ salib_gsa.py  optuna_search.py  bo_policy.py
│   └── visualization/ figures.py  interactive.py  dashboard.py
├── scripts/                         # entry points, all Hydra-decorated
│   ├── run_pipeline.py              #   THE DRIVER: orchestrates stages 0..17
│   ├── generate_world.py  make_data.py  build_graphs.py  run_abm.py
│   ├── calibrate.py  causal_analysis.py  validate_simulator.py
│   ├── train_emulator.py  eval_emulator.py  train_rl.py  train_marl.py
│   ├── run_ensemble.py  sensitivity.py  optimize_policy.py
│   ├── make_figures.py  build_report.py  dashboard.py
├── tests/                           # mirrors src/ (§12)
├── notebooks/                       # exploration only — never imported by src/
├── experiments/                     # Hydra + MLflow run dirs (gitignored)
├── artifacts/                       # observed/, oracle/, graphs, checkpoints (gitignored, DVC-trackable)
├── reports/
│   ├── FINDINGS.md                  #   auto-assembled deliverable
│   ├── figures/
│   ├── run_manifest.json
│   └── simulator_discrepancy.md     #   written only if the §10 Stage-5 gate FLAGS
├── docs/{DEVIATIONS.md, REAL_DATA.md, MINIMAL_PATH.md}
├── docker/{Dockerfile, Dockerfile.cuda, compose.yaml}
├── slurm/submit.sbatch
├── pyproject.toml  uv.lock  Makefile  .pre-commit-config.yaml  .gitignore
├── CLAUDE.md  PLAN.md  PROGRESS.md  README.md
```

**The two load-bearing conventions** (guide, Part XIII):

- *Notebooks are for looking at things.* Any code that produces a claimed result lives in `src/` under test and is invoked from `scripts/` with a Hydra config. `tests/test_layering.py` enforces that nothing in `src/` imports from `notebooks/`.
- *The world and the model are different objects.* `dgp/` is the true world; `abm/` is the estimated model that runs on *observed* data with *estimated* parameters. They share exactly one thing — the pure decision functions in `rules.py` — so the equations are written once and never drift, while the answer key stays sealed behind the firewall (§1).

---

## §5 Environment, dependencies, and portability

The pipeline must run identically on a laptop, a single cloud VM, a Slurm array, and inside Docker on a K8s pod. Sixteen tools in one dependency graph is the single largest execution risk in the whole plan — JAX and PyTorch and Ray and PyMC and EconML and BoTorch do not always co-resolve — so the strategy is designed around that risk.

### Manager and interpreter

- **uv** for everything. `uv.lock` is committed. `make setup` = `uv sync --extra dev`.
- **Python `>=3.11,<3.13`.** PyG / Ray / EconML / PyMC wheel coverage is reliable here; 3.13 is not worth the fight until you have confirmed those wheels exist.

### Core-plus-extras, with graceful degradation

`pyproject.toml` (Appendix A) declares a **core** group that must always solve, plus **extras** that are allowed to fail independently. A stage whose extra failed is marked `SKIPPED` or `DEGRADED` with a loud message — never a silent crash, and never fatal unless a downstream stage declares it a hard dependency (§15). Stage 0 installs the extras one at a time and records which ones failed to `.stage_skips`.

Floor pins carry meaning and are load-bearing: `mesa>=3.0` (the `AgentSet` API, not the removed scheduler classes), `gymnasium>=1.0` (five-tuple `step`), `stable-baselines3>=2.3` (Gymnasium 1.x compatibility), `torch-geometric>=2.6` (so `HeteroConv`/`SAGEConv`/`GATConv` run on pure-torch scatter and you never fight `torch-scatter` wheels). If a floor pin cannot be satisfied, the resolver's choice wins: record it, adapt the code, note it in `DEVIATIONS.md`.

### Fallback if the single environment will not solve

Pass `--isolated-envs` to the driver. Each stage group then runs in its own uv-managed venv as a subprocess, communicating only through files on disk (Parquet, Zarr, JSON, `.pt` checkpoints). Slower to set up, immune to resolver conflict. The pipeline is already file-mediated between stages, so this costs almost nothing architecturally. Build the single-env path first; keep this in your back pocket.

### The JAX-and-PyTorch problem

NumPyro brings JAX. **JAX and PyTorch in one process will fight over GPU memory** and on CPU merely bloat — this is a top cause of a run that OOMs at Stage 4. Therefore every JAX stage runs in a **subprocess**: `scripts/calibrate.py` is launched via `subprocess.run`, never imported into the main pipeline process. Set `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless calibration is deliberately on GPU (`calibration.device=gpu`). This is not optional hygiene; it is the difference between a run that finishes and one that does not.

### Portability contract (hard rules)

1. **No network at runtime.** All data is generated. If a module needs the network, it is wrong.
2. **CPU by default.** Torch CPU wheels. `device: auto` resolves to CUDA only if `torch.cuda.is_available()`. Bitwise GPU determinism is *not* pursued (§13).
3. **Headless everywhere.** `MPLBACKEND=Agg`, no `plt.show()` in `src/`. Streamlit runs with `--server.headless true`.
4. **Artifact root is configurable:** `REGWORLD_ARTIFACT_ROOT` (default `./artifacts`). Point it at cluster scratch or a mounted bucket; nothing else changes.
5. **Ray attaches or runs local:** `ray.init(address=os.environ.get("RAY_ADDRESS", "local"), ignore_reinit_error=True)`. Zero-config locally, cluster-native when `RAY_ADDRESS` is set. Do not initialize CUDA before `ray.init()`; use spawn, not fork.
6. **Slurm sweeps via submitit:** `make slurm` uses `hydra/launcher=submitit_slurm` (`--extra slurm`). Keep the two roles distinct — **Ray Core handles the ensemble *inside* a job; Hydra handles sweeps *across* jobs.**
7. **Seeds flow through one function.** Everything routes through `regworld.seeding.seed_everything(seed)` and explicit `np.random.default_rng(seed)` generators. **No bare `np.random.*` calls anywhere.**

Full `pyproject.toml` in **Appendix A**, `Makefile` in **B**, `docker/` in **C**, CI in **D**.

---

## §6 Configuration

Hydra composes; Pydantic validates. The guide's rule holds throughout: *"every run is a composition of config groups, and every variant is a command-line override rather than a copied script."* Every script's first two lines after `@hydra.main` are:

```python
cfg_obj = validate_config(cfg)      # Pydantic — fails in the first second on a typo'd key
seed_everything(cfg_obj.seed)
```

Misspell `emulator.latent_dim` and the run dies immediately, not after six GPU-hours. `tests/test_configs.py` composes every group value against the defaults and validates it.

### Config groups

```text
configs/
├── config.yaml                    # the defaults list (below)
├── profile/       smoke.yaml  dev.yaml  full.yaml          # size knobs ONLY (§14)
├── compute/       local.yaml  ray_local.yaml  ray_cluster.yaml  slurm.yaml
├── hydra/launcher/ basic.yaml  joblib.yaml  submitit_slurm.yaml
├── data/          synthetic.yaml  real.yaml
├── dgp/           wellspecified.yaml  confounded.yaml  misspecified.yaml
├── population/    small.yaml  base.yaml  large.yaml
├── network/       scalefree.yaml  smallworld.yaml  empirical.yaml
├── behavior/      logit_baseline.yaml  logit_sticky.yaml  bounded_rational.yaml
├── abm/           default.yaml
├── objective/     balanced.yaml  compliance_first.yaml  competition_first.yaml
├── calibration/   numpyro_nuts.yaml  smc_abc.yaml  numpyro_bsl.yaml  pymc_nuts.yaml
├── causal/        default.yaml
├── emulator/      rssm_gnn.yaml  rssm_flat.yaml  gru_baseline.yaml   # last two are ablations
├── env/           single_agent.yaml  multi_agent.yaml
├── policy/        none.yaml  uniform_low.yaml  uniform_high.yaml  targeted.yaml
│                  phased_targeted.yaml  rl_ppo.yaml  rl_dreamer.yaml  bo_optimal.yaml
├── rl/            sb3_ppo.yaml  sb3_sac.yaml  torchrl_dreamer.yaml  rllib_marl.yaml
├── ensemble/      grid_small.yaml  grid_full.yaml
├── sensitivity/   morris.yaml  sobol.yaml
├── tracking/      mlflow.yaml  wandb.yaml  none.yaml
└── eval/          fast.yaml  full.yaml
```

### `configs/config.yaml`

```yaml
defaults:
  - _self_
  - profile: smoke
  - compute: ray_local
  - data: synthetic
  - dgp: confounded            # the interesting case; wellspecified is the unit test
  - population: base
  - network: scalefree
  - behavior: logit_sticky
  - abm: default
  - objective: balanced
  - calibration: numpyro_nuts
  - causal: default
  - emulator: rssm_gnn
  - env: single_agent
  - policy: phased_targeted
  - rl: sb3_ppo
  - ensemble: grid_small
  - sensitivity: sobol
  - tracking: mlflow
  - eval: fast

seed: 0
seeds: [0, 1, 2, 3, 4]           # every headline claim runs across all of these
horizon_quarters: 24             # Regime F forecast horizon, and Regime P history length
observed_quarters: 12            # Regime P observation window (q13–24 = backtest holdout)
device: auto                     # auto | cpu | cuda

paths:
  root:      ${oc.env:REGWORLD_ARTIFACT_ROOT,${hydra:runtime.cwd}/artifacts}
  data:      ${paths.root}/data
  graphs:    ${paths.root}/graphs
  reports:   ${hydra:runtime.cwd}/reports

stages:                          # the driver reads this; any stage can be turned off
  recon: true
  data: true
  graphs: true
  abm: true
  tensorized_abm: true
  calibration: true
  causal: true
  emulator: true
  envs: true
  marl: true
  rl: true
  ensemble: true
  sensitivity: true
  figures: true
  report: true

hydra:
  run:
    dir: experiments/${now:%Y-%m-%d}/${now:%H-%M-%S}-${hydra.job.name}
  sweep:
    dir: experiments/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

All configs validate on load against Pydantic models in `types.py`. `profile/*.yaml` and `compute/*.yaml` files use `# @package _global_` and override only their own keys — **the profile is the entire scaling story, the compute group is the entire portability story, and the two are orthogonal.**

### Profiles (size knobs, and nothing else)

| Knob | `smoke` (CI, <6 min, 4 cores) | `dev` (1 node / 16 vCPU, ~2 h) | `full` (cluster) |
|---|---|---|---|
| `population.n_firms` | 200 | 2,000 | 20,000 |
| `population.n_consumer_segments` | 6 | 20 | 40 |
| `population.n_regions` (Regime P) | 4 | 8 | 8 |
| `calibration.design_points` × `replicates` | 32 × 2 | 256 × 8 | 1,024 × 16 |
| `calibration.nuts` (warmup/draws/chains) | 150/150/2 | 1000/1000/4 | 2000/2000/4 |
| `calibration.smc_abc.particles` | 256 | 2,000 | 8,000 |
| `emulator.train_episodes` / `epochs` | 64 / 3 | 2,000 / 40 | 20,000 / 150 |
| `emulator.train_steps` | 300 | 30,000 | 200,000 |
| `rl.total_timesteps` (in emulator) | 5,000 | 300,000 | 3,000,000 |
| `ensemble.posterior_draws` × policy grid × seeds | 8 × 6 × 1 | 1,000 × 7 × 3 | 4,000 × 7 × 5 |
| `sensitivity.sobol_n` (Saltelli N) | 64 | 1,024 | 8,192 |
| `sensitivity.optuna_trials` / `bo_evals` | 2 / 4 | 12 / 40 | 60 / 200 |
| `eval.abm_validation_episodes` | 8 | 64 | 256 |

Cutting the `dev` run without touching the science: `stages.marl=false` (−12 min), `calibration.crosscheck=false` (−6 min), `emulator.train_steps=15000` (−12 min), `sensitivity.optuna_trials=4` (−6 min). All four together land near 75 minutes with the headline claims intact.

---

## §7 The world: ground-truth DGP specification

**This is the science. Write it first and write it carefully — every other stage is graded against it. Implement the equations as written; do not improvise them.** Every symbol here is a config key, a calibrated parameter, or a known constant. The DGP lives in `regworld/dgp/` and the *pure decision functions* it uses live in `regworld/rules.py`, shared unchanged with the estimated ABM (`regworld/abm/`).

### 7.1 Entities

**Firms** `i = 1..F`, fixed attributes drawn once at generation:

| Symbol | Meaning | Distribution |
|---|---|---|
| `s_i` | size (revenue scale) | LogNormal(0, 1.1), normalized so median = 1 |
| `k_i` | sector | Categorical over K = 6 sectors (skewed) |
| `d_i` | data intensity | Beta(2,2), shifted by sector mean |
| `c_i` | compliance-cost coefficient | Gamma(2, 0.5), correlated with sector |
| `Q_i` | product quality | Normal(0, 1) |
| `m0_i` | baseline margin | Beta(5,20) × 0.5 |
| **`z_i`** | **latent capacity / risk tolerance** | **Normal(0,1) — UNOBSERVED. The planted confounder (§7.7).** Raises compliance propensity, is correlated with size, and drives supply-graph homophily. |

Firm state per quarter: `y_i(t) ∈ {0,1}` compliant, `alive_i(t) ∈ {0,1}`, `R_i(t)` revenue, `fines_i(t)`, `tenure_i(t)` quarters compliant.

**Consumer segments** `j = 1..S`: population weight `w_j`, privacy sensitivity `p_j ~ Beta(2,3)`, price sensitivity, trust `T_j(t) ∈ [0,1]` with `T_j(0) ~ Beta(5,3)`, budget `b_j`.

**Regulator** (single agent): audit budget `B` (fraction of firms auditable per quarter at full enforcement), fine scale `Φ`, targeting rule, phase-in schedule — all set by the **policy levers** (§7.5).

**Industry associations** `A = 4` (institution nodes): each firm belongs to one, membership probability rising with size. Each aggregates publicized enforcement into a sector-level salience signal.

### 7.2 Networks (NetworkX → PyG `HeteroData`)

Node types: `firm`, `segment`, `association`, `regulator`. **Two versions of every graph exist and both matter:** the **true** graph (used by the DGP and, at generation time, θ\*) and the **observed** graph (20% of edges missing, 3% spurious) used by calibration and the emulator. Running the emulator on the observed graph while grading it against a DGP that ran on the true graph is not a bug — it is the realistic setting, and the gap between them is a result worth reporting.

| Edge type | Generator | Meaning |
|---|---|---|
| `(firm, supplies, firm)` | Preferential attachment (m=2), directed by size rank, with **sector homophily and capacity homophily**: `P(i→j) ∝ deg(j)^α · exp(−λ_homoph·|z_i − z_j|)` | Contractual pass-through; compliant customers pressure suppliers. **`λ_homoph` is the knob that confounds peer-effect estimates.** |
| `(segment, influences, segment)` | Watts–Strogatz (k=6, p=0.1) | Trust / behavior contagion among consumers |
| `(segment, buys_from, firm)` | Preferential attachment on `s_i × sector-match`, ~8 firms per segment | Market structure; carries spend shares |
| `(firm, member_of, association)` | Star, membership prob ∝ `s_i` | Information channel |
| `(regulator, audits, firm)` | Dynamic, per-quarter | Enforcement events |

`dgp=wellspecified` sets `λ_homoph = 0`; `dgp=confounded` sets `λ_homoph = 1.5`. `graphs/analyze.py` logs assortativity-by-`z`, which should be ≈0 under `wellspecified` and clearly positive under `confounded` — a cheap sanity check that the homophily knob is doing something. `graphs/to_pyg.py` maps the four static edge types to a PyG `HeteroData` with a required NetworkX↔PyG round-trip test; node features are the **observed** attributes only.

### 7.3 Behavioral parameters θ — what calibration must recover

Sixteen parameters, split by how they are identified. This split *is* the two-part calibration design in Stage 4.

**Group A — firm-decision logit** (recoverable **exactly** by the micro-likelihood, Stage 4a — the panel contains the firm-level decisions and their lagged neighbourhood, so the likelihood is the model's own equation):

| Symbol | Config key | Meaning | Prior | θ\* (truth) |
|---|---|---|---|---|
| β₀ | `beta_0` | intercept | Normal(0, 2) | −1.2 |
| β_enf | `beta_enforce` | sensitivity to perceived enforcement risk | HalfNormal(2) | 2.5 |
| β_cost | `beta_cost` | sensitivity to compliance-cost share | HalfNormal(2) | 1.8 |
| β_peer | `beta_peer` | imitation of supply-chain neighbours (lagged) | Normal(1, 1) | 1.4 |
| β_assoc | `beta_assoc` | responsiveness to association compliance (lagged) | Normal(0.5, 1) | 0.6 |
| β_size | `beta_size` | direct size effect on compliance | Normal(0, 1) | 0.25 |
| β_cust | `beta_customer` | responsiveness to privacy-sensitive revenue share | HalfNormal(1) | 0.9 |
| φ_phase | `phi_phase` | salience of the phase-in schedule | Normal(0.5, 0.5) | 0.6 |
| β_stick | `beta_stick` | switching cost / hysteresis | HalfNormal(1) | 2.0 |
| **β_cap** | `beta_capacity` | effect of latent capacity `z_i` — **UNOBSERVED; absent from the fitted model** | (truth only) | 0.9 |

Plus two nuisance parameters the micro-model must also recover, because the reporting error is *in the data* (§7.9): misclassification rates `q₀, q₁ ~ Beta(2, 20)` (false-positive / false-negative compliance reports).

**Group B — consumer, market, and enforcement dynamics** (recoverable **approximately** by the macro-model, Stage 4b — SMC-ABC on aggregate curves, because these parameters do not appear in the firm-level likelihood):

| Symbol | Config key | Meaning | Prior | θ\* (truth) |
|---|---|---|---|---|
| γ_scale | `gamma_scale` | economies of scale in compliance cost (the backfire driver) | Beta(3,3) | 0.45 |
| ℓ_learn | `ell_learn` | learning-by-doing cost reduction | Beta(2,4) | 0.30 |
| α_trust | `alpha_trust` | consumer trust update rate | Beta(2,5) | 0.30 |
| ρ_infl | `rho_influence` | consumer social-contagion strength | Beta(2,8) | 0.15 |
| μ_priv | `mu_privacy` | privacy-driven spend reallocation toward compliers | HalfNormal(1) | 0.80 |
| δ_exit | `delta_exit` | exit-hazard scale (small-firm fragility) | HalfNormal(0.5) | 0.25 |

Fixed / known (not calibrated): audit budget `B`, fine scale `Φ`, targeting exponents `γ, ψ`, network hyperparameters `α, λ_homoph, m, k, p`, observation noise `σ_obs`, exit revenue floor `ξ`, quality weight `λ_quality`.

> **Identifiability notes — do not skip.**
> - The logit temperature is **fixed at 1**. Do not add a free temperature *and* free β's; they are not jointly identified, and NUTS will tell you so in the ugliest possible way (divergences, R-hat 1.4, a week gone).
> - `β_peer` and `β_assoc` are correlated (peer and association pressure both track sector compliance). Expect it, and expose it with an ArviZ `plot_pair` (§10, Stage 4d) rather than pretending it away.
> - `β_cap` is deliberately unrecoverable: it multiplies an unobserved variable. Under `dgp=confounded`, `β_peer` absorbs part of it through capacity homophily and is biased upward — this is C1's failure half, and it is more valuable than the success half.

### 7.4 Dynamics (one quarter)

Implemented as **pure functions in `regworld/rules.py`**, taking `(state, θ, policy, rng)` and returning the next state with no in-place agent mutation. The DGP binds θ = θ\*; the estimated ABM binds θ = posterior draw.

**Regulator action** `a(t) = (e_t, τ_t, φ_t, subsidy_t)` from the policy levers (§7.5). Targeting weight, audit probability, and expected penalty:

```
w_it   = (1 − τ_t) + τ_t · ( s_i^γ · (1 − y_{i,t−1}) )              # target large + previously non-compliant
α_it   = clip( e_t · B · F · w_it / Σ_l alive_l·w_lt , 0, 1 )       # audit probability
q_it   = α_it · Φ · s_i^ψ · φ_t · (1 + ω·publicity_{k_i}(t)) · 1[t ≥ t_start]   # perceived penalty risk
```

**Compliance-cost share** — this is the backfire mechanism (economies of scale + learning-by-doing):

```
κ_it = c_i · d_i · (s_i / s_med)^(−γ_scale) · φ_t · (1 − ℓ_learn · min(1, tenure_it / 12))
κ_it ← κ_it · (1 − subsidy_t · 1[s_i in bottom size tercile])       # cost as a share of revenue, ↓ in size
```

**Firm compliance decision** (logit, sticky; this is the equation Stage 4a must recover):

```
u_it = β₀
     + β_enf  · q_it
     − β_cost · κ_it
     + β_peer · n_{i,t−1}
     + β_assoc· m_{k,t−1}
     + β_size · log s_i
     + β_cust · x_{i,t}
     + φ_phase· φ_t
     + β_cap  · z_i                       # UNOBSERVED term — present in the DGP, absent from the fitted model
     − β_stick· 1[y_{i,t−1} = 0]          # switching cost pulls a previously non-compliant firm back

n_{i,t−1} = Σ_{l∈N_supply(i)} w_il · alive_l · y_{l,t−1} / Σ_l w_il    # lagged neighbour compliance (in+out)
m_{k,t−1} = mean lagged compliance among firm i's association members
x_{i,t}   = Σ_j p_j · spend_{ji}(t) / R_i(t)                          # privacy-weighted revenue share
y_it ~ Bernoulli( σ(u_it) )
```

Peer and association terms are **lagged deliberately**: lagging breaks the simultaneity that produces Manski's reflection problem, and it is also what a firm actually observes at decision time.

**Consumer segment update:**

```
exposure_jt = Σ_i spend_share_{ji}(t) · y_i(t)                                    # compliance exposure
T_j(t+1)    = clip( T_j + α_trust·(exposure_jt − T_j)
                        + ρ_infl · mean_{l∈N_infl(j)}(T_l − T_j) + ε , 0, 1)
v_{ji}      = λ_quality · Q_i + μ_priv · p_j · y_i(t) + ε_{ji}                    # spend utility (privacy-sensitive segments flee non-compliers)
spend_{ji}(t+1) = w_j · b_j · softmax_i(v_{ji})   over alive firms linked to j
R_i(t+1)    = Σ_j spend_{ji}(t+1)
```

**Enforcement.** The regulator draws `⌊α·F⌋` audits without replacement from `α_it`. Audited ∧ non-compliant ⇒ fine `F_i = min(Φ·R_i, f_cap·R_i)`, deducted from profit; each association's `publicity_k` is an EWMA of sector fines.

**Market and exit** (where the backfire lives):

```
μ_i(t) = m0_i − y_i(t)·κ_it − fines_i(t)/R_i(t)                     # margin
exit if rolling-3-quarter mean revenue < ξ·s_i for two consecutive quarters,
   with probability min(1, δ_exit · |μ_i| · (s_med / s_i))          # small firms are more fragile
```

Exited firms are removed from all softmaxes, so their spend redistributes to survivors and **HHI rises**. Strict enforcement plus high compliance cost drives small firms out, their demand flows to incumbents, and concentration climbs. The regulation achieves compliance and hands the market to the incumbents. That mechanism is the client's "where might this backfire," and it is **emergent, not hard-coded** — nobody writes `HHI += 0.1`.

### 7.5 Policy levers = the Gymnasium action space

```
Box(low=[0.0, −1.0, 0.0, 0.0], high=[1.0, 1.0, 1.0, 1.0], dtype=float32)
  [0] enforcement  e ∈ [0,1]     →  B(t) = e · B_max
  [1] targeting    τ ∈ [−1,1]    →  audit prob ∝ s_i^τ   (τ<0 targets small firms, τ>0 large; here folded into w_it)
  [2] phase_speed  ∈ [0,1]       →  φ(t) = min(1, t / L),  L = 12 − phase_speed·10   (2..12 quarters)
  [3] subsidy      ∈ [0,1]       →  compliance-cost subsidy for the bottom size tercile
```

(The Stage-9 strategic firms add their own per-firm action space; see §10.)

### 7.6 Outcomes, reward, and the backfire flag

Per quarter the ABM emits the **outcome vector** — also what the emulator's global head predicts:

```
compliance_rate              (unweighted, and revenue-weighted)
compliance_by_tercile[3]     (small / mid / large)
HHI                          = 10_000 · Σ_i (R_i / ΣR)²
mean_trust                   (weight-weighted)
consumer_surplus  CS(t)      = Σ_j w_j · logsumexp_i(v_{ji})        # logit inclusive value
exit_rate_cum
enforcement_cost             = audits(t) · unit_cost
```

**Regulator reward** — a superset of the natural terms, so no single number smuggles in a value judgment (`configs/objective/*.yaml`):

```
r(t) = w_c · compliance_rate(t)
     − w_h · max(0, HHI(t) − HHI(0)) / 10_000
     − w_s · max(0, CS(0) − CS(t)) / |CS(0)|
     − w_e · enforcement_cost(t) / E_max
     + w_t · (mean_trust(t) − mean_trust(0))
     − w_x · exit_rate_cum(t)

  balanced:          w = (1.0, 0.5, 0.5, 0.1, 0.3, 0.3)
  compliance_first:  w = (1.0, 0.0, 0.0, 0.1, 0.0, 0.0)
  competition_first: w = (0.4, 0.8, 1.0, 0.1, 0.3, 0.6)
```

The ensemble runs across all three weightings so the client sees a **trade-off frontier**, not one number.

**Backfire flag** — the finding the client is actually paying for:

```
backfire(t) := (compliance_rate(t) > compliance_rate(0))
             ∧ (HHI(t) > HHI(0))
             ∧ (CS(t) < CS(0))
```

Report `P(backfire at q24 | policy)` for every policy in the grid, and the compliance-vs-concentration Pareto frontier with credible bands. This pair is the headline (C5).

### 7.7 The causal structure, and the planted confounder

True DAG in the simulator (arrows the analyst assumes vs. arrows known by construction are annotated in `causal/graph.py`):

```
size ─────────────► audited            (regulator targets by size: τ)
size ─────────────► compliance_cost ──► compliant_next
size ─────────────► capacity z          (corr(z, size) ≈ +0.35)
data_intensity ───► compliance_cost
capacity z ───────► compliant_next       ONLY   (z does NOT cause audited directly)
capacity z ───────► supply-edge formation (homophily λ_homoph)  ── biases peer estimates
sector ───────────► data_intensity, quality
neighbour_compliance ► compliant_next
audited ──────────► perceived_risk ───► compliant_next          ← the effect we want to estimate
compliant ────────► revenue ──► exit ──► HHI
```

The confounder `z` is planted so it does **two** kinds of damage, and the pipeline measures both:

1. **It biases the peer coefficient in *calibration*** (Stage 4). Under `λ_homoph > 0`, firms with similar capacity are connected, and their correlated compliance looks like contagion — so `β_peer` is biased upward. This is the classic homophily-versus-contagion confound.
2. **It biases the audit effect in *causal analysis*** (Stage 5). Because `z` is correlated with size and the regulator targets on size, a naive regression of `compliant_next ~ audited` is confounded, and `z` is unmeasured, so no backdoor set built from observables closes it.

`dgp=wellspecified` disables both (`λ_homoph = 0`, `corr(z, size) = 0`); `dgp=confounded` enables both. This makes the guide's warning executable in two places at once: *"Estimating a treatment effect with EconML while a confounder sits unmeasured produces a precise estimate of the wrong quantity"* — and *"correlated compliance among connected firms looks like contagion when it is really homophily."* We can prove both, because we know the truth.

### 7.8 The historical episode (the natural experiment)

`regworld/dgp/history.py` generates **Regime P**: the analogous prior regulation, run for 24 quarters, with enforcement switched on **region by region across `R` regions at staggered quarters `t_r`**, where `t_r` is drawn **independently of firm characteristics**. Rollout timing is therefore exogenous by construction — which is exactly what makes the difference-in-differences identified. Regions not yet treated at quarter `t` are the controls for regions already treated; the pre-treatment quarters give the event study its flat pre-trends (flat because we made them flat — if they are not, the DGP has a bug). Because we wrote this world, we know the true effect the DiD should recover, and Stage 5 grades it against that.

### 7.9 The observation model (what the "data" actually contains)

`regworld/dgp/observation.py` degrades Regime P's ground truth into something that looks like a real corpus:

- **Firm registry** — `size_decile, sector, association`, plus a noisy cost proxy `cost_index_i = c_i + N(0, 0.4)`. `z_i` and the true `c_i` are **absent**.
- **Firm panel** — a 20% sample of firms, quarters 1–12, with self-reported compliance at a **one-quarter reporting lag and 5% misclassification**, `revenue_noisy`, `audited`, `fined`, `alive`, and the firm's `region` and `treatment_quarter` (the staggered-rollout structure that Stage 5 needs).
- **Consumer survey** — quarterly, a sampled **40% of segments**, with **nonresponse correlated with privacy sensitivity** (a selection problem, planted on purpose).
- **Market statistics** — quarterly aggregate revenue shares, rounded, sector-level only for small firms.
- **Supply edge list** — the observed graph: 20% of true edges missing, 3% spurious.

Stage 1's job is to clean and join this into an analysis-ready panel. The guide predicts that stage takes longer than anyone budgets, and it will. The reporting lag and misclassification are *in the data*: Stage 4 must model them (it recovers `q₀, q₁`) or eat the bias.

### 7.10 Ground-truth interventions (only the simulator can do this)

`regworld/causal/ground_truth.py` runs the DGP twice from an identical state and seed, under `do(e = e_low)` and `do(e = e_high)` (and, per firm, `do(audited = 1)` vs `do(audited = 0)`), and measures the **true ATE and the true CATE by size tercile**. Every estimator in Stage 5 is scored against this. It is the guide's *"a world model can serve as the laboratory where identification strategies get stress-tested on data whose true causal structure you control, because you wrote it"* — made into a number.

---

## §8 Data contracts

Parquet, snappy, one file per table. Schemas declared in `data/schema.py` as Polars schema dicts and validated on read — a column-type surprise at quarter three of the project is a wasted week. DuckDB gives `SELECT * FROM panel WHERE sector = 3 AND quarter > 12` over Parquet with no database to stand up; use it in notebooks and the dashboard.

```text
artifacts/data/observed/            # everything may read
  firm_registry.parquet             firm_id, sector, size_decile, data_intensity, association,
                                    cost_index (noisy proxy)        — NO capacity z, NO true betas
  firm_panel.parquet                firm_id, quarter, region, treatment_quarter,
                                    reported_compliant (bool, 1-qtr lag, 5% misclass),
                                    revenue_noisy, audited (bool), fined (bool), alive (bool)
                                    — Regime P, 20% firm sample, quarters 1..12 (calibration + DiD)
  aggregate_series.parquet          quarter, compliance_rate_obs, compliance_rate_weighted_obs,
                                    hhi_obs, mean_trust_obs, exit_rate_obs   (each + N(0, sigma_obs))
  consumer_survey.parquet           segment_id, quarter, trust_reported, privacy_bucket
                                    — sampled 40% of segments, nonresponse ∝ privacy sensitivity
  market.parquet                    quarter, sector, revenue_share_rounded
  graphs/                           supply_edges.parquet, influence_edges.parquet,
                                    market_edges.parquet, membership_edges.parquet   (OBSERVED graph)
  views.duckdb                      DuckDB views over all of the above

artifacts/oracle/                   # regworld.evaluation ONLY — see the firewall in §1
  theta_star.json                   all θ*, both DGP variants
  regime_p_full.parquet             quarters 1..24, all firms, no noise, TRUE graph
  regime_f_truth.zarr               (policy, seed, quarter, variable) ground-truth futures
  do_interventions.parquet          per-firm and aggregate do(audit / enforcement) outcomes
  true_graphs/                      the un-degraded supply/influence/market/membership graphs
```

**Ensemble output** (Stage 11), via **xarray → Zarr**:

```
dims:   (policy, draw, seed, quarter, variable)
coords: policy   = policy_id (stable hash of the lever vector)
        draw     = posterior-draw id
        variable = [compliance_rate, compliance_rate_weighted, hhi, mean_trust,
                    consumer_surplus, exit_rate, enforcement_cost, reward, backfire]
```

The coordinates travel with the data, so nobody has to remember whether axis 2 was seed or quarter. Flatten to Parquet for DuckDB queries and the dashboard. `tests/test_no_dgp_leakage.py` asserts no file under `observed/` contains `capacity`/`z_i` or the true β columns.

---

## §9 Interfaces

Contract-level signatures; the stages fill in the rest. These are the seams the whole pipeline is bolted along — an autonomous agent should treat them as fixed and build to them.

```python
# seeding.py
def seed_everything(seed: int) -> np.random.Generator: ...     # Python, NumPy, torch, JAX; returns a Generator

# types.py
def validate_config(cfg: DictConfig) -> RegWorldConfig: ...    # Pydantic; raises on any unknown/typo'd key

# dgp/  (import-restricted: evaluation + tests only)
def generate_ground_truth(cfg: RegWorldConfig) -> GenerationResult: ...   # writes observed/ + oracle/

# rules.py  (pure; shared by dgp/ and abm/)
def firm_utility(state, theta, policy, idx) -> np.ndarray: ...            # §7.4 logit, no mutation
def step_consumers(state, theta, rng) -> ConsumerState: ...
def step_market_and_exit(state, theta, rng) -> MarketState: ...

# data/
def ingest(cfg) -> None: ...                                              # Polars lazy: read→validate→join→Parquet
def validate_table(df: pl.DataFrame, spec: TableSpec) -> None: ...
def read_observed(name: str) -> pl.DataFrame: ...
def read_oracle(name: str) -> object: ...                                # raises unless caller is evaluation/ or a test

# graphs/
def build_graphs(cfg, rng) -> RegGraphs: ...                             # NetworkX bundle (true + observed)
def to_hetero_data(g: RegGraphs, node_features: dict[str, np.ndarray]) -> HeteroData: ...

# abm/model.py   (Mesa >= 3.0 — AgentSet API, NOT RandomActivation)
class RegulationModel(mesa.Model):
    def __init__(self, cfg, graphs, theta: Theta, policy: PolicyLevers, seed: int): ...
    def step(self) -> None: ...
    def run(self, quarters: int) -> Trajectory: ...                     # outcomes + firm panel + events

# abm/tensorized.py   (Stage 3b — differentiable)
def rollout_tensorized(cfg, graphs, theta, policy, seed) -> Trajectory: ...   # sparse matmuls + Bernoulli

# environments/
class AbmEnv(gym.Env): ...            # dynamics = the true ABM (slow; oracle for evaluation)
class EmulatorEnv(gym.Env): ...       # dynamics = the learned GraphRSSM (fast; policies train here)
class RegulationMARLEnv:              # PettingZoo Parallel: regulator + top-K strategic firms
    def reset(self, seed=None): ...
    def step(self, actions): ...

# calibration/
def summary_statistics(traj: Trajectory) -> np.ndarray: ...             # macro summaries (Stage 4b)
def fit_micro_numpyro(panel: pd.DataFrame, cfg) -> arviz.InferenceData: ...   # exact-likelihood NUTS
def fit_macro_smc(cfg, s_obs: np.ndarray) -> arviz.InferenceData: ...   # SMC-ABC on aggregates
def diagnose(idata: arviz.InferenceData) -> DiagnosticReport: ...       # R-hat, ESS, divergences, PPC

# causal/
def build_causal_graph(cfg, variant: str) -> str: ...                   # GML, "analyst" | "true"
def estimate_effects(panel: pd.DataFrame, gml: str, cfg) -> CausalEstimates: ...  # OLS, DML, DiD, CATE
def refute(estimate, panel, cfg) -> Refutations: ...                    # DoWhy refuters + E-value
def true_effects(cfg, graphs, theta_star) -> TrueEffects: ...           # do() in the simulator
def simulator_gate(cfg) -> GateResult: ...                              # the four-number table (Stage 5f)

# models/world_model.py
class GraphRSSM(nn.Module):
    def observe(self, obs_graph, action, state) -> tuple[State, Posterior]: ...
    def imagine(self, state, action) -> tuple[State, Prior]: ...
    def decode(self, state) -> Outcomes: ...                            # node head + global head + reward head

# agents/
def train_sb3(env_fn, cfg) -> Path: ...                                 # PPO / SAC control group
def train_dreamer(world_model, cfg) -> Path: ...                        # actor-critic on imagined rollouts
def evaluate_in_abm(policy, cfg, n_seeds, n_draws) -> PolicyScore: ...   # J_ABM — the number that matters

# ensemble/ray_ensemble.py
def run_scenarios(cfg, posterior: az.InferenceData, policies: list[PolicyLevers]) -> xr.Dataset: ...

# sensitivity/
def sobol_analysis(cfg, evaluate: Callable[[np.ndarray], float]) -> SobolResult: ...
def optimize_policy_bo(cfg) -> dict: ...                                # BoTorch, evaluated in the ABM

# tracking.py
class Tracker(Protocol):
    def start(self, run_name: str, config: dict) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...
    def log_figure(self, fig, name: str) -> None: ...
    def log_artifact(self, path: Path, name: str | None = None) -> None: ...
    def finish(self) -> None: ...
def make_tracker(cfg) -> Tracker: ...     # mlflow | wandb | none
```

---

## §10 Execution: phases and stages

The build proceeds in **seven phases**. Each phase groups several stages and ends in a **gate** — a shell command that must exit 0 before the next phase begins. The stages inside each phase are specified in full below, in build order. Every stage has the shape *Purpose → Files → Key decisions → Acceptance tests → Gate → Commit*, and every gate command is literal.

### The seven-phase arc

| Phase | Stages | Gate (must exit 0) |
|---|---|---|
| **1 · Foundation** | 0 (recon), config, seeding, types, tracking, CI skeleton | `make lint && make typecheck && uv run pytest -q` |
| **2 · World & data** | 1 (data), 2 (graphs) | `uv run pytest tests/test_data_schema.py tests/test_graph_construction.py tests/test_no_dgp_leakage.py -q` |
| **3 · Simulation** | 3 (Mesa ABM), 3b (tensorized), 8 (Gym envs), 9 (PettingZoo) | `uv run pytest tests/test_abm_contract.py tests/test_env_contract.py tests/test_marl_env.py -q` |
| **4 · Inference** | 4 (calibration), 5 (causal + the gate) | `uv run pytest -m slow tests/test_parameter_recovery.py tests/test_causal_recovers_known_effect.py -q` |
| **5 · Emulator** | 6+7 (GraphRSSM), 8 (emulator env), §11 evaluation suite | `uv run pytest tests/test_dynamics_shapes.py tests/test_smoke_train.py -q && make emulator && uv run python scripts/eval_emulator.py` |
| **6 · Control & ensemble** | 10 (RL), 11 (Ray ensemble), 14 (sensitivity) | `make rl ensemble sensitivity` and the coverage check ≥ 0.85 |
| **7 · Delivery** | 15 (figures, dashboard), 16 (Docker, CI, Slurm), 17 (report), `docs/MINIMAL_PATH.md` | `make smoke && make figures report && docker build -f docker/Dockerfile .` |

The stage numbering follows Part XIX of the guide (Stages 1–16); Stage 0 (reconnaissance) and Stage 17 (the report) sit outside the sixteen because reality demands them. Stages 6 and 7 are one artifact and are built together.

---

### Stage 0 — Reconnaissance and setup *(not in the guide; required by reality)* · Phase 1

**Purpose.** Verify the world before writing code against it, and stand up the reproducibility spine.

**Build**
- `pyproject.toml` (Appendix A) → `uv venv` → install core+dev, then each extra one at a time (§5), recording failures to `.stage_skips` → `uv lock`, commit `uv.lock`.
- `Makefile` (B), `docker/` (C), `.github/workflows/{ci,docker,nightly}.yml` (D), `.pre-commit-config.yaml`, `.gitignore` (ignore `experiments/`, `artifacts/`, `*.zarr`, `mlruns/`).
- `CLAUDE.md` (Appendix E) — write it now; it is what keeps you on-rails across context compaction.
- `regworld/{seeding,logging_conf,types,tracking}.py`. `types.py` holds Pydantic models mirroring every config group and `validate_config`.
- Full `configs/` tree (§6), all groups present even if some are stubs with the right keys.
- `tracking.py`: MLflow (file backend at `${paths.reports}/mlruns`, sqlite when several Ray workers log concurrently), W&B (offline unless `WANDB_API_KEY`), and a null backend. No stage imports `mlflow`/`wandb` directly. Captures `git rev-parse HEAD` on every run.
- `scripts/run_pipeline.py`: the driver (stub the stages; fill them in as phases land). Runs stages in order per the `stages:` map, logs each to the tracker, checkpoints outputs to disk, writes `reports/run_manifest.json` and `reports/FINDINGS.md`, and honours `--force-stage <name>` and `--isolated-envs`.
- **API probe** → `docs/DEVIATIONS.md`: import and print the version of every tool; assert Mesa exposes `model.agents`/`AgentSet` and *not* `RandomActivation`, Gymnasium `step` returns five values, PyG `HeteroConv` imports without `torch_scatter`, and record which RLlib API stack is installed (Stage 10 depends on it).

**Acceptance tests** — `test_config.py` (every profile and group combination validates; a bogus key raises), `test_seeds.py` (same seed → identical `rng.random(10)`; different seeds differ), `test_tracker.py` (null writes nothing; mlflow writes a run dir), `test_layering.py` (`src/` imports nothing from `notebooks/`), `test_no_dgp_leakage.py` (no `oracle`/`dgp` path outside `regworld.evaluation`).

**Gate**
```bash
make lint && make typecheck && uv run pytest -q && \
  uv run python scripts/run_pipeline.py profile=smoke 'stages={}' && echo GATE-0-OK
```
**Commit:** `chore: scaffold repo, hydra+pydantic config, seeding, tracking, CI, recon`

---

### Stage 1 — pandas / Polars: the data layer — *guide step 1* · Phase 2

*"pandas or Polars ingests the raw material… firm-level panel data lands in Parquet; cleaning and joining happen here, and this stage takes longer than anyone budgets."*

**Purpose.** Generate the ground-truth world, then turn its degraded observations into a clean, analysis-ready panel.

**Build**
- `dgp/world.py`, `dgp/dynamics.py`, `dgp/observation.py`, `dgp/history.py` and `rules.py` (§7). `dgp/dynamics.py` binds θ\* to `rules.py`; `history.py` produces Regime P with the staggered regional rollout. `scripts/generate_world.py` runs the DGP under Regime P's historical schedule for 24 quarters with θ\*, applies the observation model → `observed/`, and writes everything else → `oracle/`.
- `data/ingest.py`: Polars lazy pipeline — read → validate → join → write Parquet. This is the seam a real registry drops into; keep it clean. Convert to pandas only at boundaries downstream libraries demand (Mesa `DataCollector`, statsmodels, ArviZ) with `.to_pandas()` and move on.
- `data/schema.py`: column/dtype/range contracts per table (§8), `validate_table`.
- `data/store.py`: Parquet + xarray/Zarr helpers; `read_observed()`, `read_oracle()` (the latter raises unless the caller is `evaluation/` or a test — stack-frame check *and* the grep test).
- `data/duck.py`: DuckDB views over the Parquet directory.

**Acceptance tests** — `test_data_schema.py` (row counts; no nulls where forbidden; panel balanced modulo exits; join keys unique); determinism (same seed → byte-identical Parquet checksums); observation-model sanity (`observed` compliance rate within 3σ of `oracle`); `test_no_dgp_leakage.py` green.

**Gate**
```bash
make data profile=smoke && uv run pytest tests/test_data_schema.py tests/test_no_dgp_leakage.py -q && echo GATE-1-OK
```
**Commit:** `feat(data): ground-truth DGP, staggered-rollout history, observation model, Parquet contracts`
**Skip policy:** none. If this fails, everything fails.

---

### Stage 2 — NetworkX: the interaction structure — *guide step 2* · Phase 2

**Purpose.** Build the graphs the agents live on and the graphs the GNN will learn over — the true pair and the observed pair.

**Build**
- `graphs/build.py`: the five generators of §7.2, seeded, returning a `RegGraphs` dataclass holding the NetworkX graphs (true and observed) + node index maps.
- `graphs/analyze.py`: degree distributions, clustering, **assortativity-by-`z`** (≈0 under `wellspecified`, >0.2 under `confounded`), community detection (Louvain), eigenvector/betweenness centrality (top-k firms), cascade reachability. Centrality later colours the non-compliance map.
- `graphs/to_pyg.py`: `RegGraphs` + observed features → PyG `HeteroData`. Firm block `[size, data_intensity, sector 1-hot, compliant, alive, margin, cost_share]`; segment `[weight, privacy_sensitivity, trust]`; regulator `[budget_used, targeting, phase_progress]`; association `[publicity]`.

**Acceptance tests** — `test_graph_construction.py`: no self-loops; supply graph weakly connected and scale-free-ish (fit exponent in a sane band); WS graph high clustering + low path length; NetworkX↔PyG round-trip preserves node/edge counts; `HeteroConv` forward pass yields the expected shapes; assortativity-by-`z` ≈0 under `wellspecified` and >0.2 under `confounded`.

**Gate**
```bash
make graphs profile=smoke && uv run pytest tests/test_graph_construction.py -q && echo GATE-2-OK
```
**Commit:** `feat(graphs): NetworkX construction (true + observed), metrics, PyG HeteroData conversion`

---

### Stage 3 — Mesa: the agent-based model — *guide step 3* · Phase 3

*"This prototype is deliberately small, because its job is to find the model's conceptual bugs cheaply."*

**Purpose.** An interpretable, inspectable ABM implementing the same behavioral rules the DGP uses, but parameterized by *estimated* values and running on the *observed* graph. This is the "model," as distinct from the "world," and it is the ground truth for the emulator and the expensive oracle for BoTorch.

**Build**
- `abm/agents.py`: `FirmAgent`, `SegmentAgent`, `RegulatorAgent`, `AssociationAgent`.
- `abm/model.py`: `RegulationModel(mesa.Model)`. **Mesa ≥3.0 API** — `self.agents`, `AgentSet.shuffle_do("step")`, `agents_by_type`, `.select(...)`. **Do not use `RandomActivation` or `self.schedule`** — they are Mesa 2.x and gone. The firm decision calls `rules.firm_utility` — the same pure function the DGP uses.
- `abm/collect.py`: Mesa `DataCollector` for the model-level outcome vector (§7.6) and the agent-level firm panel → pandas → Parquet.
- `abm/policies.py`: the scripted policy schedules (also used as RL baselines in Stage 10).

**Key decisions.** Speed matters: budget ≈ **6 s per 24-quarter run at 2,000 firms** on one core. Profile it early; if it is >10× that, vectorize the firm decision step with NumPy across the `AgentSet` rather than looping in Python (Mesa allows this, and the guide's warning about Python-per-agent slowness is the reason).

**Acceptance tests** — `test_abm_contract.py`:
- Determinism under seed → identical trajectory.
- Invariants: spend shares sum to 1; revenue ≥ 0; exited firms never revive; audits ≤ budget; no negative margins carried forward.
- **Metamorphic / monotonicity tests** (the highest-value ABM tests, averaged over 5 seeds): enforcement ↑ ⇒ compliance at q24 non-decreasing; cost → 0 ⇒ compliance → ~1; `β_peer` ↑ ⇒ cross-firm compliance correlation on the supply graph ↑; subsidy ↑ ⇒ small-firm exit ↓.
- Performance: 200 firms × 24 quarters in **< 15 s** on one core.

**Gate**
```bash
make abm profile=smoke && uv run pytest tests/test_abm_contract.py -q && echo GATE-3-OK
```
**Commit:** `feat(abm): Mesa 3 regulation model, shared decision rules, DataCollector`

---

### Stage 3b — Tensorized / differentiable ABM — *guide: the AgentTorch destination* · Phase 3

**Purpose.** A GPU-capable, **differentiable** reimplementation of the same firm-compliance dynamics, so calibration can become gradient descent on simulation parameters and the macro calibration (Stage 4b) has a fast simulator. Optional but high-value; behind `stages.tensorized_abm`.

**Build**
- `abm/tensorized.py`: try `agent-torch`; **if it does not install cleanly, do not block** — write a pure-PyTorch version of the same `rules.py` equations, the whole firm population as tensors, the supply graph as a sparse adjacency, one quarter as a handful of sparse matmuls and a Bernoulli sample. It runs 10⁶ firms on a GPU and is differentiable end to end. Log the choice in `DEVIATIONS.md`.

**Acceptance tests** — `test_abm_agreement.py` (marked `slow`): at F = 2,000 with matched seeds, the tensorized and Mesa ABMs agree on aggregate trajectories within Monte-Carlo error (KS test on terminal compliance rate, p > 0.05 across 32 seeds). Disagreement means one of them has a bug, and finding it is cheaper now than at Stage 11.

**Gate** — folded into Phase 3's gate; runs `make abm profile=smoke` with `stages.tensorized_abm=true`.
**Commit:** `feat(abm): differentiable tensorized ABM with Mesa-agreement gate`
**Skip policy:** if neither AgentTorch nor the torch reimplementation is viable, mark `DEGRADED`; Stage 4b falls back to a GP surrogate over Mesa runs.

---

### Stage 4 — NumPyro / PyMC: Bayesian calibration — *guide steps 4, 14a* · Phase 4

*"Compliance-cost sensitivities and imitation strengths become random variables… the output is a posterior over behavioral parameters, honest uncertainty instead of point guesses."*

**Purpose.** Get a posterior over the behavioral parameters, then be honest about what that posterior is worth. **You cannot put an ABM inside NUTS** — its likelihood is intractable and non-differentiable — so calibration is done in two real pieces, matched to the parameter split in §7.3.

**Build**
- **4a · Micro-model (exact likelihood, NUTS, NumPyro).** The firm decision rule *is* a discrete-choice model and the panel *contains* the decisions, so the micro-likelihood is the model's own §7.4 equation, evaluated with `hat` quantities computed from *observed* data, plus a misclassification layer `ỹ_it ~ Bernoulli(y_it(1−q₁) + (1−y_it)q₀)` and hierarchical sector-level partial pooling on β₀ and β_cost. Priors from §7.3. **NUTS, 4 chains × 1000 draws, in a subprocess** (§5). Note what is absent from the fitted equation: `β_cap·z_i`. Under `dgp=confounded` with capacity homophily on, `β_peer` comes out biased upward — the DGP lets us measure exactly how much. `calibration/micro_numpyro.py`.
- **4b · Macro-model (approximate, simulation-based).** Parameters absent from the micro-likelihood (γ_scale, ℓ_learn, α_trust, ρ_infl, μ_priv, δ_exit, and enforcement effectiveness) are calibrated against aggregate adoption curves by **SMC-ABC** with summary statistics — terminal compliance rate, time-to-50%-compliance, terminal HHI, mean trust, exit rate, adoption-curve inflection quarter. Use the **tensorized ABM (3b)** as the simulator so thousands of runs are cheap; fall back to a GP surrogate over ~500 Mesa runs if 3b is unavailable. `calibration/macro_smc.py`, `calibration/summaries.py`. (`calibration=numpyro_bsl` offers a single Bayesian-synthetic-likelihood pass as an alternative to the micro/macro split.)
- **4c · Cross-check (PyMC).** Re-implement 4a in PyMC and confirm the posteriors agree (all marginals overlapping, |Δ posterior mean| < 0.1 SD). Two independent implementations agreeing is a real reproducibility practice; in a real project you would pick one, and the report says so. `calibration/micro_pymc.py`. Skippable via `calibration.crosscheck=false`.
- **4d · Diagnostics (ArviZ, mandatory).** R-hat, ESS bulk/tail, divergence count, energy plot, prior-predictive check (if the prior cannot produce the observed curve, stop and fix the model before sampling), posterior-predictive check against held-out quarters, and a `plot_pair` on `(β_peer, β_assoc)` to expose the identifiability trade-off. `calibration/diagnostics.py`.

**Acceptance tests / gate — the parameter-recovery gate (C1).** This is the ABM's unit test: a pipeline that fails on data it generated itself has nothing to say about reality.
- **Tiny recovery test** (fast, in `tests/`): 3 free parameters, 60 design points, 200 draws → θ\* inside the 90% CI for ≥ 2/3.
- **Full recovery gate** (`slow`, in `scripts/calibrate.py`): under `dgp=wellspecified`, the 90% credible interval covers θ\* for **≥ 12 of the 16** parameters, **all** R-hat < 1.01, ESS_bulk > 400, divergences = 0. Log posterior z-scores `(θ̂ − θ*)/sd`; |z| > 3 on any parameter is a red flag printed loudly.
- **The failure half:** under `dgp=confounded`, the same test asserts that `β_peer`'s 90% interval **does not** cover truth and records the bias. A pipeline that cannot detect its own bias is not a pipeline.

**Gate**
```bash
make calibrate profile=smoke && uv run pytest tests/test_parameter_recovery.py -q && echo GATE-4-OK
```
**Commit:** `feat(calibration): NumPyro micro-likelihood + SMC-ABC macro + PyMC cross-check + ArviZ + recovery gate`
**Skip policy:** if the `bayes` extra failed, fall back to maximum-likelihood point estimates via `statsmodels` logit with clustered SEs, mark the run `DEGRADED`, and disable every downstream claim that requires a posterior — the ensemble collapses to a point estimate and the report says so in bold.

---

### Stage 5 — DoWhy / EconML: interrogating the causal assumptions — *guide step 5* · Phase 4

*"Where the simulator disagrees with those estimates, the simulator is wrong first."* This is the intellectual core of the project. Build it carefully. `causal/__init__.py` carries the module docstring: *running a causal-inference library does not make your conclusions causal.*

**Build**
- **5a · Model.** `causal/graph.py` encodes the assumed DAG as a GML string, in **two variants**: the *analyst's* DAG (omits `capacity`, because the analyst does not know about it) and the *true* DAG (§7.7). Run the whole pipeline on the analyst's DAG; grade with the true one.
- **5b · Identify.** `identify_effect()` on the analyst's DAG returns a backdoor estimand and looks fine — that is the trap, and a code comment says so. On the true DAG the backdoor criterion fails without `capacity`. **Report both.**
- **5c · Estimate**, four ways, on Regime P's *observational* panel (`causal/estimate.py`, `causal/did.py`):
  1. Naive logit / OLS with observed controls (statsmodels). Biased.
  2. Double ML: EconML `LinearDML` and `CausalForestDML` with observed controls. Still biased, but now with a beautifully tight confidence interval — the guide's warning made flesh: *"the software will still return a number, formatted to six decimal places."*
  3. **Staggered-rollout DiD** on the historical episode: two-way fixed effects with an event-study specification, plus a staggered-adoption-robust estimator (Callaway–Sant'Anna if a package is available; otherwise a not-yet-treated-comparison event study in `linearmodels`). Rollout timing is exogenous by construction, so this one identifies. Plot the event study; check the pre-trends are flat.
  4. **Heterogeneity**: EconML CATE by firm size decile and sector — the client's real question is "for whom does this work," and the answer will be that enforcement moves mid-size firms and does almost nothing to the largest (who comply anyway) or the smallest (who exit instead).
- **5d · Refute** (`causal/refute.py`). DoWhy refuters on estimate (2): **placebo treatment** (effect → 0), **random common cause** (stable), **data subset** (stable), and **add-unobserved-common-cause** with increasing strength (the DML estimate should cross zero at a plausible confounder strength — the correct diagnosis). Report an **E-value**.
- **5e · Discovery** (`causal/discovery.py`, brief and honest). Run `causal-learn`'s PC and GES on the observables; compare the recovered graph to the true DAG by structural Hamming distance. It will be wrong — faithfulness and no-latent-confounder are violated *by construction* — so report the SHD and the reason. A two-hour experiment that inoculates the team against a very common mistake.
- **5f · THE GATE** (`causal/gate.py`, `scripts/validate_simulator.py`). Four numbers for the effect of raising enforcement from `e_low` to `e_high` on 24-quarter compliance:

  | Quantity | How it is obtained | Meaning |
  |---|---|---|
  | `τ_true` | `do(e=e_low)` vs `do(e=e_high)` in the **DGP**, 64 seeds each | the answer key |
  | `τ_abm` | the same intervention on the **calibrated Mesa ABM** | what our model believes |
  | `τ_qe` | the **staggered DiD** estimate from the historical panel | what the data says under a credible identification argument |
  | `τ_obs` | the **DML** estimate | what a careless analyst would report |

  **The gate:** `|τ_abm − τ_qe|` must lie inside the DiD's 95% CI (widened by the ABM's Monte-Carlo SE); additionally `τ_abm` and `τ_true` must agree in sign and be within 3× in magnitude. If the gate fails, the pipeline writes `reports/simulator_discrepancy.md`, marks the run **FLAGGED**, and acts on `causal.on_disagreement`:
  - `recalibrate` (default): add the DiD estimate as a moment-matching penalty in Stage 4b and re-run stages 4→5 **once**. If it still fails, stop and escalate — do not loop forever.
  - `report`: continue, but every downstream figure carries a warning banner.

  The table of all four numbers is **Figure 1 of the report**: it shows, in one frame, that the DiD recovers the truth, the DML does not, and whether our simulator is telling the truth about interventions. Most causal demos show one of those; this shows all four, because the world is synthetic and we can.

**Acceptance tests** — `test_causal_recovers_known_effect.py` (C2): DiD 95% CI covers `τ_true`; the DML estimate is biased by more than 2 of its own standard errors (assert the bias is real, so the demonstration cannot silently stop demonstrating); placebo refuter returns ≈0; `|bias(hidden)| > |bias(full)|` (if the confounder isn't confounding, the demo is dishonest).

**Gate**
```bash
make causal profile=smoke && uv run pytest tests/test_causal_recovers_known_effect.py -q && echo GATE-5-OK
```
**Commit:** `feat(causal): DoWhy identify→estimate→refute, EconML CATE, staggered DiD, do() ground truth, four-number gate`
**Skip policy:** if the `causal` extra fails, implement DiD and OLS in statsmodels alone (usually installs), skip DoWhy/EconML/causal-learn, mark `DEGRADED`. The 5f gate still runs, on statsmodels DiD.

---

### Stages 6 + 7 — PyTorch + PyTorch Geometric: the emulator — *guide steps 6, 7* · Phase 5

*"The calibrated ABM is slow, so a latent transition model learns to emulate it… each firm's next state depends on its network neighbors, so the dynamics model is a GNN over the supply and influence graphs."* These two steps are one artifact, built together: **GraphRSSM**, which predicts next-quarter system state from current state and policy settings in microseconds instead of minutes, with calibrated uncertainty.

**The factorization** (the main design decision in the whole project). Macro dynamics are genuinely uncertain and multimodal — does the compliance cascade take off or stall? That depends on early noise. Micro dynamics, *given* the macro regime and a firm's neighbourhood, are close to conditionally deterministic. So:

- **Macro path — an RSSM.** Deterministic `d_t = GRU(d_{t−1}, [z_{t−1}, a_{t−1}])`; stochastic `z_t` as **32 categorical variables × 32 classes** (DreamerV2 discrete latents, which the guide notes "turned out to fit game dynamics far better than Gaussians"), with straight-through gradients; prior `p(z_t|d_t)`, posterior `q(z_t|d_t,g_t)` where `g_t` is the encoded observation. `models/rssm.py`.
- **Micro path — a heterogeneous GNN.** `HeteroConv` over `(firm,supplies,firm)`, `(segment,buys_from,firm)`, `(segment,influences,segment)`, `(firm,member_of,association)` with `SAGEConv`/`GATConv`, 3 message-passing layers; per-node state `h_it` updated by a `GRUCell` conditioned on the broadcast macro latent `[d_t,z_t]` and the action `a_t`. Node head predicts `logit(y_{i,t+1})`. `models/gnn.py`. Config switch `emulator.stochastic_level=macro|node` (node-level latents are an ablation).
- **Encoder** `g_t` (`models/encoder.py`): the same HeteroGNN, pooled (mean + max + a learned attention pool), concatenated with hand-built aggregates (compliance rate, HHI, trust, exit rate, budget). Feeding the aggregates directly is not cheating — it is telling the model what you already know.
- **Heads** (`models/heads.py`): node compliance (BCE), aggregate observations (**symlog** MSE, per DreamerV3), reward (**two-hot** encoded), continuation flag (BCE).
- **Losses** (`training/losses.py`): reconstruction + KL with **KL balancing** (0.8 toward the prior, 0.2 toward the posterior) and **free bits** (`kl_free = 1.0` nat) — the exact DreamerV3 recipe the guide names; symlog on continuous targets; gradient clipping at 100. Without these you will tune per-config and hate your life.
- **Training** (`training/train_emulator.py`): teacher-forced posterior training + open-loop imagination loss (roll the prior forward k=8 steps and penalize drift — this is the number that actually matters). AdamW, lr 3e-4, cosine schedule. `torch.amp` on GPU; `torch.compile` behind a flag, off by default (recurrent world-model code sometimes needs coaxing). `einops.rearrange` for every `(batch, time, node, feature)` shuffle — no bare `.permute()` chains.

**Training data** (`scripts/make_emulator_dataset.py`): `train_episodes` ABM rollouts, collected in parallel with Ray, under **domain randomization** — θ sampled from the Stage-4 posterior *and* a diverse policy distribution (a mixture of random-uniform, scripted, and sinusoidal-sweep schedules, occasionally piecewise-constant within an episode, with a small amount of RL-policy data added in a second round). **This randomization is not optional:** without it the emulator memorizes one policy and every scenario-grid number is fiction. Store as Parquet + Zarr `(episode, quarter, node, feature)`; train on 24-step sequences with 8-step burn-in.

**Ablation configs to create now** (used in §11 and Stage 14): `emulator=rssm_flat` (RSSM, no GNN — features mean-pooled) and `emulator=gru_baseline` (plain GRU on aggregates, no latents, no graph). If the GNN does not beat the flat model, the graph structure was decoration and the report says so.

**Acceptance tests** — `test_dynamics_shapes.py` (forward + imagination shapes; gradients reach every parameter; no NaN); `test_smoke_train.py` (**overfit one batch: loss must fall below 0.05 of its initial value in 200 steps** — the single best deep-learning unit test there is); k-step rollout **beats a persistence baseline** ("no change") on held-out episodes; §11's emulator metrics clear their thresholds.

**Gate**
```bash
make emulator profile=smoke && uv run pytest tests/test_dynamics_shapes.py tests/test_smoke_train.py -q && echo GATE-6-OK
```
**Commit:** `feat(emulator): GraphRSSM (macro RSSM + micro GNN), DreamerV3 losses, domain-randomized dataset`

---

### Stage 8 — Gymnasium: the interface — *guide step 8* · Phases 3 & 5

**Purpose.** One contract, two worlds — everything downstream trains against the interface and does not care which world is behind it.

**Build** (`environments/{abm_env,emulator_env,wrappers}.py`)
- `AbmEnv` — steps the Mesa ABM. Slow (≈0.25 s/step). The oracle. (Built in Phase 3.)
- `EmulatorEnv` — steps the GraphRSSM (`imagine`). Fast (≈0.4 ms/step), vectorized over scenarios. Reward comes from the reward head, with a config flag to recompute it exactly from decoded outcomes instead — safer, and it lets you attribute error to dynamics vs reward modelling. (Built in Phase 5.)
- **Observation** (`Box`, ~34 dims): compliance rate (raw + revenue-weighted), HHI, ΔHHI from baseline, mean trust, consumer-surplus index, cumulative exit rate, budget remaining, `t/T`, per-sector compliance (6), compliance by size decile (10), audit rate, penalties, and a 4-dim copy of the last action. A `graph_obs=True` variant returns a `Dict` space carrying the `HeteroData` for the Dreamer agent; SB3 uses the flat `Box`.
- **Action**: the four levers (§7.5).
- **`terminated` vs `truncated`** (the non-negotiable, guide's classic silent RL bug):
  - `truncated=True` at `t == horizon_quarters` — the world continues, so bootstrap the value.
  - `terminated=True` only on systemic collapse (>40% of firms exited, or compliance <5% after q12 with budget exhausted) — an absorbing MDP end with no future value.

  Getting this backwards corrupts value functions silently. There is a test for it, and it is four lines of code that will save someone a month.

**Acceptance tests** — `test_env_contract.py`: `gymnasium.utils.env_checker.check_env` passes on both; `reset(seed=k)` twice gives identical observations; **space-identity test** asserting `AbmEnv` and `EmulatorEnv` have identical observation and action spaces (this identity is what makes the planning-utility test possible); a synthetic collapse scenario sets `terminated` and *not* `truncated`, and the time limit sets `truncated` and *not* `terminated`.

**Gate** — folded into the Phase 3 gate (`AbmEnv`) and Phase 5 gate (`EmulatorEnv`).
```bash
uv run pytest tests/test_env_contract.py -q && echo GATE-8-OK
```
**Commit:** `feat(envs): AbmEnv + EmulatorEnv with identical spaces, terminated/truncated semantics`

---

### Stage 9 — PettingZoo: when firms are strategic — *guide step 9* · Phase 3

*"PettingZoo enters if the problem is genuinely strategic."*

**Purpose.** Test whether the headline finding survives large firms that game the rule.

**Build** (`environments/marl_env.py`, PettingZoo **Parallel API**). A hybrid, per the guide's advice to use *"cheap rules for the crowd and expensive cognition for the few agents that matter"*: the **ten largest firms become learning agents** (`firm_0 … firm_9`); the other ~1,990 keep the calibrated rule-based behavior; the **regulator** is the eleventh agent (`regulator_0`).
- **Firm action** `Box(3,)`: `[comply_invest, lobby, evade]`. Lobbying reduces the effective enforcement intensity applied to that firm's association next quarter; evasion lowers audit-detection probability at a cost; compliance investment raises the firm's own compliance probability.
- **Firm reward:** profit (revenue − compliance cost − expected penalty − lobbying spend).
- **Regulator reward:** as in Stage 8. Mixed-motive by construction — nobody is purely cooperative or adversarial, which is where the interesting behavior lives.

**The decision gate the guide demands.** *"The single most common design error in this space is reaching for MARL when the question never required learned behavior at all."* So run the ensemble with strategic firms and without: if the 95% credible intervals on C5's headline numbers overlap, **report that MARL did not change the conclusion and move on** (C6). A clean negative result here is a contribution, not a failure, and it is the honest outcome for most policy questions.

**Acceptance tests** — `test_marl_env.py`: `pettingzoo.test.parallel_api_test(env, num_cycles=100)` passes; agent-death and observation-space consistency checks pass.

**Gate**
```bash
uv run pytest tests/test_marl_env.py -q && echo GATE-9-OK
```
**Commit:** `feat(envs): strategic PettingZoo parallel env (regulator + top-K firms)`

---

### Stage 10 — RL: SB3, TorchRL, RLlib — *guide step 10* · Phase 6

*"Stable-Baselines3 trains a first regulator policy inside the emulator as a sanity check; TorchRL or RLlib takes over if the policy question becomes the research question."* Three libraries, three distinct jobs; if a library cannot justify its job, it is not installed.

**Build**
- **10a · Scripted baselines** (`agents/scripted.py`, no library): `none`, `uniform_low`, `uniform_high`, `targeted`, `phased_targeted` (the status-quo policy from Regime P). One of them will probably be competitive, which is a finding.
- **10b · SB3 — the control group** (`agents/sb3_agents.py`): PPO and SAC on `EmulatorEnv` with `make_vec_env` (n_envs = cpu_count), 300k steps, 5 seeds. The guide is precise: *"It is the control group, not the experiment."* Its number exists so the Dreamer agent has something to beat.
- **10c · TorchRL — the experiment** (`agents/dreamer.py`): a **Dreamer-style actor-critic trained entirely on imagined latent rollouts** of the GraphRSSM — no environment steps at all during policy learning. Horizon H = 15, λ-returns (λ = 0.95), critic on two-hot returns, actor by backpropagating through the model's differentiable dynamics with a small entropy bonus. Assemble from TorchRL's `TensorDict`, replay buffer, and loss modules. *"A Dreamer-style agent is precisely a nonstandard assembly of standard parts, and TorchRL ships those parts."* If TorchRL's API fights you, hand-roll the loop in plain PyTorch — the components are ~300 lines and this is not the hill to die on. Log the choice.
- **10d · RLlib — the multi-agent scale-out** (`agents/marl.py`, optional `--extra rl`): independent PPO over the PettingZoo env, one policy per strategic firm + the regulator, parameter sharing across firms optional, 200k steps. **RLlib fallback (guardrail):** if RLlib's API stack churn makes this painful (check Stage 0's recon), fall back to hand-rolled **independent PPO with iterated best response** — wrap the parallel env into N single-agent views with the other agents frozen, train each with SB3 PPO, iterate 3 rounds. ~150 lines, zero new dependencies, scientifically legitimate. Mark `DEGRADED` and note it. The four-number gate does not depend on which library produced the policies.

**The gate — planning utility** (the model-based RL acid test: *does the model make an agent better?*). Every policy, from every source, is evaluated **in the true ABM** (not the emulator that trained it), across 5 seeds × 64 posterior draws:
- Every learned policy must beat `random` and `fixed_enforcement` on the `balanced` objective with a **non-overlapping 95% CI**.
- The Dreamer agent's **exploitation gap** `J_emulator − J_ABM` must be **≤ 15%**. A policy that looks brilliant in the emulator and mediocre in the ABM has found the model's errors and steered into them — exactly what the guide warns a planner will do to an overconfident model. If the gap is larger, §11's OOD analysis must explain where. That gap is a reported metric, not a footnote.

**Acceptance tests** — `test_policy.py`: every policy beats `none` in the ABM on balanced reward; the exploitation-gap assertion for the Dreamer agent.

**Gate**
```bash
make rl profile=smoke && uv run pytest tests/test_policy.py -q && echo GATE-10-OK
```
**Commit:** `feat(rl): scripted baselines, SB3 control, TorchRL Dreamer experiment, optional RLlib MARL, planning-utility gate`

---

### Stage 11 — Ray: the scenario ensemble — *guide step 11* · Phase 6

*"Thousands of simulations across the parameter posterior and a grid of policy options."* This is the thing the client actually asked for.

**The grid** (`ensemble/scenarios.py`): `dev` = 1,000 posterior draws × 7 policies × 3 seeds = 21,000 rollouts. In the emulator that is ~6 minutes; in the ABM it would be ~11 days — and that sentence is the entire justification for Stages 6 and 7. `smoke` = a 6-policy corner set × 8 draws × 1 seed. Each policy gets a stable `policy_id` hash.

**Ray usage** (`ensemble/ray_ensemble.py`):
- `@ray.remote` **actors** hold a loaded emulator (or a loaded ABM) and process rollouts in sequence, so the model is deserialized once per worker rather than once per task. `@ray.remote` **tasks** for the ABM validation subsample.
- `compute=ray_local` → `ray.init()`; `compute=ray_cluster` → `ray.init(address=cfg.compute.address)`. The code is identical — that portability is why Ray is here and `multiprocessing` is not. Batch ~64 rollouts per task so scheduling overhead doesn't dominate.

**The validation subsample (do not skip this)** (`evaluation/ood.py`): re-run a stratified random **5%** of the `(scenario, policy)` cells in the **actual ABM** and compare against the emulator's predictions, *under policy shift* (this is the OOD test). If the emulator's 90% predictive interval covers the ABM outcome **less than ~85%** of the time, the entire ensemble is decoration and the report must say so. Costs 4 minutes; it is the difference between a result and a rendering.

**Storage** (`ensemble/cube.py`): an **xarray `Dataset` backed by Zarr**, dimensioned `(policy, draw, seed, quarter, variable)` (§8). Derived quantities written alongside: `P(backfire at q24 | policy)`, credible bands per outcome, and the compliance-vs-consumer-surplus Pareto frontier.

**Acceptance tests** — `test_ensemble_shapes.py`: Zarr store has the right dims/coords; determinism under fixed seeds; a Ray task failure is retried and the run still completes; ensemble means within tolerance of a serial reference on a 4-scenario case; coverage check ≥ 0.85.

**Gate**
```bash
make ensemble profile=smoke && uv run pytest tests/test_ensemble_shapes.py -q && echo GATE-11-OK
```
**Commit:** `feat(ensemble): Ray Core scenario ensemble → xarray/Zarr cube, ABM cross-validation, P(backfire)`

---

### Stage 12 — Hydra: keeping the sprawl coherent — *guide step 12* · Phase 1 (threaded throughout)

Already threaded through every stage. What remains to verify:
- Every script is `@hydra.main(config_path="../configs", config_name="config")`.
- Every run's resolved config is saved to the run directory and logged to MLflow, along with the **git commit hash** (`git rev-parse HEAD`, captured in `tracking.py`). Every result traces to a commit or it is not a result.
- `multirun` does the sweeps:
  ```bash
  uv run python scripts/run_pipeline.py -m seed=0,1,2,3,4 policy=uniform_high,targeted,phased_targeted
  uv run python scripts/train_emulator.py -m emulator=rssm_gnn,rssm_flat,gru_baseline seed=0,1,2
  ```
- Launchers switch by config override, never code change: `basic` locally, `joblib` for a fat node, `submitit_slurm` for the cluster.

**Acceptance test** — `test_configs.py` composes every group value against the defaults and validates; the first multirun above produces 15 distinct run directories with distinct configs.

---

### Stage 13 — Experiment tracking: MLflow (default), W&B (optional) — *guide step 13* · Phase 1 (threaded throughout)

**MLflow is the default**, for a specific reason: it needs no account, no API key, and no network, so the one-command run never blocks on auth. `MLFLOW_TRACKING_URI=file:./experiments/mlruns` locally; sqlite backend when several Ray workers log concurrently; a tracking server on a cluster.

**W&B behind `tracking=wandb`**, with `WANDB_MODE=offline` as the automatic fallback when `WANDB_API_KEY` is absent. Its run comparison and media tables are better, and the imagined-vs-real rollout comparison is exactly the kind of artifact its tables were built for.

**One interface** (`tracking.py`): `log_params / log_metrics / log_figure / log_artifact / log_table`, with a null backend for tests. No stage imports `mlflow`/`wandb` directly.

**What gets logged.** All losses (reconstruction, KL, node BCE, reward, continue, actor, critic), gradient norms, learning rates, ArviZ diagnostics, the four-number causal table, the policy-comparison table, every figure, the emulator checkpoint as an artifact, the resolved config — and, per the guide's emphasis, **imagined-versus-real rollout comparisons as a media table at every checkpoint**, because that is the single most diagnostic artifact in the Dreamer tradition and loss curves lie.

---

### Stage 14 — SALib, Optuna, BoTorch: closing the loop on rigor — *guide step 14* · Phase 6

*"Global sensitivity analysis reveals that, say, four parameters drive most outcome variance, which tells the client what to measure next."*

**Build**
- **14a · SALib** (`sensitivity/salib_gsa.py`). **Morris screening first on the ABM** (cheap: `8 × (D+1)` runs over D = 16 parameters) to prune to the ~8 that move anything. Then **Sobol** on the **emulator** with Saltelli sampling at N = `sensitivity.sobol_n` (~30k evaluations, minutes on the emulator, impossible on the ABM). Outputs: terminal compliance rate, ΔHHI, cumulative exit rate, terminal trust. Report S1 and ST; a large gap means interactions matter, itself a finding. **Guard against emulator artifacts:** re-run 64 randomly chosen Sobol design points on the real ABM and confirm agreement within tolerance. Sensitivity conclusions inherited from a wrong emulator are worse than none, because they look rigorous. Feed back into calibration: ST ≈ 0 flags a parameter as unidentifiable-from-these-summaries; high ST earns a refinement pass.
- **14b · Optuna** (`sensitivity/optuna_search.py`). Emulator hyperparameters (latent dim, categorical classes, GNN layers, KL free-bits, lr, imagination horizon), TPE with `MedianPruner`, `sensitivity.optuna_trials` at a reduced budget. Also serves as the fast MAP baseline and NUTS initializer for calibration reruns (`calibration=optuna_map`). Every trial logged to MLflow.
- **14c · BoTorch / Ax** (`sensitivity/bo_policy.py`, optional `--extra opt`). Bayesian optimization of a **5-parameter scripted policy schedule** (initial enforcement, ramp rate, targeting level, phase-in speed, audit-budget split) evaluated **against the actual ABM**, ~12 s per evaluation, `sensitivity.bo_evals` evaluations — BO's home ground: expensive, noisy, low-dimensional. It gives the RL policies a genuinely different opponent, one that **never touched the emulator** and therefore cannot have been fooled by it. If the BO-optimized scripted policy beats the Dreamer agent in the ABM, that is a real and slightly embarrassing result, and it goes in the report.

**Acceptance tests** — `test_sensitivity.py`: Sobol indices ∈ [0,1]; ST ≥ S1 within Monte-Carlo error; a synthetic function with known indices (**Ishigami**) recovers them within tolerance (do this — it catches sampler-wiring bugs instantly); the emulator-vs-ABM check at 64 design points passes; BO's best beats the mean scripted baseline.

**Gate**
```bash
make sensitivity profile=smoke && uv run pytest tests/test_sensitivity.py -q && echo GATE-14-OK
```
**Commit:** `feat(sensitivity): SALib Morris→Sobol over θ×policy, Ishigami check, Optuna tuning, BoTorch policy search`

---

### Stage 15 — Plotly and Streamlit: delivering the result — *guide step 15* · Phase 7

**Matplotlib for the paper** (`visualization/figures.py`, `scripts/make_figures.py`), publication defaults, no interactivity. Thirteen figures in `reports/figures/`:

| Fig | Content |
|---|---|
| 1 | **The four-number causal table** (τ_true, τ_abm, τ_qe, τ_obs) with CIs. The project's thesis in one frame. |
| 2 | Parameter recovery: posterior marginals with true values overlaid, well-specified vs confounded. |
| 3 | ArviZ diagnostics panel (trace, energy, pair plot on β_peer/β_assoc). |
| 4 | Event study from the staggered rollout, with flat pre-trends. |
| 5 | Emulator one-step and H-step error vs horizon: GNN vs flat vs GRU ablation. |
| 6 | **Imagined-vs-real rollouts, side by side, 6 sampled trajectories** — the single most diagnostic artifact in the Dreamer tradition; make it prominent. |
| 7 | Calibration curve and 90% interval coverage. |
| 8 | **Trajectory fans**: compliance and HHI over 24 quarters, per policy, with 50/80/95% credible bands across the parameter posterior. |
| 9 | **The Pareto frontier**: terminal compliance vs ΔHHI, one point per policy, uncertainty ellipses, coloured by backfire probability. The headline. |
| 10 | Sensitivity tornado (Sobol ST). |
| 11 | Non-compliance concentration map over the supply network (node colour = P(non-compliant), size = revenue). |
| 12 | Policy comparison: `J_emulator` vs `J_ABM` per policy, exposing the exploitation gap. |
| 13 | OOD degradation: emulator error vs distance from the training action distribution. |

**Plotly for exploration** (`visualization/interactive.py`): trajectory fans, latent-space projection (PCA/UMAP of firm-node latents coloured by compliance regime — does the representation organize the world sensibly?), the network diffusion map.

**Streamlit** (`visualization/dashboard.py`, `scripts/dashboard.py`) — the thing a policy team can actually operate:
- Sliders: enforcement, targeting, phase-in speed, subsidy, fine scale.
- Live prediction from the emulator (milliseconds), showing the **trajectory fan with credible bands, not a line**; on-grid slider positions do an instant lookup from the pre-computed Zarr, off-grid do live emulator inference.
- The Pareto frontier with the current slider position marked; HHI and consumer-surplus panels; a **red backfire indicator**; the network map coloured by predicted non-compliance risk; the sensitivity tornado (so the user sees which slider even matters).
- **An out-of-distribution warning banner** that fires when the slider combination sits outside the emulator's training distribution — four lines of Mahalanobis distance against the training action distribution. A stakeholder dashboard that silently extrapolates is a liability. Runs headless: `streamlit run scripts/dashboard.py --server.headless true --server.port 8501`.

**Acceptance test** — `make figures` produces all 13 files; `make dashboard` starts and renders with no exceptions against the committed artifacts.

**Gate** — folded into Phase 7's gate.
**Commit:** `feat(viz): 13 Matplotlib figures, Plotly exploration, Streamlit dashboard with OOD banner`

---

### Stage 16 — pytest, Docker, GitHub Actions: instrument, not demo — *guide step 16* · Phases 1 & 7

**pytest.** The full suite is §12. The highest-value tests are the unassuming ones, and every one the guide names is in the suite: environment contract checks, dynamics shape and gradient checks, and a smoke test that trains for fifty steps without NaNs. Markers: `slow` (parameter recovery, ABM↔tensorized agreement, full-pipeline smoke) excluded from the fast loop, run nightly.

**ruff** (lint + format, replacing black), **mypy** (strict on `models/` and `environments/`, where a silently wrong tensor shape can masquerade as a subtle result), **pre-commit** running all three.

**Docker.** `docker/Dockerfile` (CPU, `python:3.11-slim` + uv, non-root, ~1.2 GB) and `docker/Dockerfile.cuda`; `docker/compose.yaml` brings up the dashboard and an MLflow UI together. The same image runs on the Ray cluster, which is what makes "cluster-portable" true rather than aspirational.

**GitHub Actions.**
- `ci.yml` on every push: ruff, mypy, `pytest -m "not slow"`, then `make smoke` (the full 17-stage pipeline at `profile=smoke`, ~6 min), upload `FINDINGS.md` + figures as artifacts. A pipeline not exercised end to end in CI rots within a month.
- `docker.yml` on tags: build and push both images to GHCR.
- `nightly.yml`: `pytest -m slow` plus `profile=dev` on a larger runner, results posted as an artifact.

**Makefile** and **Slurm** as in Appendix B and `slurm/submit.sbatch`.

**Gate**
```bash
make lint && make typecheck && make test && make smoke && \
  docker build -f docker/Dockerfile -t regworld:ci . && \
  docker run --rm regworld:ci python scripts/run_pipeline.py profile=smoke && echo GATE-16-OK
```
**Commit:** `feat(tooling): pytest suite, ruff/mypy/pre-commit, Docker (CPU+CUDA), CI/nightly/docker workflows, Slurm`

---

### Stage 17 — The report *(not one of the sixteen; it is why the sixteen exist)* · Phase 7

`scripts/build_report.py` (`evaluation/report.py`) assembles `reports/FINDINGS.md`:
1. **A one-paragraph disclaimer, first, before any result:** the world is synthetic, so every finding is methodological. What is demonstrated is that this pipeline recovers the truth when the truth is recoverable and fails legibly when it is not.
2. **Figure 1** and the four-number causal table.
3. **C1 through C6**, each marked SUPPORTED / REFUTED / INCONCLUSIVE, with the figure and the numbers attached.
4. **"Where this model fails"** — a *required* heading `report.py` refuses to omit: the OOD degradation curve, the `β_peer` bias under homophily, the horizon at which multi-step error exceeds the useful threshold, the policies whose ABM performance falls short of their emulator performance, and any stage that ran `DEGRADED`. The guide predicts this will be the most cited section, and that prediction is usually right.
5. **The run manifest:** every stage's status, wall clock, git hash, and config.

**Gate** — Phase 7's gate:
```bash
make smoke && make figures report && test -f reports/FINDINGS.md && \
  docker build -f docker/Dockerfile -t regworld:ci . && echo GATE-17-OK
```
**Commit:** `feat(report): FINDINGS.md generator with claims ledger and required failure section` → then tag `v0.1.0`.

---

## §11 Evaluation suite

`scripts/eval_emulator.py` and the driver write `reports/eval/{report.md, metrics.json, figures/}`. **Twelve metric families. A credible project reports several; this one reports all of them.** Thresholds are the pass criteria at `profile=dev`; every module is under `regworld/evaluation/`.

| # | Layer | Module | What it computes | Threshold |
|---|---|---|---|---|
| 1 | Predictive accuracy | `predictive.py` | 1-step node compliance AUC; aggregate MAE; **k-step rollout drift** at k ∈ {1,3,6,12,18,24} on held-out ABM trajectories | AUC ≥ 0.85; 1-step MAE ≤ 0.02; **q24 compliance MAE ≤ 0.06**; report the horizon at which error exceeds 0.10 and call that the useful range, in plain words |
| 2 | Distributional fidelity | `distributional.py` | Wasserstein-1, MMD (RBF), energy distance on terminal outcome distributions (emulator vs ABM ensemble, 256 rollouts each); NLL of ABM outcomes under the emulator | W₁(compliance) ≤ 0.03; W₁(HHI) ≤ 0.01; permutation test **fails if it rejects at p < 0.01** (we want to fail to distinguish them) |
| 3 | Calibration | `calibration_curves.py` | ECE; coverage of 50/80/90/95% predictive intervals (do 90% intervals contain reality 90% of the time?); reliability diagram | 90% coverage ∈ [0.85, 0.95]; ECE ≤ 0.05 |
| 4 | Trajectory shape | `dtw.py` | Dynamic time warping between imagined and real compliance trajectories (~30 lines of NumPy, no new dep), against a random-policy baseline | reported, not gated |
| 5 | **Planning utility** | `planning_utility.py` | `J_ABM(π_emulator)` vs π_ABM-trained vs π_fixed vs π_random; exploitation gap `J_emulator − J_ABM`. *The model-based acid test.* | learned policies beat random & fixed with non-overlapping 95% CI; Dreamer gap ≤ 15% |
| 6 | Behavioral fidelity | `behavioral_fidelity.py` | Stylized facts **never fit during calibration**: S-shaped adoption (logistic R² ≥ 0.9), heavy-tailed firm-size distribution, compliance-by-size gradient, exit-rate-vs-enforcement relationship, heavy-tailed cascade sizes | S-curve R² ≥ 0.9; gradients correct sign |
| 7 | **Parameter recovery** | `parameter_recovery.py` | θ\* coverage, bias, z-scores — the ABM's unit test (Stage 4) | ≥ 12/16 cover θ\* at 90% under `wellspecified`; β_peer miss under `confounded` |
| 8 | Causal evaluation | `causal_eval.py` | The four-number table (Stage 5f): estimated vs true ATE/CATE | DiD CI covers τ_true; DML does not |
| 9 | Out-of-distribution | `ood.py` | Emulator error vs Mahalanobis distance from the training action distribution; enforcement pushed 1.5× outside the training range | reported without spin (Fig 13) |
| 10 | Historical backtest | `backtest.py` | Train the emulator on Regime P q1–12, predict q13–24, no peeking; same metrics on the held-out window | coverage on held-out window ∈ [0.85, 0.95] |
| 11 | Ablations | `ablations.py` | `rssm_gnn` vs `rssm_flat` vs `gru_baseline`; discrete vs Gaussian latents; with/without node head; with/without KL free bits; no-calibration (prior draws) — each × ≥3 seeds | if an ablation doesn't hurt, the component was decoration — the report says so |
| 12 | Sensitivity | (Stage 14) | Sobol S1/ST table + tornado | Ishigami recovered; ST ≥ S1 |

**Every headline claim runs across ≥ 5 seeds** and reports mean ± SE, or it is not a claim.

---

## §12 Testing strategy

The guide is right that the highest-value tests here are the unassuming ones. `pytest -q -m "not slow"` runs in CI on every push; `-m slow` (the full recovery, ABM↔tensorized agreement, and planning gates) runs inside `make smoke` and nightly.

| Class | Examples |
|---|---|
| **Contract** | `check_env`, `parallel_api_test`, `terminated` vs `truncated`, Parquet schema, `AbmEnv`/`EmulatorEnv` space identity |
| **Invariant** | Spend shares sum to 1; revenue ≥ 0; exited firms stay exited; audits ≤ budget; no self-loops |
| **Metamorphic** | enforcement ↑ ⇒ compliance ↑; cost → 0 ⇒ compliance → 1; subsidy ↑ ⇒ small-firm exit ↓; β_peer ↑ ⇒ supply-graph compliance correlation ↑ |
| **Numerical** | Gradient flow to every parameter; 50-step train without NaN; overfit-one-batch (loss < 0.05× initial in 200 steps); Ishigami recovers known Sobol indices |
| **Scientific** | Parameter recovery (success + legible failure); the four-number causal gate; planning utility beats baselines; Mesa↔tensorized KS agreement |
| **Reproducibility** | Same seed → identical trajectory hash; statistical stability across 5 seeds |
| **Structural** | No `oracle/`/`dgp` access outside `evaluation/`; `src/` never imports `notebooks/` |

---

## §13 Reproducibility and seeds

`regworld/seeding.py` exposes `seed_everything(k)`, which seeds Python's `random`, NumPy's default RNG, `torch` (CPU + CUDA), and JAX's PRNG key, and **returns a seeded `np.random.Generator` that every stochastic component takes as an argument** instead of reaching for global state.

Rules:
- Gymnasium seeding goes through `reset(seed=k)`. Always.
- Every seed is logged to MLflow. Every headline claim runs across `seeds: [0,1,2,3,4]` and reports mean ± SE.
- **No bare `np.random.*`** anywhere — explicit `Generator` objects, seeded and passed.
- **Bitwise GPU determinism is not pursued** (the guide says so): it costs speed and buys nothing scientific. Statistical reproducibility across seeds is the standard, and `test_determinism.py` enforces the CPU version: same seed, same trajectory hash.
- Checkpoint model *and* optimizer state every 2,000 steps. A long run that cannot resume is a long run you will do twice.
- `logging`, not `print`, everywhere, so a dead run can be autopsied.

---

## §14 Compute profiles and wall-clock budgets

`profile=dev`, 16 vCPU, no GPU:

| Stage | Minutes |
|---|---|
| 0 setup + recon | 5 |
| 1 data | 2 |
| 2 graphs | 1 |
| 3 ABM (512 trajectories, Ray) + 3b tensorized | 5 |
| 4 calibration (NUTS + SMC-ABC + PyMC cross-check) | 17 |
| 5 causal (DML + DiD + refuters + discovery + gate) | 6 |
| 6+7 emulator training | 25 |
| 8+9 envs (build + contract tests) | 2 |
| 10 RL (SB3 8, Dreamer 6, RLlib MARL 12) | 26 |
| 11 ensemble (emulator 6 + ABM validation 4) | 10 |
| 13 tracking | (in-line) |
| 14 sensitivity (Morris 3, Sobol 5, Optuna 8, BO 8) | 24 |
| 15 figures + dashboard build | 3 |
| 16 tests | 4 |
| 17 report | 1 |
| **Total** | **≈ 2 h 10 m** |

Other configurations:
- **`profile=smoke`** ≈ 6 minutes — what CI runs on every push.
- **One GPU** (emulator 25→4, Optuna 8→3) ≈ 1 h 20 m.
- **64-core Ray cluster** (Stages 3, 11, 14 parallelize) ≈ 45 minutes.
- **`profile=full`** on a cluster: several hours, launched via `hydra/launcher=submitit_slurm`.

Knobs to cut the `dev` run (§6): `stages.marl=false`, `calibration.crosscheck=false`, `emulator.train_steps=15000`, `sensitivity.optuna_trials=4` — all four land near 75 minutes with the headline claims intact.

---

## §15 Failure policy, skip policy, run manifest

The driver (`scripts/run_pipeline.py`) does not let one broken dependency kill a two-hour run.

- Each stage is a function returning a `StageResult(status, wall_clock, outputs, notes)` with status in `{DONE, SKIPPED, DEGRADED, FAILED, BLOCKED}`.
- Stage outputs are **checkpointed to disk**. A re-run skips any stage whose outputs exist and whose config hash is unchanged. `--force-stage emulator` re-runs one stage and everything downstream of it.
- A `FAILED` stage does not stop the pipeline unless a downstream stage declares it a **hard dependency** (declared explicitly in the driver's DAG). Emulator failure is hard: Stages 10, 11, 14 depend on it. Causal failure is soft: the pipeline continues and the report notes that C2 is INCONCLUSIVE.
- At the end, the driver prints a stage table and writes `reports/run_manifest.json`. **Every stage's status appears in `FINDINGS.md`.** A degraded run that reports itself as degraded is a scientific result; a degraded run that reports itself as clean is misconduct.

---

## §16 Known hazards and guardrails

The traps that will otherwise eat a week. Most are anticipated by a floor pin, a subprocess boundary, or a test.

1. **Mesa ≥ 3.0.** Use the `AgentSet` API (`model.agents.shuffle_do("step")`, `.select(...)`, `agents_by_type`). `RandomActivation` / `self.schedule` are Mesa 2.x and gone — any tutorial using them predates your install.
2. **Gymnasium ≥ 1.0.** Five-tuple `step`; `reset(seed=...)` returns `(obs, info)`. **`truncated` (time limit → bootstrap) vs `terminated` (true end → don't).** Confusing them corrupts value functions silently; there is a test.
3. **PyG: do NOT install `torch-scatter` / `torch-sparse`.** Modern `torch_geometric` (≥2.6) uses `torch.scatter_reduce` for the layers here; compiling those wheels on a cluster is a wrong turn.
4. **Do NOT use `hydra-ray-launcher`.** Ray Core directly in `ray_ensemble.py` for the ensemble *inside* a job; Hydra's `basic`/`submitit` launcher for sweeps *across* jobs. The launcher plugin's version pins rot.
5. **JAX and Torch must not fight over the GPU.** Every JAX stage runs as a subprocess with `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless calibration is explicitly on GPU.
6. **Ray with CUDA.** Do not initialize CUDA before `ray.init()`. Use spawn, not fork. Set `runtime_env` explicitly when the cluster image differs from the driver's.
7. **Tracking must work with no credentials.** MLflow file backend by default; W&B offline unless `WANDB_API_KEY`. A cluster job that blocks on interactive login is a dead job.
8. **MLflow concurrency.** File-store MLflow with many Ray workers writing at once will corrupt. Use the sqlite backend, or log only from the driver.
9. **`MPLBACKEND=Agg`. No `plt.show()`, no `print()` in `src/`** — use `logging`. **No bare `np.random.*`** — seeded Generators, passed explicitly.
10. **Domain-randomize the emulator's training data.** Without random policy schedules it memorizes one policy and every scenario-grid number is fiction.
11. **RLlib is non-gating.** SB3 is the required path; the IPPO/iterated-best-response fallback (Stage 10) exists precisely so RLlib's API churn cannot block the build.
12. **EconML / DoWhy resolver conflicts** (numpy / scikit-learn pins) are the most likely `uv lock` failure. Relax the offending pin → if still failing, move EconML to an optional extra and use a `sklearn`-based double-ML fallback in `estimate.py`. Log it.
13. **Zarr v2 vs v3.** Pin `zarr>=2.18,<3`. If forced to v3, use `xarray.to_netcdf` (h5netcdf) instead and log the deviation.
14. **Mesa speed.** Python-per-agent is slow. If the ABM exceeds ~10 s per run at 2,000 firms, vectorize the firm decision step with NumPy across the `AgentSet` rather than looping.
15. **The reflection problem (Manski).** Peer effects from observational data are not identified in general. The lagged neighbour share helps with simultaneity; it does nothing about homophily. **Do not claim a causal peer effect.** Claim, correctly, that the estimator is biased by a measurable amount under a homophily process you planted — a stronger and more useful statement.
16. **Emulator exploitation.** A planner searching against an overconfident model will find its errors and steer into them. That is what `J_emulator − J_ABM` measures, and it is why every policy is evaluated in the ABM.
17. **Don't chase bitwise GPU determinism** (the guide says so). Statistical reproducibility across ≥ 5 seeds is the standard.
18. **The smoke budget is hard: < 6 minutes on 4 cores.** If a stage overruns, shrink the profile, not the science.
19. **The `oracle/` / `dgp` firewall is not a suggestion.** Any convenience import of ground truth into calibration or training invalidates the entire evaluation section.
20. **Over-claiming.** The world is synthetic. Every finding is about the pipeline, not about privacy regulation. `FINDINGS.md` says this first, not last.

---

## §17 What this project deliberately does not do — the minimal path

The guide's closing discipline, and the most important paragraph in it: *"The discipline that separates strong computational researchers from tool collectors is choosing the smallest stack that answers the scientific question, then stopping."*

This build is the **maximal sixteen-tool stack, chosen deliberately for pedagogy**. `README.md` says so, and `docs/MINIMAL_PATH.md` — written at the end, with the actual experience of the build in it — must state, per tool, **which specific limitation demanded it** and what you would lose by cutting it. A tool that cannot answer that question in one sentence should be removed.

| Stage | What specific limitation forced it | What you lose by cutting it |
|---|---|---|
| 1–2 pandas/Polars + NetworkX | Nothing. They are the floor. | Everything. |
| 3 Mesa | The question is about heterogeneous firms on a network; aggregates cannot express it. | The model. |
| 4 NumPyro | Point estimates give the client one trajectory and false confidence. | Honest uncertainty. The one people skip and should not. |
| **← A real project stops here, with a notebook of Matplotlib charts, and answers most of the policy question.** | | |
| 5 DoWhy/EconML | The client asked what happens *if we enforce* — a `do()` query. | The right to call any of this causal. |
| 6–7 PyTorch + PyG | The ensemble needs 21,000 rollouts; the ABM does that in eleven days. | The scenario ensemble and Stage 14's Sobol analysis. This is the one specific limitation that justifies the emulator. |
| 8 Gymnasium | Only because Stage 10 exists. | Nothing, if you have no policy question. |
| 9 PettingZoo | Only if firms are strategic. **If C6 comes back null, this stage was unnecessary — say so.** | Possibly nothing. Report the null. |
| 10 SB3/TorchRL/RLlib | Only if "what should the regulator do" is the question, not "what will happen." Plenty of world models never need PPO. | The optimized policy. Scripted policies + BoTorch (14c) get most of the way at a fraction of the complexity. |
| 11 Ray | 21,000 rollouts. | An afternoon becomes a week. |
| 12–13 Hydra + MLflow | Ninety runs and the question of which one made the figure. | Your sanity, in about a month. |
| 14 SALib | The client's follow-up is always "what should we measure next." | The most actionable output in the report. |
| 15 Plotly/Streamlit | The client is a policy team, not a Jupyter user. | Adoption. |
| 16 pytest/Docker/CI | They come back in six months with new data. | The right to call it an instrument. |

Write that table honestly at the end. If the GNN did not beat the flat model, say so. If MARL changed nothing, say so. That document is worth more to the next person than the model is.

**Out of scope, and named so nobody wonders:** the full **million-agent AgentTorch** population (Stage 3b builds only a 2,000-firm differentiable version as a calibration cross-check; the million-agent build is the scaling path, noted in the README, not built here), FLAME GPU, LLM agents, a JAX/Brax rewrite, TD-MPC/MuZero-style value-equivalent models, and real firm data.

---

## §18 Definition of done

- [ ] `make lint`, `make typecheck`, `make test`, `make smoke` all green, locally and in CI.
- [ ] `docker build` succeeds and `make smoke` passes **inside the container**.
- [ ] All sixteen rows of the §2 table have a module, a script path, and a passing test.
- [ ] `make all` (`profile=dev`) runs every stage and writes `reports/FINDINGS.md` + `reports/run_manifest.json` with no `FAILED` stage.
- [ ] `FINDINGS.md` opens with the synthetic-world disclaimer and marks **C1–C6** SUPPORTED / REFUTED / INCONCLUSIVE with evidence attached.
- [ ] **Parameter-recovery gate (C1):** ≥ 12/16 parameters cover θ\* at 90% under `wellspecified`, R-hat < 1.01, divergences = 0; and `β_peer` misses under `confounded`.
- [ ] **Four-number causal gate (C2):** DiD 95% CI covers τ_true; the DML estimate is provably biased; `|τ_abm − τ_qe|` inside the DiD CI (or the run is FLAGGED and `simulator_discrepancy.md` exists).
- [ ] **Planning-utility gate:** emulator-trained policies beat `random` and `fixed_enforcement` **in the true ABM** with non-overlapping 95% CIs; Dreamer exploitation gap ≤ 15%.
- [ ] The ensemble Zarr cube exists with dims `(policy, draw, seed, quarter, variable)`, coverage ≥ 0.85, and `P(backfire | policy)` computed for every policy.
- [ ] All 13 figures present and populated.
- [ ] `FINDINGS.md` contains a **"Where this model fails"** section with real content.
- [ ] The Streamlit dashboard launches headless and the **OOD banner fires** when the enforcement slider is dragged past its training range (verify by hand once — it is the only test that matters to the client).
- [ ] `PROGRESS.md`, `DEVIATIONS.md`, `docs/MINIMAL_PATH.md`, and `README.md` are current; the README states, per tool, which limitation demanded it.

---

## Appendix A — `pyproject.toml`

Core must always solve. Extras fail independently (§5); a stage whose extra failed is `SKIPPED`/`DEGRADED`, never a silent crash.

```toml
[project]
name = "regworld"
version = "0.1.0"
description = "A policy world model of regulatory propagation through firms, consumers, and institutions."
requires-python = ">=3.11,<3.13"
dependencies = [
  # core — must always solve
  "numpy>=1.26,<3", "scipy>=1.13", "pandas>=2.2", "polars>=1.0", "pyarrow>=16", "duckdb>=1.0",
  "networkx>=3.3", "mesa>=3.0",                         # 3.x for the AgentSet API
  "torch>=2.4", "torch-geometric>=2.6", "einops>=0.8",  # PyG ≥2.6: no compiled scatter/sparse extras
  "gymnasium>=1.0",                                     # 5-tuple step, terminated/truncated
  "xarray>=2024.6", "zarr>=2.18,<3", "h5netcdf>=1.3",
  "scikit-learn>=1.5", "statsmodels>=0.14",
  "hydra-core>=1.3", "omegaconf>=2.3", "pydantic>=2.7",
  "mlflow>=2.14",
  "matplotlib>=3.8", "plotly>=5.22",
  "tqdm>=4.66", "rich>=13.7",
]

[project.optional-dependencies]
bayes  = ["numpyro>=0.15", "jax[cpu]>=0.4.30", "pymc>=5.16", "arviz>=0.18"]
causal = ["dowhy>=0.11", "econml>=0.15", "linearmodels>=6.0", "causal-learn>=0.1.3"]
rl     = ["stable-baselines3>=2.3", "pettingzoo>=1.24", "supersuit>=3.9",
          "ray[default,rllib,tune]>=2.30", "torchrl>=0.4", "tensordict>=0.4"]
opt    = ["SALib>=1.5", "optuna>=3.6", "botorch>=0.11", "ax-platform>=0.4"]
app    = ["streamlit>=1.35"]
dev    = ["pytest>=8.2", "pytest-cov>=5.0", "pytest-xdist>=3.6", "hypothesis>=6.100",
          "ruff>=0.5", "mypy>=1.10", "pre-commit>=3.7", "types-PyYAML"]
gpu    = []                                             # install CUDA torch via index-url in Docker.cuda
tensor = ["agent-torch>=0.4"]                           # Stage 3b; pure-torch fallback if it won't install
slurm  = ["hydra-submitit-launcher>=1.2"]
dvc    = ["dvc>=3.51"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/regworld"]

[tool.ruff]
line-length = 100
target-version = "py311"
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "NPY", "RUF", "T20"]   # T20 bans print()
[tool.ruff.lint.per-file-ignores]
"notebooks/*" = ["T20", "E402"]
"scripts/*"   = ["T20"]

[tool.mypy]
python_version = "3.11"
packages = ["regworld"]
ignore_missing_imports = true
disallow_untyped_defs = true
warn_unused_ignores = true
# strict where a wrong tensor shape can masquerade as a subtle result:
[[tool.mypy.overrides]]
module = ["regworld.models.*", "regworld.environments.*"]
disallow_any_generics = true
warn_return_any = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = ["slow: long-running scientific gates"]
filterwarnings = ["ignore::DeprecationWarning"]
```

> If `uv lock` cannot resolve, see Guardrail 12 and §5's `--isolated-envs` fallback. Relax, isolate, log — never silently downgrade the stack.

---

## Appendix B — `Makefile`

```makefile
.DEFAULT_GOAL := help
PROFILE ?= smoke
RUN := uv run
export MPLBACKEND := Agg
export JAX_PLATFORMS := cpu
export XLA_PYTHON_CLIENT_PREALLOCATE := false
export PYTHONHASHSEED := 0

help:            ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

setup:           ## Install deps + hooks
	uv sync --extra dev && $(RUN) pre-commit install
lock:            ## Refresh the lockfile
	uv lock

lint:            ## ruff check + format check
	$(RUN) ruff check . && $(RUN) ruff format --check .
typecheck:       ## mypy
	$(RUN) mypy
test:            ## Fast unit tests
	$(RUN) pytest -m "not slow" -n auto
test-slow:       ## Scientific gates
	$(RUN) pytest -m slow

data:            ## Stage 1
	$(RUN) python scripts/generate_world.py profile=$(PROFILE) && $(RUN) python scripts/make_data.py profile=$(PROFILE)
graphs:          ## Stage 2
	$(RUN) python scripts/build_graphs.py profile=$(PROFILE)
abm:             ## Stage 3 (+3b)
	$(RUN) python scripts/run_abm.py profile=$(PROFILE)
calibrate:       ## Stage 4  (JAX in a subprocess)
	$(RUN) python scripts/calibrate.py profile=$(PROFILE)
causal:          ## Stage 5
	$(RUN) python scripts/causal_analysis.py profile=$(PROFILE) && $(RUN) python scripts/validate_simulator.py profile=$(PROFILE)
emulator:        ## Stages 6–7
	$(RUN) python scripts/train_emulator.py profile=$(PROFILE)
rl:              ## Stage 10
	$(RUN) python scripts/train_rl.py profile=$(PROFILE)
ensemble:        ## Stage 11
	$(RUN) python scripts/run_ensemble.py profile=$(PROFILE)
sensitivity:     ## Stage 14
	$(RUN) python scripts/sensitivity.py profile=$(PROFILE)
figures:         ## Stage 15
	$(RUN) python scripts/make_figures.py profile=$(PROFILE)
report:          ## Stage 17
	$(RUN) python scripts/build_report.py profile=$(PROFILE)

all:             ## Full pipeline, profile=dev
	$(RUN) python scripts/run_pipeline.py profile=dev
smoke:           ## Full pipeline, CPU, < 6 min  (CI gate)
	$(RUN) python scripts/run_pipeline.py profile=smoke && $(RUN) pytest -m slow -q
paper:           ## Full pipeline, profile=full (cluster)
	$(RUN) python scripts/run_pipeline.py profile=full compute=ray_cluster

sweep:           ## Hydra multirun example
	$(RUN) python scripts/run_pipeline.py -m profile=dev seed=0,1,2,3,4 emulator=rssm_gnn,rssm_flat,gru_baseline
slurm:           ## Same sweep via submitit
	$(RUN) python scripts/run_pipeline.py -m hydra/launcher=submitit_slurm profile=full seed=0,1,2,3,4

dashboard:       ## Streamlit
	$(RUN) streamlit run scripts/dashboard.py --server.headless true --server.port 8501

docker-build:
	docker build -f docker/Dockerfile -t regworld:latest .
docker-run:
	docker run --rm -v $$PWD/artifacts:/work/artifacts regworld:latest python scripts/run_pipeline.py profile=$(PROFILE)

clean:
	rm -rf experiments/* artifacts/* reports/figures/* .pytest_cache .mypy_cache

.PHONY: help setup lock lint typecheck test test-slow data graphs abm calibrate causal emulator \
        rl ensemble sensitivity figures report all smoke paper sweep slurm dashboard \
        docker-build docker-run clean
```

---

## Appendix C — `docker/`

`docker/Dockerfile` (CPU, the default CI builds):

```dockerfile
ARG BASE=python:3.11-slim
FROM ${BASE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    REGWORLD_ARTIFACT_ROOT=/work/artifacts \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# non-root
RUN useradd -m -u 1000 rw
WORKDIR /work
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./
RUN chown -R rw:rw /work
USER rw

ENTRYPOINT ["uv", "run"]
CMD ["python", "scripts/run_pipeline.py", "profile=smoke"]
```

`docker/Dockerfile.cuda` is the same with `--build-arg BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` plus a CUDA-torch index-url install step. `docker/compose.yaml` brings up the dashboard and an MLflow UI together. Keep the CPU path as the default and the one CI builds.

---

## Appendix D — GitHub Actions

`.github/workflows/ci.yml`:

```yaml
name: ci
on: { push: { branches: [main] }, pull_request: {} }

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      matrix: { python: ["3.11", "3.12"] }
    env:
      MPLBACKEND: Agg
      JAX_PLATFORMS: cpu
      PYTHONHASHSEED: "0"
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with: { enable-cache: true }
      - run: uv python install ${{ matrix.python }}
      - run: uv sync --extra dev --frozen
      - run: make lint
      - run: make typecheck
      - run: make test
      - run: make smoke
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: report-py${{ matrix.python }}
          path: |
            reports/FINDINGS.md
            reports/figures/**

  docker:
    if: startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f docker/Dockerfile -t regworld:ci .
      - run: docker run --rm regworld:ci python scripts/run_pipeline.py profile=smoke
```

`.github/workflows/nightly.yml` runs `pytest -m slow` plus `profile=dev` on a larger runner and posts results as an artifact. `.github/workflows/docker.yml` builds and pushes both images to GHCR on tags.

---

## Appendix E — `CLAUDE.md` (write this in Phase 1)

```markdown
# RegWorld — working notes for Claude Code

## What this is
A policy world model of regulatory propagation (firms × consumers × regulator × associations),
built as the maximal sixteen-tool stack from `PLAN.md`. Read `PLAN.md` §2 for the tool map, §7 for
the model equations, §10 for the phases and stages. `PROGRESS.md` says where we are.

## Non-negotiables
- Mesa >= 3.0 AgentSet API. No RandomActivation / self.schedule.
- Gymnasium >= 1.0 five-tuple. `truncated` at horizon; `terminated` only on systemic collapse.
- Nothing outside `src/regworld/evaluation/` may import `regworld.dgp` or read `artifacts/oracle/`. Ever.
- No `print()` in src/ (ruff T20). No bare `np.random.*` — seeded Generators, passed explicitly.
- No `torch-scatter` / `torch-sparse`. No `hydra-ray-launcher`.
- Every JAX stage (calibration) runs in a subprocess with JAX_PLATFORMS=cpu.
- `make lint && make typecheck && make test` before every commit. `make smoke` stays under 6 minutes.
- Never stub a stage to pass a gate. Mark BLOCKED/DEGRADED honestly in PROGRESS.md.

## Commands
make setup | lint | typecheck | test | smoke | all | sweep | slurm | dashboard | docker-build

## Where things live
src/regworld/dgp/          the answer key (import-restricted); rules.py holds the shared pure equations
src/regworld/{data,graphs,abm,calibration,causal,models,training,environments,agents,
              evaluation,ensemble,sensitivity,visualization}
configs/                   Hydra groups (profile, compute, data, dgp, population, network, behavior,
                           abm, objective, calibration, causal, emulator, env, policy, rl, ensemble,
                           sensitivity, tracking, eval)
scripts/                   Hydra entry points, one per stage; run_pipeline.py runs all of them.

## If a library API differs from PLAN.md
Follow the library. Log it in DEVIATIONS.md with one line of rationale. Do not pin backwards.
If a gate fails twice, stop and report what failed, what you tried, and what you need.
```

---

## Appendix F — `PROGRESS.md` / `DEVIATIONS.md` templates

`PROGRESS.md` (create in Phase 1, update after every stage — it is the resume point):

```markdown
# PROGRESS

Run started: <date>   Agent session: <n>   Git HEAD: <hash>

## Phase status
- [ ] 1 Foundation
- [ ] 2 World & data
- [ ] 3 Simulation
- [ ] 4 Inference
- [ ] 5 Emulator
- [ ] 6 Control & ensemble
- [ ] 7 Delivery

## Stage log
| Stage | Status (DONE/SKIPPED/DEGRADED/FAILED/BLOCKED) | Gate | Notes |
|---|---|---|---|
| 0 recon | | | |
| 1 data | | | |
| ... | | | |

## Divergences from PLAN.md
(what the installed libraries actually required; mirror docs/DEVIATIONS.md)

## Blocked / needs human
(empty if clean)

## Next action
<the single next thing a fresh session should do>
```

`docs/DEVIATIONS.md`:

```markdown
# Deviations from PLAN.md
Where the installed reality differed from this plan. Follow the library, not the plan.

| Date | Plan said | Reality | What I did | Why |
|---|---|---|---|---|
| 2026-07-17 | `mesa.time.RandomActivation` | removed in Mesa 3.x | used `agents.shuffle_do("step")` | §16 guardrail 1 anticipated this |
```

---

## Appendix G — File manifest by phase

Roughly 65 source files, 15 test files, 16 scripts. Build them in phase order (§10).

**Phase 1 (Foundation)** — `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/{ci,docker,nightly}.yml`, `docker/*`, `src/regworld/{__init__,types,seeding,logging_conf,tracking}.py`, `configs/**` (all groups, §6), `scripts/run_pipeline.py` (driver stub), `CLAUDE.md`, `PROGRESS.md`, `README.md`. Tests: `test_config.py`, `test_seeds.py`, `test_tracker.py`, `test_layering.py`, `test_no_dgp_leakage.py`.

**Phase 2 (World & data)** — `src/regworld/dgp/{world,dynamics,observation,history}.py`, `src/regworld/rules.py`; `src/regworld/data/{generate,ingest,schema,store,duck}.py`; `src/regworld/graphs/{build,analyze,to_pyg}.py`; `scripts/{generate_world,make_data,build_graphs}.py`. Tests: `test_data_schema.py`, `test_graph_construction.py`.

**Phase 3 (Simulation)** — `src/regworld/abm/{model,agents,collect,policies,tensorized}.py`; `src/regworld/environments/{abm_env,emulator_env,marl_env,wrappers}.py`; `scripts/run_abm.py`. Tests: `test_abm_contract.py`, `test_abm_agreement.py` (slow), `test_env_contract.py`, `test_marl_env.py`.

**Phase 4 (Inference)** — `src/regworld/calibration/{summaries,micro_numpyro,micro_pymc,macro_smc,diagnostics}.py`; `src/regworld/causal/{graph,estimate,did,refute,discovery,ground_truth,gate}.py`; `scripts/{calibrate,causal_analysis,validate_simulator}.py`. Tests: `test_parameter_recovery.py` (slow), `test_causal_recovers_known_effect.py`.

**Phase 5 (Emulator)** — `src/regworld/models/{encoder,rssm,gnn,heads,world_model}.py`; `src/regworld/training/{datamodule,losses,train_emulator,checkpoint}.py`; `src/regworld/evaluation/{predictive,distributional,calibration_curves,dtw,planning_utility,behavioral_fidelity,parameter_recovery,causal_eval,ood,backtest,ablations,report}.py`; `scripts/{make_emulator_dataset,train_emulator,eval_emulator}.py`. Tests: `test_dynamics_shapes.py`, `test_smoke_train.py`.

**Phase 6 (Control & ensemble)** — `src/regworld/agents/{scripted,sb3_agents,dreamer,marl}.py`; `src/regworld/training/train_policy.py`; `src/regworld/ensemble/{scenarios,ray_ensemble,cube}.py`; `src/regworld/sensitivity/{salib_gsa,optuna_search,bo_policy}.py`; `scripts/{train_rl,train_marl,run_ensemble,sensitivity,optimize_policy}.py`. Tests: `test_policy.py`, `test_ensemble_shapes.py`, `test_sensitivity.py`, `test_configs.py`, `test_determinism.py`.

**Phase 7 (Delivery)** — `src/regworld/visualization/{figures,interactive,dashboard}.py`; `scripts/{make_figures,build_report,dashboard}.py`; `docker/Dockerfile.cuda`, `docker/compose.yaml`; `slurm/submit.sbatch`; `docs/{DEVIATIONS,REAL_DATA,MINIMAL_PATH}.md`; `reports/FINDINGS.md`.

---

## Appendix H — Command reference

```bash
# build
make setup                       # uv venv + install + hooks
make lint typecheck test         # the fast loop
make smoke                       # full 17-stage pipeline, profile=smoke, ~6 min

# science
make all                         # profile=dev, ~2h10m on 16 vCPU
uv run python scripts/run_pipeline.py profile=dev dgp=confounded policy=phased_targeted
uv run python scripts/run_pipeline.py profile=dev stages.marl=false calibration.crosscheck=false

# the interesting overrides
... dgp=wellspecified                        # the unit-test world: recovery must succeed
... dgp=confounded network.homophily=0.0     # isolate the homophily effect on β_peer
... emulator=rssm_flat                       # the ablation that tests whether the GNN earns its place
... policy=uniform_high                      # the backfire case
... objective=competition_first              # re-weight the regulator's reward
... causal.on_disagreement=report            # do not auto-recalibrate; just flag

# sweeps (Hydra multirun)
uv run python scripts/run_pipeline.py -m seed=0,1,2,3,4 policy=uniform_high,targeted,phased_targeted
uv run python scripts/train_emulator.py -m emulator=rssm_gnn,rssm_flat,gru_baseline seed=0,1,2

# cluster
uv run python scripts/run_pipeline.py profile=full compute=ray_cluster \
    compute.address=ray://head:10001 hydra/launcher=submitit_slurm

# delivery
make figures report
make dashboard                   # streamlit, headless
docker compose -f docker/compose.yaml up   # dashboard + MLflow UI
```

---

## Provenance

Every numbered tool, the repository layout, the reproducibility discipline, and the evaluation taxonomy are drawn from *A Practical Guide to the World-Modeling Research Stack* (Parts XIII, XVIII, XIX). The scientific specification in §7 — the entities, the graphs, the behavioral parameters, the economies-of-scale backfire mechanism, the latent-capacity confounder with its two confounding pathways, the two-regime transportability design, and the staggered-rollout natural experiment — is the concrete instantiation this plan adds so the sixteen tools have something real to do. Where the plan and a library's installed reality disagree, the library wins, and the divergence is logged.

