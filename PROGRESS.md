# PROGRESS

Run started: 2026-07-19   Agent session: 3   Git HEAD: Stage 4 calibration (see latest git log)

## Phase status
- [x] 1 Foundation (gate green; CI runs on push)
- [x] 2 World & data (gate green)
- [x] 3 Simulation (gate green)
- [x] 4 Inference (gate green: 7 slow tests in test_parameter_recovery + test_causal_recovers_known_effect)
- [x] 5 Emulator (gate green: test_dynamics_shapes + test_smoke_train + make emulator + eval_emulator)
- [x] 6 Control & ensemble (gate green at profile=smoke; see Next action for dev-profile follow-up)
- [x] 7 Delivery (gate green at profile=smoke: full 17-stage pipeline DONE/CACHED, 13/13 figures, FINDINGS.md)

## Stage log
| Stage | Status (DONE/SKIPPED/DEGRADED/FAILED/BLOCKED) | Gate | Notes |
|---|---|---|---|
| 0 recon | DONE | GATE-0-OK | uv 0.11.29, py3.12, all extras resolved; see DEVIATIONS |
| 1 data | DONE | GATE-1-OK | Raw §8 observations, observed-only analysis hats, deterministic Parquet, DuckDB views; rollout grid retains not-yet-treated DiD controls |
| 2 graphs | DONE | GATE-2-OK | Complete-demand NetworkX graph pair, metrics, PyG static/dynamic feature contract; 9 focused tests pass |
| 3 abm | DONE | GATE-3-OK | Mesa 3.5 AgentSet model, observed-only fresh forecast world, deterministic DataCollector outputs; 10 contract tests pass |
| 3b tensorized | DONE | 32-seed KS p > 0.05 | Pure-PyTorch sparse differentiable ABM; shapes, gradients, determinism, and Mesa agreement pass |
| 4 calibration | DONE | GATE-4-OK (make calibrate exit 0) | User-authorized third `make calibrate PROFILE=smoke` succeeded in 43s: 17 fitted quantities, NumPyro micro NUTS in isolated subprocess, PyMC crosscheck, macro SMC-ABC surrogate, ArviZ energy/pair/predictive diagnostics all written. 8 focused tests green + new `test_micro_diagnostics_runs_full_arviz_and_energy_path` covering the two prior failure sites. lint + typecheck clean. PyMC rhat/ess warnings expected at smoke draws. |
| 5 causal | DONE | GATE-5-OK (make causal exit 0 + 7 C2 tests) | Full 5a-5f: two-variant DAG, four estimators (naive logit, LinearDML/CausalForestDML, C-S staggered DiD, DML-onset), DoWhy refuters (placebo -0.008) + E-value, PC/GES discovery (SHD 14-15 vs 7, wrong as designed), four-number gate PASSED: tau_true 0.415, tau_abm 0.347 (sign+3x OK), tau_qe 0.061 CI [-0.11,0.26] (tau_abm_did 0.276 inside), tau_obs 0.124 tight-and-wrong. Graded per estimand: sealed tau_did_truth 0.358 + interference gap 0.056 (user-approved adaptation; see DEVIATIONS). C2 validates estimators on a full-panel world (DiD covers, DML 11.6 SE wrong, audit confounding > 2 SE with and without z). |
| 6+7 emulator | DONE | GATE-6-OK (make emulator exit 0 + 2 test files) | GraphRSSM: macro RSSM (32x32 categorical latents, straight-through + unimix, KL-balanced 0.8/free-bits 1.0), micro HeteroConv-SAGE GNN + per-firm GRUCell, pooled encoder + hand-built aggregates, symlog/two-hot/BCE heads. Domain-randomized Zarr corpus (random/scripted/sinusoid/piecewise policies via `lever_schedule` on the tensorized twin; theta ~ Stage-4 posterior). DreamerV3 losses + k=8 open-loop imagination loss. Ablation arches rssm_flat/gru_baseline. `test_dynamics_shapes` (shapes/grads/no-NaN, 14) + `test_smoke_train` (overfit one batch < 0.05x initial in 200 steps, beats persistence, 3). §11 eval suite: 10 families run, planning-utility + sensitivity marked pending Phase 6. |
| 8 envs | DONE | GATE-8-OK | Gymnasium AbmEnv (Phase 3) + EmulatorEnv (Phase 5): identical Box spaces by construction, deterministic seeded reset, terminated (collapse) vs truncated (horizon), reward-head vs recompute flag; 9 contract tests incl. space-identity |
| 9 marl env | DONE | GATE-9-OK | PettingZoo Parallel API (100 cycles), top-K strategic firms, live pre-draw action effects and profit rewards |
| 10 rl | DONE | test_agents_contract.py green | SB3 PPO trained inside EmulatorEnv (control) + latent Dreamer-style actor-critic on imagined rollouts (experiment), shared regworld.agents.registry policy lookup used by both this stage and Stage 11. stage_rl wired (no stub). `make rl` at dev profile not yet run by the parent session — only the smoke-profile contract test is confirmed. |
| 11 ensemble | DONE | test_ensemble_contract.py green (incl. real-checkpoint e2e) | Ray-scalable (policy x posterior-draw x seed) scenario cube + ABM cross-validation subsample (regworld.ensemble). Fixed at commit time: EmulatorEnv reads meta["extras"]["n_firms"], which train_emulator.py's checkpoint never wrote — backfilled from cfg.population.n_firms (same pattern as sensitivity's policy search) in cube.py's _rollout_cell. Also rewrote the package docstring, which was tripping test_no_dgp_leakage.py's bare "oracle" grep. stage_ensemble wired (no stub). `make ensemble` at dev profile not yet run. |
| 12 hydra | | | |
| 13 tracking | | | |
| 14 sensitivity | DONE | GATE clean: ruff/mypy/pytest + scripts/sensitivity.py and scripts/eval_emulator.py both run end-to-end at profile=smoke | SALib Morris screen -> Sobol indices on the emulator + Optuna TPE policy search over the 4 regulator levers (regworld.sensitivity). Wired into stage_sensitivity and into the §11 eval driver as metric family 12 (was a placeholder status string). |
| 15 viz | DONE | 13/13 figures written by full smoke pipeline; test_visualization_contract green | regworld.visualization: figures.py (13 paper figures, graceful per-figure skip), interactive.py (Plotly fans/latent-PCA/network diffusion), dashboard.py (Streamlit, 4 real levers, Mahalanobis OOD banner — the client-critical check, unit-tested). Wired into stage_figures. Deviations: 4-lever dashboard (no fifth "fine scale" lever exists in the action space) and per-quarter fan rebuilt by re-running EmulatorEnv (cube stores terminal-only) — both in DEVIATIONS.md. |
| 16 tooling | DONE | pre-existing from Phase 1 setup; verified coherent | docker/{Dockerfile,Dockerfile.cuda,compose.yaml}, .github/workflows/{ci,docker,nightly}.yml, slurm/submit.sbatch. CI runs lint→typecheck→test→smoke and uploads FINDINGS.md + figures; docker.yml pushes to GHCR on tags; nightly runs slow tests + dev profile. `docker build` gate not run locally (needs a Docker daemon) — exercised in CI. |
| 17 report | DONE | build_report.py runs end-to-end at smoke; test_report_contract green (required-heading enforced) | regworld.evaluation.report.build_findings assembles reports/FINDINGS.md: disclaimer-first, four-number table, C1-C6 claims ledger (C2/C4/C5 SUPPORTED, C1/C3/C6 honestly INCONCLUSIVE at smoke), always-emitted "Where This Model Fails" section, run manifest. Verdicts read real artifact schemas; all reads honour cfg.paths.root (REGWORLD_ARTIFACT_ROOT-safe). docs/MINIMAL_PATH.md written. |

## Divergences from PLAN.md
See `docs/DEVIATIONS.md`; Phase 2 records the well-specified capacity control,
mandatory market coverage, exact registry/market relations, fresh Regime-F episode,
and total-regulation-onset estimand.

## Blocked / needs human
Nothing blocked. The Stage-5 estimand/power tension was resolved by the user's decision
(2026-07-20, "adapt graders to the DGP"): grade each estimator against the estimand it
identifies (sealed tau_did_truth for the DiD; tau_true for the simulator), and validate
estimator correctness on a full-panel world. Recorded in DEVIATIONS.

## Next action
All seven phases are implemented and wired end to end. The full 17-stage pipeline runs
green at profile=smoke: `run_pipeline.py profile=smoke` finishes with every stage
DONE/CACHED (0 FAILED, 0 BLOCKED), writes all 13 figures + 3 Plotly HTMLs, and
regenerates reports/FINDINGS.md with all five required sections. ruff + mypy (90 source
files) + the fast suite are green; no NotImplementedError stubs remain in stages.py.

Remaining before a v0.1.0 tag (none block the smoke gate):
- Dev/full-profile run: only profile=smoke has been exercised end to end. `make all`
  (profile=dev) and the paper/full profile on a cluster have not been run; the coverage
  >= 0.85 ensemble gate and the dev-profile claim verdicts (C1/C3/C6 are INCONCLUSIVE at
  smoke by design) are still open. **Deliberately deferred 2026-07-21**: this machine
  had swap at 94% and load 5.9/8 before the run would even start (8 CPU, ~8.6GB RAM, no
  GPU); profile=dev scales population/emulator/RL/ensemble/sensitivity 10-125x over
  smoke (e.g. emulator train_steps 300->30000), so it's a multi-hour, single-core-bound
  job with real OOM risk on this box. User chose to hold off until memory is freed
  rather than risk it. Resume with `make all` (or `python scripts/run_pipeline.py
  profile=dev`) once the user gives the go-ahead; watch for OOM and for the same class
  of checkpoint-shape bugs the smoke run surfaced (fixed so far: EmulatorEnv n_firms
  backfill).
- The Phase 7 `docker build` gate step needs Docker, which is not installed on this
  machine at all. **Attempted 2026-07-21**: `brew install --cask docker` downloads and
  installs the app but fails at a `sudo ln` step for a helper binary — needs an
  interactive password prompt Claude cannot supply. User needs to either run `brew
  install --cask docker` themselves in a real terminal (to enter the password when
  prompted) or install Docker Desktop directly from docker.com, then Claude can build +
  run the container gate. CI already builds/runs the image on every push, so this only
  blocks a *local* verification, not correctness.
- **OOD banner verified by hand 2026-07-21** (the Definition-of-Done item that says this
  is "the only test that matters to the client"): started `streamlit run
  scripts/dashboard.py` locally, drove the sliders with real click+keyboard interaction
  (not just DOM value-setting, which silently doesn't trigger Streamlit's backend), and
  confirmed the banner is genuinely reactive both ways — enforcement=1.0/targeting=1.0
  -> red "OUT OF DISTRIBUTION: Mahalanobis distance 3.16 exceeds..."; back to
  enforcement=0.5/targeting=-0.5 -> green "In distribution: ... distance 2.02". Confirms
  `ood_mahalanobis` and the dashboard wiring are both correct.
- Stage 12 (Hydra) and 13 (tracking) were threaded through Phase 1 and are functioning;
  they have no dedicated stage rows here beyond that.
