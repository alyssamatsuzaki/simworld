# PROGRESS

Run started: 2026-07-19   Agent session: 3   Git HEAD: Stage 4 calibration (see latest git log)

## Phase status
- [x] 1 Foundation (gate green; CI runs on push)
- [x] 2 World & data (gate green)
- [x] 3 Simulation (gate green)
- [x] 4 Inference (gate green: 7 slow tests in test_parameter_recovery + test_causal_recovers_known_effect)
- [ ] 5 Emulator
- [ ] 6 Control & ensemble
- [ ] 7 Delivery

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
| 6+7 emulator | | | |
| 8 envs | DONE (ABM half) | GATE-8-OK | Gymnasium AbmEnv check_env, deterministic reset, and terminated/truncated semantics; EmulatorEnv lands in Phase 5 |
| 9 marl env | DONE | GATE-9-OK | PettingZoo Parallel API (100 cycles), top-K strategic firms, live pre-draw action effects and profit rewards |
| 10 rl | | | |
| 11 ensemble | | | |
| 12 hydra | | | |
| 13 tracking | | | |
| 14 sensitivity | | | |
| 15 viz | | | |
| 16 tooling | | | |
| 17 report | | | |

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
Phase 5 (Stages 6-7): the GraphRSSM emulator — macro RSSM + PyG GNN encoder over
hetero_observed.pt, DreamerV3-style losses, domain-randomized training set from the
tensorized ABM, then the emulator evaluation suite (C1 parameter-recovery grid incl.
confounded-vs-wellspecified beta_peer bias) and EmulatorEnv with spaces identical to
AbmEnv. Gate: Phase 5 gate command in §10; commit messages per stages 6-7.
