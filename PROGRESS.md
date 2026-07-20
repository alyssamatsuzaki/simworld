# PROGRESS

Run started: 2026-07-19   Agent session: 3   Git HEAD: Stage 4 calibration (see latest git log)

## Phase status
- [x] 1 Foundation (gate green; CI runs on push)
- [x] 2 World & data (gate green)
- [x] 3 Simulation (gate green)
- [x] 4 Inference (gate green: 7 slow tests in test_parameter_recovery + test_causal_recovers_known_effect)
- [x] 5 Emulator (gate green: test_dynamics_shapes + test_smoke_train + make emulator + eval_emulator)
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
| 6+7 emulator | DONE | GATE-6-OK (make emulator exit 0 + 2 test files) | GraphRSSM: macro RSSM (32x32 categorical latents, straight-through + unimix, KL-balanced 0.8/free-bits 1.0), micro HeteroConv-SAGE GNN + per-firm GRUCell, pooled encoder + hand-built aggregates, symlog/two-hot/BCE heads. Domain-randomized Zarr corpus (random/scripted/sinusoid/piecewise policies via `lever_schedule` on the tensorized twin; theta ~ Stage-4 posterior). DreamerV3 losses + k=8 open-loop imagination loss. Ablation arches rssm_flat/gru_baseline. `test_dynamics_shapes` (shapes/grads/no-NaN, 14) + `test_smoke_train` (overfit one batch < 0.05x initial in 200 steps, beats persistence, 3). §11 eval suite: 10 families run, planning-utility + sensitivity marked pending Phase 6. |
| 8 envs | DONE | GATE-8-OK | Gymnasium AbmEnv (Phase 3) + EmulatorEnv (Phase 5): identical Box spaces by construction, deterministic seeded reset, terminated (collapse) vs truncated (horizon), reward-head vs recompute flag; 9 contract tests incl. space-identity |
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
Phase 6 (Stages 10, 11, 14): Control & ensemble. Stage 10 — SB3 PPO trained inside
EmulatorEnv + TorchRL Dreamer on imagined rollouts; the planning-utility gate evaluates
every policy in the true ABM (5 seeds x 64 draws), Dreamer exploitation gap J_emulator -
J_ABM <= 15%. Stage 11 — Ray ensemble scenario cube (posterior draws x policies x seeds)
with @ray.remote actors holding a loaded emulator; ABM cross-validation subsample. Stage
14 — SALib Morris screen -> Sobol on the emulator, Optuna policy search. This unblocks
§11 families 5 (planning utility) and 12 (sensitivity), currently reported as pending.
Gate: `make rl ensemble sensitivity` + coverage >= 0.85.
