# PROGRESS

Run started: 2026-07-19   Agent session: 3   Git HEAD: Stage 4 calibration (see latest git log)

## Phase status
- [x] 1 Foundation (gate green; CI runs on push)
- [x] 2 World & data (gate green)
- [x] 3 Simulation (gate green)
- [ ] 4 Inference
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
| 5 causal | | | |
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
Nothing blocked. Stage 4's calibration gate is green (the twice-failed integration points
are fixed and regression-tested). Phase 4's phase gate also requires Stage 5's
`test_causal_recovers_known_effect.py`, which is the next thing to build.

## Next action
Build Stage 5 (causal): `regworld/causal/` — DoWhy identify→estimate→refute, EconML CATE,
staggered DiD, `ground_truth.py` do() effects already sealed by Stage 1, and the §10 Stage-5f
four-number simulator gate (writes `reports/simulator_discrepancy.md` only if it FLAGS).
Phase 4 gate: `uv run pytest -m slow tests/test_parameter_recovery.py tests/test_causal_recovers_known_effect.py -q`.
Commit: `feat(causal): DoWhy identify→estimate→refute, EconML CATE, staggered DiD, do() ground truth, four-number gate`.
