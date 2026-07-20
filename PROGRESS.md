# PROGRESS

Run started: 2026-07-19   Agent session: 2   Git HEAD: Phase 2 gate (see latest git log)

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
| 1 data | DONE | GATE-1-OK | Raw §8 observations, observed-only analysis hats, deterministic Parquet, DuckDB views; 18 focused tests pass |
| 2 graphs | DONE | GATE-2-OK | Complete-demand NetworkX graph pair, metrics, PyG static/dynamic feature contract; 9 focused tests pass |
| 3 abm | DONE | GATE-3-OK | Mesa 3.5 AgentSet model, observed-only fresh forecast world, deterministic DataCollector outputs; 10 contract tests pass |
| 3b tensorized | DONE | 32-seed KS p > 0.05 | Pure-PyTorch sparse differentiable ABM; shapes, gradients, determinism, and Mesa agreement pass |
| 4 calibration | | | |
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
(empty if clean)

## Next action
Phase 4, Stage 4: NumPyro micro calibration + macro SMC-ABC + PyMC/ArviZ checks,
then Stage 5 causal identification and the four-number simulator gate.
