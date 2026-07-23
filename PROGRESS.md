# PROGRESS

Run started: 2026-07-23 (audit-and-finish)   Agent session: 4   Git HEAD: 2ff392d at audit start
Prior history: sessions 1–3 built Phases 1–7 (2026-07-19 → 2026-07-22, macOS arm64 then Windows x86_64).
This session: full independent audit on Linux x86_64 (4 cores, 15 GB RAM), then finish-and-publish.

## Phase status (verified this session, not inherited)

- [x] 1 Foundation — gate re-run green (lint 0, mypy 0 on 91 files, fast suite 244 passed / 1 skipped)
- [x] 2 World & data — gate re-run green (inside fast suite + smoke)
- [x] 3 Simulation — gate re-run green
- [x] 4 Inference — gate re-run green (slow suite green inside `make smoke`)
- [x] 5 Emulator — gate re-run green
- [x] 6 Control & ensemble — gate green at smoke; coverage-gate STRICT only at dev (not yet re-run at dev)
- [x] 7 Delivery — `make smoke` exit 0, all 15 pipeline stages DONE; **but see F2: figures 8/13 on a clean checkout**

## Verified baseline (2026-07-23, this box)

| Check | Result | Evidence |
|---|---|---|
| `make setup` + every extra one at a time | all resolved; `.stage_skips` empty | uv logs; bayes/causal/rl/opt/app/tensor/slurm each OK |
| `make lint` | exit 0 | ruff clean, 136 files formatted |
| `make typecheck` | exit 0 | mypy: 91 source files, no issues |
| `make test` | exit 0 | 244 passed, 1 skipped (~11 min on 4 cores) |
| `make smoke` | exit 0 | 15/15 stages DONE, slow suite green; ~14 min wall **under heavy contention** (7 audit subagents in parallel); clean re-timing pending |
| Docker | daemon started locally | `docker ps` works; build gate now locally verifiable |

## §18 Definition of Done — audited status (before fixes)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | lint/typecheck/test/smoke green | PASS (local) | above; CI runs on push |
| 2 | docker build + smoke in container | **FAIL** | Dockerfile `uv sync --frozen --no-dev` installs core only — image cannot run its own smoke CMD (F1) |
| 3 | all 16 §2 rows: module+script+test | PARTIAL | renames recorded below; Stage 14c BoTorch absent; Stage 10d not wired into pipeline (F5, F6) |
| 4 | `make all` (dev) no FAILED stage | UNKNOWN | not yet run on this box (prior sessions deferred; see Next action) |
| 5 | FINDINGS disclaimer + C1–C6 | PASS (structure) | verified; C1 verdict logic gap (F7) |
| 6 | Parameter-recovery gate (C1) | PARTIAL | reported but thresholds asserted nowhere; β_peer must-miss unasserted (F7) |
| 7 | Four-number causal gate (C2) | PASS at smoke | gate PASSED; C2 slow tests green |
| 8 | Planning-utility gate | PARTIAL | asserted in test_policy.py; xfail-documented at smoke, STRICT at dev — needs dev run |
| 9 | Ensemble Zarr cube (policy,draw,seed,quarter,variable) + coverage ≥0.85 + P(backfire) | **FAIL** | cube is terminal-only Parquet, no quarter dim, no Zarr (F3); coverage gate exists and is honest |
| 10 | All 13 figures present | **FAIL** (clean run) | 8/13; figs 2,5,7,12,13 skip because `reports/eval/metrics.json` is only written by `eval_emulator`, which the driver never runs (F2) |
| 11 | "Where this model fails" real content | PASS | verified in report.py + FINDINGS.md |
| 12 | Dashboard launches + OOD banner fires | PENDING MANUAL VERIFICATION | prior session hand-verified (2026-07-21); this session will re-verify headless launch only and leave the hand-check to the user |
| 13 | PROGRESS/DEVIATIONS/MINIMAL_PATH/README current | PASS (being maintained) | this file; DEVIATIONS gains rows in Phase B |

## §2 sixteen-tool map — verified (all rows have module+script+test; renames noted)

Rows 1–9, 11–13, 15–16: present and gate-passing (see audit reports). Renames vs Appendix G, functionality present:
`agents/scripted.py`→`abm/policies.py`; `agents/sb3_agents.py`→`agents/ppo.py`; `training/train_policy.py`→`agents/train_rl` path;
`ensemble/{scenarios,ray_ensemble}.py`→`ensemble/cube.py`; `sensitivity/salib_gsa.py`→`sensitivity/screen.py`;
`sensitivity/optuna_search.py`→`sensitivity/policy_search.py` (repurposed — see F6); tests `test_ensemble_shapes/test_sensitivity/test_configs/test_determinism`
→ `test_ensemble_contract/test_sensitivity_contract/test_config/test_seeds`. Genuinely absent: `sensitivity/bo_policy.py`,
`scripts/optimize_policy.py` (Stage 14c), Optuna emulator-HP tuning (14b as specified), SAC path (`rl.algo` unread).

## Prioritized fix plan (Phase B, strict phase order)

**P0 — §18 blockers**
- F1 [Phase 7, independent] Dockerfile/Dockerfile.cuda `--no-dev` core-only → sync all extras; compose MLflow URI → sqlite.
- F2 [Phase 5→7, shared-core] Wire the §11 eval suite into the driver (new `evaluation` stage) so figs 2/5/7/12/13 have inputs on a clean run; assert 13/13 in the slow e2e.
- F3 [Phase 6, shared-core] Rebuild ensemble cube as per-quarter xarray→Zarr with dims (policy,draw,seed,quarter,variable); keep terminal Parquet for DuckDB/dashboard; supersede DEVIATIONS row on terminal-only cube; figures/dashboard fan reads the cube.
- F4 [Phase 4+7, shared-core] Make the recovery gate assertable: count hdi-90 coverage (≥12), assert β_peer must-miss under confounded (dev/full recovery run), fix report.py C1 verdict to count coverage.
- F5 [Phase 6, independent] Wire Stage 10d (train_marl IPPO fallback) into the pipeline `marl` stage (smoke-scale budget), recorded DEGRADED-vs-RLlib as sanctioned.
- F6 [Phase 6, independent] Build Stage 14c `sensitivity/bo_policy.py` + `scripts/optimize_policy.py` (BoTorch over the 5-param scripted schedule vs the ABM, `bo_evals` budget) and a real 14b Optuna emulator-HP tuning entry point; read `rl.algo` (SAC).

**P1 — correctness (science-affecting)**
- F7 [Phase 4/7] E-value from the real DML CI (refute.py fabricates one); pin JAX_PLATFORMS=cpu in scripts/calibrate.py unless calibration.device=gpu.
- F8 [Phase 5] gru_baseline imagination feeds symlog aggregates to a natural-units encoder (world_model.py:361) — symexp first; fixes ablation fairness.
- F9 [Phase 5, shared-core] Imagination-time node-path off-by-one (world_model.py:375) — align with teacher forcing; needs retrain (smoke retrain in pipeline re-run).
- F10 [Phase 3, shared-core] Env model factory uses prior-center Theta; use posterior-mean theta when posterior.nc exists (biases exploitation gap + C6 otherwise).
- F11 [Phase 3, shared-core] `reset(seed=None)` re-pins cfg.seed in AbmEnv/EmulatorEnv — derive from self.np_random so SB3 auto-resets vary.
- F12 [Phase 3] Baseline outcome computed noise-free vs noisy stepped quarters — make consistent (backfire CS leg bias); dashboard CS/backfire leg same issue [Phase 7].
- F13 [Phase 5] Checkpoint lacks n_firms; write it, drop inconsistent backfills.
- F14 [Phase 6] Sensitivity 64-point emulator-vs-ABM check compares J vs terminal compliance — compare like-for-like with a tolerance; assert.
- F15 [Phase 4] Implement `on_disagreement=recalibrate` (DiD moment penalty into 4b, one 4→5 re-run) — currently dead config.
- F16 [Phase 4] Add-unobserved-common-cause refuter sweep; identify on the true DAG too ("report both").
- F17 [Phase 3] Lobbying applies same-quarter vs PLAN's next-quarter — fix to lagged; budget-exhaustion collapse branch dead code — real threshold.

**P2 — dead knobs / missing small features (implement or record)**
- F18 [Phase 1] `--isolated-envs` is a silent no-op — implement minimally for real (per-group venv via UV_PROJECT_ENVIRONMENT) or record + hard-error; add configs/hydra/launcher yamls (+ joblib launcher dep); pin dev profile sizes explicitly in profile/dev.yaml; force_stage typo → error; driver caching tests.
- F19 [Phase 3] `graph_obs`/`abm.vectorized` dead flags; EmulatorEnv missing from environments/__init__.
- F20 [Phase 4] `calibration.method` dead (numpyro_bsl) — dispatch or record.
- F21 [Phase 5] RL second-round corpus promised in datamodule docstring — implement or scope out; dead `emulator.epochs` knob; eval seeds from cfg.seed.
- F22 [Phase 6] run_ensemble.py script enforces coverage gate; ray_cluster address honored; sensitivity.py summary clobber; MedianPruner trial.report.
- F23 [Phase 2] Firewall regexes miss `from ..dgp import`/importlib forms; scripts/ never greped for oracle; store.write_observed silently skips unknown names.

**P3 — honesty bookkeeping**
- F24 DEVIATIONS rows for every unrecorded adaptation found by audit: peer-share denominator, phase-onset +1, Φ split + dead fine_cap, reward exit delta, PyG static/dynamic restructure, multiplicative hhi noise, 4σ sanity test, ground_truth.py oracle exception rationale, do()-at-generation, 5f de-attenuation + smoke n_do=16, backtest filter-not-train, DTW shuffled baseline, no torch.amp, draw=RNG-seed θ-marginal cube semantics, CI all-extras sync, CACHED status addition.
- F25 report.py: fig01 filename mismatch; exploitation-gap list in failure section; fig8 95% band; report integration test slow-not-skip; tracker minors (TimeoutExpired, wandb git hash); seeding PYTHONHASHSEED no-op removal; ObservedWorld cache; lever_schedule + live-margin tests.

## Blocked / needs human
Nothing blocked. One §18 item is deliberately left PENDING MANUAL VERIFICATION (dashboard OOD banner by hand — instructions below at delivery).

## Next action
Phase B: work Phases 1→7 applying F1–F25 in phase order, gate after each phase, commit at each green gate.
Then Phase C: full §18 re-run, `make all` (profile=dev) on this box (multi-hour on 4 cores — run in background),
docker build + container smoke, publish per instructions.
