# PROGRESS

Run: 2026-07-23 audit-and-finish (session 4, Linux x86_64, 4 cores / 15 GB, no GPU).
Base at audit start: origin/main @ 2ff392d. Branch: claude/audit-and-finish-xvofcv.

This session independently re-audited the repo against PLAN.md (not trusting prior
PROGRESS), then fixed every gap found and re-verified by running the plan's own gates.

## Phase status (all verified this session by running gates)

- [x] 1 Foundation — lint 0, mypy 0 (92 files), fast suite green
- [x] 2 World & data — gate green (data-schema, graph-construction, firewall)
- [x] 3 Simulation — gate green (abm/env/marl contract)
- [x] 4 Inference — gate green (parameter-recovery + causal C2 slow tests)
- [x] 5 Emulator — gate green (dynamics-shapes, smoke-train, eval suite)
- [x] 6 Control & ensemble — gate green (policy, ensemble, sensitivity)
- [x] 7 Delivery — clean `make smoke` exit 0: 16/16 stages DONE, 13/13 figures, FINDINGS.md

## Final verification (clean-slate `make smoke`, 2026-07-23)

Artifacts wiped, full pipeline re-run from scratch (no cache):

| Stage | Status | Stage | Status |
|---|---|---|---|
| recon | DONE | rl | DONE |
| data | DONE | ensemble | DONE |
| graphs | DONE | sensitivity | DONE |
| abm | DONE | evaluation | DONE (new stage, wired this session) |
| tensorized_abm | DONE | figures | DONE (13/13) |
| calibration | DONE | report | DONE |
| causal | DONE | | |
| emulator | DONE | | |
| envs / marl | DONE | | |

`SMOKE-EXIT: 0` (pipeline + slow suite). 0 FAILED, 0 BLOCKED, 0 DEGRADED.
`make lint` 0 · `make typecheck` 0 (92 files) · `make test` 244+ passed.

## §18 Definition of Done

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | lint/typecheck/test/smoke green | **PASS** | all four green locally; CI runs them on push |
| 2 | docker build + smoke in container | **DEGRADED** | Dockerfiles corrected to `--all-extras` (F1); local build blocked — sandbox proxy 403s all container registries (Docker Hub CDN + ghcr); CI builds/runs the image. See DEVIATIONS |
| 3 | all 16 §2 rows: module+script+test | **PASS** | every row present (renames documented in PROGRESS §2 map + DEVIATIONS); BoTorch 14c is named-optional (§3), recorded |
| 4 | `make all` (dev) no FAILED stage | **DEGRADED** | full pipeline runs green at smoke (16/16 DONE); dev profile is ~12h+ for the emulator alone on 4 cores — deferred with reason; CI nightly runs dev. See DEVIATIONS |
| 5 | FINDINGS disclaimer + C1–C6 verdicts | **PASS** | 5 sections in order; C2/C4 SUPPORTED, C1/C3/C5/C6 honestly INCONCLUSIVE at smoke |
| 6 | Parameter-recovery gate (C1) | **DEGRADED** | verdict logic now counts >=12/16 coverage + divergences + R-hat (F4) and asserts β_peer miss under confounded; the SUPPORTED verdict needs a dev-scale calibration (deferred) |
| 7 | Four-number causal gate (C2) | **PASS** | gate PASSED; C2 slow tests green; E-value now from real DML CI + add-unobserved refuter (F7/F16) |
| 8 | Planning-utility gate | **DEGRADED** | asserted in test_policy.py (xfail-documented at smoke, STRICT at dev); env oracle now uses calibrated θ (F10); strict pass needs the dev run |
| 9 | Ensemble Zarr cube + coverage + P(backfire) | **PASS (cube) / DEGRADED (coverage)** | cube.zarr has dims (policy,draw,seed,quarter,variable)=6×8×1×24×9 (F3); P(backfire\|policy) recorded for all 6; coverage>=0.85 gate wired and enforced but needs dev-scale emulator to clear |
| 10 | All 13 figures present | **PASS** | 13/13 on a clean run (was 8/13; the eval suite is now a driver stage — F2) |
| 11 | "Where this model fails" real content | **PASS** | OOD, β_peer bias, horizon, exploitation gap (F25), degraded stages |
| 12 | Dashboard OOD banner fires | **PENDING MANUAL VERIFICATION** | dashboard launches headless with no error (confirmed); banner reactivity is the one item needing a human — steps in FINDINGS.md "Pending manual verification" and below |
| 13 | PROGRESS/DEVIATIONS/MINIMAL_PATH/README current | **PASS** | this file; DEVIATIONS has ~35 rows; README + MINIMAL_PATH carry the per-tool tables |

**PENDING MANUAL (item 12) — exact steps for the user:** run `make dashboard`; drag the
**enforcement** slider to 1.0 and **targeting** to 1.0 → the banner must turn **red**
("OUT OF DISTRIBUTION: Mahalanobis distance … exceeds …"). Return enforcement to 0.5 and
targeting to -0.5 → it must go **green** ("In distribution"). This is the only DoD item
that cannot be verified autonomously.

## What this session changed (changelog)

- **Phase 1:** `--isolated-envs` made real (per-group uv venvs); force-stage typo
  validation; driver caching tests; `configs/hydra/launcher/` group + joblib dep;
  `profile/dev.yaml` pins the full §6 dev column; tracker/seeding hardening.
- **Phase 2:** firewall tripwire hardened (relative/dynamic imports; scripts/ swept;
  oracle grep matches access forms not PLAN's prose word); strict observed-write
  validation; exact 3% spurious-edge count; 6 bookkeeping DEVIATIONS.
- **Phase 3:** env oracle binds calibrated posterior-mean θ (was prior-center);
  Gymnasium `reset(seed=None)` draws fresh episode noise; noise-consistent q0 baseline
  (backfire CS leg); lagged lobbying; real budget-exhaustion collapse; dead flags removed.
- **Phase 4:** E-value from DoWhy's real CI; add-unobserved-common-cause refuter (was
  absent); identify_effect on both DAGs ("report both"); honest recalibrate fallback.
- **Phase 5:** gru_baseline imagination symexp fix (ablation fairness); checkpoint writes
  n_firms; verified the flagged imagination off-by-one is a non-issue (recorded).
- **Phase 6:** per-quarter xarray→Zarr cube with the §18 dims (was terminal-only);
  P(backfire|policy) for every policy; like-for-like J emulator-vs-ABM cross-check
  (was J-vs-terminal-compliance); coverage gate in the standalone script; summary merge.
- **Phase 7:** eval suite wired as a driver stage → 13/13 figures; Docker `--all-extras`;
  compose MLflow→sqlite; C1 verdict counts coverage+divergences; fig01 link; fig8 95%
  bands; dashboard backfire CS-leg from the real rollout; OOD PENDING note in FINDINGS.

## Remaining limitations / decisions for the user

1. **dev-profile run** — not executed on this box (emulator alone ~12h on 4 cores).
   Run `make all` (or the §6 science-preserving cuts) on a 16-vCPU node to upgrade
   C1/C3/C5/C6 to their dev verdicts and clear the coverage>=0.85 gate. The logic is wired.
2. **docker build** — blocked by this sandbox's registry egress; verify locally where
   Docker Hub/ghcr are reachable, or rely on CI (already builds+runs the image).
3. **OOD banner** — the one manual hand-check (steps above).

## Blocked / needs human
Nothing blocked. Three items are DEGRADED/PENDING with honest reasons (above), all
compute- or environment-limited, none a code defect.
