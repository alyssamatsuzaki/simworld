# SimWorld — Senior Engineering Review & Handoff

**Date:** 2026-07-24 · **Branch:** `claude/audit-and-finish-xvofcv`

A public-repo review pass focused on **presentation, hygiene, and code quality** — making
the repository land well with recruiters, engineering managers, and practitioners without
touching the science or the honesty of the graded results. This document is the handoff: what
changed, the prioritized engineering backlog, and the repository-metadata recommendations.

---

## 1. What changed in this pass

### New meta files (were missing)

| File | Purpose |
|---|---|
| `LICENSE` | MIT license — a public repo without one is a red flag; MIT is the portfolio-friendly default. |
| `.env.example` | Documents every (all-optional) environment variable with defaults. SimWorld runs fully offline with **zero secrets**; this file makes that explicit and shows where the networked backends plug in. |
| `CONTRIBUTING.md` | Setup, the **enforced** ground rules (the leakage firewall, "never stub a stage to pass a gate"), the pre-commit gate, and PR/commit conventions. |
| `CODE_OF_CONDUCT.md` | Contributor Covenant v2.1. |
| `SECURITY.md` | Vulnerability-reporting policy and a note on the (small) attack surface. |

### README presentation layer

- **Badges** (CI, Python, License, ruff, mypy, uv, reproducibility) directly under the title.
- **"At a glance"** block: a What / Why / headline / reproducibility table, a one-line tech-stack
  chip list, and the six-claim scorecard as a compact chip row — everything a skimming reviewer
  needs in the first screen, above the (already excellent, retained) deep sections.
- **Live-demo callout** pointing to `make dashboard` and the 13 figures, with a screenshot/GIF
  placeholder for the landing view.
- **`.env.example`** wired into the setup steps; a new **Contributing & license** section (§14)
  with a BibTeX citation block; the doc index now lists the community files.

### `pyproject.toml` metadata

Added `readme`, `license`, `authors`, `keywords`, PyPI `classifiers` (incl. `Typing :: Typed`),
and a `[project.urls]` table. Verified: `uv sync` rebuilds the package cleanly with the new
metadata (hatchling accepts `license = { file = "LICENSE" }`).

### Code-quality fixes applied (verified: ruff + mypy + fast suite green)

These three are mechanical, low-risk, and fully gate-verified:

| # | Fix | Files |
|---|---|---|
| **F2** | Extracted `rules.objective_weights(cfg)` — the six regulator-objective weights were copy-pasted (with an ugly `getattr`-loop + `cast`) in **5 modules**. Now one typed source of truth; removed 4 `cast`s and 1 `type: ignore`. | `rules.py`, `training/datamodule.py`, `sensitivity/screen.py`, `environments/{abm_env,marl_env,emulator_env}.py` |
| **F9** | Added the missing class docstring to `WorldModel` (the most central model class), naming the three ablation arches. | `models/world_model.py` |
| **F11** | Replaced the obscure `x == x  # not NaN` self-comparison with `not math.isnan(x)`. | `evaluation/report.py` |

---

## 2. Prioritized engineering backlog (deferred, recommended)

A full read of the central/largest modules found **no correctness bugs and no firewall
violations** — this is a high-craft codebase (dense purposeful docstrings, honest degradation
paths, named gate constants, seeded RNGs threaded explicitly). The items below are
maintainability/consistency improvements a linter and type-checker structurally cannot see.
They were **deferred rather than applied** because each touches subprocess-backed pipeline
stages that the fast suite does not fully exercise, so they warrant a focused PR with a
`make smoke` run — not an autonomous batch edit. Each cites a real `file:line`.

### P0 — maintainability risk

- **B1 — Private env state read across module boundaries.** `EmulatorEnv._aggregates`
  (`environments/emulator_env.py:102`) is read directly by `ensemble/cube.py:172`,
  `visualization/figures.py:397`, and `visualization/dashboard.py:149`; `AbmEnv._outcome`/
  `_baseline` are read by `agents/marl.py:359`. These classes are actively evolving and the
  consumers live in stages the fast suite may not hit, so a rename surfaces only as a runtime
  `AttributeError` deep in a stage. **Fix:** expose the natural-unit aggregate row as a public
  property (`env.aggregates`) or return it in the `step()` `info` dict; update the four consumers.

### P1 — clear improvements

- **B2 — Tensor→`QuarterOutcome` conversion duplicated.** `_terminal_tensor_outcome`
  (`ensemble/validation.py:198`) and the nested `_to_outcome` (`sensitivity/screen.py:448`) both
  rebuild `rules.QuarterOutcome` field-by-field; the screen.py copy carries ~10
  `# type: ignore[attr-defined]`. **Fix:** one `to_quarter_outcome(tensor_outcome)` helper on the
  tensorized-ABM side (which owns the real type, killing the ignores).
- **B3 — `run_screening`/`run_sobol` share a near-identical eval loop.**
  `sensitivity/screen.py:337-353` and `:393-409` (load checkpoint → backfill `n_firms` → build env
  → loop with the same progress-log expression). **Fix:** extract
  `_evaluate_design(cfg, samples, seed_base)` + the meta-backfill.
- **B4 — Episode-collapse predicate duplicated across 3 modules with bare literals.**
  `emulator_env.py:152-163`, `training/datamodule.py:254-255`, and `abm_env.py`'s own `_collapsed`
  each encode the same `exit>0.40 or (elapsed>12 and compliance<0.05 and exhausted)`. **Fix:** one
  `is_collapsed(...)` predicate with the four thresholds as named constants.
- **B5 — `build_findings` is a ~600-line function** mixing verdict logic with Markdown templating
  (`evaluation/report.py:53-656`; the C1 block alone spans ~211-270). **Fix:** extract one pure
  `_verdict_cN(artifacts) -> (verdict, evidence)` per claim; keep `build_findings` as the assembler.
- **B6 — `RegulationModel._step_impl` is a ~227-line method** doing ~10 jobs
  (`abm/model.py:519-746`). **Fix:** peel off `_enforcement(...)`, `_build_outcome(...)`,
  `_record(...)` along the physics boundaries already visible in the code.
- **B7 — Node-GRU recurrence hand-inlined in 3 `WorldModel` methods**
  (`models/world_model.py:231-239`, `:335-341`, `:383-385`) — drift risk between the training and
  imagination paths. **Fix:** an `_advance_node_gru(firm_emb, context, hidden)` helper.

### P2 — polish

- **B8 — Aggregate-vector indexing inconsistent between siblings.** `emulator_env.py:31-35` names
  the indices; `datamodule.py:255,272-282` uses bare literals (`agg[5]`, `agg[6]`, `agg[0]`) on the
  same vector. Reuse the named constants in `aggregate_to_outcome`.
- **B9 — `MicroData.subset` hardcodes the field list** (`calibration/micro_numpyro.py:50-67`) — a
  new field is silently dropped. Derive from `dataclasses.fields(self)` minus `n_sectors`.
- **B10 — `build_world_model(cfg: Any, ...)`** (`models/world_model.py:476`) drops config typing;
  it only touches `cfg.emulator`. Type it `SimWorldConfig` (or a narrow Protocol).
- **B11 — Broad `except Exception`** in best-effort readers (`evaluation/report.py:29`,
  `stages.py:125`). Defensible, but narrowing to `(OSError, json.JSONDecodeError)` /
  `(ImportError, AttributeError)` stops them masking genuine bugs.

---

## 3. Repository metadata recommendations (GitHub Settings → About)

These are set in the GitHub web UI / API, not in git — apply them on the repo's **About** panel.

**Description (≤ 350 chars):**

> A synthetic world-model of regulatory propagation: a 16-tool research stack (ABM · Bayesian
> calibration · causal inference · graph-RSSM emulator · RL) graded end-to-end against a
> planted ground truth. Fully reproducible, CPU-only, in under 6 minutes.

**Topics (add all):**

`world-models` · `agent-based-modeling` · `causal-inference` · `bayesian-inference` ·
`reinforcement-learning` · `graph-neural-networks` · `simulation` · `policy-analysis` ·
`reproducible-research` · `machine-learning` · `python` · `mesa` · `gymnasium` · `numpyro` ·
`pytorch` · `hydra`

**Also worth enabling:** the repo's **social preview image** (Settings → General) — a rendered
`fig01_four_numbers` or the dashboard makes a strong card when the link is shared.

---

## 4. Remaining work (compute-/environment-bound — pre-existing, not introduced here)

Unchanged by this pass; carried from `PROGRESS.md` for completeness:

1. **`dev`-scale run** (`make all` on a multi-vCPU node, ~12h+ for the emulator on 4 cores) to
   upgrade **C1 / C3 / C5** from INCONCLUSIVE to their `dev` verdicts and clear the ≥ 0.85
   ensemble-coverage and strict planning-utility gates. All wiring is in place and tested.
2. **Docker build** — blocked in the current sandbox by container-registry egress; CI builds and
   runs the image, and it builds where Docker Hub/ghcr are reachable.
3. **OOD dashboard banner** — the one manual hand-check (steps in `PROGRESS.md` §18 item 12).

None are code defects. Nothing in the pipeline was stubbed to pass a gate.

---

## 5. Verification

- `make lint` — ruff check + format check: **clean**.
- `make typecheck` — mypy: **Success, 92 source files**.
- `make test` — fast suite: **green** (see the commit that lands this handoff).
- `uv sync` rebuilds the local package cleanly with the new `pyproject.toml` metadata.
