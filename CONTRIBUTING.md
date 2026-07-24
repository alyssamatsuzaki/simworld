# Contributing to SimWorld

Thanks for your interest. SimWorld is a research-grade codebase with a strict
"the pipeline must never lie about what it found" ethos. These conventions keep
that guarantee intact. Please read them before opening a pull request.

## Ground rules (the non-negotiables)

These are enforced by tests and CI, not just etiquette:

1. **The leakage firewall is sacred.** Nothing outside `src/simworld/evaluation/`
   may import `simworld.dgp` or read `artifacts/oracle/`. The `dgp/` package is the
   sealed answer key; importing it into calibration, training, or the emulator would
   invalidate every graded result. This is enforced mechanically by
   `tests/test_no_dgp_leakage.py` (a static + dynamic import grep) and a runtime
   stack-frame check in `data/store.py::read_oracle()`. Do not work around either.
2. **Never stub a stage to pass a gate.** If a claim's evidence needs research-scale
   compute you don't have, mark it `INCONCLUSIVE` / `DEGRADED` / `BLOCKED` **honestly**
   in `PROGRESS.md` and `reports/FINDINGS.md`. A green checkmark must mean the evidence
   is real.
3. **Modern APIs only.** Mesa ≥ 3.0 AgentSet API (no `RandomActivation` /
   `self.schedule`); Gymnasium ≥ 1.0 five-tuple (`truncated` at the horizon,
   `terminated` only on systemic collapse).
4. **No `print()` in `src/`** (ruff `T20`) — use the loggers in `logging_conf.py`.
   **No bare `np.random.*`** — seed `Generator`s and pass them explicitly.
5. **If a library's real API differs from `PLAN.md`, follow the library** and log the
   deviation in `docs/DEVIATIONS.md` with one line of rationale. Do not pin backwards.

## Development setup

```bash
git clone https://github.com/alyssamatsuzaki/simworld.git
cd simworld
make setup        # uv sync (core + all extras) + pre-commit hooks
cp .env.example .env   # optional; defaults run fully offline
make smoke        # end-to-end sanity check, CPU-only, < 6 min
```

Python 3.11 or 3.12 (see `.python-version`). Dependencies and the locked resolution
are managed by [uv](https://github.com/astral-sh/uv); use `make lock` to refresh
`uv.lock`.

## Before every commit

The pre-commit hook runs a subset of these, but run the full set yourself:

```bash
make lint         # ruff check + ruff format --check
make typecheck    # mypy (strict on models.* and environments.*)
make test         # fast unit suite (pytest -m "not slow" -n auto)
```

`make smoke` should stay under **6 minutes** — it is the CI gate. If you touch a
scientific gate, also run `make test-slow`.

## Branch, commit, and PR conventions

- Branch off `main`; use a descriptive branch name (e.g. `feat/…`, `fix/…`, `docs/…`).
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat(C1): …`, `fix(C6): …`, `docs: …`, `refactor: …`). Scope with the claim
  (`C1`–`C6`) or stage when relevant.
- Keep PRs focused. Explain **what evidence changed**, not just what code changed —
  if a claim's verdict moves, say why and cite the numbers.
- CI (`.github/workflows/ci.yml`) runs lint, typecheck, the fast suite, and `make smoke`
  on Python 3.11 and 3.12. All must be green before merge.

## Adding or changing a stage

Every stage in the [sixteen-tool stack](README.md#4-architecture-the-sixteen-tool-stack)
should have three things present and wired: a **module** in `src/simworld/…`, a
**script** entry point in `scripts/…`, and a **test**. A stage that runs but is
orphaned from the driver (`scripts/run_pipeline.py`) is not done — add a regression
test that asserts it is wired in.

## Reporting bugs and asking questions

Open a [GitHub issue](https://github.com/alyssamatsuzaki/simworld/issues) with a
minimal reproduction: the `make` target or script invocation, the profile
(`smoke`/`dev`/`full`), the seed, and the observed vs. expected behavior. For anything
touching a graded result, include the relevant snippet of `reports/FINDINGS.md`.

By contributing you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
