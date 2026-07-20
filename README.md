# RegWorld

A policy world model of how a data-privacy regulation propagates through firms, consumers,
and institutions — built as the maximal sixteen-tool stack from *A Practical Guide to the
World-Modeling Research Stack*, Part XIX. The world is synthetic and known, so every stage
is graded against planted ground truth. See `PLAN.md` for the full specification,
`PROGRESS.md` for build status, and `reports/FINDINGS.md` for results.

```bash
make setup     # uv sync + hooks
make smoke     # full 17-stage pipeline, CPU, < 6 min
make all       # the real run (profile=dev)
```

Per-tool justification lives in `docs/MINIMAL_PATH.md` (written at the end, honestly).
