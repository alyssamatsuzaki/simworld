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

## Which limitation demanded each tool

One line per tool: the specific limitation that forced it in. The full treatment — including
what you lose by cutting each, and where a real project should stop — is in
`docs/MINIMAL_PATH.md`.

| Stage | Tool(s) | The limitation that demanded it |
|---|---|---|
| 1 | pandas / Polars (+ pyarrow, DuckDB) | The floor: raw observations must land in a validated, analysis-ready panel. |
| 2 | NetworkX | Firms interact through supply, market, and membership edges; aggregates cannot express them. |
| 3 | Mesa (≥3.0) | Firm heterogeneity on a network is the question; a representative agent answers a different one. |
| 3b | PyTorch (tensorized ABM) | Calibration and emulator training need thousands of fast, differentiable rollouts. |
| 4 | NumPyro + PyMC (+ ArviZ, SMC-ABC) | Point estimates hand the client one trajectory and false confidence. |
| 5 | DoWhy + EconML (+ linearmodels, causal-learn) | The client asked what happens *if we enforce* — a `do()` query, not a correlation. |
| 6–7 | PyTorch + PyTorch Geometric | The ensemble needs ~21,000 rollouts; the ABM would take ~11 days. |
| 8 | Gymnasium (≥1.0) | Only because Stage 10 exists: one env contract over two different worlds. |
| 9 | PettingZoo | Only if the largest firms game the rule rather than follow it. |
| 10 | SB3 → TorchRL (→ RLlib opt.) | Only if the question is "what should the regulator do," not "what will happen." |
| 11 | Ray | 21,000 rollouts serially is a week; parallel it is an afternoon. |
| 12–13 | Hydra + MLflow (+ OmegaConf, Pydantic) | Ninety runs, and the question of which one made the figure. |
| 14 | SALib + Optuna (+ BoTorch/Ax opt.) | The client's follow-up is always "what should we measure next?" |
| 15 | Plotly + Streamlit (+ Matplotlib) | The client is a policy team, not a Jupyter user. |
| 16 | pytest + Docker + GitHub Actions (+ ruff, mypy, uv, Make) | They come back in six months with new data. |

This is the **maximal** stack, chosen deliberately for pedagogy — not the stack a real project
should build. `docs/MINIMAL_PATH.md` says where a real project stops (Stage 4) and why.
