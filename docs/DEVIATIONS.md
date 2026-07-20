# Deviations from PLAN.md
Where the installed reality differed from this plan. Follow the library, not the plan.

| Date | Plan said | Reality | What I did | Why |
|---|---|---|---|---|
| 2026-07-19 | Docker build gates (Stage 16, §18) | No Docker daemon on this machine (macOS, not installed) | Dockerfiles + compose + CI workflows written; local `docker build` gates recorded as SKIPPED (env-limited); CI runs them on push | Cannot install Docker Desktop without the user; CI covers it |
| 2026-07-19 | §6 defaults list order: `profile` first, `dgp` before `network` | Hydra merges defaults in list order — later entries override earlier ones, so `profile` first would be overridden by later groups | Moved `dgp` after `network` and `profile` last in `configs/config.yaml` defaults | Profile is the scaling story and must win on size knobs; dgp variant must be able to set `network.homophily` |
| 2026-07-19 | `paths` use `${hydra:runtime.cwd}` | The `hydra:` resolver only resolves inside a Hydra app, which breaks config validation in plain pytest | Used relative paths + `${oc.env:REGWORLD_ARTIFACT_ROOT,artifacts}`; `hydra.job.chdir=false` keeps cwd at repo root | Same behavior, testable everywhere |
| 2026-07-19 | `causal` extra as listed in Appendix A | uv backtracked econml→sparse→numba to numba 0.53/llvmlite 0.36, unbuildable on py3.12 | Added `numba>=0.60` floor to the causal extra; everything then resolves | Guardrail 12: relax/steer the offending pin, don't drop EconML |
| 2026-07-19 | `dowhy>=0.11` | dowhy 0.11 imports `numpy.distutils`, removed in numpy 2.x | Floor raised to `dowhy>=0.12` (installs 0.12, imports clean) | Follow the library |
| 2026-07-19 | `arviz>=0.18` (open-ended) | arviz 1.2 removed `concat` and other APIs pymc 5.26 needs | Capped `arviz>=0.18,<1` (installs 0.23.4) | Two libraries must agree; 0.x is the API the plan's diagnostics use |

## Installed stack (Stage 0 probe, 2026-07-19, macOS arm64, Python 3.12)
numpy 2.4.6 · scipy 1.18.0 · pandas 2.3.3 · polars 1.42.1 · pyarrow 24.0 · duckdb 1.5.4 ·
networkx 3.6.1 · mesa 3.5.1 (AgentSet API confirmed; no RandomActivation) · torch 2.13.0 ·
torch-geometric 2.8.0 (HeteroConv imports, no torch_scatter) · gymnasium 1.2.2 (5-tuple confirmed) ·
xarray 2026.7.0 · zarr 2.18.7 · sklearn 1.6.1 · statsmodels 0.14.6 · hydra 1.3.4 · pydantic 2.13.4 ·
mlflow 3.14.0 · numpyro 0.21.0 · jax 0.11.0 · pymc 5.26.1 · arviz 0.23.4 · dowhy 0.12 · econml 0.16.0 ·
linearmodels 7.0 · causal-learn (imports) · stable-baselines3 2.9.0 · pettingzoo 1.26.1 · ray 2.56.1
(RLlib present) · torchrl 0.13.3 · tensordict 0.13.0 · SALib (imports) · optuna 4.9.0 · botorch 0.18.1 ·
ax 1.3.1 · streamlit 1.59.2 · agent-torch (imports; usability probed at Stage 3b) · matplotlib 3.11.1 ·
plotly 6.9.0. `.stage_skips` is empty — every extra resolved.
| 2026-07-19 | MLflow file backend (`file:./experiments/mlruns`) | MLflow 3.14 raises: filesystem store is in maintenance mode | Default tracking URI is `sqlite:///experiments/mlflow.db` | Still credential-free and offline; guardrail 8 wanted sqlite under concurrency anyway |
| 2026-07-19 | Gate 0 no-op override `'stages={}'` | Hydra merges `{}` into the all-true defaults — a no-op | No-op runs use `'~stages'` (delete node → StagesCfg all-False defaults) | Same intent, real Hydra semantics |
| 2026-07-19 | §7.4 q_it has no own-audit term, but the §7.7 DAG draws `audited → perceived_risk → compliant_next` | The two sections are inconsistent as written | Added a known constant `(1 + 0.8·audited_{t-1})` factor to q_it; the factor appears in the panel's perceived_risk column so the fitted micro model stays well specified | Makes per-firm do(audited) well defined, which Stage 5's CATE grading needs |
| 2026-07-19 | §10 5f: "raising enforcement from e_low to e_high"; causal defaults 0.2/0.8 | The DiD identifies the ATT of enforcement ONSET at Regime P's level (0 → 0.6); 0.2→0.8 is not what the natural experiment measures | causal/default.yaml sets e_low=0.0, e_high=0.6 so tau_true, tau_abm, tau_qe, tau_obs are estimates of the SAME estimand | The four-number table must compare like with like |
