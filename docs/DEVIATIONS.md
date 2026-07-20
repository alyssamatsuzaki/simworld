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
| 2026-07-19 | §7.4 q_it has no own-audit term, but the §7.7 DAG draws `audited → perceived_risk → compliant_next` | The two sections are inconsistent as written | Added a known constant `(1 + 0.8·audited_{t-1})` factor to q_it; Stage 1 reconstructs the same documented factor from observed audit history | Makes per-firm do(audited) well defined, which Stage 5's CATE grading needs |
| 2026-07-19 | §10 5f calls the staggered DiD “raising enforcement from e_low to e_high” | Regime-P onset simultaneously activates enforcement, phase salience, and compliance cost; it is not an enforcement-only intervention | Set the numeric anchors to 0.0/0.6 and label the answer-key estimand `total_regulation_onset_att`; Stage 5 must compare the same onset intervention or report the audit-only effect separately | The causal table must not claim a narrower intervention than the DGP ran |
| 2026-07-19 | `dgp=wellspecified` disables homophily and corr(z,size), but retains the omitted `beta_capacity*z` term | Even an independent omitted term changes a logistic link (non-collapsibility), so the fitted micro-likelihood cannot recover the planted slopes exactly | Omit the latent-capacity term only in the recovery-control world; keep it in `confounded` and `misspecified` | Makes “well specified” mathematically true while preserving the planted failure case |
| 2026-07-19 | About eight market edges per consumer segment | At smoke/dev/full sizes this leaves most firms with structurally zero demand, mechanically causing exits and concentration before policy acts | Give every firm one sector-matched market edge, then add the configured eight preferential links per segment | The backfire finding must arise from policy costs, not an uncovered graph |
| 2026-07-19 | Two degraded versions of every graph | Association membership is directly observed in the registry, and the current observation model does not define noise for market links | Keep membership and market edges shared/exact for now; degrade supply and influence, and report this limitation | Avoid inventing an unplanned error model; market/membership degradation remains a documented extension |
| 2026-07-19 | Relationship between Regime P's terminal state and Regime F's initial state is underspecified | Carrying P's quarter/compliance state into the distinct future CDPA makes phase-in start at q24 and begins near full compliance | Start Regime F as a fresh episode on the same generated entities/true graph, with a common q0 baseline and common random numbers across policies | P is an analogous past regulation, not the same compliance obligation |
