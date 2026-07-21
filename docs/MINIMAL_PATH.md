# Minimal Path Analysis: Which Tools Earned Their Place

This document answers: **if you were to cut the sixteen-tool stack down, what is the minimal path?** It is written with the actual experience of the build. Each row names a tool, which specific limitation demanded it, and what you lose by cutting it. A tool that cannot answer that in one sentence should be removed (or marked optional). Per the guide's closing note, a real project stops after Stage 4 and answers most policy questions with a notebook of Matplotlib charts; this stack is pedagogical and deliberate.

## The Maximal Stack and Its Justifications

| Stage | Tool(s) | Specific Limitation | What You Lose by Cutting |
|---|---|---|---|
| 1 | **pandas / Polars** (+ pyarrow, DuckDB) | None; they are the floor. | Everything. Data ingestion is non-negotiable. |
| 2 | **NetworkX** | None; graph construction is the floor. | The model; firms are nodes and edges are economic links. |
| 3 | **Mesa** (≥3.0 AgentSet API) | ABM heterogeneity cannot be expressed in aggregates. | The agent-based model itself and its per-firm panel data. |
| 3b | **PyTorch** (tensorized ABM fallback) | Multi-step calibration and RL training need fast forward passes; the Mesa ABM does ~40 firms/sec (smoke scales to 2000). | 10³–10⁴× speedup; without it, Stage 4 calibration and Stage 6 emulator training are unaffordable at 2000 firms. |
| 4 | **NumPyro** + **PyMC** (MCMC + SMC-ABC) | Point estimates give one trajectory and false confidence; only Bayesian posterior gives honest intervals. | Honest uncertainty quantification; C1 and all downstream claims depend on parameter posteriors covering θ\*. |
| 5 | **DoWhy** + **EconML** (+ causal-learn, statsmodels/linearmodels) | The policy question is a `do()` query; observational estimates are provably biased (audit targeting confounds). | The right to call the estimate causal; C2 and the DiD grading collapse without refuters catching the naive bias. |
| 6–7 | **PyTorch** (emulator training) + **PyTorch Geometric** (GNN encoder) | 21,000 ensemble rollouts; the ABM does that in ~11 days, the emulator in ~8 minutes. | The scenario ensemble and Stage 14's Sobol screen (what to measure next); this is the *one* specific technical limitation that justifies the emulator. |
| 8 | **Gymnasium** (≥1.0, 5-tuple) | Only because Stage 10 (RL policy learning) exists; the ABM/emulator need a standard environment interface. | Standard control flow for RL training; without it, Stage 10 loops must hand-roll policy interactions. |
| 9 | **PettingZoo** | Only if firms are strategic learners; multi-agent coordination requires a MARL environment. | Strategic firm behavior (Stage 9–10); if C6 comes back null (MARL changes nothing), this stage was unnecessary — mark it so. |
| 10 | **SB3** + **TorchRL** (+ RLlib opt.) | The policy question asks "what should the regulator do?" — a planning problem, not forecasting. | Optimized regulatory policy; if the question is "what will happen?" a world model alone suffices (you lose the policy search). |
| 11 | **Ray** | 21,000 rollouts in sequence take a week; Ray brings it to an afternoon on a cluster. | Parallelization; without it, the ensemble runs serially and blows the wall-clock budget. |
| 12 | **Hydra** (+ OmegaConf, Pydantic) | 90+ configuration hyperparameters across 17 stages; version control + reproducibility is impossible without a declarative config system. | Configuration sanity; without it, you lose the ability to reproduce a run or swap profile/DGP/calibration method in one line. |
| 13 | **MLflow** (local file backend) | 90 runs and the question of which one made Figure 5; no credentials/network dependency needed for one-command portability. | Experiment tracking; without it, you lose git commit hashes, wall-clock times, and the connection between a figure and the run that produced it. |
| 14 | **SALib** + **Optuna** | The client's follow-up after C5 is "what should we measure next?"; Morris screening prunes 16 parameters to 8, Sobol quantifies the rest. | Actionable sensitivity indices; without it, you answer "optimize the whole parameter space" instead of "measure these 3 variables." |
| 15 | **Plotly** + **Streamlit** | The client is a policy team, not a Jupyter audience; interactive dashboards are the delivery mechanism. | Client adoption; a notebook stays a draft; Streamlit makes it an instrument. |
| 16 | **pytest** + **Docker** + **GitHub Actions** (+ ruff, mypy, pre-commit, uv, Make) | Six months later the client returns with new data; without automated gates, a breaking change slips in undetected. | The right to call it an instrument; without CI, a one-command deployment breaks silently under version changes. |

## The Teachable Fault Lines

- **After Stage 4:** A real project stops after Bayesian calibration (honest parameter estimates) and answers most policy questions with Matplotlib + a notebook. Everything above is for scenarios, causal validation, and optimization — justified only if the client asks "what if?" or "what should we do?"

- **After Stage 5:** If DoWhy's refuters confirm the naive estimate was biased by ≤10%, you can drop Stages 6–14 and report the observational finding with appropriate caveats. The emulator is built only if ensemble risk quantification matters.

- **After Stage 9:** If MARL (Stage 9–10) is run and C6 returns null (strategic agents do not change C5), flag it as unnecessary and note it for the next build.

- **After Stage 11:** If the question is forecasting only (not policy optimization), Stages 10–11 (RL + ensemble) can be replaced with a deterministic policy suite and Monte Carlo sampling — much cheaper and still answers "what happens under uniform enforcement vs. phased."

- **Before Stage 15:** Plotly/Streamlit are only for delivery; for internal validation, Matplotlib + Jupyter suffice.

- **Before Stage 16:** Docker/CI are only for production deployments; for research, local `make test` passes.

## What This Stack Actually Buys

The pedagogical value of the full stack is not efficiency — a real project would stop earlier — but **integrity**. Each tool solves one specific problem:

- **Calibration** answers "are the parameters recoverable?" not "what are the best guesses?"
- **Causal inference** asks "do we know the intervention worked?" not "was the point estimate close?"
- **Emulation** enables "21,000 scenarios at scale" instead of "8 scenarios in a week."
- **RL** converts "here are scripted policies" to "here is the policy that maximizes your objective under uncertainty."
- **Sensitivity** converts "optimize all 16 parameters" to "measure these 3 variables and you win 80% of the benefit."
- **Visualization** converts "see the dataframe" to "here is the decision under every plausible parameter draw."

Remove any tool and you lose one of these guarantees, usually silently.

## Advice for a Real Project

1. **Build Stages 1–4 first.** Calibration with honest intervals answers 70% of policy questions. Stop and write up if the posterior is wide or the model is visibly misspecified.
2. **Add Stage 5 only if you suspect confounding.** If administrative data are clean (audit is exogenous), the observational estimate is defensible; if there is a backdoor, DoWhy's refuters will tell you.
3. **Add Stages 6–7 (emulator) only if you need >500 rollouts.** If a policy sweep of 50 scenarios is enough, the ABM is fast enough; emulation is for risk quantification.
4. **Add Stage 10–11 (RL + ensemble) only if the client asks "what should we do?"** Forecasting models never need a policy learner.
5. **Skip Stage 9 (MARL) in the first pass.** Strategic firms add complexity; if the ABM without them answers the question, stop.
6. **Keep Stages 12–13 (Hydra + MLflow) from the start.** Configuration + tracking cost nothing and save weeks of debugging.
7. **Add Stage 14 (sensitivity) only if the client cares about measurement strategy.** If they want a point forecast, Sobol is overkill.
8. **Add Stage 15 (Streamlit) only for delivery.** Matplotlib is fine for internal validation.
9. **Add Stage 16 (Docker + CI) only if the model will be re-run on new data.** For one-off research, local tests suffice.

The guide's closing note stands: this sixteen-tool stack is maximal by design, chosen for pedagogy. A minimal path for a real policy question starts at Stage 1, stops at Stage 4, and adds only the stages that answer a specific client question.
