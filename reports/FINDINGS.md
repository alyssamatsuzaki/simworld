# SimWorld Findings

## Disclaimer

This world model is entirely synthetic with known ground truth. Every finding is methodological: what is demonstrated is that this pipeline recovers the truth when the truth is recoverable and fails legibly when it is not. The policy insights below emerge from a constructed regulatory environment whose parameters and causal structure are known in full. A real policy deployment would require validation against observed data, external cross-checks, and expert judgment; this report's value is in exposing the methodology and the seams where it breaks.

## The Four-Number Causal Table

Figure 1 (see reports/figures/fig01_four_numbers.png) and the table below report the four key causal estimates:

| Estimand | Value |
|---|---|
| τ_true (do() ATT, ground truth) | 0.4146 |
| τ_abm (simulator DIL rollout) | 0.3653 |
| τ_qe (observational DML) | 0.0612 [95% CI: -0.1133, 0.2616] |
| τ_obs (naive panel contrast) | 0.1245 [95% CI: 0.0308, 0.2182] |

## The Six Claims

### C1

**Claim:** Bayesian calibration recovers the true behavioral parameters when the model is well specified, and fails *legibly* (a visibly biased peer coefficient β_peer) when supply-network capacity homophily is switched on.

**Verdict:** INCONCLUSIVE

**Evidence:** Max R-hat=1.020 across 11 fitted parameters, divergences=0, 15/17 parameters cover θ* at 90% — convergence is not clean at this profile's draw count; recovery not yet assertable (dev-profile gate). Under confounded, β_peer covers truth (the C1 failure half).

### C2

**Claim:** The observational estimate of the enforcement effect is confidently wrong when audit targeting correlates with unobserved firm capacity. The staggered-rollout DiD recovers the true effect; DoWhy's refuters catch the naive estimate.

**Verdict:** SUPPORTED

**Evidence:** Four-number gate passed: naive observational τ_obs=0.125 is confidently wrong against τ_true=0.415, while the DiL simulator/DiD path recovers τ_abm=0.365 (sign and DiD agreement OK).

### C3

**Claim:** The graph-RSSM emulator reproduces the ABM's *distribution* of outcomes within tolerance at 10³-10⁴x the speed, and degrades honestly out of distribution.

**Verdict:** INCONCLUSIVE

**Evidence:** W1 distance=0.178, OOD error growth=1.45x, but the Stage-11 ABM cross-check covers only 0.00% of outcomes (threshold 85%), so the emulator this rests on is not validated.

### C4

**Claim:** Of ~16 uncertain parameters, a small handful drive most outcome variance — which tells the client what to measure next.

**Verdict:** SUPPORTED

**Evidence:** Morris elementary effects over 15 behavioral parameters on the tensorized ABM (64 rollouts) rank the drivers beta_enforce, beta_0, delta_exit; the top three carry 53% of mean mu* share, so a small handful dominate. (15 of the 16 §7.3 parameters enter the forecast dynamics; beta_capacity is answer-key-only and q0/q1 are observation-model-only, so screening them on the ABM would manufacture guaranteed zeros.)

### C5

**Claim:** Aggressive uniform enforcement maximizes compliance and backfires on market concentration: small firms exit, HHI rises. Phased, targeted enforcement buys nearly the same compliance for materially less concentration. Reported as a Pareto frontier with credible intervals across the parameter posterior.

**Verdict:** INCONCLUSIVE

**Evidence:** Scenario cube built over 48 cells / 6 policies; backfire probability 0.00%. Verdict withheld: the Stage-11 ABM cross-check covers only 0.00% of outcomes (threshold 85%), so the emulator this rests on is not validated.

### C6

**Claim:** Modeling the ten largest firms as strategic learners (MARL) either changes C5 or does not. Report which.

**Verdict:** INCONCLUSIVE

**Evidence:** Artifact `artifacts/marl/c6_comparison.json` not found; the Stage-10d MARL ablation has not run, so C6 is unanswered.

## Where This Model Fails

The pipeline is honest about its seams and the stages at which it cannot generalize:

- **Out-of-distribution:** When enforcement is pushed 1.5x beyond training range, compliance MAE grows from 0.201 to 0.291 (1.5x growth). The emulator has not learned to extrapolate.
- **Horizon limits:** Multi-step compliance forecasting is useful only within 0 quarters. Beyond this horizon, the model's open-loop drift exceeds a 10% mean absolute error threshold.
- **Emulator exploitation:** the Dreamer policy's exploitation gap J_emulator - J_ABM is +2.5% (within the 15% budget) (J_emulator=12.934 vs J_ABM=12.624) — the planner steers into the model's errors to exactly the extent this gap is positive.


### Pending manual verification

- **Streamlit OOD banner (§18):** launch `make dashboard`, set enforcement and targeting sliders to 1.0 — the banner must turn red ("OUT OF DISTRIBUTION: Mahalanobis distance … exceeds …"); return them to enforcement 0.5 / targeting -0.5 and it must go green ("In distribution"). The dashboard is confirmed to launch headless without error; this reactivity check is the single item that requires a human.

## Run Manifest

**Profile:** smoke
**Seed:** 0
**Git commit:** 8c9623cb46eb4161b80aac9e38ab9bd588718d6b
**Total wall-clock time:** 860.4 seconds

### Stage-by-stage status

| Stage | Status | Wall clock (s) | Notes |
|---|---|---|---|
| abm | DONE | 0.10 |  |
| calibration | DONE | 68.82 |  |
| causal | DONE | 69.49 |  |
| data | DONE | 5.20 |  |
| emulator | DONE | 438.12 |  |
| ensemble | DONE | 10.62 |  |
| envs | DONE | 0.26 |  |
| evaluation | DONE | 117.38 |  |
| figures | DONE | 19.71 |  |
| graphs | DONE | 1.63 |  |
| marl | DONE | 0.02 |  |
| recon | DONE | 0.57 |  |
| report | DONE | 0.00 |  |
| rl | DONE | 45.93 |  |
| sensitivity | DONE | 82.41 |  |
| tensorized_abm | DONE | 0.08 |  |
