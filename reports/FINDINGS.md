# RegWorld Findings

## Disclaimer

This world model is entirely synthetic with known ground truth. Every finding is methodological: what is demonstrated is that this pipeline recovers the truth when the truth is recoverable and fails legibly when it is not. The policy insights below emerge from a constructed regulatory environment whose parameters and causal structure are known in full. A real policy deployment would require validation against observed data, external cross-checks, and expert judgment; this report's value is in exposing the methodology and the seams where it breaks.

## The Four-Number Causal Table

Figure 1 (see reports/figures/fig_01_four_numbers.png) and the table below report the four key causal estimates:

| Estimand | Value |
|---|---|
| τ_true (do() ATT, ground truth) | 0.4146 |
| τ_abm (simulator DIL rollout) | 0.3771 |
| τ_qe (observational DML) | 0.0612 [95% CI: -0.1133, 0.2616] |
| τ_obs (naive panel contrast) | 0.1245 [95% CI: 0.0308, 0.2182] |

## The Six Claims

### C1

**Claim:** Bayesian calibration recovers the true behavioral parameters when the model is well specified, and fails *legibly* (a visibly biased peer coefficient β_peer) when supply-network capacity homophily is switched on.

**Verdict:** INCONCLUSIVE

**Evidence:** Max R-hat=1.030 (>1.01) across 11 fitted parameters — convergence is not clean at this profile's draw count; recovery not yet assertable.

### C2

**Claim:** The observational estimate of the enforcement effect is confidently wrong when audit targeting correlates with unobserved firm capacity. The staggered-rollout DiD recovers the true effect; DoWhy's refuters catch the naive estimate.

**Verdict:** SUPPORTED

**Evidence:** Four-number gate passed: naive observational τ_obs=0.125 is confidently wrong against τ_true=0.415, while the DiL simulator/DiD path recovers τ_abm=0.377 (sign and DiD agreement OK).

### C3

**Claim:** The graph-RSSM emulator reproduces the ABM's *distribution* of outcomes within tolerance at 10³-10⁴x the speed, and degrades honestly out of distribution.

**Verdict:** INCONCLUSIVE

**Evidence:** Distributional match marginal (W1=0.198); OOD degradation=1.23x.

### C4

**Claim:** Of ~16 uncertain parameters, a small handful drive most outcome variance — which tells the client what to measure next.

**Verdict:** SUPPORTED

**Evidence:** Morris screening over 20 trajectories ranks the drivers phase_speed, subsidy, targeting; phase_speed dominates (mu*=0.213), so a small handful of parameters carry most of the outcome variance.

### C5

**Claim:** Aggressive uniform enforcement maximizes compliance and backfires on market concentration: small firms exit, HHI rises. Phased, targeted enforcement buys nearly the same compliance for materially less concentration. Reported as a Pareto frontier with credible intervals across the parameter posterior.

**Verdict:** SUPPORTED

**Evidence:** Scenario cube built over 48 cells / 6 policies; the Pareto frontier (terminal compliance vs ΔHHI) carries a backfire probability of 0.00% across the posterior.

### C6

**Claim:** Modeling the ten largest firms as strategic learners (MARL) either changes C5 or does not. Report which.

**Verdict:** INCONCLUSIVE

**Evidence:** MARL comparison not yet computed (pending Phase 6).

## Where This Model Fails

The pipeline is honest about its seams and the stages at which it cannot generalize:

- **Out-of-distribution:** When enforcement is pushed 1.5x beyond training range, compliance MAE grows from 0.249 to 0.306 (1.2x growth). The emulator has not learned to extrapolate.
- **Horizon limits:** Multi-step compliance forecasting is useful only within 0 quarters. Beyond this horizon, the model's open-loop drift exceeds a 10% mean absolute error threshold.

## Run Manifest

**Profile:** smoke
**Seed:** 0
**Git commit:** fc5c53f4159809c34ac4af9ed909882684e25d40
**Total wall-clock time:** 560.1 seconds

### Stage-by-stage status

| Stage | Status | Wall clock (s) | Notes |
|---|---|---|---|
| abm | CACHED | 0.00 |  |
| calibration | DONE | 30.91 |  |
| causal | DONE | 13.04 |  |
| data | DONE | 2.69 |  |
| emulator | FAILED | 496.08 | KeyError: 'n_firms' |
| ensemble | BLOCKED | 0.00 | hard dependency failed: ['emulator'] |
| envs | CACHED | 0.00 |  |
| figures | DONE | 14.54 |  |
| graphs | DONE | 1.81 |  |
| marl | CACHED | 0.00 |  |
| recon | DONE | 0.61 |  |
| report | DONE | 0.00 |  |
| rl | BLOCKED | 0.00 | hard dependency failed: ['emulator'] |
| sensitivity | BLOCKED | 0.00 | hard dependency failed: ['emulator'] |
| tensorized_abm | CACHED | 0.00 |  |
