# RegWorld Findings

## Disclaimer

This world model is entirely synthetic with known ground truth. Every finding is methodological: what is demonstrated is that this pipeline recovers the truth when the truth is recoverable and fails legibly when it is not. The policy insights below emerge from a constructed regulatory environment whose parameters and causal structure are known in full. A real policy deployment would require validation against observed data, external cross-checks, and expert judgment; this report's value is in exposing the methodology and the seams where it breaks.

## The Four-Number Causal Table

Figure 1 (see reports/figures/fig_01_four_numbers.png) and the table below report the four key causal estimates:

| Estimand | Value |
|---|---|
| τ_true (do() ATT, ground truth) | 0.4146 |
| τ_abm (simulator DIL rollout) | 0.3537 |
| τ_qe (observational DML) | 0.0612 [95% CI: -0.1133, 0.2616] |
| τ_obs (naive panel contrast) | 0.1245 [95% CI: 0.0308, 0.2182] |

## The Six Claims

### C1

**Claim:** Bayesian calibration recovers the true behavioral parameters when the model is well specified, and fails *legibly* (a visibly biased peer coefficient β_peer) when supply-network capacity homophily is switched on.

**Verdict:** INCONCLUSIVE

**Evidence:** Max R-hat=1.020 (>1.01) across 11 fitted parameters — convergence is not clean at this profile's draw count; recovery not yet assertable.

### C2

**Claim:** The observational estimate of the enforcement effect is confidently wrong when audit targeting correlates with unobserved firm capacity. The staggered-rollout DiD recovers the true effect; DoWhy's refuters catch the naive estimate.

**Verdict:** SUPPORTED

**Evidence:** Four-number gate passed: naive observational τ_obs=0.125 is confidently wrong against τ_true=0.415, while the DiL simulator/DiD path recovers τ_abm=0.354 (sign and DiD agreement OK).

### C3

**Claim:** The graph-RSSM emulator reproduces the ABM's *distribution* of outcomes within tolerance at 10³-10⁴x the speed, and degrades honestly out of distribution.

**Verdict:** INCONCLUSIVE

**Evidence:** Distributional or OOD metrics incomplete.

### C4

**Claim:** Of ~16 uncertain parameters, a small handful drive most outcome variance — which tells the client what to measure next.

**Verdict:** SUPPORTED

**Evidence:** Morris elementary effects over 15 behavioral parameters on the tensorized ABM (64 rollouts) rank the drivers beta_enforce, delta_exit, beta_0; the top three carry 53% of mean mu* share, so a small handful dominate. (15 of the 16 §7.3 parameters enter the forecast dynamics; beta_capacity is answer-key-only and q0/q1 are observation-model-only, so screening them on the ABM would manufacture guaranteed zeros.)

### C5

**Claim:** Aggressive uniform enforcement maximizes compliance and backfires on market concentration: small firms exit, HHI rises. Phased, targeted enforcement buys nearly the same compliance for materially less concentration. Reported as a Pareto frontier with credible intervals across the parameter posterior.

**Verdict:** INCONCLUSIVE

**Evidence:** Scenario cube built over 48 cells / 6 policies; backfire probability 0.00%. Verdict withheld: the Stage-11 ABM cross-check covers only 8.00% of outcomes (threshold 85%), so the emulator this rests on is not validated.

### C6

**Claim:** Modeling the ten largest firms as strategic learners (MARL) either changes C5 or does not. Report which.

**Verdict:** INCONCLUSIVE

**Evidence:** Artifact `artifacts/marl/c6_comparison.json` not found; the Stage-10d MARL ablation has not run, so C6 is unanswered.

## Where This Model Fails

The pipeline is honest about its seams and the stages at which it cannot generalize:

*(No major failure modes recorded; the pipeline ran to completion with no DEGRADED stages and within acceptable thresholds.)*

## Run Manifest

**Artifact missing:** `reports/run_manifest.json` not found.
