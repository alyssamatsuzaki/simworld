"""Four ways to estimate a treatment effect on the observational panel (5c).

Two effects live here. The *audit* effect (do(audited) -> perceived_risk ->
compliant_next) is the one the planted confounder ``z`` biases: naive logit and
double-ML on observed controls both return a precise estimate of the wrong number,
and only conditioning on the (unobserved) ``z`` closes the gap. The *onset* effect
is what the four-number gate calls ``tau_obs`` - the DML number a careless analyst
would report for turning enforcement on, biased because it ignores the staggered
timing that the DiD exploits.

The software will still return a number, formatted to six decimal places.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# Observed controls the analyst conditions on for the audit effect. `perceived_risk`
# is deliberately excluded: it is the mediator, and conditioning on it would block
# the very effect we are trying to estimate.
_AUDIT_CONTROLS = (
    "cost_share",
    "neighbor_compliant_share",
    "assoc_compliant_share",
    "privacy_rev_share",
    "phase_phi",
    "compliant_lag",
    "log_size_proxy",
    "size_decile",
    "data_intensity",
    "cost_index",
)
_ONSET_CONTROLS = (
    "cost_share",
    "neighbor_compliant_share",
    "assoc_compliant_share",
    "privacy_rev_share",
    "compliant_lag",
    "log_size_proxy",
    "size_decile",
    "data_intensity",
    "cost_index",
)


@dataclass(frozen=True)
class PointEstimate:
    """An effect estimate on the probability scale with a 95% CI."""

    estimate: float
    ci_low: float
    ci_high: float
    method: str

    @property
    def se(self) -> float:
        return (self.ci_high - self.ci_low) / (2.0 * 1.959964)


@dataclass(frozen=True)
class CateByGroup:
    """Average CATE within size deciles and sectors (5c-4)."""

    by_size_decile: dict[int, float]
    by_sector: dict[int, float]


def _ensure_econml_importable() -> None:
    """Stub ``shap`` before importing econml (DEVIATIONS 2026-07-20).

    econml eagerly imports shap, whose 0.46 colour-conversion module crashes on
    numpy 2.4. shap is only used inside ``.shap_values()``, which we never call, so
    a placeholder module lets econml import while keeping pandas at 2.x.
    """
    import sys
    import types

    if "shap" in sys.modules:
        return
    try:
        import shap  # noqa: F401
    except Exception:
        sys.modules["shap"] = types.ModuleType("shap")


def _matrix(panel: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([panel[c].to_numpy().astype(np.float64) for c in columns])


def naive_logit_audit(panel: pl.DataFrame) -> PointEstimate:
    """Logit of report on audited_prev + observed controls; average marginal effect.

    Biased for the audit effect because ``z`` is omitted and correlates with both
    size (hence audit targeting) and compliance.
    """
    import statsmodels.api as sm

    y = panel["outcome_reported"].to_numpy().astype(np.float64)
    t = panel["audited_prev"].to_numpy().astype(np.float64)
    controls = _matrix(panel, _AUDIT_CONTROLS)
    design = np.column_stack([np.ones(len(y)), t, controls])
    model = sm.Logit(y, design)
    fit = model.fit(disp=0, maxiter=200)
    # Average marginal effect of the treatment on the probability scale.
    beta = fit.params
    lin = design @ beta
    p = 1.0 / (1.0 + np.exp(-lin))
    ame = float(np.mean(p * (1.0 - p) * beta[1]))
    # Delta-method SE for the treatment coefficient scaled by mean p(1-p).
    scale = float(np.mean(p * (1.0 - p)))
    se_coef = float(fit.bse[1])
    se = scale * se_coef
    return PointEstimate(ame, ame - 1.959964 * se, ame + 1.959964 * se, "naive_logit_ame")


def _ridge_regressor() -> object:
    """Cross-validated ridge: fast, deterministic nuisance model for DML residuals."""
    from sklearn.linear_model import RidgeCV

    return RidgeCV(alphas=np.logspace(-3.0, 3.0, 7))


def _fit_linear_dml(
    y: np.ndarray, t: np.ndarray, w: np.ndarray, *, seed: int, method: str
) -> PointEstimate:
    _ensure_econml_importable()
    from econml.dml import LinearDML
    from sklearn.linear_model import LogisticRegression

    est = LinearDML(
        model_y=_ridge_regressor(),
        model_t=LogisticRegression(max_iter=500),
        discrete_treatment=True,
        cv=3,
        random_state=seed,
    )
    est.fit(y, t, X=None, W=w)
    ate = float(np.asarray(est.ate(X=None)).reshape(-1)[0])
    low, high = est.ate_interval(X=None, alpha=0.05)
    return PointEstimate(
        ate,
        float(np.asarray(low).reshape(-1)[0]),
        float(np.asarray(high).reshape(-1)[0]),
        method,
    )


def dml_audit(
    panel: pl.DataFrame, *, seed: int = 0, extra_control: np.ndarray | None = None
) -> PointEstimate:
    """LinearDML for the audit effect on observed controls (5c-2).

    Pass ``extra_control`` (the sealed true ``z``, supplied by an allowlisted grader)
    to close the backdoor: the estimate then moves toward the truth, which is the point.
    """
    y = panel["outcome_reported"].to_numpy().astype(np.float64)
    t = panel["audited_prev"].to_numpy().astype(np.float64)
    w = _matrix(panel, _AUDIT_CONTROLS)
    if extra_control is not None:
        w = np.column_stack([w, np.asarray(extra_control, dtype=np.float64)])
    method = "dml_audit_full" if extra_control is not None else "dml_audit_observed"
    return _fit_linear_dml(y, t, w, seed=seed, method=method)


def dml_onset(panel: pl.DataFrame, *, seed: int = 0) -> PointEstimate:
    """LinearDML for the enforcement-onset effect ignoring staggered timing (tau_obs).

    Treatment is the post-onset indicator; controls are observed firm covariates but
    *not* calendar time, so the secular compliance rise contaminates the estimate.
    """
    y = panel["outcome_reported"].to_numpy().astype(np.float64)
    t = ((panel["treated"] == 1.0) & (panel["event_time"] >= 0)).to_numpy().astype(np.float64)
    w = _matrix(panel, _ONSET_CONTROLS)
    return _fit_linear_dml(y, t, w, seed=seed, method="dml_onset_observed")


def cate_by_group(panel: pl.DataFrame, *, seed: int = 0) -> CateByGroup:
    """Causal-forest audit CATE averaged by size decile and sector (5c-4)."""
    _ensure_econml_importable()
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LogisticRegression

    y = panel["outcome_reported"].to_numpy().astype(np.float64)
    t = panel["audited_prev"].to_numpy().astype(np.float64)
    features = _matrix(panel, ("size_decile", "data_intensity", "cost_index", "sector"))
    w = _matrix(panel, _AUDIT_CONTROLS)
    est = CausalForestDML(
        model_y=_ridge_regressor(),
        model_t=LogisticRegression(max_iter=500),
        discrete_treatment=True,
        cv=3,
        n_estimators=200,
        random_state=seed,
    )
    est.fit(y, t, X=features, W=w)
    effects = np.asarray(est.effect(features)).reshape(-1)
    size_decile = panel["size_decile"].to_numpy()
    sector = panel["sector"].to_numpy()
    by_size = {
        int(d): float(effects[size_decile == d].mean()) for d in sorted(set(size_decile.tolist()))
    }
    by_sector = {int(s): float(effects[sector == s].mean()) for s in sorted(set(sector.tolist()))}
    return CateByGroup(by_size, by_sector)
