"""§14 Stage 14: fast hermetic tests for sensitivity analysis.

Covers the two distinct designs Stage 14a specifies — the Morris screen over the §7.3
behavioral parameters theta (claim C4) and the Sobol analysis over the policy levers —
plus the Ishigami recovery gate (§11 family 12) and Optuna. No dependency on artifacts/
or a real checkpoint.
"""

from __future__ import annotations

import numpy as np
import optuna
from SALib.analyze.morris import analyze as morris_analyze
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.morris import sample as morris_sample
from SALib.sample.sobol import sample as sobol_sample

from regworld.rules import Theta
from regworld.sensitivity.screen import (
    THETA_NAMES,
    _salib_problem,
    _theta_from_vector,
    _theta_problem,
    theta_bounds,
)
from regworld.training.datamodule import ACTION_HIGH, ACTION_LOW


class TestSALibProblem:
    """Validate the SALib problem dict."""

    def test_problem_dict_well_formed(self) -> None:
        problem = _salib_problem()
        assert problem["num_vars"] == 4
        assert len(problem["names"]) == 4
        assert problem["names"] == ["enforcement", "targeting", "phase_speed", "subsidy"]
        assert len(problem["bounds"]) == 4

    def test_problem_bounds_correct(self) -> None:
        problem = _salib_problem()
        expected_low = [float(x) for x in ACTION_LOW]
        expected_high = [float(x) for x in ACTION_HIGH]
        for i, (name, bounds) in enumerate(zip(problem["names"], problem["bounds"], strict=True)):
            assert bounds[0] == expected_low[i], f"{name} lower bound mismatch"
            assert bounds[1] == expected_high[i], f"{name} upper bound mismatch"


class TestMorrisScreening:
    """Morris screening on a toy objective."""

    def test_morris_sample_shape(self) -> None:
        problem = _salib_problem()
        samples = morris_sample(problem, N=4, num_levels=4, seed=0)
        # Morris gives (D+1) * N samples
        expected_count = (problem["num_vars"] + 1) * 4
        assert samples.shape[0] == expected_count
        assert samples.shape[1] == problem["num_vars"]

    def test_morris_within_bounds(self) -> None:
        problem = _salib_problem()
        samples = morris_sample(problem, N=4, num_levels=4, seed=0)
        for i, (name, bounds) in enumerate(zip(problem["names"], problem["bounds"], strict=True)):
            assert np.all(samples[:, i] >= bounds[0]), f"{name} samples below lower bound"
            assert np.all(samples[:, i] <= bounds[1]), f"{name} samples above upper bound"

    def test_morris_analyze_output(self) -> None:
        problem = _salib_problem()
        samples = morris_sample(problem, N=4, num_levels=4, seed=0)
        # Toy objective: linear in enforcement, zero in others
        outputs = samples[:, 0] + 0.1 * np.random.RandomState(0).randn(len(samples))
        result = morris_analyze(problem, samples, outputs, seed=0)

        assert "mu" in result
        assert "sigma" in result
        assert "mu_star" in result
        assert len(result["mu"]) == 4
        assert np.all(np.isfinite(result["mu"]))


class TestSobolAnalysis:
    """Sobol first/total-order indices on a toy objective."""

    def test_sobol_sample_shape(self) -> None:
        problem = _salib_problem()
        samples = sobol_sample(problem, N=64, calc_second_order=False, seed=0)
        # SALib Sobol sample shape (N * (2*D+2) is for Saltelli, but may vary)
        assert samples.shape[0] > 0
        assert samples.shape[1] == problem["num_vars"]

    def test_sobol_within_bounds(self) -> None:
        problem = _salib_problem()
        samples = sobol_sample(problem, N=64, calc_second_order=False, seed=0)
        for i, (name, bounds) in enumerate(zip(problem["names"], problem["bounds"], strict=True)):
            assert np.all(samples[:, i] >= bounds[0]), f"{name} samples below lower bound"
            assert np.all(samples[:, i] <= bounds[1]), f"{name} samples above upper bound"

    def test_sobol_analyze_output(self) -> None:
        """Shape/ordering contract on the lever problem with a nonlinear toy response.

        This is deliberately NOT the Ishigami check — the lever bounds are not
        U(-pi, pi) and there are four factors, not three. The real Ishigami recovery
        test lives in :class:`TestIshigamiRecovery`.
        """
        problem = _salib_problem()
        samples = sobol_sample(problem, N=64, calc_second_order=False, seed=0)
        outputs = (
            np.sin(3.0 * samples[:, 0])
            + 2.0 * samples[:, 1] ** 2
            + 0.5 * samples[:, 2] * samples[:, 3]
        )
        result = sobol_analyze(problem, outputs, seed=0, calc_second_order=False)

        assert "S1" in result
        assert "ST" in result
        assert len(result["S1"]) == 4
        assert np.all(np.isfinite(result["S1"]))
        assert np.all(np.isfinite(result["ST"]))
        # ST >= S1 within Monte Carlo error
        assert np.all(result["ST"] >= result["S1"] - 0.15)

    def test_sobol_indices_in_range(self) -> None:
        problem = _salib_problem()
        samples = sobol_sample(problem, N=32, calc_second_order=False, seed=0)
        outputs = np.sum(samples, axis=1)  # Simple linear combination
        result = sobol_analyze(problem, outputs, seed=0, calc_second_order=False)

        assert np.all(result["S1"] >= 0.0)
        assert np.all(result["S1"] <= 1.0)
        assert np.all(result["ST"] >= 0.0)
        assert np.all(result["ST"] <= 1.0)


# --------------------------------------------------------------------------------------
# The Ishigami gate (PLAN Stage 14a acceptance tests, §11 family 12)
# --------------------------------------------------------------------------------------

ISHIGAMI_A = 7.0
ISHIGAMI_B = 0.1

# Analytic Sobol indices of the Ishigami function on U(-pi, pi)^3 at a=7, b=0.1.
ISHIGAMI_S1_TRUE = np.array([0.3139, 0.4424, 0.0])
ISHIGAMI_ST_TRUE = np.array([0.5576, 0.4424, 0.2437])

# Empirically achieved at N=8192 is max|error| < 0.002 across seeds 0-5; 0.01 leaves
# headroom for SALib sampler changes without being satisfiable by a wrong wiring
# (a mis-scaled or mis-ordered design misses by >0.05).
ISHIGAMI_N = 8192
ISHIGAMI_TOL = 0.01


def ishigami_problem() -> dict[str, object]:
    """The canonical Ishigami problem: exactly 3 factors, each U(-pi, pi)."""
    return {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
    }


def ishigami(x: np.ndarray, a: float = ISHIGAMI_A, b: float = ISHIGAMI_B) -> np.ndarray:
    """f(x) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1)."""
    return (
        np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * (x[:, 2] ** 4) * np.sin(x[:, 0])
    ).astype(np.float64)


class TestIshigamiRecovery:
    """Sobol must recover the known Ishigami indices — this catches sampler wiring bugs."""

    def test_ishigami_problem_is_canonical(self) -> None:
        problem = ishigami_problem()
        assert problem["num_vars"] == 3
        assert len(problem["bounds"]) == 3
        for lo, hi in problem["bounds"]:
            assert lo == -np.pi
            assert hi == np.pi

    def test_sobol_recovers_ishigami_first_order(self) -> None:
        problem = ishigami_problem()
        samples = sobol_sample(problem, N=ISHIGAMI_N, calc_second_order=False, seed=0)
        # Saltelli without second order: N * (D + 2) rows.
        assert samples.shape == (ISHIGAMI_N * (3 + 2), 3)
        result = sobol_analyze(problem, ishigami(samples), calc_second_order=False, seed=0)

        s1 = np.asarray(result["S1"], dtype=np.float64)
        errors = np.abs(s1 - ISHIGAMI_S1_TRUE)
        assert np.all(errors < ISHIGAMI_TOL), (
            f"S1 = {s1} vs analytic {ISHIGAMI_S1_TRUE} (abs error {errors})"
        )

    def test_sobol_recovers_ishigami_total_order(self) -> None:
        problem = ishigami_problem()
        samples = sobol_sample(problem, N=ISHIGAMI_N, calc_second_order=False, seed=0)
        result = sobol_analyze(problem, ishigami(samples), calc_second_order=False, seed=0)

        st = np.asarray(result["ST"], dtype=np.float64)
        errors = np.abs(st - ISHIGAMI_ST_TRUE)
        assert np.all(errors < ISHIGAMI_TOL), (
            f"ST = {st} vs analytic {ISHIGAMI_ST_TRUE} (abs error {errors})"
        )

    def test_ishigami_x3_is_pure_interaction(self) -> None:
        """x3 has zero first-order effect but a large total effect — the diagnostic case."""
        problem = ishigami_problem()
        samples = sobol_sample(problem, N=ISHIGAMI_N, calc_second_order=False, seed=0)
        result = sobol_analyze(problem, ishigami(samples), calc_second_order=False, seed=0)

        s1 = np.asarray(result["S1"], dtype=np.float64)
        st = np.asarray(result["ST"], dtype=np.float64)
        assert abs(s1[2]) < ISHIGAMI_TOL
        assert st[2] > 0.2
        assert np.all(st >= s1 - ISHIGAMI_TOL)


# --------------------------------------------------------------------------------------
# Stage 14a-i — the theta screen (claim C4)
# --------------------------------------------------------------------------------------


class TestThetaScreenProblem:
    """The Morris screen must run over the behavioral parameters, not the policy levers."""

    def test_theta_problem_is_the_behavioral_parameters(self) -> None:
        problem = _theta_problem()
        assert problem["num_vars"] == len(THETA_NAMES)
        assert problem["num_vars"] >= 15, "claim C4 screens the ~16 behavioral parameters"
        assert problem["names"] == list(THETA_NAMES)
        assert len(problem["bounds"]) == problem["num_vars"]
        # These are theta, not levers.
        assert "enforcement" not in problem["names"]
        assert "beta_enforce" in problem["names"]

    def test_theta_names_are_real_theta_fields(self) -> None:
        fields = set(Theta.__dataclass_fields__)
        for name in THETA_NAMES:
            assert name in fields, f"{name} is not a rules.Theta field"

    def test_theta_names_cover_both_prior_groups(self) -> None:
        theta = Theta()
        screened = set(THETA_NAMES)
        # Every Group B parameter is screenable.
        assert set(theta.group_b_names()) <= screened
        # Group A minus the observation-model nuisance pair (q0/q1 do not enter dynamics).
        group_a = set(theta.group_a_names()) - {"q0", "q1"}
        assert group_a <= screened
        # beta_capacity is answer-key-only and absent from the fitted model (PLAN 7.3).
        assert "beta_capacity" not in screened

    def test_theta_bounds_bracket_the_prior_centres(self) -> None:
        bounds = theta_bounds()
        assert len(bounds) == len(THETA_NAMES)
        for name, (lo, hi) in zip(THETA_NAMES, bounds, strict=True):
            assert hi > lo, f"{name} has a degenerate box"
            assert np.isfinite(lo) and np.isfinite(hi), f"{name} has a non-finite bound"
        by_name = dict(zip(THETA_NAMES, bounds, strict=True))
        # HalfNormal / Beta supports are non-negative.
        for name in ("beta_enforce", "beta_cost", "beta_customer", "beta_stick"):
            assert by_name[name][0] >= 0.0
        for name in ("gamma_scale", "ell_learn", "alpha_trust", "rho_influence"):
            assert 0.0 <= by_name[name][0] < by_name[name][1] <= 1.0

    def test_theta_morris_samples_within_bounds(self) -> None:
        problem = _theta_problem()
        samples = morris_sample(problem, N=4, num_levels=4, seed=0)
        assert samples.shape == (4 * (problem["num_vars"] + 1), problem["num_vars"])
        for i, (name, (lo, hi)) in enumerate(zip(problem["names"], problem["bounds"], strict=True)):
            assert np.all(samples[:, i] >= lo), f"{name} below prior box"
            assert np.all(samples[:, i] <= hi), f"{name} above prior box"

    def test_theta_from_vector_binds_every_screened_field(self) -> None:
        problem = _theta_problem()
        row = morris_sample(problem, N=4, num_levels=4, seed=0)[0]
        base = Theta()
        bound = _theta_from_vector(base, row)
        for name, value in zip(THETA_NAMES, row, strict=True):
            assert getattr(bound, name) == float(value)
        # Unscreened fields are untouched.
        assert bound.beta_capacity == base.beta_capacity
        assert bound.q0 == base.q0
        assert bound.q1 == base.q1


class TestOptuna:
    """Optuna TPE policy optimization."""

    def test_optuna_study_creation(self) -> None:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=0),
            pruner=optuna.pruners.MedianPruner(),
        )
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_optuna_optimization_with_toy_objective(self) -> None:
        problem = _salib_problem()

        def objective(trial: optuna.Trial) -> float:
            enforcement = trial.suggest_float(
                "enforcement", problem["bounds"][0][0], problem["bounds"][0][1]
            )
            # Toy: maximize enforcement
            return float(enforcement)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=0),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=4, show_progress_bar=False)

        assert study.best_trial is not None
        assert study.best_trial.value > 0.0
        assert study.best_trial.params["enforcement"] > 0.5  # Should optimize toward 1

    def test_optuna_best_params_in_bounds(self) -> None:
        problem = _salib_problem()

        def objective(trial: optuna.Trial) -> float:
            enforcement = trial.suggest_float(
                "enforcement", problem["bounds"][0][0], problem["bounds"][0][1]
            )
            targeting = trial.suggest_float(
                "targeting", problem["bounds"][1][0], problem["bounds"][1][1]
            )
            phase_speed = trial.suggest_float(
                "phase_speed", problem["bounds"][2][0], problem["bounds"][2][1]
            )
            subsidy = trial.suggest_float(
                "subsidy", problem["bounds"][3][0], problem["bounds"][3][1]
            )
            return float(enforcement + targeting + phase_speed + subsidy)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=0),
        )
        study.optimize(objective, n_trials=3, show_progress_bar=False)

        best = study.best_trial
        assert float(best.params["enforcement"]) >= 0.0 and float(best.params["enforcement"]) <= 1.0
        assert float(best.params["targeting"]) >= -1.0 and float(best.params["targeting"]) <= 1.0
        assert float(best.params["phase_speed"]) >= 0.0 and float(best.params["phase_speed"]) <= 1.0
        assert float(best.params["subsidy"]) >= 0.0 and float(best.params["subsidy"]) <= 1.0
