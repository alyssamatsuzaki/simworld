"""§14 Stage 14: fast hermetic tests for sensitivity analysis.

Tests validate the SALib problem dict, Morris and Sobol execution on a toy objective,
and Optuna optimization. No dependency on artifacts/ or real checkpoint.
"""

from __future__ import annotations

import numpy as np
import optuna
from SALib.analyze.morris import analyze as morris_analyze
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.morris import sample as morris_sample
from SALib.sample.sobol import sample as sobol_sample

from regworld.sensitivity.screen import _salib_problem
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
        problem = _salib_problem()
        samples = sobol_sample(problem, N=64, calc_second_order=False, seed=0)
        # Ishigami-like: a = 7, b = 0.1
        # y = sin(a*x1) + a*sin²(b*x2) + b*x3⁴*sin(a*x1)
        # True S1: [0.31, 0.44, 0.0]
        a, b = 7.0, 0.1
        outputs = (
            np.sin(a * samples[:, 0])
            + a * (np.sin(b * samples[:, 1]) ** 2)
            + b * (samples[:, 2] ** 4) * np.sin(a * samples[:, 0])
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
