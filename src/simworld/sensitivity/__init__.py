"""§14 Stage 14: SALib Morris → Sobol sensitivity analysis + Optuna policy search.

Sensitivity analysis on the four policy levers (enforcement, targeting, phase_speed, subsidy)
over the trained GraphRSSM emulator. Morris screening ranks the factors, Sobol quantifies
first-order (S1) and total-order (ST) indices. Emulator-vs-ABM cross-check validates that
the sensitivity surface agrees with the true model.

Optuna policy optimization (TPE) over the same four levers to maximize the regulator
objective J in the emulator.
"""

from __future__ import annotations

__all__: list[str] = []
