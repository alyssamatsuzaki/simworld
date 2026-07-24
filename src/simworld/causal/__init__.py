"""Stage 5 - interrogating the causal assumptions.

Running a causal-inference library does not make your conclusions causal. Every
estimator in this package is graded against the DGP's sealed ``do()`` ground truth
(``causal/ground_truth.py``); where the simulator disagrees with a credibly
identified estimate, the simulator is wrong first (``causal/gate.py``).
"""

from __future__ import annotations
