"""Stage 15 (§10 Phase 7): delivering the result.

Matplotlib for the paper (:mod:`regworld.visualization.figures`), Plotly for
exploration (:mod:`regworld.visualization.interactive`), and a Streamlit
operator dashboard (:mod:`regworld.visualization.dashboard`). Every public
entry point degrades gracefully when its input artifact is missing: it logs
a warning and returns ``None`` (or skips that one item), rather than raising,
so one absent artifact never takes the rest of a run down with it.

Nothing in this package imports ``regworld.dgp`` or reads the sealed
answer-key tree (PLAN.md §1); where a figure needs a "true" value for
comparison, it reads the comparison the evaluation suite already computed
and wrote to ``reports/eval/metrics.json``.
"""

from __future__ import annotations

from regworld.visualization.dashboard import ood_mahalanobis
from regworld.visualization.figures import make_all_figures
from regworld.visualization.interactive import make_all_interactive

__all__ = ["make_all_figures", "make_all_interactive", "ood_mahalanobis"]
