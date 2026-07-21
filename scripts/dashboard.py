"""§10 Stage 15 driver: the Streamlit operator dashboard.

Streamlit can't take a Hydra decorator cleanly, so the config is composed
inside :func:`regworld.visualization.dashboard.load_default_config` via
``hydra.initialize_config_dir`` + ``compose`` instead of ``@hydra.main``; the
profile is chosen from a sidebar selectbox rather than a CLI override.

Run headless: ``streamlit run scripts/dashboard.py --server.headless true --server.port 8501``.

All UI-building logic lives in :mod:`regworld.visualization.dashboard`; this
file only calls it under the ``__main__`` guard so ``import scripts.dashboard``
stays safe for tests and tooling.
"""

from __future__ import annotations

from regworld.visualization.dashboard import main

if __name__ == "__main__":
    main()
