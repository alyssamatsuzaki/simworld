"""§11 evaluation suite: twelve metric families, all reported, none hidden.

This package is the ONLY code outside the world builders allowed to import
``regworld.dgp`` or read ``artifacts/oracle`` — it grades everything against
the answer key. Thresholds are the pass criteria at ``profile=dev``; at smoke
the numbers are reported alongside their thresholds without gating.
"""
