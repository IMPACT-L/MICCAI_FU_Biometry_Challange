#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Focused search around the best observed hidden recipe:
# - keep the safe edge-snapped anchor fixed;
# - test moderate A4C/AOP/PSAX pulls because these drove the 24.64 gain;
# - keep FUGC small because larger shifts damaged prior submissions;
# - leave fetal_femur fixed to avoid hidden-measurement regressions.
sweep.OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_focused_v1"
sweep.GRID = {
    "A4C": [0.12, 0.15, 0.18],
    "AOP": [0.18, 0.22, 0.26],
    "FUGC": [0.04, 0.06, 0.08],
    "PLAX": [0.00, 0.06],
    "PSAX": [0.08, 0.10, 0.12],
    "fetal_femur": [0.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
