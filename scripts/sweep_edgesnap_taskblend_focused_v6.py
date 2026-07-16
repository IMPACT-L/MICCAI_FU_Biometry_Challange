#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Sixth focused search after the 24.29 hidden result. v5_top1 and v5_top2 tied,
# so PLAX is not the main lever. Search A4C/AOP/PSAX around the v5 point, keep
# FUGC near 0.08, and leave fetal_femur anchored.
sweep.OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_focused_v6"
sweep.GRID = {
    "A4C": [0.48, 0.52, 0.56, 0.60],
    "AOP": [0.66, 0.70, 0.74, 0.78],
    "FUGC": [0.06, 0.08],
    "PLAX": [0.06, 0.08],
    "PSAX": [0.26, 0.30, 0.34],
    "fetal_femur": [0.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
