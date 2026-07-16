#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Seventh focused search after the 24.27 hidden result. Gains are now small, so
# this is a local sweep around v6_top2 instead of another wide push.
sweep.OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_focused_v7"
sweep.GRID = {
    "A4C": [0.56, 0.60, 0.64, 0.68],
    "AOP": [0.74, 0.78, 0.82, 0.86],
    "FUGC": [0.06, 0.08],
    "PLAX": [0.04, 0.06],
    "PSAX": [0.30, 0.34, 0.38],
    "fetal_femur": [0.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
