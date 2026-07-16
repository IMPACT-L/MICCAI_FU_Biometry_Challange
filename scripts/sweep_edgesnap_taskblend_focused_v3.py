#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Third focused search after the 24.45 hidden result. The previous best was at
# the upper edge for A4C/AOP/PSAX, so this tests a bounded push while keeping
# FUGC conservative and fetal_femur anchored.
sweep.OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_focused_v3"
sweep.GRID = {
    "A4C": [0.24, 0.28, 0.32],
    "AOP": [0.34, 0.38, 0.42, 0.46],
    "FUGC": [0.06, 0.08, 0.10],
    "PLAX": [0.04, 0.06],
    "PSAX": [0.14, 0.16, 0.18],
    "fetal_femur": [0.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
