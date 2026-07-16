#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Second focused search after the 24.51 hidden result. This explores only the
# neighborhood that improved CodaBench: stronger A4C/AOP/PSAX movement, bounded
# FUGC movement, no fetal-femur specialist pull.
sweep.OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_focused_v2"
sweep.GRID = {
    "A4C": [0.16, 0.18, 0.21, 0.24],
    "AOP": [0.24, 0.26, 0.30, 0.34],
    "FUGC": [0.04, 0.06, 0.08],
    "PLAX": [0.00, 0.04, 0.06],
    "PSAX": [0.10, 0.12, 0.14],
    "fetal_femur": [0.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
