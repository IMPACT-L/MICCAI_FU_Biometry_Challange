#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Fifth focused search after the 24.32 hidden result. The confirmed best kept
# FUGC at 0.08 and gained by pushing A4C/AOP/PSAX. This grid performs a local
# search around that point rather than jumping to the most aggressive v4 option.
sweep.OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_focused_v5"
sweep.GRID = {
    "A4C": [0.36, 0.40, 0.44, 0.48],
    "AOP": [0.54, 0.58, 0.62, 0.66],
    "FUGC": [0.06, 0.08],
    "PLAX": [0.04, 0.06, 0.08],
    "PSAX": [0.20, 0.22, 0.24, 0.26],
    "fetal_femur": [0.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
