#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Fourth focused search after the 24.38 hidden result. v3_top1 was at the upper
# edge for A4C/AOP/FUGC/PLAX/PSAX, so this tests a further bounded push. Keep
# fetal_femur anchored because every useful hidden gain so far came without a
# fetal-femur specialist pull.
sweep.OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_focused_v4"
sweep.GRID = {
    "A4C": [0.32, 0.36, 0.40],
    "AOP": [0.46, 0.50, 0.54, 0.58],
    "FUGC": [0.08, 0.10, 0.12],
    "PLAX": [0.04, 0.06, 0.08],
    "PSAX": [0.18, 0.20, 0.22],
    "fetal_femur": [0.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
