#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Backoff search after v7 regressed to 24.48. The best confirmed point is v6_top2:
# A4C=0.60, AOP=0.78, FUGC=0.08, PLAX=0.06, PSAX=0.34. Search around it with
# smaller A4C/PSAX movement and avoid larger AOP/FUGC/PLAX shifts.
sweep.OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_focused_v8_backoff"
sweep.GRID = {
    "A4C": [0.54, 0.56, 0.58, 0.60, 0.62],
    "AOP": [0.74, 0.78],
    "FUGC": [0.06, 0.08],
    "PLAX": [0.04, 0.06],
    "PSAX": [0.28, 0.30, 0.32, 0.34, 0.36],
    "fetal_femur": [0.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
