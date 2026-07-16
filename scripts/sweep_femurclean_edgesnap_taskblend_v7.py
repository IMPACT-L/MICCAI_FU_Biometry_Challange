#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sweep_edgesnap_taskblend_candidates as sweep


REPO = Path(__file__).resolve().parents[1]

# Local search around the 24.22 public best. The anchor is the fetal-femur-clean
# offset128 model after conservative edge snapping. fetal_femur itself is locked
# back to the previous best because the direct femur-clean output produced one
# large validation shift outlier.
sweep.ANCHOR_JSON = (
    REPO
    / "output/submissions/dinov3_vitb_hidden_context_offset128_v1_femurclean_edge_snap_safe_seed42/regression_predictions.json"
)
sweep.REFERENCE_JSON = (
    REPO
    / "output/submissions/dinov3_vitb_hidden_context_offset128_v1_femurclean_edgesnap_taskblend_v6_keepbestfemur_seed42/regression_predictions.json"
)
sweep.OUT_ROOT = REPO / "output/submissions/sweep_femurclean_edgesnap_taskblend_v7"
sweep.SOURCES = {
    "A4C": REPO / "output/submissions/dinov3_vitb_taskfpn_canonical_pairs/regression_predictions.json",
    "AOP": REPO / "output/submissions/vitlarge_dinov3_taskfpn_grouped_strongaug_v1/regression_predictions.json",
    "FUGC": REPO / "output/submissions/dinov3_vitb_fpn_deep_taskweighted/regression_predictions.json",
    "PLAX": REPO / "output/submissions/vitlarge_dinov3_taskfpn_v1/regression_predictions.json",
    "PSAX": REPO / "output/submissions/dinov3_vitb_taskfpn_cardiac_graph_augv1/regression_predictions.json",
    "fetal_femur": (
        REPO
        / "output/submissions/dinov3_vitb_hidden_context_offset128_v1_femurclean_edgesnap_taskblend_v6_keepbestfemur_seed42/regression_predictions.json"
    ),
}
sweep.GRID = {
    "A4C": [0.56, 0.60, 0.62, 0.64],
    "AOP": [0.74, 0.78, 0.80, 0.82],
    "FUGC": [0.06, 0.08, 0.10],
    "PLAX": [0.04, 0.06],
    "PSAX": [0.30, 0.34, 0.36, 0.38],
    "fetal_femur": [1.00],
}


if __name__ == "__main__":
    raise SystemExit(sweep.main())
