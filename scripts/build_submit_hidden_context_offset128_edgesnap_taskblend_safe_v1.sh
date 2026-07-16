#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="${1:-output/submissions/dinov3_vitb_hidden_context_offset128_edgesnap_taskblend_safe_v1_seed42}"
ANCHOR_JSON="${2:-output/submissions/dinov3_vitb_hidden_context_offset128_edge_snap_safe_v1_seed42/regression_predictions.json}"

mkdir -p "${SUBMISSION_DIR}"

python scripts/blend_submission_predictions_by_task.py \
  --anchor-json "${ANCHOR_JSON}" \
  --task-blend A4C=0.12:output/submissions/dinov3_vitb_taskfpn_canonical_pairs/regression_predictions.json \
  --task-blend AOP=0.12:output/submissions/vitlarge_dinov3_taskfpn_grouped_strongaug_v1/regression_predictions.json \
  --task-blend FUGC=0.06:output/submissions/dinov3_vitb_fpn_deep_taskweighted/regression_predictions.json \
  --task-blend IVC=0.02:output/submissions/dinov3_vitb_fpn_deep128_measured/regression_predictions.json \
  --task-blend PLAX=0.12:output/submissions/vitlarge_dinov3_taskfpn_v1/regression_predictions.json \
  --task-blend PSAX=0.06:output/submissions/dinov3_vitb_taskfpn_cardiac_graph_augv1/regression_predictions.json \
  --task-blend fetal_femur=0.06:output/submissions/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42/regression_predictions.json \
  --output-json "${SUBMISSION_DIR}/regression_predictions.json"

python - <<'PY' "${SUBMISSION_DIR}"
import sys
import zipfile
from pathlib import Path

out = Path(sys.argv[1])
with zipfile.ZipFile(out / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(out / "regression_predictions.json", arcname="regression_predictions.json")
PY

cat > "${SUBMISSION_DIR}/blend_sources.txt" <<'EOF'
anchor: dinov3_vitb_hidden_context_offset128_edge_snap_safe_v1_seed42
A4C alpha=0.12 source=dinov3_vitb_taskfpn_canonical_pairs
AOP alpha=0.12 source=vitlarge_dinov3_taskfpn_grouped_strongaug_v1
FA anchor
FUGC alpha=0.06 source=dinov3_vitb_fpn_deep_taskweighted
HC anchor edge-snap
IVC alpha=0.02 source=dinov3_vitb_fpn_deep128_measured + edge-snap anchor
PLAX alpha=0.12 source=vitlarge_dinov3_taskfpn_v1 + edge-snap anchor
PSAX alpha=0.06 source=dinov3_vitb_taskfpn_cardiac_graph_augv1 + edge-snap anchor
fetal_femur alpha=0.06 source=dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42 + edge-snap anchor
EOF

bash scripts/audit_submission_before_upload.sh "${SUBMISSION_DIR}"
