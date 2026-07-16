#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANCHOR_JSON = REPO / "output/submissions/dinov3_vitb_hidden_context_roi_hcivcplax_v1_seed42/regression_predictions.json"
STUDENT_JSON = REPO / "output/submissions/dinov3_vitb_hidden_context_offset128_current_anchor_student_v1_seed42/regression_predictions.json"
OUT_ROOT = REPO / "output/submissions/current_anchor_student_blends_v2"


CANDIDATES = {
    # v1 winner: A4C=0.10,AOP=0.08,HC=0.20,IVC=0.18,PLAX=0.16,PSAX=0.10.
    "top1_repeat": {"A4C": 0.10, "AOP": 0.08, "HC": 0.20, "IVC": 0.18, "PLAX": 0.16, "PSAX": 0.10},
    "aop_lower": {"A4C": 0.10, "AOP": 0.05, "HC": 0.20, "IVC": 0.18, "PLAX": 0.16, "PSAX": 0.10},
    "aop_higher": {"A4C": 0.10, "AOP": 0.11, "HC": 0.20, "IVC": 0.18, "PLAX": 0.16, "PSAX": 0.10},
    "hcivc_higher": {"A4C": 0.10, "AOP": 0.08, "HC": 0.25, "IVC": 0.22, "PLAX": 0.16, "PSAX": 0.10},
    "hcivc_lower": {"A4C": 0.10, "AOP": 0.08, "HC": 0.16, "IVC": 0.14, "PLAX": 0.16, "PSAX": 0.10},
    "psax_lower": {"A4C": 0.10, "AOP": 0.08, "HC": 0.20, "IVC": 0.18, "PLAX": 0.16, "PSAX": 0.06},
    "psax_higher": {"A4C": 0.10, "AOP": 0.08, "HC": 0.20, "IVC": 0.18, "PLAX": 0.16, "PSAX": 0.14},
    "a4c_lower": {"A4C": 0.06, "AOP": 0.08, "HC": 0.20, "IVC": 0.18, "PLAX": 0.16, "PSAX": 0.10},
    "a4c_higher": {"A4C": 0.14, "AOP": 0.08, "HC": 0.20, "IVC": 0.18, "PLAX": 0.16, "PSAX": 0.10},
    "plax_lower": {"A4C": 0.10, "AOP": 0.08, "HC": 0.20, "IVC": 0.18, "PLAX": 0.10, "PSAX": 0.10},
    "plax_higher": {"A4C": 0.10, "AOP": 0.08, "HC": 0.20, "IVC": 0.18, "PLAX": 0.22, "PSAX": 0.10},
}


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def zip_submission(candidate_dir: Path) -> None:
    with zipfile.ZipFile(candidate_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(candidate_dir / "regression_predictions.json", arcname="regression_predictions.json")


def main() -> int:
    if not ANCHOR_JSON.is_file():
        raise FileNotFoundError(ANCHOR_JSON)
    if not STUDENT_JSON.is_file():
        raise FileNotFoundError(STUDENT_JSON)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, weights in CANDIDATES.items():
        candidate_dir = OUT_ROOT / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        output_json = candidate_dir / "regression_predictions.json"
        command = [
            sys.executable,
            "scripts/blend_submission_predictions_by_task.py",
            "--anchor-json",
            str(ANCHOR_JSON),
            "--output-json",
            str(output_json),
        ]
        for task_id, alpha in weights.items():
            command.extend(["--task-blend", f"{task_id}={alpha}:{STUDENT_JSON}"])
        run(command)
        zip_submission(candidate_dir)
        audit_json = candidate_dir / "pre_submit_audit.json"
        run(
            [
                sys.executable,
                "scripts/audit_submission_before_upload.py",
                "--candidate-json",
                str(output_json),
                "--reference-json",
                str(ANCHOR_JSON),
                "--output-json",
                str(audit_json),
            ]
        )
        audit = json.loads(audit_json.read_text())
        (candidate_dir / "blend_weights.json").write_text(json.dumps(weights, indent=2, sort_keys=True))
        summary.append(
            {
                "name": name,
                "dir": str(candidate_dir.relative_to(REPO)),
                "status": audit["status"],
                "overall_mean_task_shift_px": audit["overall_mean_task_shift_px"],
                "weights": weights,
            }
        )

    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
