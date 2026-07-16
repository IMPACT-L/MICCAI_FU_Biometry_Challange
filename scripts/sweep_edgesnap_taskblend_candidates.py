#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import subprocess
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ANCHOR_JSON = REPO / "output/submissions/dinov3_vitb_hidden_context_offset128_edge_snap_safe_v1_seed42/regression_predictions.json"
REFERENCE_JSON = REPO / "output/submissions/dinov3_vitb_hidden_context_offset128_v1_seed42/regression_predictions.json"
OUT_ROOT = REPO / "output/submissions/sweep_edgesnap_taskblend_v1"

SOURCES = {
    "A4C": REPO / "output/submissions/dinov3_vitb_taskfpn_canonical_pairs/regression_predictions.json",
    "AOP": REPO / "output/submissions/vitlarge_dinov3_taskfpn_grouped_strongaug_v1/regression_predictions.json",
    "FUGC": REPO / "output/submissions/dinov3_vitb_fpn_deep_taskweighted/regression_predictions.json",
    "PLAX": REPO / "output/submissions/vitlarge_dinov3_taskfpn_v1/regression_predictions.json",
    "PSAX": REPO / "output/submissions/dinov3_vitb_taskfpn_cardiac_graph_augv1/regression_predictions.json",
    "fetal_femur": REPO / "output/submissions/dinov3_vitb_taskfpn_hidden_context_pseudo_hardtasks_v1_seed42/regression_predictions.json",
}

# Narrow sweep around the first successful 24.64 recipe. IVC is left at edge-snap
# anchor because increasing IVC blend helped local audit little and risks hidden drift.
GRID = {
    "A4C": [0.10, 0.12, 0.15],
    "AOP": [0.12, 0.16, 0.20, 0.24],
    "FUGC": [0.04, 0.06, 0.08],
    "PLAX": [0.00, 0.06, 0.10, 0.12],
    "PSAX": [0.06, 0.08, 0.10],
    "fetal_femur": [0.00, 0.02, 0.06],
}


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=REPO,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def zip_submission(candidate_dir: Path) -> None:
    with zipfile.ZipFile(candidate_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(candidate_dir / "regression_predictions.json", arcname="regression_predictions.json")


def candidate_name(index: int, weights: dict[str, float]) -> str:
    parts = [f"{task}{int(round(weights[task] * 100)):02d}" for task in sorted(weights)]
    return f"candidate_{index:03d}_" + "_".join(parts)


def audit_candidate(candidate_dir: Path) -> dict:
    audit_json = candidate_dir / "pre_submit_audit.json"
    run_command(
        [
            "python",
            "scripts/audit_submission_before_upload.py",
            "--candidate-json",
            str(candidate_dir / "regression_predictions.json"),
            "--reference-json",
            str(REFERENCE_JSON),
            "--output-json",
            str(audit_json),
        ],
        check=False,
    )
    return json.loads(audit_json.read_text())


def build_candidate(index: int, weights: dict[str, float]) -> dict:
    candidate_dir = OUT_ROOT / candidate_name(index, weights)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        "scripts/blend_submission_predictions_by_task.py",
        "--anchor-json",
        str(ANCHOR_JSON),
        "--output-json",
        str(candidate_dir / "regression_predictions.json"),
    ]
    for task, alpha in sorted(weights.items()):
        if alpha <= 0.0:
            continue
        command.extend(["--task-blend", f"{task}={alpha}:{SOURCES[task]}"])

    run_command(command)
    zip_submission(candidate_dir)
    audit = audit_candidate(candidate_dir)
    (candidate_dir / "blend_weights.json").write_text(json.dumps(weights, indent=2, sort_keys=True))

    task_shift = {
        row["task_id"]: float(row["mean_shift_px"])
        for row in audit.get("tasks", [])
    }
    # Conservative rank: prefer PASS, moderate movement, and higher alpha on tasks
    # where the 24.64 row improved hidden score. This is not a hidden-label estimate;
    # it is a risk-aware submission ordering heuristic.
    expected_push = (
        1.6 * weights.get("AOP", 0.0)
        + 1.4 * weights.get("A4C", 0.0)
        + 1.2 * weights.get("PSAX", 0.0)
        + 0.8 * weights.get("FUGC", 0.0)
        + 0.4 * weights.get("PLAX", 0.0)
        - 0.8 * weights.get("fetal_femur", 0.0)
    )
    risk = float(audit.get("overall_mean_task_shift_px", 999.0))
    score = expected_push - 0.18 * risk
    return {
        "name": candidate_dir.name,
        "dir": str(candidate_dir.relative_to(REPO)),
        "status": audit.get("status"),
        "score": score,
        "overall_shift": risk,
        "weights": weights,
        "task_shift": task_shift,
        "issues": audit.get("issues", []),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = list(GRID)
    rows = []
    for index, values in enumerate(itertools.product(*(GRID[task] for task in tasks)), start=1):
        weights = dict(zip(tasks, values))
        row = build_candidate(index, weights)
        rows.append(row)

    pass_rows = [row for row in rows if row["status"] == "PASS"]
    pass_rows.sort(key=lambda row: row["score"], reverse=True)
    payload = {
        "anchor_json": str(ANCHOR_JSON.relative_to(REPO)),
        "reference_json": str(REFERENCE_JSON.relative_to(REPO)),
        "num_candidates": len(rows),
        "num_pass": len(pass_rows),
        "top_pass": pass_rows[:25],
    }
    summary_path = OUT_ROOT / "sweep_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"Built {len(rows)} candidates; {len(pass_rows)} passed audit.")
    print(f"Wrote {summary_path.relative_to(REPO)}")
    print("Top candidates:")
    for row in pass_rows[:10]:
        print(
            f"{row['name']} score={row['score']:.4f} "
            f"shift={row['overall_shift']:.4f} weights={row['weights']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
