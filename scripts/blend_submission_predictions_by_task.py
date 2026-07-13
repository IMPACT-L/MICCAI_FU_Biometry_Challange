#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_predictions(path: Path) -> dict[tuple[str, str], dict]:
    data = json.loads(path.read_text())
    return {
        (str(item["task_id"]), str(item["image_path"])): item
        for item in data
    }


def _parse_task_blends(values: list[str]) -> dict[str, tuple[float, Path]]:
    blends: dict[str, tuple[float, Path]] = {}
    for value in values:
        if "=" not in value or ":" not in value:
            raise ValueError(f"Expected TASK=ALPHA:path, got: {value}")
        task_id, rhs = value.split("=", 1)
        alpha_text, path_text = rhs.split(":", 1)
        alpha = float(alpha_text)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha} for {task_id}")
        blends[task_id.strip()] = (alpha, Path(path_text).resolve())
    return blends


def _blend_vector(anchor: list[float], source: list[float], alpha: float) -> list[float]:
    if len(anchor) != len(source):
        raise ValueError(f"Vector length mismatch: {len(anchor)} != {len(source)}")
    return [
        round((1.0 - alpha) * float(a) + alpha * float(s), 6)
        for a, s in zip(anchor, source)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soft-blend task predictions toward specialist submissions while keeping an anchor model."
    )
    parser.add_argument("--anchor-json", required=True)
    parser.add_argument(
        "--task-blend",
        action="append",
        default=[],
        help="TASK=ALPHA:path/to/regression_predictions.json. Example: IVC=0.25:output/submissions/run/regression_predictions.json",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    anchor_path = Path(args.anchor_json).resolve()
    output_path = Path(args.output_json).resolve()
    task_blends = _parse_task_blends(list(args.task_blend))

    anchor = _load_predictions(anchor_path)
    source_cache: dict[Path, dict[tuple[str, str], dict]] = {}
    for _, source_path in task_blends.values():
        if source_path not in source_cache:
            source_cache[source_path] = _load_predictions(source_path)

    merged = []
    used_counts: dict[str, int] = {}
    for key in sorted(anchor.keys(), key=lambda item: (item[0], item[1])):
        task_id, _ = key
        base_item = dict(anchor[key])
        if task_id in task_blends:
            alpha, source_path = task_blends[task_id]
            source_map = source_cache[source_path]
            if key not in source_map:
                raise KeyError(f"Missing key {key} in {source_path}")
            source_item = source_map[key]
            base_item["predicted_points_normalized"] = _blend_vector(
                base_item["predicted_points_normalized"],
                source_item["predicted_points_normalized"],
                alpha,
            )
            base_item["predicted_points_pixels"] = _blend_vector(
                base_item["predicted_points_pixels"],
                source_item["predicted_points_pixels"],
                alpha,
            )
            used_counts[task_id] = used_counts.get(task_id, 0) + 1
        merged.append(base_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))

    summary = {
        "anchor_json": str(anchor_path),
        "output_json": str(output_path),
        "task_blends": {
            task_id: {"alpha": alpha, "source": str(path)}
            for task_id, (alpha, path) in sorted(task_blends.items())
        },
        "used_counts": used_counts,
        "num_predictions": len(merged),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
