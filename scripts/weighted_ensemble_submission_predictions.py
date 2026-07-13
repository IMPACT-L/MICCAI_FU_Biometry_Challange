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


def _parse_source(value: str) -> tuple[float, Path]:
    if ":" not in value:
        raise ValueError(f"Expected WEIGHT:path, got: {value}")
    weight_text, path_text = value.split(":", 1)
    weight = float(weight_text)
    if weight <= 0.0:
        raise ValueError(f"Weight must be positive, got {weight}")
    return weight, Path(path_text).resolve()


def _weighted_mean(vectors: list[list[float]], weights: list[float]) -> list[float]:
    length = len(vectors[0])
    if any(len(vector) != length for vector in vectors):
        raise ValueError("Prediction vector length mismatch.")
    weight_sum = sum(weights)
    return [
        round(sum(weights[i] * float(vectors[i][j]) for i in range(len(vectors))) / weight_sum, 6)
        for j in range(length)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted-average multiple submission prediction JSON files.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Weighted source in the form WEIGHT:path/to/regression_predictions.json.",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    sources = [_parse_source(value) for value in args.source]
    weights = [weight for weight, _ in sources]
    prediction_maps = [_load_predictions(path) for _, path in sources]
    reference_keys = set(prediction_maps[0].keys())
    for (_, path), pred_map in zip(sources[1:], prediction_maps[1:]):
        if set(pred_map.keys()) != reference_keys:
            missing = sorted(reference_keys - set(pred_map.keys()))[:5]
            extra = sorted(set(pred_map.keys()) - reference_keys)[:5]
            raise ValueError(f"Key mismatch for {path}; missing={missing}, extra={extra}")

    merged = []
    for key in sorted(reference_keys, key=lambda item: (item[0], item[1])):
        items = [pred_map[key] for pred_map in prediction_maps]
        base_item = dict(items[0])
        base_item["predicted_points_normalized"] = _weighted_mean(
            [list(map(float, item["predicted_points_normalized"])) for item in items],
            weights,
        )
        base_item["predicted_points_pixels"] = _weighted_mean(
            [list(map(float, item["predicted_points_pixels"])) for item in items],
            weights,
        )
        merged.append(base_item)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))
    print(
        json.dumps(
            {
                "output_json": str(output_path.resolve()),
                "num_predictions": len(merged),
                "sources": [
                    {"weight": weight, "path": str(path)}
                    for weight, path in sources
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
