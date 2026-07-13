#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_predictions(path: Path) -> dict[tuple[str, str], dict]:
    data = json.loads(path.read_text())
    mapping: dict[tuple[str, str], dict] = {}
    for item in data:
        key = (str(item["task_id"]), str(item["image_path"]))
        mapping[key] = item
    return mapping


def _parse_task_sources(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected TASK=path[,path2,...], got: {value}")
        task_id, raw_paths = value.split("=", 1)
        paths = [Path(part.strip()).resolve() for part in raw_paths.split(",") if part.strip()]
        if not paths:
            raise ValueError(f"No paths provided for task {task_id}")
        result[str(task_id).strip()] = paths
    return result


def _mean_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("No vectors to average.")
    length = len(vectors[0])
    for vector in vectors:
        if len(vector) != length:
            raise ValueError("Vector length mismatch while averaging pseudo labels.")
    return [
        round(sum(vector[idx] for vector in vectors) / float(len(vectors)), 6)
        for idx in range(length)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge submission prediction JSON files task-wise into one pseudo-teacher JSON."
    )
    parser.add_argument(
        "--fallback-json",
        required=True,
        help="Base regression_predictions.json used for every task unless overridden.",
    )
    parser.add_argument(
        "--task-source",
        action="append",
        default=[],
        help="Override a task with one or more source JSONs, e.g. AOP=path1 or HC=path1,path2",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output merged regression_predictions.json path.",
    )
    args = parser.parse_args()

    fallback_path = Path(args.fallback_json).resolve()
    output_path = Path(args.output_json).resolve()
    task_sources = _parse_task_sources(list(args.task_source))

    fallback = _load_predictions(fallback_path)
    loaded_sources: dict[Path, dict[tuple[str, str], dict]] = {}
    for paths in task_sources.values():
        for path in paths:
            if path not in loaded_sources:
                loaded_sources[path] = _load_predictions(path)

    task_to_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in fallback.keys():
        task_to_keys[key[0]].append(key)

    merged = []
    for task_id in sorted(task_to_keys.keys()):
        source_paths = task_sources.get(task_id, [fallback_path])
        source_maps = [fallback if path == fallback_path else loaded_sources[path] for path in source_paths]
        for key in sorted(task_to_keys[task_id], key=lambda item: item[1]):
            items = []
            for source_map in source_maps:
                if key not in source_map:
                    raise KeyError(f"Missing key {key} in one of the source submissions.")
                items.append(source_map[key])

            base_item = dict(items[0])
            base_item["predicted_points_normalized"] = _mean_vectors(
                [list(map(float, item["predicted_points_normalized"])) for item in items]
            )
            base_item["predicted_points_pixels"] = _mean_vectors(
                [list(map(float, item["predicted_points_pixels"])) for item in items]
            )
            merged.append(base_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))

    summary = {
        "fallback_json": str(fallback_path),
        "output_json": str(output_path),
        "task_sources": {
            task_id: [str(path) for path in paths]
            for task_id, paths in sorted(task_sources.items())
        },
        "num_predictions": len(merged),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
