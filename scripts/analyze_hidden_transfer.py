#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from statistics import mean


RUN_ALIAS_TO_LOCAL = {
    "dinov3_vitb_taskfpn_canonical_pairs": "dinov3_vitb_taskfpn/evaluation_summary_canonical_pairs.txt",
    "dinov3_vitb_taskfpn": "dinov3_vitb_taskfpn/evaluation_summary.txt",
}

TASK_ORDER = [
    "A4C",
    "AOP",
    "FA",
    "FUGC",
    "HC",
    "IVC",
    "PLAX",
    "PSAX",
    "fetal_femur",
]


def parse_local_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    avg_match = re.search(r"Average MRE:\s*([0-9.]+)", text)
    if not avg_match:
        return None
    tasks = {}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z0-9_]+): MRE=([0-9.]+)", line)
        if match:
            tasks[match.group(1)] = float(match.group(2))
    return {"avg": float(avg_match.group(1)), "tasks": tasks, "path": str(path)}


def parse_codabench_line(path: Path) -> dict | None:
    line = path.read_text(errors="ignore").strip()
    if not line:
        return None
    parts = line.split("\t")
    numeric = []
    for value in parts[4:]:
        try:
            numeric.append(float(value))
        except ValueError:
            pass
    if not numeric:
        return None
    return {
        "overall": numeric[0],
        "mre_col": numeric[1] if len(numeric) > 1 else None,
        "raw": line,
        "path": str(path),
    }


def resolve_local_summary(run_name: str, output_root: Path) -> Path | None:
    alias = RUN_ALIAS_TO_LOCAL.get(run_name)
    if alias is not None:
        path = output_root / "runs" / alias
        return path if path.exists() else None
    candidate = output_root / "runs" / run_name / "evaluation_summary.txt"
    return candidate if candidate.exists() else None


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    rx = rank(xs)
    ry = rank(ys)
    mx = mean(rx)
    my = mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    output_root = Path("output")
    rows = []
    for result_path in sorted((output_root / "submissions").glob("*/codabench_result_*.txt")):
        run_name = result_path.parent.name
        local_path = resolve_local_summary(run_name, output_root)
        local = parse_local_summary(local_path) if local_path is not None else None
        server = parse_codabench_line(result_path)
        if local is None or server is None:
            continue
        row = {
            "run": run_name,
            "server_overall": server["overall"],
            "server_mre_col": server["mre_col"],
            "local_avg": local["avg"],
        }
        row.update(local["tasks"])
        rows.append(row)

    if not rows:
        print("No overlapping local/server runs found.")
        return

    rows.sort(key=lambda row: row["server_overall"])
    top_k = min(5, len(rows))
    top_rows = rows[:top_k]
    rest_rows = rows[top_k:] if len(rows) > top_k else rows[top_k:]

    print("Top hidden-validation runs")
    print("run\tserver_overall\tlocal_avg\tA4C\tAOP\tFA\tFUGC\tHC\tIVC\tPLAX\tPSAX\tfetal_femur")
    for row in top_rows:
        values = [
            row["run"],
            f"{row['server_overall']:.2f}",
            f"{row['local_avg']:.6f}",
        ]
        for task_id in TASK_ORDER:
            values.append(f"{row.get(task_id, float('nan')):.6f}")
        print("\t".join(values))

    print("\nTask-level Spearman correlation with hidden leaderboard score")
    print("(positive means lower local task MRE tends to align with better hidden score)")
    for metric_name in ["local_avg"] + TASK_ORDER:
        xs = [row[metric_name] for row in rows if metric_name in row]
        ys = [row["server_overall"] for row in rows if metric_name in row]
        print(f"{metric_name:14s}\t{spearman(xs, ys):+.3f}")

    if rest_rows:
        print("\nTop-cluster vs rest: mean local MRE")
        for task_id in TASK_ORDER:
            top_value = mean(row[task_id] for row in top_rows)
            rest_value = mean(row[task_id] for row in rest_rows)
            print(
                f"{task_id:14s}\tbest{top_k}={top_value:.3f}\trest={rest_value:.3f}\tdelta={rest_value - top_value:+.3f}"
            )

    print("\nRecommended next ablation policy")
    print("- Freeze the architecture at the best stable family: ViT-B DINOv3 + task-specific FPN + uniform decoder.")
    print("- Do not change augmentation profile away from baseline until a hidden score win is observed.")
    print("- Prioritize mild reweighting/fine-tuning around HC, IVC, fetal_femur, AOP, and FA.")
    print("- Treat A4C as a checkpoint-selection problem first, not a decoder rewrite problem.")


if __name__ == "__main__":
    main()
