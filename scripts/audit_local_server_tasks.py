#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


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


def main() -> None:
    output_root = Path("output")
    rows = []
    for result_path in sorted((output_root / "submissions").glob("*/codabench_result_*.txt")):
        run_name = result_path.parent.name
        local_path = resolve_local_summary(run_name, output_root)
        if local_path is None:
            continue
        local = parse_local_summary(local_path)
        server = parse_codabench_line(result_path)
        if local is None or server is None:
            continue
        rows.append(
            {
                "run": run_name,
                "server_overall": server["overall"],
                "local_avg": local["avg"],
                "tasks": local["tasks"],
            }
        )

    if not rows:
        print("No overlapping local/server runs found.")
        return

    rows.sort(key=lambda row: row["server_overall"])

    header = ["run", "server_overall", "local_avg"] + TASK_ORDER
    print("\t".join(header))
    for row in rows:
        values = [
            row["run"],
            f"{row['server_overall']:.2f}",
            f"{row['local_avg']:.6f}",
        ]
        for task_id in TASK_ORDER:
            task_value = row["tasks"].get(task_id)
            values.append("NA" if task_value is None else f"{task_value:.6f}")
        print("\t".join(values))

    if len(rows) >= 2:
        best = rows[0]
        worst = rows[-1]
        print("\nBest server run:", best["run"], f"({best['server_overall']:.2f})")
        print("Worst server run:", worst["run"], f"({worst['server_overall']:.2f})")
        print("\nPer-task local MRE delta (worst - best):")
        for task_id in TASK_ORDER:
            best_value = best["tasks"].get(task_id)
            worst_value = worst["tasks"].get(task_id)
            if best_value is None or worst_value is None:
                continue
            delta = worst_value - best_value
            print(f"  {task_id}: {delta:+.6f}")

        sorted_deltas = sorted(
            (
                (task_id, worst["tasks"][task_id] - best["tasks"][task_id])
                for task_id in TASK_ORDER
                if task_id in best["tasks"] and task_id in worst["tasks"]
            ),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        print("\nLargest local task shifts between best and worst server runs:")
        for task_id, delta in sorted_deltas[:5]:
            direction = "worse" if delta > 0 else "better"
            print(f"  {task_id}: {delta:+.6f} ({direction} in the worse server run)")


if __name__ == "__main__":
    main()
