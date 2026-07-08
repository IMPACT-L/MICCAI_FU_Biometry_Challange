#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


RUN_ALIAS_TO_LOCAL = {
    "dinov3_vitb_taskfpn_canonical_pairs": "dinov3_vitb_taskfpn/evaluation_summary_canonical_pairs.txt",
    "dinov3_vitb_taskfpn": "dinov3_vitb_taskfpn/evaluation_summary.txt",
}


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
        local = parse_local_summary(local_path) if local_path is not None else None
        server = parse_codabench_line(result_path)
        if server is None:
            continue
        rows.append(
            {
                "run": run_name,
                "local_avg": local["avg"] if local else None,
                "server_overall": server["overall"],
                "server_mre_col": server["mre_col"],
                "local_path": local["path"] if local else "missing",
                "server_path": server["path"],
            }
        )

    print("run\tlocal_avg_mre\tserver_overall\tserver_mre_col")
    for row in rows:
        local_avg = "NA" if row["local_avg"] is None else f"{row['local_avg']:.6f}"
        server_mre = "NA" if row["server_mre_col"] is None else f"{row['server_mre_col']:.2f}"
        print(f"{row['run']}\t{local_avg}\t{row['server_overall']:.2f}\t{server_mre}")

    ranked_rows = sorted(rows, key=lambda row: row["server_overall"])
    best = ranked_rows[0]
    best_local = "NA" if best["local_avg"] is None else f"{best['local_avg']:.6f}"
    best_server_mre = "NA" if best["server_mre_col"] is None else f"{best['server_mre_col']:.2f}"
    print("\nBest known server run:")
    print(
        f"- {best['run']} | local_avg_mre={best_local} | "
        f"server_overall={best['server_overall']:.2f} | server_mre_col={best_server_mre}"
    )

    print("\nInterpretation:")
    print("- Lower local_avg_mre is better locally.")
    print("- Lower server_overall is better on CodaBench.")
    print("- Large disagreements indicate that the local split is not predictive enough.")


if __name__ == "__main__":
    main()
