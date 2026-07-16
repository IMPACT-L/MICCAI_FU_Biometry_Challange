#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")


DEFAULT_TASK_PAIR_ORDER = [
    "A4C",
    "AOP",
    "FA",
    "HC",
    "FUGC",
    "IVC",
    "PLAX",
    "PSAX",
    "fetal_femur",
]


@dataclass
class Row:
    source: Path
    rank: int | None
    user: str | None
    submission_id: str | None
    values: list[float]
    text: str


def parse_row(path: Path) -> Row | None:
    text = path.read_text(errors="ignore").strip()
    if not text:
        return None

    # Prefer raw CodaBench rows. They contain rank, user, date, id, then metrics.
    raw_line = None
    for line in text.splitlines():
        if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", line) and len(FLOAT_RE.findall(line)) >= 8:
            raw_line = line
            break
    if raw_line is None and len(FLOAT_RE.findall(text)) >= 8:
        raw_line = text.splitlines()[0]
    if raw_line is None:
        score_match = re.search(r"(?:overall[_ ]score|score|overall)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        if score_match is None:
            return None
        return Row(path, None, None, None, [float(score_match.group(1))], text)

    fields = re.split(r"\s+", raw_line.strip())
    rank = None
    user = None
    submission_id = None
    try:
        rank = int(fields[0])
    except Exception:
        pass
    if len(fields) > 1:
        user = fields[1].strip()
    if len(fields) > 4 and fields[4].isdigit():
        submission_id = fields[4]

    numbers = [float(item) for item in FLOAT_RE.findall(raw_line)]
    # Raw row numbers include rank, date fragments, time fragments, and submission id.
    # Metrics start after the submission id when it is present.
    if submission_id is not None:
        try:
            id_float = float(submission_id)
            id_index = numbers.index(id_float)
            values = numbers[id_index + 1 :]
        except ValueError:
            values = numbers[-20:]
    else:
        values = numbers[-20:]
    if len(values) < 1:
        return None
    return Row(path, rank, user, submission_id, values, text)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def metric_label(index: int, task_pair_order: list[str]) -> str:
    if index == 0:
        return "overall"
    if index == 1:
        return "aggregate"
    pair_index = (index - 2) // 2
    metric_index = (index - 2) % 2
    if pair_index >= len(task_pair_order):
        return f"col{index:02d}"
    metric_name = "metric1" if metric_index == 0 else "metric2"
    return f"{task_pair_order[pair_index]}_{metric_name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize saved CodaBench leaderboard response columns.")
    parser.add_argument("--manual-results-dir", default="output/submissions/manual_results")
    parser.add_argument("--current-score", type=float, default=24.18)
    parser.add_argument(
        "--top-row",
        default=(
            "22.61 27.84 23.67 17.12 14.44 9.6 20.95 94.63 20.44 19.49 "
            "3.59 2.51 42.02 66.05 24.36 13.73 16.27 8.07 37.75 19.38"
        ),
        help="Reference top competitor metric values after rank/user/date/id, if known.",
    )
    parser.add_argument(
        "--task-pair-order",
        default=",".join(DEFAULT_TASK_PAIR_ORDER),
        help=(
            "Comma-separated task order for the 18 task metric columns after overall and aggregate. "
            "Default matches the task-pair interpretation used in the saved analysis notes."
        ),
    )
    args = parser.parse_args()
    task_pair_order = [item.strip() for item in args.task_pair_order.split(",") if item.strip()]

    root = Path(args.manual_results_dir)
    rows = [row for path in sorted(root.glob("*.txt")) if (row := parse_row(path)) is not None]
    rows = [row for row in rows if row.values]
    if not rows:
        raise SystemExit(f"No parseable result rows under {root}")

    full_rows = [row for row in rows if len(row.values) >= 20]
    score_rows = [row for row in full_rows if row.values] or [row for row in rows if row.values[0] > 10.0]
    best_score = min(score_rows, key=lambda row: row.values[0])
    print(f"Parsed rows: {len(rows)} ({len(full_rows)} full metric rows)")
    print(f"Best saved score: {best_score.values[0]:.3f} from {best_score.source.name}")
    print()

    if full_rows:
        current = min(full_rows, key=lambda row: abs(row.values[0] - args.current_score))
        top_values = [float(v) for v in args.top_row.split()]
        if len(top_values) < len(current.values):
            top_values = top_values + [float("nan")] * (len(current.values) - len(top_values))

        print("Column gaps for current saved best versus provided top-row reference:")
        print("col\tlabel\tcurrent\ttop_ref\tgap(current-top)\tbest_ours\tbest_file")
        gap_rows = []
        for idx in range(min(len(current.values), 20)):
            best_col = min(full_rows, key=lambda row: row.values[idx] if idx < len(row.values) else float("inf"))
            top = top_values[idx] if idx < len(top_values) else None
            gap = current.values[idx] - top if top is not None else None
            gap_rows.append((idx, metric_label(idx, task_pair_order), gap, current.values[idx], top, best_col))
            print(
                f"{idx:02d}\t{metric_label(idx, task_pair_order)}\t"
                f"{fmt(current.values[idx])}\t{fmt(top)}\t{fmt(gap)}\t"
                f"{fmt(best_col.values[idx])}\t{best_col.source.name}"
            )
        print()

        actionable = [
            row for row in gap_rows[2:]
            if row[2] is not None and row[2] > 1.0
        ]
        actionable.sort(key=lambda row: row[2], reverse=True)
        if actionable:
            print("Largest positive task-metric gaps versus top reference:")
            print("label\tgap\tcurrent\ttop_ref\tbest_ours\tbest_file")
            for idx, label, gap, current_value, top_value, best_col in actionable[:8]:
                print(
                    f"{label}\t{fmt(gap)}\t{fmt(current_value)}\t{fmt(top_value)}\t"
                    f"{fmt(best_col.values[idx])}\t{best_col.source.name}"
                )
            print()

    print("Best saved full rows:")
    for row in sorted(full_rows, key=lambda item: item.values[0])[:12]:
        print(f"{row.values[0]:.3f}\t{row.source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
