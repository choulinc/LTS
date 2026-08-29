#!/usr/bin/env python3
"""Render the frozen DTOB6-0826 paper CSV as compact LaTeX rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def value(row: dict[str, str], stem: str, digits: int) -> str:
    return (
        f"${float(row[f'{stem}_mean']):.{digits}f} "
        f"\\pm {float(row[f'{stem}_sample_std']):.{digits}f}$"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/main_tables/dtob6_0826_paper.csv"),
    )
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        fields = [
            row["model"],
            value(row, "position_rel_l2_percent", 4),
            value(row, "displacement_rel_l2_percent", 4),
            value(row, "stress_rel_l2_percent", 4),
            value(row, "training_wall_hours", 3),
            f"${float(row['peak_train_allocated_gib']):.3f}$",
            value(row, "inference_latency_ms_per_99step_trajectory", 3),
            f"${float(row['parameters_million']):.3f}$",
        ]
        print(" & ".join(fields) + r" \\")


if __name__ == "__main__":
    main()
