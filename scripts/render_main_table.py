#!/usr/bin/env python3
"""Render the source-derived seven-row DropTest Table 1 as LaTeX."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "displacement_rel_l2_percent", "stress_rel_l2_percent",
    "train_step_latency_ms", "peak_train_allocated_gib",
    "inference_latency_ms_per_99step_trajectory", "parameters_million",
)


def render(rows: list[dict[str, str]]) -> str:
    ranks = {}
    for metric in METRICS:
        key = metric if metric in rows[0] else metric + "_mean"
        ranks[metric] = sorted({float(row[key]) for row in rows})[:2]
    lines = []
    for i, row in enumerate(rows):
        if i == 5:
            lines.append(r"    \midrule")
        model = row["model"]
        fields = [r"\textbf{LTS (Full)}" if model == "LTS (Full)" else model]
        for metric in METRICS:
            key = metric if metric in row else metric + "_mean"
            number = float(row[key])
            cell = f"{number:.2f}"
            if metric + "_sample_std" in row:
                cell += rf" \pm {float(row[metric + '_sample_std']):.2f}"
            if number == ranks[metric][0]:
                cell = r"\mathbf{" + cell + "}"
            elif number == ranks[metric][1]:
                cell = r"\underline{" + cell + "}"
            fields.append("$" + cell + "$")
        lines.append("    " + " & ".join(fields) + r" \\")
    deltas = [r"\scriptsize vs. Transolver"]
    for metric in METRICS:
        key = metric if metric in rows[-1] else metric + "_mean"
        a, b = float(rows[0][key]), float(rows[-1][key])
        direction = r"\uparrow" if b > a else r"\downarrow"
        percent = metric.endswith("percent")
        delta = abs(b - a) if percent else 100 * abs(b - a) / a
        unit = r"\,\mathrm{pp}" if percent else r"\%"
        digits = 2 if percent else 1
        deltas.append(r"\scriptsize $" + direction + f" {delta:.{digits}f}" + unit + "$")
    lines.append("    " + " & ".join(deltas) + r" \\")
    return r"""% Generated from frozen per-seed metrics and original profiling exports.
\begin{table*}[!t]
  \centering
  \caption{Main results on DropTest under a unified evaluation and single-H200 profiling protocol.}
  \label{tab:main_results}
  \setlength{\tabcolsep}{5pt}
  \renewcommand{\arraystretch}{1.12}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{lcccccc}
    \toprule
    & \multicolumn{2}{c}{Accuracy} & \multicolumn{4}{c}{Efficiency} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-7}
    Model & Disp. Rel. $\mathrm{L}_2\downarrow$ & Stress Rel. $\mathrm{L}_2\downarrow$
    & Train-step Latency $\downarrow$ & Peak Train Alloc. Mem. $\downarrow$
    & Inference Latency $\downarrow$ & Params. \\
    & (\%) & (\%) & (ms/step) & (GiB) & (ms/traj.) & (M) \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}%
  }
  \par\smallskip
  \begin{minipage}{\textwidth}
  \small
  \textbf{Notes.} Accuracy and 99-query latency are mean $\pm$ sample s.d.
  over seeds $\{7,17,27\}$. Seed-7 H200 training latency uses 50 timed steps
  after 20 warm-up steps, with Python cyclic GC excluded; peak memory is the
  maximum allocated during profiling.
  Base disables CA, Fourier, and D-Scale; Full enables all three. Only CA uses
  component labels. Detailed settings are in Appendix B.1.
  \end{minipage}
\end{table*}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results/main_tables/dtob6_0826_paper.csv",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = render(rows)
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
