#!/usr/bin/env python3
"""Recompute the paper CSV from frozen per-seed results and profiling records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("transolver", "Transolver"), ("geotransolver", "GeoTransolver"),
    ("geots_flare", "GeoTS-FLARE"), ("transolver_plus", "Transolver++"),
    ("transolver3", "Transolver-3"), ("lts", "LTS (Base)"),
    ("lts_ca_fourier_tenv", "LTS (Full)"),
)


def build(raw: list[dict[str, str]], evidence: dict) -> list[dict]:
    result = []
    for model, label in MODELS:
        seeds = [r for r in raw if r["model"] == model]
        if sorted(int(r["seed"]) for r in seeds) != [7, 17, 27]:
            raise ValueError(f"Expected exactly seeds 7, 17, 27 for {model}")
        profile = evidence["profiles"][model]
        parameters = {int(float(r["total_parameters"])) for r in seeds}
        if parameters != {profile["parameter_count"]}:
            raise ValueError(f"Profile parameter identity mismatch for {model}")
        samples = profile.get("raw_train_step_latency_ms")
        if samples is not None:
            if len(samples) != 50 or not all(math.isfinite(v) for v in samples):
                raise ValueError(f"Invalid measured steps for {model}")
            for key, observed in (
                ("train_step_latency_ms_mean", statistics.mean(samples)),
                ("train_step_latency_ms_sample_std", statistics.stdev(samples)),
            ):
                if not math.isclose(profile[key], observed, rel_tol=1e-12):
                    raise ValueError(f"Profiling summary mismatch for {model}: {key}")
        row = {"model": label}
        for source, target, scale in (
            ("displacement_physical_relative_l2_macro", "displacement_rel_l2_percent", 100),
            ("stress_physical_relative_l2_macro", "stress_rel_l2_percent", 100),
            ("full_99_step_inference_seconds", "inference_latency_ms_per_99step_trajectory", 1000),
        ):
            values = [float(r[source]) * scale for r in seeds]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite seed metric for {model}: {source}")
            row[target + "_mean"] = statistics.mean(values)
            row[target + "_sample_std"] = statistics.stdev(values)
        row.update({k: profile[k] for k in (
            "train_step_latency_ms_mean", "train_step_latency_ms_sample_std",
            "peak_train_allocated_gib",
        )})
        row["parameters_million"] = parameters.pop() / 1e6
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/main_tables/dtob6_0826_paper.csv")
    args = parser.parse_args()
    raw_path = ROOT / "results/main_tables/dtob6_0826_per_seed_raw.csv"
    profile_path = ROOT / "results/profiling/dtob6_0826_gc_controlled.json"
    with raw_path.open(newline="") as stream:
        rows = build(list(csv.DictReader(stream)), json.loads(profile_path.read_text()))
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "authority": "Frozen machine-readable per-seed metrics and original profiling export records",
        "inputs_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (raw_path, profile_path)
        },
        "calculation": "Unrounded per-seed values; arithmetic mean and sample standard deviation; ratio to percent x100; seconds to milliseconds x1000; parameters to millions /1e6.",
        "display": "Round once, to two decimal places, only in the LaTeX renderer. CSV retains source precision.",
        "accuracy_and_inference": {"seeds": [7, 17, 27], "dispersion": "sample standard deviation"},
        "training_profile": {"warmup_steps": 20, "measured_steps": 50, "python_cyclic_gc_excluded": True, "source_limit": "Five baselines retain exported summaries and original profile hashes; two LTS variants retain all measured-step samples. See profiling evidence for source types."},
        "paper_csv_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    if args.output == ROOT / "results/main_tables/dtob6_0826_paper.csv":
        (args.output.parent / "paper_table_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
