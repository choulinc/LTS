"""Aggregate the official-backbone DropTest K=1/2/3/4/6/8/12 study."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from examples.structural_mechanics.openradioss_dataset_gen import (
    droptest_official_backbones_k1248_20260827 as experiment,
)


METRICS = (
    "displacement_physical_relative_l2_macro",
    "stress_physical_relative_l2_macro",
)
EFFICIENCY = (
    "total_training_wall_seconds",
    "full_99_step_inference_seconds",
    "inference_latency_per_sample_seconds",
    "samples_per_second",
    "peak_allocated_max_all_ranks_bytes",
    "total_parameters",
)


def _source(k: int, seed: int) -> tuple[Path, str]:
    if k == 1:
        root = experiment.K1_OUTPUT_ROOT
        model = "lts_ca_fourier_tenv"
    else:
        root = experiment.OUTPUT_ROOT
        model = f"lts_ca_fourier_tenv_k{k}"
    return root / "runs" / model / f"seed_{seed}", model


def _load(k: int, seed: int) -> tuple[dict[str, Any] | None, str]:
    directory, source_model = _source(k, seed)
    marker = directory / "TEST_COMPLETED.json"
    config_path = directory / "config_snapshot.json"
    if not marker.is_file() or not config_path.is_file():
        return None, str(marker)
    payload = json.loads(marker.read_text())
    config = json.loads(config_path.read_text())
    if (
        payload.get("status") != "TEST_COMPLETED"
        or payload.get("model") != source_model
        or int(payload.get("seed", -1)) != seed
    ):
        return None, str(marker)
    identity = config["model_identity"]
    backbone = identity["backbone"]
    expected_k = int(backbone["resampling_count"])
    if expected_k != k:
        raise RuntimeError(f"K mismatch in {config_path}: {expected_k} != {k}")
    invariants = {
        "layers": 12,
        "hidden_dim": 256,
        "heads": 8,
        "physics_slices": 256,
        "component_aware": True,
        "fourier_bands": 10,
        "time_envelope_width": 64,
        "block_film": False,
    }
    for key, expected in invariants.items():
        if backbone.get(key) != expected:
            raise RuntimeError(
                f"Backbone mismatch in {config_path}: {key}="
                f"{backbone.get(key)!r} != {expected!r}"
            )
    row: dict[str, Any] = {
        "k": k,
        "model": f"lts_ca_fourier_tenv_k{k}",
        "source_model": source_model,
        "seed": seed,
        "source_directory": str(directory),
        "model_config_sha256": config["model_config_sha256"],
    }
    for name in METRICS:
        row[name] = float(payload["metrics"][name])
    efficiency = payload.get("efficiency", {})
    for name in EFFICIENCY:
        value = efficiency.get(name)
        row[name] = float(value) if value is not None else None
    wall = row["total_training_wall_seconds"]
    row["training_gpu_hours"] = wall / 3600.0 if wall is not None else None
    peak = row["peak_allocated_max_all_ranks_bytes"]
    row["peak_allocated_gib"] = peak / (2**30) if peak is not None else None
    return row, str(marker)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for k in (1, 2, 3, 4, 6, 8, 12):
        rows = []
        for seed in experiment.SEEDS:
            row, path = _load(k, seed)
            if row is None:
                missing.append({"k": k, "seed": seed, "path": path})
            else:
                rows.append(row)
                records.append(row)
        if len(rows) != len(experiment.SEEDS):
            continue
        summary: dict[str, Any] = {
            "k": k,
            "model": f"lts_ca_fourier_tenv_k{k}",
            "seeds": list(experiment.SEEDS),
        }
        for name in METRICS:
            values = [float(row[name]) for row in rows]
            summary[f"{name}_mean"] = statistics.fmean(values)
            summary[f"{name}_sample_std"] = statistics.stdev(values)
            summary[f"{name}_per_seed"] = values
        for name in (*EFFICIENCY, "training_gpu_hours", "peak_allocated_gib"):
            values = [float(row[name]) for row in rows if row[name] is not None]
            summary[f"{name}_mean"] = (
                statistics.fmean(values) if len(values) == len(rows) else None
            )
        summaries.append(summary)
    result = {
        "status": "INCOMPLETE" if missing else "BENCHMARK_COMPLETED",
        "format": experiment.FORMAT,
        "aggregation": "mean +/- sample standard deviation over seeds 7,17,27",
        "controlled_variable": (
            "K only; K=1 reused from official-backbone main table"
        ),
        "main_table": summaries,
        "per_seed": records,
        "missing": missing,
    }
    experiment.official.atomic_json(
        experiment.OUTPUT_ROOT / "combined_results.json", result
    )
    experiment.official.atomic_json(
        experiment.OUTPUT_ROOT / "aggregation_status.json", result
    )
    _write_csv(experiment.OUTPUT_ROOT / "k_table.csv", summaries)
    _write_csv(experiment.OUTPUT_ROOT / "per_seed.csv", records)
    return result


if __name__ == "__main__":
    payload = aggregate()
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "BENCHMARK_COMPLETED" else 2)
