import csv
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_table_has_five_baselines_and_base_full_lts() -> None:
    path = ROOT / "results/main_tables/dtob6_0826_paper.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["model"] for row in rows] == [
        "Transolver",
        "GeoTransolver",
        "GeoTS-FLARE",
        "Transolver++",
        "Transolver-3",
        "LTS (Base)",
        "LTS (Full)",
    ]


def test_paper_table_uses_training_allocated_memory() -> None:
    path = ROOT / "results/main_tables/dtob6_0826_paper.csv"
    with path.open(newline="") as stream:
        rows = {row["model"]: row for row in csv.DictReader(stream)}
    assert round(float(rows["Transolver"]["peak_train_allocated_gib"]), 2) == 28.15
    assert round(float(rows["LTS (Base)"]["peak_train_allocated_gib"]), 2) == 4.84
    assert round(float(rows["LTS (Full)"]["peak_train_allocated_gib"]), 2) == 4.93


def test_train_step_profile_matches_paper_and_has_correct_seed_boundary():
    path = ROOT / "results/main_tables/dtob6_0826_paper.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [round(float(r["train_step_latency_ms_mean"]), 2) for r in rows] == [
        193.12, 362.14, 276.43, 26.91, 177.19, 24.06, 29.75,
    ]
    evidence = json.loads((path.parent / "paper_table_provenance.json").read_text())
    assert evidence["paper_csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for name, digest in evidence["inputs_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert evidence["training_profile"]["warmup_steps"] == 20
    assert evidence["training_profile"]["measured_steps"] == 50
    assert evidence["accuracy_and_inference"]["seeds"] == [7, 17, 27]
    spec = importlib.util.spec_from_file_location("table_renderer", ROOT / "scripts/render_main_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (path.parent / "dtob6_0826_main_table.tex").read_text() == module.render(rows)


def test_paper_values_are_recomputed_from_seed_records():
    with (ROOT / "results/main_tables/dtob6_0826_per_seed_raw.csv").open(newline="") as f:
        raw = list(csv.DictReader(f))
    with (ROOT / "results/main_tables/dtob6_0826_paper.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    labels = {"LTS (Base)": "lts", "LTS (Full)": "lts_ca_fourier_tenv",
              "Transolver": "transolver", "GeoTransolver": "geotransolver",
              "GeoTS-FLARE": "geots_flare", "Transolver++": "transolver_plus",
              "Transolver-3": "transolver3"}
    for row in rows:
        seeds = [r for r in raw if r["model"] == labels[row["model"]]]
        assert {int(r["seed"]) for r in seeds} == {7, 17, 27}
        for source, target, factor in (
            ("displacement_physical_relative_l2_macro", "displacement_rel_l2_percent", 100),
            ("stress_physical_relative_l2_macro", "stress_rel_l2_percent", 100),
            ("full_99_step_inference_seconds", "inference_latency_ms_per_99step_trajectory", 1000),
        ):
            samples = [float(r[source]) * factor for r in seeds]
            assert math.isclose(float(row[target + "_mean"]), statistics.mean(samples), rel_tol=1e-14)
            assert math.isclose(float(row[target + "_sample_std"]), statistics.stdev(samples), rel_tol=1e-14)


def test_original_lts_profile_samples_reproduce_reported_statistics():
    evidence = json.loads((ROOT / "results/profiling/dtob6_0826_gc_controlled.json").read_text())
    for model in ("lts", "lts_ca_fourier_tenv"):
        profile = evidence["profiles"][model]
        values = profile["raw_train_step_latency_ms"]
        assert len(values) == 50
        assert math.isclose(statistics.mean(values), profile["train_step_latency_ms_mean"], rel_tol=1e-12)
        assert math.isclose(statistics.stdev(values), profile["train_step_latency_ms_sample_std"], rel_tol=1e-12)
