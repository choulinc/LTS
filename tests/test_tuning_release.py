from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validation_search_contains_twelve_effective_candidates() -> None:
    path = ROOT / "results/tuning/dtob6_0828_validation_search.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    winner = min(rows, key=lambda row: float(row["validation_metric_mean"]))
    assert winner["fourier_bands"] == "16"
    assert winner["temporal_envelope_width"] == "128"
    assert winner["detach_component_pool_source"] == "true"


def test_tuned_heldout_result_contains_exactly_three_seeds() -> None:
    path = ROOT / "results/main_tables/dtob6_0828_tuned_per_seed_raw.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["seed"]) for row in rows] == [7, 17, 27]
    assert {int(float(row["total_parameters"])) for row in rows} == {8_348_175}


def test_tuned_matched_profile_identity() -> None:
    path = ROOT / "results/profiling/dtob6_0828_lts_tuned_matched_h200.json"
    payload = json.loads(path.read_text())
    assert payload["status"] == "PASS"
    assert payload["hardware"] == "NVIDIA H200"
    assert payload["warmup_steps"] == 20
    assert payload["measured_steps"] == 50
    assert payload["parameter_count"] == 8_348_175
    assert payload["peak_allocated_gib"] == 4.939683437347412
