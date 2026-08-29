import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_table_has_five_baselines_and_lts() -> None:
    path = ROOT / "results/main_tables/dtob6_0826_paper.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["model"] for row in rows] == [
        "Transolver",
        "GeoTransolver",
        "GeoTS-FLARE",
        "Transolver++",
        "Transolver-3",
        "LTS",
    ]


def test_paper_table_uses_training_allocated_memory() -> None:
    path = ROOT / "results/main_tables/dtob6_0826_paper.csv"
    with path.open(newline="") as stream:
        rows = {row["model"]: row for row in csv.DictReader(stream)}
    assert float(rows["Transolver"]["peak_train_allocated_gib"]) == 28.154
    assert float(rows["LTS"]["peak_train_allocated_gib"]) == 4.932


def test_paper_table_uses_tuned_lts_and_matched_profile() -> None:
    path = ROOT / "results/main_tables/dtob6_0828_tuned_paper.csv"
    with path.open(newline="") as stream:
        rows = {row["model"]: row for row in csv.DictReader(stream)}
    lts = rows["LTS"]
    assert float(lts["displacement_rel_l2_percent_mean"]) == 0.6306
    assert float(lts["stress_rel_l2_percent_mean"]) == 2.3498
    assert float(lts["train_step_latency_ms_mean"]) == 27.723
    assert float(lts["parameters_million"]) == 8.348
