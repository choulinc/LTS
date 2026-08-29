from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ablations" / "droptest"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_dtob6_component_ablation_is_nested_and_complete() -> None:
    rows = _rows("dtob6_0826_components.csv")
    assert [row["model"] for row in rows] == [
        "lts",
        "lts_ca",
        "lts_ca_fourier_tenv",
    ]
    assert {row["seeds"] for row in rows} == {"7;17;27"}


def test_dtob6_routing_ablation_preserves_all_completed_k_values() -> None:
    paper = _rows("dtob6_0826_routing_k_paper.csv")
    raw = _rows("dtob6_0826_routing_k_raw.csv")
    expected = ["1", "2", "3", "4", "6", "8", "12"]
    assert [row["k"] for row in paper] == expected
    assert [row["k"] for row in raw] == expected
    assert all(float(row["parameters_m"]) == 8.335055 for row in paper)


def test_legacy_factorial_contains_all_binary_cells() -> None:
    rows = _rows("legacy_0813_ca_fourier_tenv_factorial_raw.csv")
    assert {row["ca_fourier_tenv_bits"] for row in rows} == {
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    }


def test_ablation_import_hashes_match_provenance() -> None:
    provenance = json.loads((RESULTS / "source_provenance.json").read_text())
    for source in provenance["sources"]:
        payload = (RESULTS / source["repository_file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["repository_sha256"]


def test_ablation_common_protocol_references_resolve() -> None:
    config_root = ROOT / "configs" / "droptest" / "ablations" / "dtob6_0826"
    for config in config_root.glob("*.yaml"):
        line = next(
            item
            for item in config.read_text().splitlines()
            if item.startswith("common_protocol:")
        )
        relative = line.split(":", 1)[1].strip()
        assert (config.parent / relative).resolve().is_file()
