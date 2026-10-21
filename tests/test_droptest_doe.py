"""CPU-only release checks; no solver, torch, raw data, or network required."""

import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "doe", ROOT / "scripts/prepare_droptest_doe.py"
)
doe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doe)
FIXTURE = ROOT / "tests/fixtures/droptest_doe"


def splits():
    return {
        key: (ROOT / f"metadata/splits/dtob6_0826_{key}.txt").read_text().splitlines()
        for key in ("train", "validation", "test")
    }


def test_metadata_matches_frozen_bytes_and_full_factorial(tmp_path):
    _generator, rows = doe.generate_metadata(tmp_path / "metadata")
    canonical = json.loads((doe.META / "planned_summary.json").read_text())
    assert json.loads(json.dumps(rows)) == canonical
    assert [r["run_id"] for r in rows] == [f"run{i:04d}" for i in range(1, 193)]
    combinations = {tuple(r["parameters"]["e_scales"].values()) for r in rows}
    assert combinations == set(itertools.product([0.8, 1.2], repeat=5))
    assert [r["parameters"]["rwall_orientation_deg"] for r in rows[:6]] == [
        [0, 0, 0],
        [10, 0, 0],
        [-10, 0, 0],
        [0, 10, 0],
        [0, -10, 0],
        [0, 0, 10],
    ]
    assert all(r["timestamp"] is None for r in rows)
    for path in doe.META.glob("*.json"):
        if path.name in {"planned_summary.json", "global_features.json"}:
            assert (tmp_path / "metadata" / path.name).read_bytes() == path.read_bytes()
    doe.validate_splits(rows, splits())


@pytest.mark.parametrize("corruption", ["overlap", "missing", "unknown", "leakage"])
def test_corrupted_splits_rejected(corruption):
    rows = doe.planned_runs(doe.modules()[0])
    partitions = splits()
    if corruption == "overlap":
        partitions["test"][0] = partitions["train"][0]
    elif corruption == "missing":
        partitions["train"].pop()
    elif corruption == "unknown":
        partitions["test"][0] = "run9999"
    else:
        partitions["train"][0], partitions["test"][0] = (
            partitions["test"][0],
            partitions["train"][0],
        )
    with pytest.raises(ValueError):
        doe.validate_splits(rows, partitions)


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    templates = tmp_path / "templates"
    shutil.copytree(FIXTURE, templates)
    monkeypatch.setattr(
        doe,
        "TEMPLATE_HASHES",
        {name: doe.sha256(templates / name) for name in doe.TEMPLATE_HASHES},
    )
    output = tmp_path / "output"
    output.mkdir()
    return templates, output


@pytest.mark.parametrize("index", [0, 1, 3, 191])
def test_real_mutator_scales_materials_rotates_wall_and_preserves_engine(
    synthetic, index
):
    templates, output = synthetic
    generator = doe.modules()[0]
    row = doe.planned_runs(generator)[index]
    case = doe.prepare_case(generator, row, templates, output)
    receipt = json.loads((case / "preparation_receipt.json").read_text())
    scales = row["parameters"]["e_scales"]
    for mid, base in {1: 2000, 4: 2000, 5: 70000, 8: 15000, 9: 2000}.items():
        assert (
            receipt["verification"]["materials"][str(mid)]["actual_E"]
            == base * scales[mid]
        )
    assert receipt["verification"]["walls"]["/RWALL/PLANE/1"][
        "vector"
    ] == pytest.approx(
        {
            0: [0, 0, 1],
            1: [0, -0.173648, 0.984808],
            3: [0.173648, 0, 0.984808],
            191: [0, 0, 1],
        }[index],
        abs=1e-6,
    )
    engine = "Cell_Phone_Drop_0001.rad"
    assert (case / engine).read_bytes() == (templates / engine).read_bytes()
    assert receipt["historical_dynamic_targets_verified"] is False
    with pytest.raises(FileExistsError):
        doe.prepare_case(generator, row, templates, output)


def test_template_tamper_rejected_before_case_created(synthetic):
    templates, output = synthetic
    (templates / "Cell_Phone_Drop_0001.rad").write_text("tampered")
    generator = doe.modules()[0]
    with pytest.raises(ValueError, match="Template hash"):
        doe.prepare_case(generator, doe.planned_runs(generator)[0], templates, output)
    assert not list(output.iterdir())


def test_silent_noop_mutation_rejected(synthetic, monkeypatch):
    templates, output = synthetic
    generator = doe.modules()[0]
    monkeypatch.setattr(
        generator,
        "modify_radioss_file",
        lambda source, dest, *args: shutil.copyfile(source, dest),
    )
    with pytest.raises(ValueError, match="not correctly changed"):
        doe.prepare_case(generator, doe.planned_runs(generator)[0], templates, output)
    assert not (output / "run0001/preparation_receipt.json").exists()


def test_source_tamper_rejected(tmp_path, monkeypatch):
    source = tmp_path / "upstream"
    shutil.copytree(doe.SOURCE, source)
    (source / "LICENSE.txt").write_text("changed")
    monkeypatch.setattr(doe, "SOURCE", source)
    with pytest.raises(ValueError, match="Upstream source changed"):
        doe.modules()


def test_metadata_output_is_not_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        doe.generate_metadata(output)
    assert sentinel.read_text() == "keep"
