#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce official DropTest192 metadata or prepare independently supplied decks.

Standard library only. This command does not run a solver or create targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path

from droptest_deck_checks import engine_timing, physical_inputs, verify_case

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/physicsnemo_doe"
META = ROOT / "metadata/doe/droptest192"
DROP = SOURCE / "examples/structural_mechanics/openradioss_dataset_gen/drop_test"
TEMPLATE_HASHES = {
    "Cell_Phone_Drop_0000.rad": "435499d744ad560a843d9a21210483bbe74c9fc6e2775da4a232ce609cf097e2",
    "Cell_Phone_Drop_0001.rad": "1a090204d2913c1759429c45a65d8b67852a1889ed8cac73d6e5eb38cb6d98ff",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def modules():
    receipt = json.loads((SOURCE / "source_receipt.json").read_text())
    for entry in receipt["files"]:
        if sha256(SOURCE / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Upstream source changed: {entry['path']}")
    loaded = []
    for name in ("generate_dataset", "restructure_global_features"):
        spec = importlib.util.spec_from_file_location(name, DROP / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def planned_runs(generator):
    setup = generator.DEFAULT_EXPERIMENT_SETUP
    mids = sorted(setup["e_scales"])
    combos = [
        dict(zip(mids, row))
        for row in itertools.product(*(setup["e_scales"][mid] for mid in mids))
    ]
    return [
        {
            "run_id": f"run{i:04d}",
            "timestamp": None,
            "parameters": {"e_scales": scales, "rwall_orientation_deg": list(angles)},
        }
        for i, (scales, angles) in enumerate(
            itertools.product(combos, setup["rwall_orientations"]), 1
        )
    ]


def validate_splits(runs, splits):
    expected = {row["run_id"] for row in runs}
    if len(runs) != len(expected) or len(runs) != 192:
        raise ValueError("Expected 192 unique DOE cases")
    if {key: len(ids) for key, ids in splits.items()} != {
        "train": 132,
        "validation": 30,
        "test": 30,
    }:
        raise ValueError("Unexpected frozen split counts")
    all_ids = list(itertools.chain.from_iterable(splits.values()))
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != expected:
        raise ValueError("Split overlap, duplicate, missing or unknown case")
    # Every six-orientation material combination stays within one partition.
    groups = {}
    by_id = {row["run_id"]: row for row in runs}
    for split, ids in splits.items():
        for case_id in ids:
            scales = by_id[case_id]["parameters"]["e_scales"]
            key = tuple(scales[mid] for mid in sorted(scales))
            if key in groups and groups[key] != split:
                raise ValueError("Material-combination leakage between splits")
            groups[key] = split


def generate_metadata(output):
    generator, converter = modules()
    runs = planned_runs(generator)
    splits = {
        key: (ROOT / f"metadata/splits/dtob6_0826_{key}.txt").read_text().splitlines()
        for key in ("train", "validation", "test")
    }
    validate_splits(runs, splits)
    frozen = json.loads((ROOT / "metadata/datasets/dtob6_0826.json").read_text())
    for key, ids in splits.items():
        digest = hashlib.sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != frozen["ordered_split_id_sha256"][key]:
            raise ValueError(f"Frozen ordered split hash changed: {key}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "planned_summary.json", runs)
    with (output / "planned_doe.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(generator._flatten_row(runs[0]))
        )
        writer.writeheader()
        writer.writerows(generator._flatten_row(row) for row in runs)
    converter.convert_global_features(
        output / "planned_summary.json", output / "global_features.json"
    )
    hashes = json.loads((META / "SHA256.json").read_text())
    for name, expected in hashes.items():
        if sha256(output / name) != expected:
            raise ValueError(f"Canonical metadata mismatch: {name}")
    if sha256(output / "global_features.json") != frozen["global_features_sha256"]:
        raise ValueError("Frozen global-feature hash mismatch")
    return generator, runs


def prepare_case(generator, row, templates, output):
    # Check both originals before making a case directory. Reject includes and
    # missing E headers rather than trusting the upstream mutator's silent no-op.
    for name, digest in TEMPLATE_HASHES.items():
        if sha256(templates / name) != digest:
            raise ValueError(f"Template hash mismatch: {name}")
    starter_name, engine_name = TEMPLATE_HASHES
    starter, engine = templates / starter_name, templates / engine_name
    material, walls = physical_inputs(starter)
    timing = engine_timing(engine)
    case = output / row["run_id"]
    case.mkdir(exist_ok=False)
    params = row["parameters"]
    generator.modify_radioss_file(
        str(starter),
        str(case / starter_name),
        params["e_scales"],
        tuple(params["rwall_orientation_deg"]),
    )
    shutil.copyfile(engine, case / engine_name)
    verification = verify_case(
        case / starter_name,
        case / engine_name,
        material,
        walls,
        params,
        generator,
        TEMPLATE_HASHES[engine_name],
    )
    write_json(case / f"{row['run_id']}.json", row)
    write_json(
        case / "preparation_receipt.json",
        {
            "schema_version": 1,
            "status": "decks_prepared_not_simulated",
            "case": row,
            "template_sha256": TEMPLATE_HASHES,
            "source_receipt_sha256": sha256(SOURCE / "source_receipt.json"),
            "wrapper_sha256": sha256(Path(__file__)),
            "checks_sha256": sha256(
                Path(__file__).with_name("droptest_deck_checks.py")
            ),
            "command": [sys.executable, *sys.argv],
            "timing_s": timing,
            "verification": verification,
            "historical_dynamic_targets_verified": False,
        },
    )
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "--templates", type=Path, help="Independently obtained original decks"
    )
    parser.add_argument("--case", default="run0001", help="Canonical run ID, or all")
    args = parser.parse_args()
    if args.case != "all" and args.case not in {f"run{i:04d}" for i in range(1, 193)}:
        parser.error("--case must be run0001..run0192 or all")
    output = args.output.resolve()
    # Keep outputs out of the source release, including restricted deck copies.
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the repository")
    generator, runs = generate_metadata(output)
    if args.templates:
        for row in runs:
            if args.case in (row["run_id"], "all"):
                prepare_case(generator, row, args.templates.resolve(), output)
    print(f"PASS: 192 canonical designs validated; output: {output}")


if __name__ == "__main__":
    main()
