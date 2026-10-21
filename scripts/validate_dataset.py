#!/usr/bin/env python3
"""Validate a local DropTest copy against the frozen DTOB6-0826 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def split_ids(root: Path, split: str) -> list[str]:
    directory = root / split
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return [path.stem for path in sorted(directory.glob("*.vtu"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    expected = config["dataset"]
    observed: dict[str, list[str]] = {}
    failures: list[str] = []

    for split in ("train", "validation", "test"):
        ids = split_ids(args.data_root, split)
        observed[split] = ids
        if len(ids) != expected["split_counts"][split]:
            failures.append(
                f"{split}: count {len(ids)} != "
                f"{expected['split_counts'][split]}"
            )
        digest = sha256_json(ids)
        if digest != expected["split_id_hashes"][split]:
            failures.append(
                f"{split}: ordered ID hash {digest} != "
                f"{expected['split_id_hashes'][split]}"
            )

    overlap = (
        set(observed["train"]) & set(observed["validation"])
        | set(observed["train"]) & set(observed["test"])
        | set(observed["validation"]) & set(observed["test"])
    )
    if overlap:
        failures.append(f"split overlap: {sorted(overlap)}")

    for filename, key in (
        ("split_manifest.json", "split_manifest_sha256"),
        ("global_features.json", "global_features_sha256"),
    ):
        path = args.data_root / filename
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        digest = sha256_file(path)
        if digest != expected[key]:
            failures.append(f"{filename}: SHA-256 {digest} != {expected[key]}")

    if failures:
        raise SystemExit("FAIL\n- " + "\n- ".join(failures))
    print("PASS: dataset matches DTOB6-0826 split and metadata hashes")


if __name__ == "__main__":
    main()
