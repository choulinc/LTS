"""Validate frozen experiment protocol files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REQUIRED_COMMON_KEYS = {
    "protocol",
    "dataset",
    "task",
    "normalization",
    "training",
    "checkpoint_selection",
    "evaluation",
    "profiling",
}


def load_protocol(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    missing = REQUIRED_COMMON_KEYS - payload.keys()
    if missing:
        raise ValueError(f"{source} is missing keys: {sorted(missing)}")
    return payload


def validate_protocol(payload: dict[str, Any]) -> None:
    counts = payload["dataset"]["split_counts"]
    if counts != {"train": 132, "validation": 30, "test": 30}:
        raise ValueError(f"unexpected DTOB6-0826 split counts: {counts}")
    training = payload["training"]
    if training["epochs"] != 25 or training["seeds"] != [7, 17, 27]:
        raise ValueError("DTOB6-0826 requires 25 epochs and seeds 7/17/27")
    if payload["checkpoint_selection"].get("uses_test") is not False:
        raise ValueError("held-out test data must not select checkpoints")
    if payload["task"]["prediction_steps"] != 99:
        raise ValueError("DTOB6-0826 requires 99 prediction steps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    payload = load_protocol(args.config)
    validate_protocol(payload)
    print(f"PASS: {args.config}")


if __name__ == "__main__":
    main()
