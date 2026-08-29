#!/usr/bin/env python3
"""Construct the complete DTOB6-0826 LTS and verify its parameter identity."""

from __future__ import annotations

import torch

from lts.models import LTSConfig, build_lts


EXPECTED_PARAMETERS = 8_335_055


def main() -> None:
    # The parameter count is topology-size independent. One contiguous node per
    # component is sufficient for a construction-only identity check.
    node_component_index = torch.arange(17, dtype=torch.long)
    torch.manual_seed(7)
    model = build_lts(
        LTSConfig(), node_component_index=node_component_index
    )
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != EXPECTED_PARAMETERS:
        raise SystemExit(
            f"FAIL: parameter count {observed:,} != {EXPECTED_PARAMETERS:,}"
        )
    print(f"PASS: complete DTOB6-0826 LTS has {observed:,} parameters")


if __name__ == "__main__":
    main()
