# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prototype fixed-run PART mean; intentionally not used by production Route 2."""

from __future__ import annotations

import torch

from .route2_run_amax_prototype import FixedRunMetadata


def fixed_run_mean(
    node_features: torch.Tensor,
    metadata: FixedRunMetadata,
) -> torch.Tensor:
    """Mean each contiguous source run, then reorder to dense PART IDs."""

    if node_features.ndim != 3 or not torch.is_floating_point(node_features):
        raise ValueError("node_features must be a floating tensor [B,N,C]")
    if node_features.shape[1] != metadata.node_count:
        raise ValueError(
            f"Expected {metadata.node_count} nodes, got "
            f"{node_features.shape[1]}"
        )
    if metadata.dense_to_run.device != node_features.device:
        raise ValueError("Fixed-run metadata must be on the feature device")

    run_means = torch.stack(
        [
            torch.mean(
                node_features[:, start : start + length, :],
                dim=1,
            )
            for start, length in zip(
                metadata.run_starts, metadata.run_lengths
            )
        ],
        dim=1,
    )
    if not metadata.has_empty_parts:
        return run_means.index_select(1, metadata.dense_to_run)

    zero_part = node_features.new_zeros(
        (node_features.shape[0], 1, node_features.shape[2])
    )
    extended = torch.cat((run_means, zero_part), dim=1)
    empty_index = run_means.shape[1]
    safe_reorder = torch.where(
        metadata.dense_to_run >= 0,
        metadata.dense_to_run,
        torch.full_like(metadata.dense_to_run, empty_index),
    )
    return extended.index_select(1, safe_reorder)


def fixed_run_mean_fp32_accum(
    node_features: torch.Tensor,
    metadata: FixedRunMetadata,
) -> torch.Tensor:
    """Accumulate fixed-run means in FP32, then restore the input dtype."""

    if node_features.ndim != 3 or not torch.is_floating_point(node_features):
        raise ValueError("node_features must be a floating tensor [B,N,C]")
    if node_features.dtype == torch.float32:
        return fixed_run_mean(node_features, metadata)
    return fixed_run_mean(
        node_features.to(dtype=torch.float32), metadata
    ).to(dtype=node_features.dtype)


__all__ = ["fixed_run_mean", "fixed_run_mean_fp32_accum"]
