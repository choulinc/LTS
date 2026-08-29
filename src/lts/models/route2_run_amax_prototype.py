# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prototype fixed-run PART max; intentionally not used by production Route 2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch


EmptyPartPolicy = Literal["error", "zeros"]


@dataclass(frozen=True)
class FixedRunMetadata:
    """CPU-validated fixed run boundaries and a device-movable dense reorder."""

    run_starts: tuple[int, ...]
    run_lengths: tuple[int, ...]
    run_part_ids: tuple[int, ...]
    dense_to_run: torch.Tensor
    num_parts: int
    node_count: int
    empty_part_policy: EmptyPartPolicy

    @property
    def has_empty_parts(self) -> bool:
        return len(self.run_starts) != self.num_parts

    def to(self, device: torch.device | str) -> "FixedRunMetadata":
        return replace(
            self,
            dense_to_run=self.dense_to_run.to(device=device),
        )


def build_fixed_run_metadata(
    node_part_index: torch.Tensor,
    *,
    num_parts: int,
    empty_part_policy: EmptyPartPolicy = "error",
) -> FixedRunMetadata:
    """Validate that every nonempty PART occupies exactly one source run."""

    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if empty_part_policy not in {"error", "zeros"}:
        raise ValueError("empty_part_policy must be 'error' or 'zeros'")
    source = torch.as_tensor(node_part_index)
    if (
        source.dtype == torch.bool
        or torch.is_floating_point(source)
        or torch.is_complex(source)
    ):
        raise TypeError("node_part_index must use an integer dtype")
    if source.ndim != 1 or source.numel() == 0:
        raise ValueError("fixed-run topology requires nonempty [N] indices")
    indices = source.detach().to(device="cpu", dtype=torch.long)
    if bool((indices < 0).any()) or bool((indices >= num_parts).any()):
        raise ValueError(f"PART indices must be in [0,{num_parts - 1}]")

    starts = torch.nonzero(
        torch.cat(
            (
                torch.ones(1, dtype=torch.bool),
                indices[1:] != indices[:-1],
            )
        ),
        as_tuple=False,
    ).reshape(-1)
    ends = torch.cat((starts[1:], torch.tensor([indices.numel()])))
    run_parts = indices[starts]
    runs_per_part = torch.bincount(run_parts, minlength=num_parts)
    fragmented = torch.nonzero(runs_per_part > 1, as_tuple=False).reshape(-1)
    if fragmented.numel():
        raise ValueError(
            "Fixed-run pooling requires one contiguous run per PART; "
            f"fragmented PARTs: {fragmented.tolist()}"
        )
    empty = torch.nonzero(runs_per_part == 0, as_tuple=False).reshape(-1)
    if empty.numel() and empty_part_policy == "error":
        raise ValueError(f"Empty PARTs: {empty.tolist()}")

    dense_to_run = torch.full((num_parts,), -1, dtype=torch.long)
    dense_to_run[run_parts] = torch.arange(run_parts.numel())
    return FixedRunMetadata(
        run_starts=tuple(int(value) for value in starts),
        run_lengths=tuple(
            int(end - start)
            for start, end in zip(starts.tolist(), ends.tolist())
        ),
        run_part_ids=tuple(int(value) for value in run_parts),
        dense_to_run=dense_to_run.to(device=source.device),
        num_parts=int(num_parts),
        node_count=int(indices.numel()),
        empty_part_policy=empty_part_policy,
    )


def fixed_run_amax(
    node_features: torch.Tensor,
    metadata: FixedRunMetadata,
) -> torch.Tensor:
    """Reduce each contiguous source run, then reorder runs to dense PART IDs."""

    if node_features.ndim != 3 or not torch.is_floating_point(node_features):
        raise ValueError("node_features must be a floating tensor [B,N,C]")
    if node_features.shape[1] != metadata.node_count:
        raise ValueError(
            f"Expected {metadata.node_count} nodes, got "
            f"{node_features.shape[1]}"
        )
    if metadata.dense_to_run.device != node_features.device:
        raise ValueError("Fixed-run metadata must be on the feature device")

    run_maxima = torch.stack(
        [
            torch.amax(
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
        return run_maxima.index_select(1, metadata.dense_to_run)

    zero_part = node_features.new_zeros(
        (node_features.shape[0], 1, node_features.shape[2])
    )
    extended = torch.cat((run_maxima, zero_part), dim=1)
    empty_index = run_maxima.shape[1]
    safe_reorder = torch.where(
        metadata.dense_to_run >= 0,
        metadata.dense_to_run,
        torch.full_like(metadata.dense_to_run, empty_index),
    )
    return extended.index_select(1, safe_reorder)


__all__ = [
    "FixedRunMetadata",
    "build_fixed_run_metadata",
    "fixed_run_amax",
]
