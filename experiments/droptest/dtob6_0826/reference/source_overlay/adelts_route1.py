# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Shared-backbone ADELT Route 1 stress residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from physicsnemo.models.transolver.light_transolver import LightTransolver


EmptyPartPolicy = Literal["error", "zeros"]
MaxPoolBackend = Literal["scatter", "segmented"]


@dataclass(frozen=True)
class PartMeanMaxPool:
    """Memory-safe PART statistics without a node-by-PART one-hot tensor."""

    mean: torch.Tensor
    maximum: torch.Tensor
    counts: torch.Tensor

    @property
    def summary(self) -> torch.Tensor:
        return torch.cat((self.mean, self.maximum), dim=-1)


class FixedPartTopology(nn.Module):
    """Validated, device-movable cache for a fixed node-to-PART mapping.

    Validation and empty-PART discovery happen once on CPU.  Non-persistent
    buffers preserve checkpoint compatibility while following the parent model
    across device moves.
    """

    def __init__(
        self,
        node_part_index: torch.Tensor,
        *,
        num_parts: int,
        empty_part_policy: EmptyPartPolicy = "error",
    ) -> None:
        super().__init__()
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
        if source.ndim not in {1, 2} or source.numel() == 0:
            raise ValueError("node_part_index must be non-empty [N] or [B,N]")

        indices = source.detach().to(device="cpu", dtype=torch.long).clone()
        index_batches = indices.unsqueeze(0) if indices.ndim == 1 else indices
        if index_batches.shape[1] <= 0:
            raise ValueError("Fixed PART topology requires at least one node")
        if bool((index_batches < 0).any()) or bool(
            (index_batches >= num_parts).any()
        ):
            raise ValueError(f"PART indices must be in [0,{num_parts - 1}]")

        mapping_batches = int(index_batches.shape[0])
        offsets = (
            torch.arange(mapping_batches, dtype=torch.long) * int(num_parts)
        )
        flat_index = (
            index_batches + offsets[:, None]
        ).reshape(-1)
        counts = torch.bincount(
            flat_index, minlength=mapping_batches * int(num_parts)
        ).reshape(mapping_batches, int(num_parts))
        part_permutation = torch.argsort(
            index_batches, dim=1, stable=True
        )
        segment_lengths = counts.clone()
        segment_offsets = torch.cumsum(segment_lengths, dim=1) - segment_lengths
        empty = counts == 0
        has_empty_parts = bool(empty.any())
        if has_empty_parts and empty_part_policy == "error":
            locations = torch.nonzero(empty, as_tuple=False).tolist()
            raise ValueError(
                "Empty PART entries at [batch, part] "
                f"{locations}"
            )

        self.num_parts = int(num_parts)
        self.node_count = int(index_batches.shape[1])
        self.shared_across_batch = indices.ndim == 1
        self.mapping_batch_size = None if indices.ndim == 1 else mapping_batches
        self.has_empty_parts = has_empty_parts
        buffer_device = source.device
        self.register_buffer(
            "node_part_index",
            indices.to(device=buffer_device),
            persistent=False,
        )
        self.register_buffer(
            "flat_part_index",
            flat_index.to(device=buffer_device),
            persistent=False,
        )
        self.register_buffer(
            "part_counts", counts.to(device=buffer_device), persistent=False
        )
        self.register_buffer(
            "empty_part_mask",
            empty.to(device=buffer_device),
            persistent=False,
        )
        self.register_buffer(
            "part_permutation",
            part_permutation.to(device=buffer_device),
            persistent=False,
        )
        self.register_buffer(
            "part_segment_offsets",
            segment_offsets.to(device=buffer_device),
            persistent=False,
        )
        self.register_buffer(
            "part_segment_lengths",
            segment_lengths.to(device=buffer_device),
            persistent=False,
        )

    def _check_shape(self, *, batch_size: int, node_count: int) -> None:
        if node_count != self.node_count:
            raise ValueError(
                f"Expected {self.node_count} fixed-topology nodes, "
                f"got {node_count}"
            )
        if (
            not self.shared_across_batch
            and batch_size != self.mapping_batch_size
        ):
            raise ValueError(
                "Fixed batched PART mapping expects batch size "
                f"{self.mapping_batch_size}, got {batch_size}"
            )

    def expanded_indices(
        self, *, batch_size: int, node_count: int
    ) -> torch.Tensor:
        """Return ``[B,N]`` indices without value-dependent device checks."""

        self._check_shape(batch_size=batch_size, node_count=node_count)
        if self.shared_across_batch:
            return self.node_part_index.unsqueeze(0).expand(batch_size, -1)
        return self.node_part_index

    def flat_indices(
        self, *, batch_size: int, node_count: int
    ) -> torch.Tensor:
        """Return batch-offset flat indices, reusing the common B=1 cache."""

        indices = self.expanded_indices(
            batch_size=batch_size, node_count=node_count
        )
        if not self.shared_across_batch or batch_size == 1:
            return self.flat_part_index
        offsets = (
            torch.arange(
                batch_size,
                device=indices.device,
                dtype=torch.long,
            )
            * self.num_parts
        )
        return (indices + offsets[:, None]).reshape(-1)

    def counts_for_batch(self, *, batch_size: int) -> torch.Tensor:
        """Return cached ``[B,P]`` node counts."""

        if self.shared_across_batch:
            return self.part_counts.expand(batch_size, -1)
        if batch_size != self.mapping_batch_size:
            raise ValueError(
                "Fixed batched PART mapping expects batch size "
                f"{self.mapping_batch_size}, got {batch_size}"
            )
        return self.part_counts

    def empty_mask_for_batch(self, *, batch_size: int) -> torch.Tensor:
        """Return the initialization-time empty-PART mask."""

        if self.shared_across_batch:
            return self.empty_part_mask.expand(batch_size, -1)
        if batch_size != self.mapping_batch_size:
            raise ValueError(
                "Fixed batched PART mapping expects batch size "
                f"{self.mapping_batch_size}, got {batch_size}"
            )
        return self.empty_part_mask

    def permutation_for_batch(
        self, *, batch_size: int, node_count: int
    ) -> torch.Tensor:
        """Return cached node permutations that make each PART contiguous."""

        self._check_shape(batch_size=batch_size, node_count=node_count)
        if self.shared_across_batch:
            return self.part_permutation.expand(batch_size, -1)
        return self.part_permutation

    def segment_offsets_for_batch(
        self, *, batch_size: int
    ) -> torch.Tensor:
        """Return cached starts for the contiguous PART segments."""

        if self.shared_across_batch:
            return self.part_segment_offsets.expand(batch_size, -1)
        if batch_size != self.mapping_batch_size:
            raise ValueError(
                "Fixed batched PART mapping expects batch size "
                f"{self.mapping_batch_size}, got {batch_size}"
            )
        return self.part_segment_offsets

    def segment_lengths_for_batch(
        self, *, batch_size: int
    ) -> torch.Tensor:
        """Return cached lengths for the contiguous PART segments."""

        if self.shared_across_batch:
            return self.part_segment_lengths.expand(batch_size, -1)
        if batch_size != self.mapping_batch_size:
            raise ValueError(
                "Fixed batched PART mapping expects batch size "
                f"{self.mapping_batch_size}, got {batch_size}"
            )
        return self.part_segment_lengths


def _expanded_part_index(
    node_part_index: torch.Tensor,
    *,
    batch_size: int,
    node_count: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.as_tensor(node_part_index, device=device)
    if (
        indices.dtype == torch.bool
        or torch.is_floating_point(indices)
        or torch.is_complex(indices)
    ):
        raise TypeError("node_part_index must use an integer dtype")
    indices = indices.to(dtype=torch.long)
    if indices.ndim == 1:
        if indices.shape != (node_count,):
            raise ValueError(
                f"Expected node_part_index [{node_count}], got "
                f"{tuple(indices.shape)}"
            )
        return indices.unsqueeze(0).expand(batch_size, -1)
    if indices.ndim == 2:
        if indices.shape != (batch_size, node_count):
            raise ValueError(
                "Expected node_part_index "
                f"[{batch_size},{node_count}], got {tuple(indices.shape)}"
            )
        return indices
    raise ValueError("node_part_index must have shape [N] or [B,N]")


def part_mean_max_pool(
    node_features: torch.Tensor,
    node_part_index: torch.Tensor,
    *,
    num_parts: int,
    empty_part_policy: EmptyPartPolicy = "error",
) -> PartMeanMaxPool:
    """Pool ``[B,N,C]`` node features into mean/max ``[B,P,C]`` tensors."""

    if node_features.ndim != 3 or not torch.is_floating_point(node_features):
        raise ValueError("node_features must be a floating tensor [B,N,C]")
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if empty_part_policy not in {"error", "zeros"}:
        raise ValueError("empty_part_policy must be 'error' or 'zeros'")

    batch_size, node_count, channels = node_features.shape
    indices = _expanded_part_index(
        node_part_index,
        batch_size=batch_size,
        node_count=node_count,
        device=node_features.device,
    )
    if indices.numel() == 0:
        raise ValueError("PART pooling requires at least one node")
    if bool((indices < 0).any()) or bool((indices >= num_parts).any()):
        raise ValueError(f"PART indices must be in [0,{num_parts - 1}]")

    batch_offsets = (
        torch.arange(batch_size, device=node_features.device, dtype=torch.long)
        * num_parts
    )
    flat_index = (indices + batch_offsets[:, None]).reshape(-1)
    flat_features = node_features.reshape(-1, channels)
    pooled_shape = (batch_size * num_parts, channels)

    sums = node_features.new_zeros(pooled_shape)
    sums.index_add_(0, flat_index, flat_features)
    counts = torch.zeros(
        batch_size * num_parts,
        device=node_features.device,
        dtype=torch.long,
    )
    counts.index_add_(0, flat_index, torch.ones_like(flat_index))

    maximum = node_features.new_full(pooled_shape, -torch.inf)
    maximum = maximum.scatter_reduce(
        0,
        flat_index[:, None].expand(-1, channels),
        flat_features,
        reduce="amax",
        include_self=True,
    )

    empty = counts == 0
    if bool(empty.any()) and empty_part_policy == "error":
        locations = torch.nonzero(
            empty.reshape(batch_size, num_parts), as_tuple=False
        ).detach()
        raise ValueError(
            "Empty PART entries at [batch, part] "
            f"{locations.cpu().tolist()}"
        )

    mean = sums / counts.clamp_min(1).to(node_features.dtype)[:, None]
    if bool(empty.any()):
        maximum = maximum.masked_fill(empty[:, None], 0.0)

    return PartMeanMaxPool(
        mean=mean.reshape(batch_size, num_parts, channels),
        maximum=maximum.reshape(batch_size, num_parts, channels),
        counts=counts.reshape(batch_size, num_parts),
    )


def broadcast_part_features(
    part_features: torch.Tensor,
    node_part_index: torch.Tensor,
) -> torch.Tensor:
    """Broadcast ``[B,P,C]`` PART features to matching nodes as ``[B,N,C]``."""

    if part_features.ndim != 3:
        raise ValueError("part_features must have shape [B,P,C]")
    batch_size, num_parts, channels = part_features.shape
    node_count = int(torch.as_tensor(node_part_index).shape[-1])
    indices = _expanded_part_index(
        node_part_index,
        batch_size=batch_size,
        node_count=node_count,
        device=part_features.device,
    )
    if indices.numel() == 0:
        raise ValueError("PART broadcast requires at least one node")
    if bool((indices < 0).any()) or bool((indices >= num_parts).any()):
        raise ValueError(f"PART indices must be in [0,{num_parts - 1}]")
    return torch.gather(
        part_features,
        dim=1,
        index=indices[:, :, None].expand(-1, -1, channels),
    )


def fixed_part_segmented_max(
    node_features: torch.Tensor,
    topology: FixedPartTopology,
) -> torch.Tensor:
    """Max pool fixed PARTs after one cached node permutation."""

    if node_features.ndim != 3 or not torch.is_floating_point(node_features):
        raise ValueError("node_features must be a floating tensor [B,N,C]")
    batch_size, node_count, channels = node_features.shape
    permutation = topology.permutation_for_batch(
        batch_size=batch_size, node_count=node_count
    )
    if topology.shared_across_batch:
        ordered = node_features.index_select(1, permutation[0])
    else:
        ordered = torch.gather(
            node_features,
            dim=1,
            index=permutation[:, :, None].expand(-1, -1, channels),
        )
    lengths = topology.segment_lengths_for_batch(
        batch_size=batch_size
    ).reshape(-1)
    maximum = torch.segment_reduce(
        ordered.reshape(-1, channels),
        reduce="max",
        lengths=lengths,
        axis=0,
        unsafe=True,
    ).reshape(batch_size, topology.num_parts, channels)
    if topology.has_empty_parts:
        empty = topology.empty_mask_for_batch(batch_size=batch_size)
        maximum = maximum.masked_fill(empty[:, :, None], 0.0)
    return maximum


def fixed_part_mean_max_pool(
    node_features: torch.Tensor,
    topology: FixedPartTopology,
    *,
    max_backend: MaxPoolBackend = "scatter",
) -> PartMeanMaxPool:
    """Mean/max pool with initialization-time mapping validation and counts."""

    if node_features.ndim != 3 or not torch.is_floating_point(node_features):
        raise ValueError("node_features must be a floating tensor [B,N,C]")
    if max_backend not in {"scatter", "segmented"}:
        raise ValueError("max_backend must be 'scatter' or 'segmented'")
    batch_size, node_count, channels = node_features.shape
    flat_index = topology.flat_indices(
        batch_size=batch_size, node_count=node_count
    )
    flat_features = node_features.reshape(-1, channels)
    pooled_shape = (batch_size * topology.num_parts, channels)

    sums = node_features.new_zeros(pooled_shape)
    sums.index_add_(0, flat_index, flat_features)
    if max_backend == "scatter":
        maximum = node_features.new_full(pooled_shape, -torch.inf)
        maximum = maximum.scatter_reduce(
            0,
            flat_index[:, None].expand(-1, channels),
            flat_features,
            reduce="amax",
            include_self=True,
        ).reshape(batch_size, topology.num_parts, channels)
    else:
        maximum = fixed_part_segmented_max(node_features, topology)

    counts = topology.counts_for_batch(batch_size=batch_size)
    flat_counts = counts.reshape(-1)
    mean = sums / flat_counts.clamp_min(1).to(node_features.dtype)[:, None]
    mean = mean.reshape(batch_size, topology.num_parts, channels)
    if topology.has_empty_parts and max_backend == "scatter":
        empty = topology.empty_mask_for_batch(batch_size=batch_size)
        maximum = maximum.masked_fill(empty[:, :, None], 0.0)
    return PartMeanMaxPool(mean=mean, maximum=maximum, counts=counts)


def fixed_broadcast_part_features(
    part_features: torch.Tensor,
    topology: FixedPartTopology,
) -> torch.Tensor:
    """Gather fixed-topology ``[B,P,C]`` features to ``[B,N,C]``."""

    if part_features.ndim != 3:
        raise ValueError("part_features must have shape [B,P,C]")
    batch_size, num_parts, channels = part_features.shape
    if num_parts != topology.num_parts:
        raise ValueError(
            f"Expected {topology.num_parts} PART features, got {num_parts}"
        )
    indices = topology.expanded_indices(
        batch_size=batch_size, node_count=topology.node_count
    )
    return torch.gather(
        part_features,
        dim=1,
        index=indices[:, :, None].expand(-1, -1, channels),
    )


class SharedResidualADELT(nn.Module):
    """One shared LTS plus a dense PART-conditioned stress residual."""

    model_family = "adelts_route1_residual"

    def __init__(
        self,
        shared_lts: LightTransolver,
        node_part_index: torch.Tensor,
        *,
        num_parts: int = 17,
        residual_hidden_dim: int | None = None,
        empty_part_policy: EmptyPartPolicy = "error",
    ) -> None:
        super().__init__()
        if not isinstance(shared_lts, LightTransolver):
            raise TypeError("SharedResidualADELT requires one LightTransolver")
        if int(shared_lts.out_dim) != 4:
            raise ValueError("ADELT Route 1 requires a four-channel LTS output")
        if num_parts <= 0:
            raise ValueError("num_parts must be positive")
        if empty_part_policy not in {"error", "zeros"}:
            raise ValueError("empty_part_policy must be 'error' or 'zeros'")

        indices = torch.as_tensor(node_part_index)
        if indices.ndim not in {1, 2} or indices.numel() == 0:
            raise ValueError("node_part_index must be non-empty [N] or [B,N]")
        if (
            indices.dtype == torch.bool
            or torch.is_floating_point(indices)
            or torch.is_complex(indices)
        ):
            raise TypeError("node_part_index must use an integer dtype")

        self.shared_lts = shared_lts
        self.num_parts = int(num_parts)
        self.empty_part_policy = empty_part_policy
        self.register_buffer(
            "node_part_index",
            indices.detach().clone().to(dtype=torch.long),
            persistent=False,
        )

        hidden_dim = int(shared_lts.n_hidden)
        residual_width = (
            max(32, hidden_dim // 2)
            if residual_hidden_dim is None
            else int(residual_hidden_dim)
        )
        if residual_width <= 0:
            raise ValueError("residual_hidden_dim must be positive")

        # Pool input is [decoded features, motion xyz, coarse stress], so the
        # mean/max PART summary has 2 * (C + 4) channels. The node head receives
        # decoded features, that summary, motion, and coarse stress.
        residual_input_dim = hidden_dim + 2 * (hidden_dim + 4) + 4
        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, residual_width),
            nn.GELU(),
            nn.Linear(residual_width, 1),
        )
        output_layer = self.residual_head[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    @property
    def motion_expert(self) -> LightTransolver:
        return self.shared_lts

    @property
    def deformation_expert(self) -> None:
        return None

    def forward(
        self,
        fx: torch.Tensor,
        embedding: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_prediction, intermediates = self.shared_lts(
            fx=fx,
            embedding=embedding,
            return_intermediates=True,
        )
        decoded = intermediates.decoded_node_features
        motion = raw_prediction[..., :3]
        coarse_stress = raw_prediction[..., 3:4]

        pool_input = torch.cat((decoded, motion, coarse_stress), dim=-1)
        pooled = part_mean_max_pool(
            pool_input,
            self.node_part_index,
            num_parts=self.num_parts,
            empty_part_policy=self.empty_part_policy,
        )
        part_summary = pooled.summary
        node_part_summary = broadcast_part_features(
            part_summary, self.node_part_index
        )
        residual_input = torch.cat(
            (decoded, node_part_summary, motion, coarse_stress), dim=-1
        )
        delta_stress = self.residual_head(residual_input)
        final_stress = coarse_stress + delta_stress
        output = torch.cat((motion, final_stress), dim=-1)

        if not return_details:
            return output
        return output, {
            "raw_prediction": raw_prediction,
            "decoded_node_features": decoded,
            "part_mean": pooled.mean,
            "part_maximum": pooled.maximum,
            "part_counts": pooled.counts,
            "part_summary": part_summary,
            "node_part_summary": node_part_summary,
            "coarse_stress": coarse_stress,
            "delta_stress": delta_stress,
        }


__all__ = [
    "FixedPartTopology",
    "MaxPoolBackend",
    "PartMeanMaxPool",
    "SharedResidualADELT",
    "broadcast_part_features",
    "fixed_broadcast_part_features",
    "fixed_part_mean_max_pool",
    "fixed_part_segmented_max",
    "part_mean_max_pool",
]
