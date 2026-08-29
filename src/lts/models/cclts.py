# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Component-conditioned LightTransolver (CCLTS) prototype."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from .light_transolver import LightTransolver

from .adelts_route1 import (
    EmptyPartPolicy,
    FixedPartTopology,
    fixed_broadcast_part_features,
)
from .cclts_contiguous_backward import (
    contiguous_run_affine_custom_backward,
)
from .route2_run_amax_prototype import (
    FixedRunMetadata,
    build_fixed_run_metadata,
    fixed_run_amax,
)
from .route2_run_mean_prototype import fixed_run_mean


CCLTS_MODEL_ID = "cclts_component_conditioned"
CCLTS_PROJECTOR_WIDTH = 128
TrainingMaxReduction = Literal["max_values", "amax"]
TrainingAffineImplementation = Literal[
    "contiguous_custom_backward",
    "contiguous_runs",
    "gather",
]


def fixed_run_max_values(
    node_features: torch.Tensor,
    metadata: FixedRunMetadata,
) -> torch.Tensor:
    """Fixed-run max with first-maximum training gradients.

    Values match ``torch.amax``. Unlike ``amax``, ``torch.max(...).values``
    routes an equal-maximum gradient to the first maximum instead of dividing
    it across every tied maximum.
    """

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
            node_features[:, start : start + length, :].max(
                dim=1
            ).values
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


class CCLTS(nn.Module):
    """One LTS with zero-init component-conditioned output modulation.

    Fixed topology is validated once during construction. Every nonempty PART
    must occupy one contiguous node run; the number of PARTs is supplied by the
    topology/config rather than encoded in the model.
    """

    model_family = CCLTS_MODEL_ID

    def __init__(
        self,
        shared_lts: LightTransolver,
        node_part_index: torch.Tensor,
        *,
        num_parts: int,
        empty_part_policy: EmptyPartPolicy = "error",
    ) -> None:
        super().__init__()
        if not isinstance(shared_lts, LightTransolver):
            raise TypeError("CCLTS requires exactly one LightTransolver")
        if num_parts <= 0:
            raise ValueError("num_parts must be positive")
        if empty_part_policy not in {"error", "zeros"}:
            raise ValueError("empty_part_policy must be 'error' or 'zeros'")

        indices = torch.as_tensor(node_part_index)
        if indices.ndim != 1:
            raise ValueError(
                "CCLTS contiguous-run fast path requires fixed [N] "
                "node_part_index"
            )
        self.shared_lts = shared_lts
        self.num_parts = int(num_parts)
        self.empty_part_policy = empty_part_policy
        # These are execution-policy flags, not parameters or persistent
        # buffers, so legacy CCLTS checkpoints retain identical state_dicts.
        self.training_max_reduction: TrainingMaxReduction = "amax"
        self.detach_training_pool_source = False
        self.training_affine_implementation: TrainingAffineImplementation = (
            "gather"
        )
        self.part_topology = FixedPartTopology(
            indices,
            num_parts=self.num_parts,
            empty_part_policy=empty_part_policy,
        )
        metadata = build_fixed_run_metadata(
            indices,
            num_parts=self.num_parts,
            empty_part_policy=empty_part_policy,
        )
        self._run_starts = metadata.run_starts
        self._run_lengths = metadata.run_lengths
        self._run_part_ids = metadata.run_part_ids
        self.register_buffer(
            "_dense_to_run",
            metadata.dense_to_run.detach().clone(),
            persistent=False,
        )

        hidden_dim = int(shared_lts.n_hidden)
        self.component_projector = nn.Sequential(
            nn.Linear(2 * hidden_dim, CCLTS_PROJECTOR_WIDTH),
            nn.GELU(),
            nn.Linear(CCLTS_PROJECTOR_WIDTH, hidden_dim),
        )
        self.component_affine = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.component_affine.weight)
        nn.init.zeros_(self.component_affine.bias)

    @property
    def motion_expert(self) -> LightTransolver:
        return self.shared_lts

    @property
    def deformation_expert(self) -> None:
        return None

    @property
    def node_part_index(self) -> torch.Tensor:
        return self.part_topology.node_part_index

    def _run_metadata(self) -> FixedRunMetadata:
        """Return cached run metadata with its reorder tensor on this device."""

        return FixedRunMetadata(
            run_starts=self._run_starts,
            run_lengths=self._run_lengths,
            run_part_ids=self._run_part_ids,
            dense_to_run=self._dense_to_run,
            num_parts=self.num_parts,
            node_count=self.part_topology.node_count,
            empty_part_policy=self.empty_part_policy,
        )

    def component_context(
        self, node_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Pool contiguous PART runs and project mean/max to ``[B,P,C]``."""

        metadata = self._run_metadata()
        pool_source = (
            node_embedding.detach()
            if self.training and self.detach_training_pool_source
            else node_embedding
        )
        part_mean = fixed_run_mean(pool_source, metadata)
        if self.training and self.training_max_reduction == "max_values":
            part_maximum = fixed_run_max_values(pool_source, metadata)
        else:
            # Inference intentionally retains the original CCLTS definition.
            part_maximum = fixed_run_amax(pool_source, metadata)
        pooled = torch.cat((part_mean, part_maximum), dim=-1)
        context = self.component_projector(pooled)
        return context, {
            "part_mean": part_mean,
            "part_maximum": part_maximum,
            "pooled_component_summary": pooled,
        }

    def configure_training_pooling(
        self,
        *,
        max_reduction: TrainingMaxReduction,
        detach_source: bool = False,
    ) -> None:
        """Select a training-only pooling gradient implementation."""

        if max_reduction not in {"max_values", "amax"}:
            raise ValueError(
                "max_reduction must be 'max_values' or 'amax'"
            )
        self.training_max_reduction = max_reduction
        self.detach_training_pool_source = bool(detach_source)

    def configure_training_affine(
        self,
        *,
        implementation: TrainingAffineImplementation,
    ) -> None:
        """Select the parameter-free training affine execution path."""

        if implementation not in {
            "contiguous_custom_backward",
            "contiguous_runs",
            "gather",
        }:
            raise ValueError(
                "implementation must be 'contiguous_custom_backward', "
                "'contiguous_runs', or 'gather'"
            )
        self.training_affine_implementation = implementation

    def _contiguous_custom_backward_affine(
        self,
        node_hidden: torch.Tensor,
        part_gamma: torch.Tensor,
        part_beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the fixed-run affine with FP32 custom gradient reductions."""

        return contiguous_run_affine_custom_backward(
            node_hidden,
            part_gamma,
            part_beta,
            run_starts=self._run_starts,
            run_lengths=self._run_lengths,
            run_part_ids=self._run_part_ids,
            num_parts=self.num_parts,
        )

    def _contiguous_run_affine(
        self,
        node_hidden: torch.Tensor,
        part_gamma: torch.Tensor,
        part_beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply PART affine modulation directly over cached node runs."""

        def refine_run(start: int, length: int, part_id: int) -> torch.Tensor:
            hidden = node_hidden[:, start : start + length, :]
            gamma = part_gamma[:, part_id : part_id + 1, :]
            beta = part_beta[:, part_id : part_id + 1, :]
            return hidden + gamma * hidden + beta

        chunks = [
            refine_run(start, length, part_id)
            for start, length, part_id in zip(
                self._run_starts,
                self._run_lengths,
                self._run_part_ids,
            )
        ]
        return torch.cat(chunks, dim=1)

    def _gather_affine(
        self,
        node_hidden: torch.Tensor,
        part_gamma: torch.Tensor,
        part_beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original two-gather CCLTS affine path."""

        node_gamma = fixed_broadcast_part_features(
            part_gamma, self.part_topology
        )
        node_beta = fixed_broadcast_part_features(
            part_beta, self.part_topology
        )
        refined = node_hidden + node_gamma * node_hidden + node_beta
        return refined, node_gamma, node_beta

    def _refine_node_features(
        self,
        node_hidden: torch.Tensor,
        node_embedding: torch.Tensor,
        details: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context, pool_details = self.component_context(node_embedding)
        part_gamma, part_beta = self.component_affine(context).chunk(2, dim=-1)
        training_affine = (
            self.training_affine_implementation
            if self.training
            else "gather"
        )
        if training_affine == "contiguous_custom_backward":
            refined = self._contiguous_custom_backward_affine(
                node_hidden, part_gamma, part_beta
            )
            node_gamma = None
            node_beta = None
        elif training_affine == "contiguous_runs":
            refined = self._contiguous_run_affine(
                node_hidden, part_gamma, part_beta
            )
            node_gamma = None
            node_beta = None
        else:
            refined, node_gamma, node_beta = self._gather_affine(
                node_hidden, part_gamma, part_beta
            )
        if details is not None:
            if node_gamma is None or node_beta is None:
                # Diagnostics remain API-compatible, but the clean training
                # path does not materialize these node-resolution tensors.
                node_gamma = fixed_broadcast_part_features(
                    part_gamma, self.part_topology
                )
                node_beta = fixed_broadcast_part_features(
                    part_beta, self.part_topology
                )
            details.update(pool_details)
            details.update(
                {
                    "node_embedding_before_slicing": node_embedding,
                    "decoded_node_features": node_hidden,
                    "component_context": context,
                    "part_gamma": part_gamma,
                    "part_beta": part_beta,
                    "node_gamma": node_gamma,
                    "node_beta": node_beta,
                    "refined_node_features": refined,
                }
            )
        return refined

    def forward(
        self,
        fx: torch.Tensor,
        embedding: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        details: dict[str, torch.Tensor] | None = {} if return_details else None

        def refine(
            node_hidden: torch.Tensor, node_embedding: torch.Tensor
        ) -> torch.Tensor:
            return self._refine_node_features(
                node_hidden, node_embedding, details
            )

        output = self.shared_lts(
            fx=fx,
            embedding=embedding,
            node_feature_refiner=refine,
        )
        if not return_details:
            return output
        assert details is not None
        details["prediction"] = output
        return output, details


__all__ = [
    "CCLTS",
    "CCLTS_MODEL_ID",
    "CCLTS_PROJECTOR_WIDTH",
    "TrainingAffineImplementation",
    "TrainingMaxReduction",
    "fixed_run_max_values",
]
