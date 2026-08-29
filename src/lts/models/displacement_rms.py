# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-level displacement-RMS output adapter for phone-drop models.

The wrapped solver keeps its existing architecture and canonical phone-drop
I/O.  During training only, its normalized absolute-position prediction is
converted to physical displacement divided by a train-split RMS.  Evaluation
converts through the same representation and reconstructs the canonical
normalized absolute position consumed by the existing evaluator.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


class DisplacementRMSModelAdapter(nn.Module):
    """Apply RMS-only physical-displacement supervision to any phone model."""

    normalization_mode = "train_displacement_rms_only"

    def __init__(
        self,
        model: nn.Module,
        *,
        displacement_rms: Sequence[float],
        position_std: Sequence[float],
    ) -> None:
        super().__init__()
        self.model = model
        rms = torch.as_tensor(displacement_rms, dtype=torch.float32).reshape(-1)
        pos_std = torch.as_tensor(position_std, dtype=torch.float32).reshape(-1)
        if rms.shape != (3,) or pos_std.shape != (3,):
            raise ValueError("displacement_rms and position_std must be [3]")
        if not bool(torch.isfinite(rms).all()) or not bool(
            torch.isfinite(pos_std).all()
        ):
            raise ValueError("normalization scales must be finite")
        if bool((rms <= 0).any()) or bool((pos_std <= 0).any()):
            raise ValueError("normalization scales must be positive")
        self.register_buffer("displacement_rms", rms, persistent=True)
        self.register_buffer("position_std", pos_std, persistent=True)

    @staticmethod
    def _broadcast_coords(
        coords: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if coords.shape[-1] != 3 or output.shape[-1] != 4:
            raise ValueError("Phone-drop coordinates/output must end in 3/4")
        expanded = coords
        while expanded.ndim < output.ndim:
            expanded = expanded.unsqueeze(-2)
        if expanded.shape[:-1] != output.shape[:-1] and any(
            left not in (1, right)
            for left, right in zip(expanded.shape[:-1], output.shape[:-1])
        ):
            raise ValueError(
                f"Coordinates {coords.shape} cannot broadcast to {output.shape}"
            )
        return expanded

    def canonical_to_training_space(
        self, coords: torch.Tensor, canonical: torch.Tensor
    ) -> torch.Tensor:
        """Convert normalized absolute xyz to ``D_physical / RMS``."""

        expanded = self._broadcast_coords(coords, canonical)
        scale = self.position_std.to(canonical) / self.displacement_rms.to(
            canonical
        )
        displacement = (canonical[..., :3] - expanded) * scale
        return torch.cat((displacement, canonical[..., 3:4]), dim=-1)

    def training_to_canonical_space(
        self, coords: torch.Tensor, training_output: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct normalized absolute xyz from ``D_physical / RMS``."""

        expanded = self._broadcast_coords(coords, training_output)
        scale = self.displacement_rms.to(training_output) / self.position_std.to(
            training_output
        )
        position = expanded + training_output[..., :3] * scale
        return torch.cat((position, training_output[..., 3:4]), dim=-1)

    def target_to_training_space(self, sample: Any) -> torch.Tensor:
        target = sample.node_target
        if target.ndim != 2:
            raise ValueError("Training target must be one-step [N,4]")
        return self.canonical_to_training_space(
            sample.node_features["coords"], target
        )

    def forward(
        self, sample: Any, data_stats: dict[str, Any] | None = None
    ) -> torch.Tensor:
        canonical = self.model(sample=sample, data_stats=data_stats)
        coords = sample.node_features["coords"]
        rms_output = self.canonical_to_training_space(coords, canonical)
        if self.training:
            return rms_output
        # Keep the established validation/test API while making the physical
        # reconstruction P_hat = P0 + RMS * D_hat_norm explicit.
        return self.training_to_canonical_space(coords, rms_output)


__all__ = ["DisplacementRMSModelAdapter"]
