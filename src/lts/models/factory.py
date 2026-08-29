# SPDX-License-Identifier: Apache-2.0
"""Construct LTS and the validation-selected DropTest paper model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .cclts import CCLTS
from .features import install_fourier_time_envelope
from .light_transolver import LightTransolver


@dataclass(frozen=True)
class LTSConfig:
    functional_dim: int = 12
    output_dim: int = 4
    geometry_dim: int = 3
    layers: int = 12
    hidden_dim: int = 256
    heads: int = 8
    slices: int = 256
    mlp_ratio: int = 4
    dropout: float = 0.0
    routing_cycles: int = 1
    component_aware: bool = True
    num_components: int = 17
    fourier: bool = True
    trajectory_envelope: bool = True
    fourier_bands: int = 16
    envelope_width: int = 128
    detach_component_pool_source: bool = True
    activation_checkpointing: bool = False


def build_lts(
    config: LTSConfig | None = None,
    *,
    node_component_index: torch.Tensor | None = None,
) -> nn.Module:
    """Build an LTS backbone.

    The default configuration is the complete DTOB6-0828 paper model. A fixed
    one-dimensional node-to-component mapping is therefore required by
    default. Set ``component_aware=False`` to construct the plain LTS ablation.

    Seeding is intentionally owned by the caller, matching the formal runner.
    """

    cfg = config or LTSConfig()
    light = LightTransolver(
        functional_dim=cfg.functional_dim,
        out_dim=cfg.output_dim,
        embedding_dim=cfg.geometry_dim,
        n_layers=cfg.layers,
        n_hidden=cfg.hidden_dim,
        dropout=cfg.dropout,
        n_head=cfg.heads,
        act="gelu",
        mlp_ratio=cfg.mlp_ratio,
        slice_num=cfg.slices,
        plus=False,
        activation_checkpointing=cfg.activation_checkpointing,
        checkpoint_scope=(
            "latent_blocks" if cfg.activation_checkpointing else "none"
        ),
        debug_routing_counters=False,
        slicing_cycles=cfg.routing_cycles,
    )

    if cfg.component_aware:
        if node_component_index is None:
            raise ValueError(
                "node_component_index is required when component_aware=True"
            )
        model: nn.Module = CCLTS(
            light,
            node_component_index,
            num_parts=cfg.num_components,
        )
        model.configure_training_pooling(
            max_reduction="amax",
            detach_source=cfg.detach_component_pool_source,
        )
        model.configure_training_affine(
            implementation="contiguous_custom_backward"
        )
    else:
        model = light

    install_fourier_time_envelope(
        light,
        fourier=cfg.fourier,
        trajectory_envelope=cfg.trajectory_envelope,
        fourier_bands=cfg.fourier_bands,
        envelope_width=cfg.envelope_width,
    )
    return model


__all__ = ["LTSConfig", "build_lts"]
