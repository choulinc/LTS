# SPDX-License-Identifier: Apache-2.0
"""Frozen Fourier and trajectory-envelope add-ons used by DTOB6-0826."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .light_transolver import LightTransolver


DTOB6_FOURIER_BANDS = 10
DTOB6_TIME_ENVELOPE_WIDTH = 64
DTOB6_ARCH_SEED_OFFSET = 771013


def _initialize_like_transolver(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


def _zero_last_linear(module: nn.Sequential) -> None:
    layer = next(
        child for child in reversed(list(module.children()))
        if isinstance(child, nn.Linear)
    )
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def fourier_expand(value: torch.Tensor, bands: int) -> torch.Tensor:
    """Apply powers-of-two sine/cosine features without retaining raw input."""

    if bands <= 0:
        raise ValueError("bands must be positive")
    frequencies = torch.pow(
        value.new_tensor(2.0),
        torch.arange(bands, device=value.device, dtype=value.dtype),
    )
    angles = value.unsqueeze(-1) * frequencies * torch.pi
    return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)


def install_fourier_time_envelope(
    lts: LightTransolver,
    *,
    fourier: bool = True,
    trajectory_envelope: bool = True,
    fourier_bands: int = DTOB6_FOURIER_BANDS,
    envelope_width: int = DTOB6_TIME_ENVELOPE_WIDTH,
    architecture_seed_offset: int = DTOB6_ARCH_SEED_OFFSET,
) -> None:
    """Install the exact zero-init residual features used by the paper model.

    The LTS preprocess receives ``cat(embedding, functional_input)``. Under the
    DropTest adapter this is 15-D: geometry coordinates (3), functional
    coordinates (3), eight design variables, and target time. The envelope is
    conditioned on the final nine values (design variables and time).

    The hooks preserve the historical state-dict names
    ``fourier_projector`` and ``time_envelope``.
    """

    if not fourier and not trajectory_envelope:
        return
    if fourier_bands < 3:
        raise ValueError("fourier_bands must be at least three")
    if envelope_width <= 0:
        raise ValueError("envelope_width must be positive")

    context: dict[str, torch.Tensor] = {}

    def capture(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        raw = inputs[0]
        if raw.shape[-1] != 15:
            raise RuntimeError(
                f"DTOB6-0826 LTS preprocess input must be 15-D, got {raw.shape}"
            )
        context["raw"] = raw
        context["conditioning"] = raw[:, :1, 6:]

    lts.preprocess.register_forward_pre_hook(capture)

    if fourier:
        time_bands = fourier_bands - 2
        dimension = 3 * fourier_bands * 2 + time_bands * 2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                torch.initial_seed() + architecture_seed_offset + 2
            )
            projector = nn.Sequential(
                nn.Linear(dimension, lts.n_hidden),
                nn.GELU(),
                nn.Linear(lts.n_hidden, lts.n_hidden),
            )
            _initialize_like_transolver(projector)
            _zero_last_linear(projector)
        lts.fourier_projector = projector

        def augment(
            _module: nn.Module,
            _inputs: Any,
            output: torch.Tensor,
        ) -> torch.Tensor:
            raw = context["raw"]
            features = torch.cat(
                (
                    fourier_expand(raw[..., :3], fourier_bands),
                    fourier_expand(raw[..., -1:], time_bands),
                ),
                dim=-1,
            )
            return output + projector(features)

        lts.preprocess.register_forward_hook(augment)

    if trajectory_envelope:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                torch.initial_seed() + architecture_seed_offset + 4
            )
            envelope = nn.Sequential(
                nn.Linear(9, envelope_width),
                nn.GELU(),
                nn.Linear(envelope_width, 3),
            )
            _initialize_like_transolver(envelope)
            _zero_last_linear(envelope)
        lts.time_envelope = envelope

        def rescale(
            _module: nn.Module,
            _inputs: Any,
            output: torch.Tensor,
        ) -> torch.Tensor:
            scale = 1.0 + envelope(context["conditioning"])
            return torch.cat(
                (output[..., :3] * scale, output[..., 3:]), dim=-1
            )

        lts.output_head.register_forward_hook(rescale)


__all__ = [
    "DTOB6_ARCH_SEED_OFFSET",
    "DTOB6_FOURIER_BANDS",
    "DTOB6_TIME_ENVELOPE_WIDTH",
    "fourier_expand",
    "install_fourier_time_envelope",
]
