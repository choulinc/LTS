# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

r"""Light Transolver with persistent physics-aware slices (LTS-v1).

LTS keeps the irregular-mesh slicing and deslicing semantics of Transolver. Its
formal baseline routes once; the slicing-count ablation can repartition the same
six independent latent blocks across up to six shared slice/deslice cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.transolver.transolver import _TransolverMlp
from physicsnemo.nn.module.physics_attention import (
    _compute_slices_from_projections,
    _project_input,
)


SLICING_CYCLE_LAYOUTS = {
    1: (6,),
    2: (3, 3),
    3: (2, 2, 2),
    4: (1, 2, 2, 1),
    5: (1, 1, 2, 1, 1),
    6: (1, 1, 1, 1, 1, 1),
}


@dataclass
class LightTransolverMetaData(ModelMetaData):
    r"""Metadata for :class:`LightTransolver`."""

    name: str = "LightTransolver"
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


@dataclass(frozen=True)
class LightTransolverIntermediates:
    r"""Opt-in node-resolution features from a LightTransolver forward pass."""

    node_embedding_before_slicing: torch.Tensor
    decoded_node_features: torch.Tensor


class _GeometryRoutingProjector(nn.Module):
    r"""Position-only trunk for LNO-inspired slice assignments."""

    def __init__(
        self,
        geometry_dim: int,
        hidden_dim: int,
        *,
        width: int,
        depth: int,
        act: str,
    ) -> None:
        super().__init__()
        if geometry_dim <= 0 or hidden_dim <= 0 or width <= 0:
            raise ValueError("geometry routing dimensions must be positive")
        if depth <= 0:
            raise ValueError("geometry routing depth must be positive")

        activation = nn.GELU if act.lower() == "gelu" else nn.ReLU
        layers: list[nn.Module] = []
        in_features = geometry_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(in_features, width), activation()])
            in_features = width
        layers.append(nn.Linear(in_features, hidden_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self, geometry: Float[torch.Tensor, "batch points geometry"]
    ) -> Float[torch.Tensor, "batch points hidden"]:
        return self.network(geometry)


class _PhysicsSliceEncoder(nn.Module):
    r"""One standard-Transolver irregular-mesh slicing operation."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        slice_num: int,
        *,
        plus: bool,
        geometry_dim: int = 3,
        geometry_only_routing: bool = False,
        act: str = "gelu",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = hidden_dim // num_heads
        self.slice_num = slice_num
        self.plus = plus
        self.geometry_only_routing = bool(geometry_only_routing)
        if self.geometry_only_routing and plus:
            raise ValueError(
                "geometry_only_routing requires plus=False so assignments "
                "and token values use separate projections"
            )
        if self.geometry_only_routing:
            # Do not advance the shared initialization stream merely because
            # an ablation-only module is present.
            with torch.random.fork_rng(devices=[]):
                self.geometry_routing_projector = _GeometryRoutingProjector(
                    geometry_dim,
                    hidden_dim,
                    width=hidden_dim,
                    depth=2,
                    act=act,
                )
        else:
            self.geometry_routing_projector = None

        self.in_project_x = nn.Linear(hidden_dim, hidden_dim)
        if not plus:
            self.in_project_fx = nn.Linear(hidden_dim, hidden_dim)
        self.in_project_slice = nn.Linear(self.dim_head, slice_num)
        self.temperature = nn.Parameter(
            torch.ones(1, 1, num_heads, 1, dtype=torch.float32) * 0.5
        )
        if plus:
            self.proj_temperature = nn.Sequential(
                nn.Linear(self.dim_head, slice_num),
                nn.GELU(),
                nn.Linear(slice_num, 1),
                nn.GELU(),
            )

    def _project_routing_and_values(
        self,
        point_state: Float[torch.Tensor, "batch points hidden"],
        geometry: Float[torch.Tensor, "batch points geometry"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project assignment keys and token values through separate paths."""

        if self.geometry_routing_projector is None:
            projected = _project_input(
                point_state,
                self.in_project_x,
                self.num_heads,
                self.dim_head,
                "B N (H D) -> B N H D",
                project_fx=None if self.plus else self.in_project_fx,
            )
            if self.plus:
                return projected, projected
            return projected

        if geometry is None:
            raise ValueError(
                "geometry_only_routing requires per-point geometry"
            )
        geometry_state = self.geometry_routing_projector(geometry)
        routing_state = rearrange(
            self.in_project_x(geometry_state),
            "b n (h d) -> b n h d",
            h=self.num_heads,
            d=self.dim_head,
        )
        value_state = rearrange(
            self.in_project_fx(point_state),
            "b n (h d) -> b n h d",
            h=self.num_heads,
            d=self.dim_head,
        )
        return routing_state, value_state

    def forward(
        self,
        point_state: Float[torch.Tensor, "batch points hidden"],
        geometry: Float[torch.Tensor, "batch points geometry"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch points heads slices"],
        Float[torch.Tensor, "batch slices hidden"],
    ]:
        x_mid, fx_mid = self._project_routing_and_values(
            point_state, geometry
        )

        slice_projection = self.in_project_slice(x_mid)
        slice_weights, slice_tokens = _compute_slices_from_projections(
            slice_projection,
            fx_mid,
            self.temperature,
            self.plus,
            self.proj_temperature if self.plus else None,
        )
        latent = rearrange(slice_tokens, "b h m d -> b m (h d)")
        return slice_weights, latent


class _PersistentSliceBlock(nn.Module):
    r"""A pre-normalized Transformer block operating only on slice tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        act: str,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = hidden_dim // num_heads

        self.layer_norm_attention = nn.LayerNorm(hidden_dim)
        # Match Transolver physics attention: one head-local QKV projection is
        # shared across heads, followed by standard scaled dot-product attention.
        self.qkv_project = nn.Linear(self.dim_head, 3 * self.dim_head, bias=False)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = _TransolverMlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * mlp_ratio,
            out_features=hidden_dim,
            act_layer=act,
            use_te=False,
        )

    def forward(
        self, latent: Float[torch.Tensor, "batch slices hidden"]
    ) -> Float[torch.Tensor, "batch slices hidden"]:
        normalized = self.layer_norm_attention(latent)
        head_state = rearrange(
            normalized,
            "b m (h d) -> b h m d",
            h=self.num_heads,
            d=self.dim_head,
        )
        qkv = self.qkv_project(head_state)
        query, key, value = rearrange(
            qkv, "b h m (q d) -> q b h m d", q=3, d=self.dim_head
        ).unbind(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=False
        )
        attended = rearrange(attended, "b h m d -> b m (h d)")
        latent = latent + self.out_dropout(self.out_linear(attended))
        return latent + self.ffn(self.layer_norm_ffn(latent))


class _PhysicsSliceDecoder(nn.Module):
    r"""One standard-Transolver weighted deslicing operation."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = hidden_dim // num_heads
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        latent: Float[torch.Tensor, "batch slices hidden"],
        slice_weights: Float[torch.Tensor, "batch points heads slices"],
    ) -> Float[torch.Tensor, "batch points hidden"]:
        slice_tokens = rearrange(
            latent,
            "b m (h d) -> b h m d",
            h=self.num_heads,
            d=self.dim_head,
        )
        # This is PhysicsAttentionBase._project_attention_outputs before its
        # output projection: reuse the entry assignment for the one-time decode.
        decoded = torch.einsum("bnhm,bhmd->bnhd", slice_weights, slice_tokens)
        decoded = rearrange(decoded, "b n h d -> b n (h d)")
        return self.out_dropout(self.out_linear(decoded))


class _IndependentCoordinateDecoder(nn.Module):
    r"""Decode latent tokens with coordinate-only, independently learned routing.

    Unlike :class:`_PhysicsSliceDecoder`, this module does not consume the
    encoder's head-local slicing assignments.  Normalized reference coordinates
    produce one distribution over the full latent-token axis for every node.
    """

    def __init__(
        self,
        geometry_dim: int,
        hidden_dim: int,
        slice_num: int,
        dropout: float,
        act: str,
    ) -> None:
        super().__init__()
        activation = nn.GELU if act.lower() == "gelu" else nn.ReLU
        self.geometry_dim = int(geometry_dim)
        self.hidden_dim = int(hidden_dim)
        self.slice_num = int(slice_num)
        self.routing_projector = nn.Sequential(
            nn.Linear(self.geometry_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.slice_num),
        )
        self.out_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_dropout = nn.Dropout(dropout)

    def routing_weights(
        self,
        reference_coordinates: Float[
            torch.Tensor, "batch points geometry"
        ],
    ) -> Float[torch.Tensor, "batch points slices"]:
        """Return node-to-latent decoder weights normalized over tokens."""

        if reference_coordinates.ndim != 3:
            raise ValueError(
                "reference_coordinates must have shape (B, N, geometry_dim)"
            )
        if reference_coordinates.shape[-1] != self.geometry_dim:
            raise ValueError(
                f"Expected geometry_dim={self.geometry_dim}, got "
                f"{reference_coordinates.shape[-1]}"
            )
        return torch.softmax(
            self.routing_projector(reference_coordinates), dim=-1
        )

    def forward(
        self,
        latent: Float[torch.Tensor, "batch slices hidden"],
        reference_coordinates: Float[
            torch.Tensor, "batch points geometry"
        ],
    ) -> Float[torch.Tensor, "batch points hidden"]:
        if latent.ndim != 3:
            raise ValueError("latent must have shape (B, M, hidden_dim)")
        if latent.shape[1:] != (self.slice_num, self.hidden_dim):
            raise ValueError(
                "Expected latent shape (*, "
                f"{self.slice_num}, {self.hidden_dim}), got {tuple(latent.shape)}"
            )
        if latent.shape[0] != reference_coordinates.shape[0]:
            raise ValueError("latent and coordinates must share batch size")
        decoder_weights = self.routing_weights(reference_coordinates)
        decoded = torch.einsum("bnm,bmc->bnc", decoder_weights, latent)
        return self.out_dropout(self.out_linear(decoded))


class _PointRefinement(nn.Module):
    r"""The single point-resolution FFN used after deslicing."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        act: str,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = _TransolverMlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * mlp_ratio,
            out_features=hidden_dim,
            act_layer=act,
            use_te=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, point_state: Float[torch.Tensor, "batch points hidden"]
    ) -> Float[torch.Tensor, "batch points hidden"]:
        return point_state + self.dropout(self.ffn(self.layer_norm(point_state)))


class LightTransolver(Module):
    r"""LTS-v1: Transolver with one physical slice route and persistent slices.

    The first version intentionally supports only the irregular-mesh, non-TE
    configuration used by the unified Bumper Transolver baseline.  Architecture
    width, heads, slices, depth, activation, and FFN ratio remain configurable so
    the formal LTS config can exactly match that baseline.
    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        embedding_dim: int | None = None,
        n_layers: int = 6,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 128,
        unified_pos: bool = False,
        ref: int = 8,
        structured_shape: tuple[int, ...] | None = None,
        use_te: bool = False,
        time_input: bool = False,
        plus: bool = False,
        activation_checkpointing: bool = True,
        checkpoint_scope: str = "latent_blocks",
        debug_routing_counters: bool = False,
        geometry_only_routing: bool = False,
        symmetric_latent_skip: bool = False,
        slicing_cycles: int = 1,
        independent_final_decoder: bool = False,
        independent_cycle_decoders: bool = False,
    ) -> None:
        super().__init__(meta=LightTransolverMetaData())
        self.__name__ = "LightTransolver"

        if functional_dim <= 0 or out_dim <= 0:
            raise ValueError("functional_dim and out_dim must be positive")
        if n_hidden <= 0 or n_head <= 0 or n_hidden % n_head != 0:
            raise ValueError("n_hidden must be positive and divisible by n_head")
        if n_layers <= 0 or slice_num <= 0:
            raise ValueError("n_layers and slice_num must be positive")
        if embedding_dim is None or embedding_dim <= 0:
            raise ValueError("LTS-v1 requires a positive embedding_dim")
        if structured_shape is not None or unified_pos:
            raise ValueError("LTS-v1 currently supports irregular meshes only")
        if use_te:
            raise ValueError("LTS-v1 currently requires use_te=False")
        if time_input:
            raise ValueError("LTS-v1 does not yet support time_input=True")
        if checkpoint_scope not in {"latent_blocks", "none"}:
            raise ValueError(
                "checkpoint_scope must be either 'latent_blocks' or 'none'"
            )
        if symmetric_latent_skip and n_layers != 6:
            raise ValueError(
                "symmetric_latent_skip implements B1->B6 and B2->B5 "
                "and therefore requires n_layers=6"
            )
        if (
            not isinstance(slicing_cycles, int)
            or isinstance(slicing_cycles, bool)
            or slicing_cycles not in SLICING_CYCLE_LAYOUTS
        ):
            raise ValueError("slicing_cycles must be an integer from 1 to 6")
        if slicing_cycles > 1 and n_layers != 6:
            raise ValueError(
                "multi-cycle slicing keeps the formal six latent blocks fixed "
                "and therefore requires n_layers=6"
            )
        if slicing_cycles > 1 and symmetric_latent_skip:
            raise ValueError(
                "symmetric_latent_skip is only defined for slicing_cycles=1"
            )
        if independent_final_decoder and independent_cycle_decoders:
            raise ValueError(
                "independent_final_decoder and independent_cycle_decoders "
                "are mutually exclusive"
            )

        # ``ref`` is accepted for Transolver config compatibility. It is not
        # used when unified_pos=False.
        self.ref = ref
        self.functional_dim = functional_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.head_dim = n_hidden // n_head
        self.slice_num = slice_num
        self.n_layers = n_layers
        self.structured_shape = None
        self.unified_pos = False
        self.use_te = False
        self.time_input = False
        self.plus = plus
        self.activation_checkpointing = activation_checkpointing
        self.checkpoint_scope = checkpoint_scope
        self.debug_routing_counters = debug_routing_counters
        self.geometry_only_routing = bool(geometry_only_routing)
        self.symmetric_latent_skip = bool(symmetric_latent_skip)
        self.slicing_cycles = int(slicing_cycles)
        self.use_independent_final_decoder = bool(independent_final_decoder)
        self.use_independent_cycle_decoders = bool(
            independent_cycle_decoders
        )
        self.slicing_cycle_layout = SLICING_CYCLE_LAYOUTS[self.slicing_cycles]
        self._routing_counts: dict[str, int | list[int]] = {
            "slicing": 0,
            "deslicing": 0,
            "latent_blocks": 0,
            "blocks_per_cycle": [],
        }
        self.last_latent_shape: tuple[int, ...] | None = None

        self.preprocess = _TransolverMlp(
            in_features=functional_dim + embedding_dim,
            hidden_features=n_hidden * 2,
            out_features=n_hidden,
            act_layer=act,
            use_te=False,
        )
        self.slicer = _PhysicsSliceEncoder(
            n_hidden,
            n_head,
            slice_num,
            plus=plus,
            geometry_dim=embedding_dim,
            geometry_only_routing=self.geometry_only_routing,
            act=act,
        )
        self.latent_blocks = nn.ModuleList(
            [
                _PersistentSliceBlock(
                    n_hidden, n_head, dropout, act, mlp_ratio
                )
                for _ in range(n_layers)
            ]
        )
        self.symmetric_latent_skip_pairs = (
            ((0, 5), (1, 4)) if n_layers == 6 else ()
        )
        if self.symmetric_latent_skip:
            self.symmetric_latent_skip_gates = nn.Parameter(torch.zeros(2))
        else:
            # A None parameter does not add a state-dict key, so old strict
            # checkpoint loading remains unchanged in the baseline mode.
            self.register_parameter("symmetric_latent_skip_gates", None)
        self.decoder = _PhysicsSliceDecoder(n_hidden, n_head, dropout)
        if self.use_independent_final_decoder:
            # Keep all shared-module initialization identical to the baseline
            # for a fixed seed.  The ablation-only decoder receives its own
            # deterministic RNG stream below.
            with torch.random.fork_rng(devices=[]):
                self.independent_final_decoder = _IndependentCoordinateDecoder(
                    embedding_dim,
                    n_hidden,
                    slice_num,
                    dropout,
                    act,
                )
        else:
            self.independent_final_decoder = None
        if self.use_independent_cycle_decoders:
            # Each cycle owns a distinct coordinate decoder.  Construction is
            # isolated from the common RNG stream so enabling the ablation
            # cannot perturb any pre-existing LTS/CCLTS parameter.
            with torch.random.fork_rng(devices=[]):
                self.independent_cycle_decoders = nn.ModuleList(
                    [
                        _IndependentCoordinateDecoder(
                            embedding_dim,
                            n_hidden,
                            slice_num,
                            dropout,
                            act,
                        )
                        for _ in range(self.slicing_cycles)
                    ]
                )
        else:
            self.independent_cycle_decoders = None
        self.point_refinement = _PointRefinement(
            n_hidden, dropout, act, mlp_ratio
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, out_dim),
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        r"""Use the same Linear/LayerNorm initialization policy as Transolver."""

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        optional_geometry = self.slicer.geometry_routing_projector
        optional_decoder = self.independent_final_decoder
        optional_cycle_decoders = self.independent_cycle_decoders
        optional_modules = tuple(
            module
            for module in (
                optional_geometry,
                optional_decoder,
                optional_cycle_decoders,
            )
            if module is not None
        )

        def initialize_common(module: nn.Module) -> None:
            for child in module.children():
                if all(child is not optional for optional in optional_modules):
                    initialize_common(child)
            initialize(module)

        initialize_common(self)
        if optional_geometry is not None:
            # Initialize the optional trunk deterministically without changing
            # baseline/shared-backbone RNG consumption.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() + 104729)
                optional_geometry.apply(initialize)
        if optional_decoder is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() + 130363)
                optional_decoder.apply(initialize)
        if optional_cycle_decoders is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() + 155921)
                optional_cycle_decoders.apply(initialize)

    def _run_latent_blocks_with_symmetric_skip(
        self, latent: torch.Tensor
    ) -> torch.Tensor:
        """Run B1..B6 with zero-init B1->B6 and B2->B5 residuals."""

        if self.symmetric_latent_skip_gates is None:
            raise RuntimeError("symmetric latent skip is not enabled")
        cached: dict[int, torch.Tensor] = {}
        destination_to_gate = {5: 0, 4: 1}
        for block_index, block in enumerate(self.latent_blocks):
            gate_index = destination_to_gate.get(block_index)
            if gate_index is not None:
                source_index = self.symmetric_latent_skip_pairs[gate_index][0]
                latent = (
                    latent
                    + self.symmetric_latent_skip_gates[gate_index]
                    * cached[source_index]
                )
            if (
                self.training
                and self.activation_checkpointing
                and self.checkpoint_scope == "latent_blocks"
            ):
                latent = checkpoint(block, latent, use_reentrant=False)
            else:
                latent = block(latent)
            if block_index in (0, 1):
                cached[block_index] = latent
        return latent

    def reset_routing_counters(self) -> None:
        r"""Reset opt-in routing counters used by tests and GPU smoke runs."""

        self._routing_counts = {
            "slicing": 0,
            "deslicing": 0,
            "latent_blocks": 0,
            "blocks_per_cycle": [],
        }
        self.last_latent_shape = None

    def get_routing_counters(self) -> dict[str, int | list[int] | None]:
        r"""Return a copy of routing diagnostics without exposing mutable state."""

        return {
            **self._routing_counts,
            "last_latent_shape": (
                list(self.last_latent_shape)
                if self.last_latent_shape is not None
                else None
            ),
        }

    @staticmethod
    def _combine_point_residual(
        point_state: torch.Tensor, decoded_correction: torch.Tensor
    ) -> torch.Tensor:
        return point_state + decoded_correction

    def forward(
        self,
        fx: Float[torch.Tensor, "batch points functional"],
        embedding: Float[torch.Tensor, "batch points embedding"] | None = None,
        time: torch.Tensor | None = None,
        *,
        latent_context: torch.Tensor | None = None,
        return_intermediates: bool = False,
        node_feature_refiner: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
    ) -> (
        Float[torch.Tensor, "batch points output"]
        | tuple[
            Float[torch.Tensor, "batch points output"],
            LightTransolverIntermediates,
        ]
    ):
        if time is not None:
            raise NotImplementedError("LTS-v1 does not implement time input")
        if embedding is None:
            raise ValueError("LTS-v1 requires per-point geometry embedding")
        if fx.ndim != 3 or embedding.ndim != 3:
            raise ValueError("fx and embedding must have shape (B, N, C)")
        if fx.shape[:2] != embedding.shape[:2]:
            raise ValueError("fx and embedding must share batch and point dimensions")
        if fx.shape[-1] != self.functional_dim:
            raise ValueError(
                f"Expected functional_dim={self.functional_dim}, got {fx.shape[-1]}"
            )
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {embedding.shape[-1]}"
            )

        point_state = self.preprocess(torch.cat((embedding, fx), dim=-1))
        node_embedding_before_slicing = point_state
        if self.slicing_cycles == 1:
            # Keep the formal one-cycle LTS baseline call path byte-for-byte in
            # spirit: one slice, all existing blocks, one deslice.  In
            # particular, do not route this branch through a new helper.
            if self.geometry_only_routing:
                slice_weights, latent = self.slicer(point_state, embedding)
            else:
                slice_weights, latent = self.slicer(point_state)
            if self.debug_routing_counters:
                self._routing_counts["slicing"] += 1
                self.last_latent_shape = tuple(latent.shape)

            if self.symmetric_latent_skip_gates is not None:
                latent = self._run_latent_blocks_with_symmetric_skip(latent)
            else:
                for block in self.latent_blocks:
                    if (
                        self.training
                        and self.activation_checkpointing
                        and self.checkpoint_scope == "latent_blocks"
                    ):
                        if latent_context is None:
                            latent = checkpoint(block, latent, use_reentrant=False)
                        else:
                            latent = checkpoint(
                                block,
                                latent,
                                latent_context,
                                use_reentrant=False,
                            )
                    else:
                        latent = (
                            block(latent)
                            if latent_context is None
                            else block(latent, latent_context)
                        )
                    if self.debug_routing_counters:
                        self._routing_counts["latent_blocks"] += 1

            if self.debug_routing_counters:
                blocks_per_cycle = self._routing_counts["blocks_per_cycle"]
                assert isinstance(blocks_per_cycle, list)
                blocks_per_cycle.append(len(self.latent_blocks))

            if self.independent_cycle_decoders is not None:
                decoded_correction = self.independent_cycle_decoders[0](
                    latent, embedding
                )
            elif self.independent_final_decoder is not None:
                decoded_correction = self.independent_final_decoder(
                    latent, embedding
                )
            else:
                decoded_correction = self.decoder(latent, slice_weights)
            if self.debug_routing_counters:
                self._routing_counts["deslicing"] += 1
            point_state = self._combine_point_residual(
                point_state, decoded_correction
            )
        else:
            block_start = 0
            for cycle_index, blocks_in_cycle in enumerate(
                self.slicing_cycle_layout
            ):
                if self.geometry_only_routing:
                    slice_weights, latent = self.slicer(
                        point_state, embedding
                    )
                else:
                    slice_weights, latent = self.slicer(point_state)
                if self.debug_routing_counters:
                    self._routing_counts["slicing"] += 1
                    self.last_latent_shape = tuple(latent.shape)

                block_stop = block_start + blocks_in_cycle
                for block_index in range(block_start, block_stop):
                    block = self.latent_blocks[block_index]
                    if (
                        self.training
                        and self.activation_checkpointing
                        and self.checkpoint_scope == "latent_blocks"
                    ):
                        if latent_context is None:
                            latent = checkpoint(block, latent, use_reentrant=False)
                        else:
                            latent = checkpoint(
                                block,
                                latent,
                                latent_context,
                                use_reentrant=False,
                            )
                    else:
                        latent = (
                            block(latent)
                            if latent_context is None
                            else block(latent, latent_context)
                        )
                    if self.debug_routing_counters:
                        self._routing_counts["latent_blocks"] += 1
                block_start = block_stop

                if self.debug_routing_counters:
                    blocks_per_cycle = self._routing_counts[
                        "blocks_per_cycle"
                    ]
                    assert isinstance(blocks_per_cycle, list)
                    blocks_per_cycle.append(blocks_in_cycle)

                is_final_cycle = cycle_index == self.slicing_cycles - 1
                if self.independent_cycle_decoders is not None:
                    decoded_correction = self.independent_cycle_decoders[
                        cycle_index
                    ](latent, embedding)
                elif (
                    is_final_cycle
                    and self.independent_final_decoder is not None
                ):
                    decoded_correction = self.independent_final_decoder(
                        latent, embedding
                    )
                else:
                    decoded_correction = self.decoder(latent, slice_weights)
                if self.debug_routing_counters:
                    self._routing_counts["deslicing"] += 1
                # Parameter-free mesh-state handoff. Point refinement, CA and
                # the output head remain outside this loop and run once.
                point_state = self._combine_point_residual(
                    point_state, decoded_correction
                )
            if block_start != len(self.latent_blocks):
                raise RuntimeError(
                    "slicing cycle layout did not execute every latent block"
                )
        point_state = self.point_refinement(point_state)
        if node_feature_refiner is not None:
            refined = node_feature_refiner(
                point_state, node_embedding_before_slicing
            )
            if refined.shape != point_state.shape:
                raise ValueError(
                    "node_feature_refiner must preserve the decoded node "
                    f"feature shape {tuple(point_state.shape)}, got "
                    f"{tuple(refined.shape)}"
                )
            point_state = refined
        prediction = self.output_head(point_state)
        if return_intermediates:
            return prediction, LightTransolverIntermediates(
                node_embedding_before_slicing=node_embedding_before_slicing,
                decoded_node_features=point_state
            )
        return prediction


__all__ = [
    "LightTransolver",
    "LightTransolverIntermediates",
    "LightTransolverMetaData",
    "SLICING_CYCLE_LAYOUTS",
]
