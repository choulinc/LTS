# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Custom-backward prototype for the fixed-run CCLTS affine."""

from __future__ import annotations

import torch


class _ContiguousRunAffine(torch.autograd.Function):
    """Apply the CCLTS affine without node-resolution PART tensors.

    ``run_starts``, ``run_lengths``, and ``run_part_ids`` are validated and
    cached by :class:`CCLTS` before this function is called. They deliberately
    remain Python tuples: no device tensor is read back or inspected here.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        node_hidden: torch.Tensor,
        part_gamma: torch.Tensor,
        part_beta: torch.Tensor,
        run_starts: tuple[int, ...],
        run_lengths: tuple[int, ...],
        run_part_ids: tuple[int, ...],
        num_parts: int,
    ) -> torch.Tensor:
        ctx.run_starts = run_starts
        ctx.run_lengths = run_lengths
        ctx.run_part_ids = run_part_ids
        ctx.num_parts = num_parts
        ctx.part_beta_dtype = part_beta.dtype
        ctx.save_for_backward(node_hidden, part_gamma)

        chunks = []
        for start, length, part_id in zip(
            run_starts, run_lengths, run_part_ids
        ):
            hidden = node_hidden[:, start : start + length, :]
            gamma = part_gamma[:, part_id : part_id + 1, :]
            beta = part_beta[:, part_id : part_id + 1, :]
            # Preserve the gather implementation's operation order exactly.
            chunks.append(hidden + gamma * hidden + beta)
        return torch.cat(chunks, dim=1)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
    ]:
        node_hidden, part_gamma = ctx.saved_tensors
        batch_size, _, channels = node_hidden.shape
        accumulation_shape = (batch_size, ctx.num_parts, channels)
        grad_gamma_fp32 = torch.zeros(
            accumulation_shape,
            dtype=torch.float32,
            device=node_hidden.device,
        )
        grad_beta_fp32 = torch.zeros_like(grad_gamma_fp32)
        grad_hidden_chunks = []

        for start, length, part_id in zip(
            ctx.run_starts,
            ctx.run_lengths,
            ctx.run_part_ids,
        ):
            stop = start + length
            grad_chunk = grad_output[:, start:stop, :]
            hidden_chunk = node_hidden[:, start:stop, :]
            gamma_chunk = part_gamma[:, part_id : part_id + 1, :]

            grad_hidden_chunks.append(grad_chunk * (1 + gamma_chunk))
            grad_gamma_fp32[:, part_id, :].copy_(
                (grad_chunk.float() * hidden_chunk.float()).sum(dim=1)
            )
            grad_beta_fp32[:, part_id, :].copy_(
                grad_chunk.sum(dim=1, dtype=torch.float32)
            )

        return (
            torch.cat(grad_hidden_chunks, dim=1),
            grad_gamma_fp32.to(dtype=part_gamma.dtype),
            grad_beta_fp32.to(dtype=ctx.part_beta_dtype),
            None,
            None,
            None,
            None,
        )


def contiguous_run_affine_custom_backward(
    node_hidden: torch.Tensor,
    part_gamma: torch.Tensor,
    part_beta: torch.Tensor,
    *,
    run_starts: tuple[int, ...],
    run_lengths: tuple[int, ...],
    run_part_ids: tuple[int, ...],
    num_parts: int,
) -> torch.Tensor:
    """Run the fixed-topology affine with an FP32-reduction custom backward."""

    if node_hidden.ndim != 3:
        raise ValueError("node_hidden must have shape [B,N,C]")
    if part_gamma.shape != part_beta.shape or part_gamma.ndim != 3:
        raise ValueError("part_gamma and part_beta must have shape [B,P,C]")
    if (
        node_hidden.shape[0] != part_gamma.shape[0]
        or node_hidden.shape[2] != part_gamma.shape[2]
    ):
        raise ValueError("Node and PART affine batch/channel shapes differ")
    if part_gamma.shape[1] != num_parts:
        raise ValueError(f"Expected {num_parts} PARTs")
    if not (
        len(run_starts) == len(run_lengths) == len(run_part_ids)
    ):
        raise ValueError("Run metadata lengths differ")
    if sum(run_lengths) != node_hidden.shape[1]:
        raise ValueError("Run metadata does not cover every node")

    return _ContiguousRunAffine.apply(
        node_hidden,
        part_gamma,
        part_beta,
        run_starts,
        run_lengths,
        run_part_ids,
        num_parts,
    )


__all__ = ["contiguous_run_affine_custom_backward"]
