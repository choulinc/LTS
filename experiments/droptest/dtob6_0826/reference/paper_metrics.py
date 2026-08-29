"""Physical-space paper metrics for the official 192-run DropTest benchmark.

This module extends the established DropTest evaluator with physical position
Relative-L2 while retaining its exact displacement, stress, per-component,
per-trajectory, and per-timestep sufficient statistics.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from examples.structural_mechanics.openradioss_dataset_gen.phone_drop_phase1.me_lts_final_benchmark_v1 import (
    benchmark_metrics as base,
)


def _stats_tensor(
    stats: dict[str, Any], key: str, *, shape: tuple[int, ...]
) -> torch.Tensor:
    return (
        torch.as_tensor(stats[key], dtype=torch.float64, device="cpu")
        .flatten()
        .reshape(shape)
    )


def trajectory_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    initial_coords: torch.Tensor,
    *,
    sample_id: str,
    node_stats: dict[str, Any],
    dynamic_target_stats: dict[str, Any],
    node_part_index: torch.Tensor,
    component_ids: Iterable[int],
    epsilon: float,
) -> dict[str, Any]:
    """Evaluate one trajectory over all 76,065 nodes and 99 target steps."""

    result = base.trajectory_metrics(
        prediction,
        target,
        initial_coords,
        sample_id=sample_id,
        node_stats=node_stats,
        dynamic_target_stats=dynamic_target_stats,
        node_part_index=node_part_index,
        component_ids=component_ids,
        epsilon=epsilon,
    )
    pred = prediction.detach().to(device="cpu", dtype=torch.float64)
    truth = target.detach().to(device="cpu", dtype=torch.float64)
    nodes = int(pred.shape[0])
    pos_mean = _stats_tensor(node_stats, "pos_mean", shape=(1, 1, 3))
    pos_std = _stats_tensor(node_stats, "pos_std", shape=(1, 1, 3))
    pred_position = pred[..., :3] * pos_std + pos_mean
    truth_position = truth[..., :3] * pos_std + pos_mean
    error = torch.square(pred_position - truth_position)
    denominator = torch.square(truth_position)
    result["position_physical"] = base.metric(
        error.sum().item(),
        denominator.sum().item(),
        nodes * 99 * 3,
        epsilon,
    )
    for step in range(99):
        result["timesteps"][step]["position_physical"] = base.metric(
            error[:, step].sum().item(),
            denominator[:, step].sum().item(),
            nodes * 3,
            epsilon,
        )
    return result


def aggregate_trajectories(
    trajectories: list[dict[str, Any]], *, epsilon: float
) -> dict[str, Any]:
    result = base.aggregate_trajectories(trajectories, epsilon=epsilon)
    result["position_physical"] = base._aggregate(  # noqa: SLF001
        [row["position_physical"] for row in trajectories], epsilon
    )
    for step in range(99):
        result["timesteps"][step]["position_physical"] = base._aggregate(  # noqa: SLF001
            [row["timesteps"][step]["position_physical"] for row in trajectories],
            epsilon,
        )
    return result


def summary_metrics(aggregate: dict[str, Any]) -> dict[str, Any]:
    result = base.summary_metrics(aggregate)
    result.update(
        {
            "position_physical_relative_l2_macro": aggregate[
                "position_physical"
            ]["relative_l2_macro"],
            "position_physical_relative_l2_micro": aggregate[
                "position_physical"
            ]["relative_l2_micro"],
            "displacement_physical_relative_l2_macro": aggregate[
                "displacement_physical"
            ]["relative_l2_macro"],
            "displacement_physical_relative_l2_micro": aggregate[
                "displacement_physical"
            ]["relative_l2_micro"],
            "stress_physical_relative_l2_macro": aggregate[
                "stress_physical"
            ]["relative_l2_macro"],
            "stress_physical_relative_l2_micro": aggregate[
                "stress_physical"
            ]["relative_l2_micro"],
        }
    )
    return result


def timestep_rows(
    aggregate: dict[str, Any], *, model: str, seed: int, split: str
) -> list[dict[str, Any]]:
    rows = base.timestep_rows(aggregate, model=model, seed=seed, split=split)
    for timestep in aggregate["timesteps"]:
        rows.append(
            {
                "model": model,
                "seed": seed,
                "split": split,
                "prediction_step": timestep["prediction_step"],
                "field": "position_physical",
                **timestep["position_physical"],
            }
        )
    return rows


def trajectory_rows(
    aggregate: dict[str, Any], *, model: str, seed: int, split: str
) -> list[dict[str, Any]]:
    rows = base.trajectory_rows(aggregate, model=model, seed=seed, split=split)
    metrics = {
        row["sample_id"]: row["position_physical"]
        for row in aggregate["per_trajectory"]
    }
    for row in rows:
        position = metrics[row["sample_id"]]
        row["position_physical_mse"] = position["mse"]
        row["position_physical_relative_l2"] = position["relative_l2"]
    return rows


part_rows = base.part_rows


__all__ = [
    "aggregate_trajectories",
    "part_rows",
    "summary_metrics",
    "timestep_rows",
    "trajectory_metrics",
    "trajectory_rows",
]
