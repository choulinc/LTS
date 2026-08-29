"""Official-backbone DropTest K=1/2/3/4/6/8/12 resampling study.

K=1 is reused from the completed 20260826 official-backbone main table. This
runner trains K=2/3/4/6/8/12 and holds the 12-layer, 256-slice full LTS model,
initialization, data protocol, optimizer, and seeds fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

from examples.structural_mechanics.openradioss_dataset_gen import (
    droptest_official_backbones_20260826 as base,
)


official = base.official
FORMAT = "droptest_official192_official_backbones_k12346812_v2"
OUTPUT_ROOT = Path(
    "/work/choulin420/physicsnemo_data/experiments/"
    "droptest_official_backbones_k12346812_20260827"
)
PREFLIGHT_ROOT = OUTPUT_ROOT / "preflight"
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
COMPONENT_MAPPING_EVIDENCE = OUTPUT_ROOT / "component_mapping_evidence.json"
K1_OUTPUT_ROOT = base.OUTPUT_ROOT

MODELS = (
    "lts_ca_fourier_tenv_k2",
    "lts_ca_fourier_tenv_k3",
    "lts_ca_fourier_tenv_k4",
    "lts_ca_fourier_tenv_k6",
    "lts_ca_fourier_tenv_k8",
    "lts_ca_fourier_tenv_k12",
)
MODEL_K = {
    "lts_ca_fourier_tenv_k2": 2,
    "lts_ca_fourier_tenv_k3": 3,
    "lts_ca_fourier_tenv_k4": 4,
    "lts_ca_fourier_tenv_k6": 6,
    "lts_ca_fourier_tenv_k8": 8,
    "lts_ca_fourier_tenv_k12": 12,
}
MODEL_LABELS = {
    model: f"LTS + CA + Fourier + TEnv (K={k})"
    for model, k in MODEL_K.items()
}
SEEDS = base.SEEDS
GPU_MODE = base.GPU_MODE
EXPECTED_PARAMETER_COUNT = base.EXPECTED_PARAMETER_COUNTS[
    "lts_ca_fourier_tenv"
]


def cycle_layout(k: int, layers: int = 12) -> tuple[int, ...]:
    """Return a symmetric, nonempty balanced partition of latent blocks."""

    if k not in (1, 2, 3, 4, 6, 8, 12) or k > layers:
        raise ValueError((k, layers))
    small, remainder = divmod(layers, k)
    layout = [small] * k
    start = (k - remainder) // 2
    for index in range(start, start + remainder):
        layout[index] += 1
    result = tuple(layout)
    if len(result) != k or min(result) < 1 or sum(result) != layers:
        raise RuntimeError(f"Invalid K={k} layout: {result}")
    return result


class PhoneDropLTSKOfficial(base.PhoneDropLTSOfficial):
    """Exact official-scale full LTS cell with only routing K changed."""

    def __init__(self, node_part_index: torch.Tensor, *, k: int) -> None:
        super().__init__(
            node_part_index,
            component_aware=True,
            fourier_tenv=True,
        )
        light = self.motion_expert
        light.slicing_cycles = int(k)
        light.slicing_cycle_layout = cycle_layout(k, light.n_layers)


def backbone_identity(model_id: str) -> dict[str, Any]:
    if model_id not in MODELS:
        raise ValueError(model_id)
    identity = dict(base.backbone_identity("lts_ca_fourier_tenv"))
    k = MODEL_K[model_id]
    identity.update(
        {
            "resampling_count": k,
            "resampling_share_weights": True,
            "resampling_cycle_layout": list(cycle_layout(k)),
            "resampling_handoff": "parameter-free point residual",
            "k_only_change_from_reused_k1": True,
        }
    )
    return identity


def model_identity(model_id: str) -> dict[str, Any]:
    if model_id not in MODELS:
        raise ValueError(model_id)
    return {
        "model": model_id,
        "label": MODEL_LABELS[model_id],
        "backbone": backbone_identity(model_id),
        "conditioning": base.model_identity("lts_ca_fourier_tenv")[
            "conditioning"
        ],
        "normalization": official.normalization_spec(),
        "warm_start": False,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
    }


def model_config_sha256(model_id: str) -> str:
    return official.sha256_json(model_identity(model_id))


def resolved_config(
    model_id: str, seed: int, gpu_mode: str, output_root: Path
) -> DictConfig:
    if model_id not in MODELS or seed not in SEEDS or gpu_mode != GPU_MODE:
        raise ValueError((model_id, seed, gpu_mode))
    cfg = official.load_config()
    directory = official.run_directory(output_root, gpu_mode, model_id, seed)
    with open_dict(cfg):
        cfg.training.seed = seed
        cfg.training.epochs = official.MAX_EPOCHS
        cfg.training.scheduler_t_max = official.MAX_EPOCHS
        cfg.training.phone_ddp.config_name = FORMAT
        cfg.training.phone_ddp.run_name = model_id
        cfg.training.phone_ddp.model_id = model_id
        cfg.training.phone_ddp.model_label = MODEL_LABELS[model_id]
        cfg.training.phone_ddp.output_root = str(output_root)
        cfg.training.phone_ddp.output_dir = str(directory)
        cfg.training.phone_ddp.optimizer_updates_per_epoch = (
            official.TRAIN_TIMESTEP_SAMPLES
        )
        cfg.training.phone_ddp.checkpoint_selection = {
            "name": official.SELECTION_METRIC_NAME,
            "description": official.CHECKPOINT_SELECTION_DESCRIPTION,
            "uses_test": False,
        }
        cfg.model = OmegaConf.create(
            {
                "_target_": (
                    "examples.structural_mechanics.openradioss_dataset_gen."
                    "droptest_official_backbones_k1248_20260827."
                    "instantiate_model"
                ),
                "model_id": model_id,
                "identity_sha256": model_config_sha256(model_id),
            }
        )
    return cfg


def instantiate_model(model_id: str, seed: int, runtime: Any) -> nn.Module:
    if model_id not in MODELS:
        raise ValueError(model_id)
    official.formal.seed_model(seed)
    normalization = official.normalization_spec()
    backbone = PhoneDropLTSKOfficial(
        runtime.part_mapping.node_part_index,
        k=MODEL_K[model_id],
    )
    conditioned = official.OfficialConditioningAdapter(backbone)
    model = official.DisplacementRMSModelAdapter(
        conditioned,
        displacement_rms=normalization["displacement_rms"],
        position_std=normalization["position_std"],
    )
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"{model_id} parameter count {observed} != "
            f"frozen {EXPECTED_PARAMETER_COUNT}"
        )
    light = backbone.motion_expert
    expected_layout = cycle_layout(MODEL_K[model_id])
    if (
        light.n_layers != 12
        or light.slice_num != 256
        or light.slicing_cycles != MODEL_K[model_id]
        or tuple(light.slicing_cycle_layout) != expected_layout
    ):
        raise RuntimeError("Official-backbone K invariant failed")
    return model


def model_target(model_id: str) -> str:
    if model_id not in MODELS:
        raise ValueError(model_id)
    return (
        "examples.structural_mechanics.openradioss_dataset_gen."
        "droptest_official_backbones_k1248_20260827"
    )


def task_mapping(
    task_id: int, output_root: Path = OUTPUT_ROOT
) -> dict[str, Any]:
    task_id = int(task_id)
    count = len(MODELS) * len(SEEDS)
    if not 0 <= task_id < count:
        raise ValueError(f"Formal task must be 0..{count - 1}")
    model_id = MODELS[task_id // len(SEEDS)]
    seed = SEEDS[task_id % len(SEEDS)]
    return {
        "task_id": task_id,
        "model": model_id,
        "model_label": MODEL_LABELS[model_id],
        "k": MODEL_K[model_id],
        "cycle_layout": list(cycle_layout(MODEL_K[model_id])),
        "seed": seed,
        "gpu_mode": GPU_MODE,
        "world_size": 1,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "effective_global_batch": 1,
        "output": str(
            official.run_directory(output_root, GPU_MODE, model_id, seed)
        ),
    }


def launch_manifest(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    protocol = official.protocol_snapshot(
        resolved_config(MODELS[0], SEEDS[0], GPU_MODE, output_root),
        model=MODELS[0],
        gpu_mode=GPU_MODE,
    )
    return {
        "format": FORMAT,
        "status": "READY_AFTER_6_MODEL_PREFLIGHT",
        "source_file": str(Path(__file__).resolve()),
        "source_sha256": official.sha256_file(Path(__file__)),
        "dataset": {
            "paths": official.exact_paths(),
            "counts": {"train": 132, "validation": 30, "test": 30},
            "timesteps": 100,
            "prediction_steps": 99,
            "nodes": 76065,
            **official.validate_dataset_metadata(),
        },
        "protocol": protocol,
        "controlled_variable": "shared-route resampling count K only",
        "fixed_backbone": {
            "layers": 12,
            "hidden_dim": 256,
            "heads": 8,
            "physics_slices": 256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "k1_reuse": {
            "model": "lts_ca_fourier_tenv",
            "source_root": str(K1_OUTPUT_ROOT),
            "seeds": list(SEEDS),
            "cycle_layout": [12],
        },
        "new_models": {model: model_identity(model) for model in MODELS},
        "tasks": [
            task_mapping(index, output_root)
            for index in range(len(MODELS) * len(SEEDS))
        ],
        "gpu_packing": {
            "preflight": "6 models x 1 GPU",
            "formal": "3 seed jobs x 6 models x 1 GPU",
            "one_model_per_gpu": True,
            "maximum_gpus_per_job": 6,
        },
        "slurm": {},
    }


def configure_runner() -> None:
    official.FORMAT = FORMAT
    official.OUTPUT_ROOT = OUTPUT_ROOT
    official.PREFLIGHT_ROOT = PREFLIGHT_ROOT
    official.MANIFEST_PATH = MANIFEST_PATH
    official.COMPONENT_MAPPING_EVIDENCE = COMPONENT_MAPPING_EVIDENCE
    official.MODELS = MODELS
    official.MODEL_LABELS = MODEL_LABELS
    official.model_identity = model_identity
    official.model_config_sha256 = model_config_sha256
    official.resolved_config = resolved_config
    official.instantiate_model = instantiate_model
    official.model_target = model_target
    official.configure_runner()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--gpu-mode", choices=(GPU_MODE,), default=GPU_MODE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-updates", type=int, choices=(2, 3), default=2)
    parser.add_argument("--audit-task", type=int)
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args(argv)
    if not args.print_manifest and args.audit_task is None:
        if args.model is None or args.seed is None:
            parser.error("execution requires --model and --seed")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_runner()
    if args.print_manifest:
        print(json.dumps(launch_manifest(args.output_root), indent=2, sort_keys=True))
        return
    if args.audit_task is not None:
        row = task_mapping(args.audit_task, args.output_root)
        if args.model is not None and args.model != row["model"]:
            raise RuntimeError("Task/model mapping mismatch")
        if args.seed is not None and args.seed != row["seed"]:
            raise RuntimeError("Task/seed mapping mismatch")
        if Path(row["output"]).exists():
            raise RuntimeError(f"Refusing to overwrite {row['output']}")
        print(json.dumps(row, indent=2, sort_keys=True))
        return
    official.shared.parse_args = lambda: args
    official.shared.main()


if __name__ == "__main__":
    main()
