"""DropTest official-192 extension for the frozen 0813 LTS studies.

This module deliberately reuses the audited official-192 dataset, train-only
statistics, evaluator and training loop.  It adds only the missing 2^3
CA/Fourier/TEnv cells and the controlled K=1/2/4/8 resampling study.  The six
already scheduled main-table models remain in the upstream official runner and
are not duplicated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from examples.structural_mechanics.openradioss_dataset_gen.drop_test.official_192_benchmark import (
    benchmark as official,
)
from omegaconf import OmegaConf, open_dict
from physicsnemo.models.transolver.light_transolver import LightTransolver

FORMAT = "droptest_official192_0813_full_ablation_v1"
OUTPUT_ROOT = Path(
    "/work/choulin420/physicsnemo_data/experiments/"
    "droptest_official192_0813_full_20260825"
)
PREFLIGHT_ROOT = OUTPUT_ROOT / "preflight"
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
COMPONENT_MAPPING_EVIDENCE = OUTPUT_ROOT / "component_mapping_evidence.json"
UPSTREAM_OUTPUT_ROOT = official.OUTPUT_ROOT

SEEDS = official.SEEDS
GPU_MODE = official.GPU_MODE

# The full CA+Fourier+TEnv cell is already in the upstream six-model formal
# chain as cclts_fourier_tenv, so the extension contains only missing cells.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "lts": {"ca": False, "fourier": False, "tenv": False, "layers": 6, "k": 1},
    "lts_ca": {"ca": True, "fourier": False, "tenv": False, "layers": 6, "k": 1},
    "lts_fourier": {"ca": False, "fourier": True, "tenv": False, "layers": 6, "k": 1},
    "lts_tenv": {"ca": False, "fourier": False, "tenv": True, "layers": 6, "k": 1},
    "lts_ca_fourier": {"ca": True, "fourier": True, "tenv": False, "layers": 6, "k": 1},
    "lts_ca_tenv": {"ca": True, "fourier": False, "tenv": True, "layers": 6, "k": 1},
    "lts_fourier_tenv": {"ca": False, "fourier": True, "tenv": True, "layers": 6, "k": 1},
    # K must divide latent depth and K=8 must not contain zero-block segments.
    # Therefore this isolated study holds depth=8 and every parameter fixed
    # while changing only how often the shared mesh<->latent route is rebuilt.
    "lts_ca_fourier_tenv_k1_l8": {"ca": True, "fourier": True, "tenv": True, "layers": 8, "k": 1},
    "lts_ca_fourier_tenv_k2_l8": {"ca": True, "fourier": True, "tenv": True, "layers": 8, "k": 2},
    "lts_ca_fourier_tenv_k4_l8": {"ca": True, "fourier": True, "tenv": True, "layers": 8, "k": 4},
    "lts_ca_fourier_tenv_k8_l8": {"ca": True, "fourier": True, "tenv": True, "layers": 8, "k": 8},
}
MODELS = tuple(MODEL_SPECS)
MODEL_LABELS = {
    "lts": "LTS",
    "lts_ca": "LTS + CA",
    "lts_fourier": "LTS + Fourier",
    "lts_tenv": "LTS + TEnv",
    "lts_ca_fourier": "LTS + CA + Fourier",
    "lts_ca_tenv": "LTS + CA + TEnv",
    "lts_fourier_tenv": "LTS + Fourier + TEnv",
    "lts_ca_fourier_tenv_k1_l8": "LTS + CA + Fourier + TEnv (K=1, L=8)",
    "lts_ca_fourier_tenv_k2_l8": "LTS + CA + Fourier + TEnv (K=2, L=8)",
    "lts_ca_fourier_tenv_k4_l8": "LTS + CA + Fourier + TEnv (K=4, L=8)",
    "lts_ca_fourier_tenv_k8_l8": "LTS + CA + Fourier + TEnv (K=8, L=8)",
}

FACTORIAL_CELLS = {
    "000": {"model": "lts", "source_root": str(OUTPUT_ROOT)},
    "100": {"model": "lts_ca", "source_root": str(OUTPUT_ROOT)},
    "010": {"model": "lts_fourier", "source_root": str(OUTPUT_ROOT)},
    "001": {"model": "lts_tenv", "source_root": str(OUTPUT_ROOT)},
    "110": {"model": "lts_ca_fourier", "source_root": str(OUTPUT_ROOT)},
    "101": {"model": "lts_ca_tenv", "source_root": str(OUTPUT_ROOT)},
    "011": {"model": "lts_fourier_tenv", "source_root": str(OUTPUT_ROOT)},
    "111": {"model": "cclts_fourier_tenv", "source_root": str(UPSTREAM_OUTPUT_ROOT)},
}


class PhoneDropLTSVariant8(nn.Module):
    """Official eight-field time-conditional wrapper for one LTS variant."""

    def __init__(self, spec: dict[str, Any], node_part_index: torch.Tensor) -> None:
        super().__init__()
        layers = int(spec["layers"])
        k = int(spec["k"])
        if layers % k != 0:
            raise ValueError(f"K={k} must divide latent depth L={layers}")
        light = LightTransolver(
            functional_dim=12,
            out_dim=4,
            embedding_dim=3,
            n_layers=layers,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=4,
            slice_num=128,
            plus=False,
            activation_checkpointing=False,
            checkpoint_scope="none",
            debug_routing_counters=False,
            slicing_cycles=1,
        )
        # No parameters are added here.  All K arms share the same slicer,
        # decoder and six/eight latent-block parameterization at a fixed seed.
        light.slicing_cycles = k
        light.slicing_cycle_layout = tuple([layers // k] * k)

        if bool(spec["ca"]):
            backbone: nn.Module = official.CCLTS(
                light, node_part_index, num_parts=17
            )
            backbone.configure_training_pooling(
                max_reduction="amax", detach_source=False
            )
            backbone.configure_training_affine(
                implementation="contiguous_custom_backward"
            )
        else:
            backbone = light

        install_optional_features(
            light,
            fourier=bool(spec["fourier"]),
            tenv=bool(spec["tenv"]),
        )
        self.backbone = backbone
        self.rollout_steps = official.PREDICTION_STEPS

    def _step(self, sample: Any) -> torch.Tensor:
        coords = sample.node_features["coords"]
        time_value = sample.node_features.get("time")
        if time_value is None or time_value.numel() != 1:
            raise RuntimeError("LTS variant requires one normalized target time")
        globals_ = torch.stack(
            [sample.global_features[name] for name in official.GLOBAL_FEATURES]
        ).to(coords)
        raw = torch.cat(
            (
                coords,
                globals_.reshape(1, 8).expand(coords.shape[0], 8),
                time_value.to(coords).reshape(1, 1).expand(coords.shape[0], 1),
            ),
            dim=-1,
        )
        output = self.backbone(
            fx=raw.unsqueeze(0), embedding=coords.unsqueeze(0)
        ).squeeze(0)
        return torch.cat((coords + output[:, :3], output[:, 3:4]), dim=-1)

    def forward(
        self, sample: Any, data_stats: dict[str, Any] | None = None
    ) -> torch.Tensor:
        del data_stats
        if self.training:
            return self._step(sample)
        coords = sample.node_features["coords"]
        original = sample.node_features.get("time")
        had_time = "time" in sample.node_features
        outputs = []
        try:
            for index in range(official.PREDICTION_STEPS):
                sample.node_features["time"] = coords.new_tensor(
                    index / official.PREDICTION_STEPS
                )
                outputs.append(self._step(sample))
        finally:
            if had_time:
                sample.node_features["time"] = original
            else:
                sample.node_features.pop("time", None)
        return torch.stack(outputs, dim=1)


def install_optional_features(
    lts: LightTransolver, *, fourier: bool, tenv: bool
) -> None:
    """Install exactly the frozen Fourier-10 and/or TEnv-64 residual add-ons."""

    if not fourier and not tenv:
        return
    context: dict[str, torch.Tensor] = {}

    def capture(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module
        raw = inputs[0]
        if raw.shape[-1] != 15:
            raise RuntimeError(f"LTS preprocess input must be 15-D, got {raw.shape}")
        context["raw"] = raw
        context["conditioning"] = raw[:, :1, 6:]

    lts.preprocess.register_forward_pre_hook(capture)

    if fourier:
        time_bands = official.FOURIER_BANDS - 2
        dimension = 3 * official.FOURIER_BANDS * 2 + time_bands * 2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                torch.initial_seed() + official.ARCH_SEED_OFFSET + 2
            )
            projector = nn.Sequential(
                nn.Linear(dimension, lts.n_hidden),
                nn.GELU(),
                nn.Linear(lts.n_hidden, lts.n_hidden),
            )
            official._init_like_transolver(projector)
            official._zero_last_linear(projector)
        lts.fourier_projector = projector

        def augment(
            module: nn.Module, inputs: Any, output: torch.Tensor
        ) -> torch.Tensor:
            del module, inputs
            raw = context["raw"]
            features = torch.cat(
                (
                    official._fourier_expand(
                        raw[..., :3], official.FOURIER_BANDS
                    ),
                    official._fourier_expand(raw[..., -1:], time_bands),
                ),
                dim=-1,
            )
            return output + projector(features)

        lts.preprocess.register_forward_hook(augment)

    if tenv:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                torch.initial_seed() + official.ARCH_SEED_OFFSET + 4
            )
            envelope = nn.Sequential(
                nn.Linear(9, official.TIME_ENVELOPE_WIDTH),
                nn.GELU(),
                nn.Linear(official.TIME_ENVELOPE_WIDTH, 3),
            )
            official._init_like_transolver(envelope)
            official._zero_last_linear(envelope)
        lts.time_envelope = envelope

        def rescale(
            module: nn.Module, inputs: Any, output: torch.Tensor
        ) -> torch.Tensor:
            del module, inputs
            scale = 1.0 + envelope(context["conditioning"])
            return torch.cat(
                (output[..., :3] * scale, output[..., 3:]), dim=-1
            )

        lts.output_head.register_forward_hook(rescale)


def backbone_identity(model_id: str) -> dict[str, Any]:
    spec = dict(MODEL_SPECS[model_id])
    return {
        "implementation": "LightTransolver + optional CCLTS/Fourier/TEnv",
        "hidden_dim": 256,
        "layers": spec["layers"],
        "heads": 8,
        "physics_slices": 128,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "functional_dim": 12,
        "component_aware": spec["ca"],
        "fourier_bands": official.FOURIER_BANDS if spec["fourier"] else None,
        "time_envelope_width": official.TIME_ENVELOPE_WIDTH if spec["tenv"] else None,
        "resampling_count": spec["k"],
        "resampling_share_weights": True,
        "block_film": False,
    }


def model_identity(model_id: str) -> dict[str, Any]:
    if model_id not in MODELS:
        raise ValueError(model_id)
    return {
        "model": model_id,
        "label": MODEL_LABELS[model_id],
        "backbone": backbone_identity(model_id),
        "conditioning": {
            "official_feature_order": list(official.GLOBAL_FEATURES),
            "material_preprocessing": "(E_scale - 1.0) / 0.2",
            "orientation_preprocessing": "angle_deg / 10.0",
            "time": "prediction_index / 99 for indices 0..98",
        },
        "normalization": official.normalization_spec(),
        "warm_start": False,
    }


def model_config_sha256(model_id: str) -> str:
    return official.sha256_json(model_identity(model_id))


def resolved_config(
    model_id: str, seed: int, gpu_mode: str, output_root: Path
) -> Any:
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
                    "droptest_official192_0813_full.instantiate_model"
                ),
                "model_id": model_id,
                "identity_sha256": model_config_sha256(model_id),
            }
        )
    return cfg


def instantiate_model(model_id: str, seed: int, runtime: Any) -> nn.Module:
    official.formal.seed_model(seed)
    normalization = official.normalization_spec()
    backbone = PhoneDropLTSVariant8(
        MODEL_SPECS[model_id], runtime.part_mapping.node_part_index
    )
    conditioned = official.OfficialConditioningAdapter(backbone)
    return official.DisplacementRMSModelAdapter(
        conditioned,
        displacement_rms=normalization["displacement_rms"],
        position_std=normalization["position_std"],
    )


def model_target(model_id: str) -> str:
    if model_id not in MODELS:
        raise ValueError(model_id)
    return (
        "examples.structural_mechanics.openradioss_dataset_gen."
        "droptest_official192_0813_full.PhoneDropLTSVariant8"
    )


def task_mapping(task_id: int, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    task_id = int(task_id)
    if not 0 <= task_id < len(MODELS) * len(SEEDS):
        raise ValueError(f"Formal task must be 0..{len(MODELS) * len(SEEDS) - 1}")
    model_id = MODELS[task_id // len(SEEDS)]
    seed = SEEDS[task_id % len(SEEDS)]
    return {
        "task_id": task_id,
        "model": model_id,
        "model_label": MODEL_LABELS[model_id],
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
    metadata = official.validate_dataset_metadata()
    main_table = {
        "transolver": {"source_root": str(UPSTREAM_OUTPUT_ROOT)},
        "transolver_plus": {"source_root": str(UPSTREAM_OUTPUT_ROOT)},
        "transolver3": {"source_root": str(UPSTREAM_OUTPUT_ROOT)},
        "geotransolver": {"source_root": str(UPSTREAM_OUTPUT_ROOT)},
        "geots_flare": {"source_root": str(UPSTREAM_OUTPUT_ROOT)},
        "lts": {"source_root": str(output_root)},
        "lts_ca": {"source_root": str(output_root)},
        "cclts_fourier_tenv": {"source_root": str(UPSTREAM_OUTPUT_ROOT)},
    }
    return {
        "format": FORMAT,
        "status": "READY_AFTER_PREFLIGHT",
        "dataset": {
            "paths": official.exact_paths(),
            "counts": {"train": 132, "validation": 30, "test": 30},
            "timesteps": 100,
            "prediction_steps": 99,
            "nodes": 76065,
            **metadata,
        },
        "protocol": {
            "source": "frozen DropTest 0813 recipe on official-192 data",
            "seeds": list(SEEDS),
            "epochs": 25,
            "optimizer": "Adam(lr=1e-4, weight_decay=0)",
            "scheduler": "CosineAnnealingLR(T_max=25, eta_min=3e-7)",
            "precision": "BF16 autocast, no GradScaler",
            "batch_size": 1,
            "gradient_accumulation": 1,
            "checkpoint_selection": official.CHECKPOINT_SELECTION_DESCRIPTION,
            "test_used_during_training_or_selection": False,
        },
        "normalization": official.normalization_spec(),
        "main_table": main_table,
        "factorial_bit_order": ["CA", "Fourier", "TEnv"],
        "factorial_cells": FACTORIAL_CELLS,
        "resampling_study": {
            "models": [
                f"lts_ca_fourier_tenv_k{k}_l8" for k in (1, 2, 4, 8)
            ],
            "latent_layers": 8,
            "shared_routing_weights": True,
            "reason_separate_from_main_table": (
                "K=8 requires at least eight nonempty latent segments; the "
                "frozen 0813 main-table architecture has six latent layers"
            ),
        },
        "extension_models": {
            model: model_identity(model) for model in MODELS
        },
        "tasks": [
            task_mapping(index, output_root)
            for index in range(len(MODELS) * len(SEEDS))
        ],
        "gpu_packing": {
            "one_model_per_gpu": True,
            "maximum_gpus_per_job": 8,
            "new_formal_cells": len(MODELS) * len(SEEDS),
            "packed8_cells": 32,
            "tail_cells": 1,
        },
        "upstream_six_model_output_root": str(UPSTREAM_OUTPUT_ROOT),
        "slurm": {},
    }


def configure_runner() -> None:
    # Redirect every write into the choulin-owned extension root while retaining
    # the already frozen/read-only train-132 statistics from the audited runner.
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
