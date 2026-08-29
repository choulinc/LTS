"""DropTest official-192 main table with source-native backbone profiles.

Only model construction changes relative to the audited official-192 runner.
Dataset splits, train-only normalization, optimization, validation-only
checkpoint selection, held-out testing, and metrics are delegated unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from physicsnemo.models.transolver import Transolver
from physicsnemo.models.transolver.light_transolver import LightTransolver

from examples.structural_mechanics.openradioss_dataset_gen import (
    droptest_official192_0813_full as lts_features,
)
from examples.structural_mechanics.openradioss_dataset_gen.drop_test.official_192_benchmark import (
    benchmark as official,
)


FORMAT = "droptest_official192_official_backbones_8model_v1"
OUTPUT_ROOT = Path(
    "/work/choulin420/physicsnemo_data/experiments/"
    "droptest_official_backbones_20260826"
)
PREFLIGHT_ROOT = OUTPUT_ROOT / "preflight"
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
COMPONENT_MAPPING_EVIDENCE = OUTPUT_ROOT / "component_mapping_evidence.json"

MODELS = (
    "transolver",
    "transolver_plus",
    "transolver3",
    "geotransolver",
    "geots_flare",
    "lts",
    "lts_ca",
    "lts_ca_fourier_tenv",
)
MODEL_LABELS = {
    "transolver": "Transolver",
    "transolver_plus": "Transolver++",
    "transolver3": "Transolver-3",
    "geotransolver": "GeoTransolver",
    "geots_flare": "GeoTS-FLARE",
    "lts": "LTS",
    "lts_ca": "LTS + CA",
    "lts_ca_fourier_tenv": "LTS + CA + Fourier + TEnv",
}
SEEDS = official.SEEDS
GPU_MODE = official.GPU_MODE

PHYSICSNEMO_PROFILE_COMMIT = "dc244e6747d2fba0bf42656422059349f4861fbc"
TRANSOLVER_PLUS_COMMIT = official.external.EXTERNAL_COMMITS[
    official.external.TRANSOLVER_PLUS_MODEL_ID
]
TRANSOLVER3_COMMIT = official.external.EXTERNAL_COMMITS[
    official.external.TRANSOLVER3_MODEL_ID
]

EXPECTED_PARAMETER_COUNTS = {
    "transolver": 8_967_012,
    "transolver_plus": 1_743_784,
    "transolver3": 7_603_844,
    "geotransolver": 27_575_024,
    "geots_flare": 26_325_968,
    "lts": 8_018_444,
    "lts_ca": 8_248_716,
    "lts_ca_fourier_tenv": 8_335_055,
}

TRANSOLVER_SCALE = {
    "hidden_dim": 256,
    "layers": 12,
    "heads": 8,
    "physics_slices": 256,
    "mlp_ratio": 4,
    "dropout": 0.0,
}
GEO_RADII = [0.01, 0.05, 0.25, 1.0, 2.5, 5.0]
GEO_NEIGHBORS = [8, 16, 32, 32, 64, 128]


class PhoneDropTransolverOfficial(official.PhoneDropTransolver8):
    """Official volume Transolver scale with DropTest task I/O."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.rollout_steps = official.PREDICTION_STEPS
        self.backbone = Transolver(
            functional_dim=12,
            out_dim=4,
            embedding_dim=3,
            n_layers=12,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=4,
            slice_num=256,
            unified_pos=False,
            structured_shape=None,
            use_te=False,
            time_input=False,
            plus=False,
        )


class PhoneDropLTSOfficial(official.PhoneDropCCLTS8):
    """LTS variants matched to the official volume Transolver scale."""

    def __init__(
        self,
        node_part_index: torch.Tensor,
        *,
        component_aware: bool,
        fourier_tenv: bool,
    ) -> None:
        nn.Module.__init__(self)
        self.rollout_steps = official.PREDICTION_STEPS
        light = LightTransolver(
            functional_dim=12,
            out_dim=4,
            embedding_dim=3,
            n_layers=12,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=4,
            slice_num=256,
            plus=False,
            activation_checkpointing=False,
            checkpoint_scope="none",
            debug_routing_counters=False,
            slicing_cycles=1,
        )
        if component_aware:
            self.backbone = official.CCLTS(
                light, node_part_index, num_parts=17
            )
            self.backbone.configure_training_pooling(
                max_reduction="amax", detach_source=False
            )
            self.backbone.configure_training_affine(
                implementation="contiguous_custom_backward"
            )
        else:
            self.backbone = light
        if fourier_tenv:
            lts_features.install_optional_features(
                light, fourier=True, tenv=True
            )


def geo_config(model_id: str) -> DictConfig:
    if model_id not in {"geotransolver", "geots_flare"}:
        raise ValueError(model_id)
    return OmegaConf.create(
        {
            "_target_": "rollout.GeoTransolverTimeConditional",
            "_convert_": "all",
            "functional_dim": 4,
            "out_dim": 4,
            "geometry_dim": 3,
            "global_dim": 8,
            "slice_num": 256,
            "n_layers": 12,
            "n_hidden": 256,
            "dropout": 0.0,
            "n_head": 8,
            "act": "gelu",
            "mlp_ratio": 4,
            "n_hidden_local": 32,
            "radii": list(GEO_RADII),
            "neighbors_in_radius": list(GEO_NEIGHBORS),
            "attention_type": (
                "GALE" if model_id == "geotransolver" else "GALE_FA"
            ),
            "state_mixing_mode": "weighted",
            "use_te": False,
            "time_input": False,
            "plus": False,
            "include_local_features": True,
            "num_time_steps": 100,
        }
    )


def backbone_identity(model_id: str) -> dict[str, Any]:
    task_adaptation = {
        "task_io_only": True,
        "drop_test_input": "coords(3) + official globals(8) + time(1)",
        "drop_test_output": "displacement(3) + Von Mises stress(1)",
    }
    if model_id == "transolver":
        return {
            "implementation": "physicsnemo.models.transolver.Transolver",
            "official_profile": "transolver_volume.yaml",
            "official_profile_commit": PHYSICSNEMO_PROFILE_COMMIT,
            **TRANSOLVER_SCALE,
            "functional_dim": 12,
            "embedding_dim": 3,
            "activation_checkpointing": False,
            **task_adaptation,
        }
    if model_id == "transolver_plus":
        return {
            "implementation": str(
                official.external.external_source_path(
                    official.external.TRANSOLVER_PLUS_MODEL_ID
                )
            ),
            "official_profile": "author aircraft native",
            "upstream_commit": TRANSOLVER_PLUS_COMMIT,
            "native_config": {
                **official.external.TRANSOLVER_PLUS_KWARGS,
                "space_dim": 12,
            },
            **task_adaptation,
        }
    if model_id == "transolver3":
        return {
            "implementation": str(
                official.external.external_source_path(
                    official.external.TRANSOLVER3_MODEL_ID
                )
            ),
            "official_profile": "author DrivAerML volume native",
            "upstream_commit": TRANSOLVER3_COMMIT,
            "native_config": {
                **official.external.TRANSOLVER3_KWARGS,
                "space_dim": 12,
            },
            "native_training_activation_checkpointing": True,
            **task_adaptation,
        }
    if model_id in {"geotransolver", "geots_flare"}:
        return {
            "implementation": (
                "physicsnemo.experimental.models.geotransolver.GeoTransolver"
            ),
            "official_profile": "geotransolver_volume.yaml",
            "official_profile_commit": PHYSICSNEMO_PROFILE_COMMIT,
            "config": OmegaConf.to_container(
                geo_config(model_id), resolve=True
            ),
            **task_adaptation,
        }
    if model_id in {"lts", "lts_ca", "lts_ca_fourier_tenv"}:
        component_aware = model_id != "lts"
        full = model_id == "lts_ca_fourier_tenv"
        return {
            "implementation": "LightTransolver + optional CCLTS/Fourier/TEnv",
            "matched_reference": "Transolver official volume scale",
            "matched_reference_commit": PHYSICSNEMO_PROFILE_COMMIT,
            **TRANSOLVER_SCALE,
            "functional_dim": 12,
            "embedding_dim": 3,
            "activation_checkpointing": False,
            "resampling_count": 1,
            "component_aware": component_aware,
            "component_count": 17 if component_aware else 0,
            "ca_reductions": ["mean", "amax"] if component_aware else [],
            "ca_max_backward": (
                "contiguous_custom_backward" if component_aware else None
            ),
            "ca_detach_source": False if component_aware else None,
            "fourier_bands": official.FOURIER_BANDS if full else None,
            "time_envelope_width": (
                official.TIME_ENVELOPE_WIDTH if full else None
            ),
            "block_film": False,
            **task_adaptation,
        }
    raise ValueError(model_id)


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
            "old_14d_padding": False,
        },
        "normalization": official.normalization_spec(),
        "warm_start": False,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNTS[model_id],
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
                    "droptest_official_backbones_20260826.instantiate_model"
                ),
                "model_id": model_id,
                "identity_sha256": model_config_sha256(model_id),
            }
        )
    return cfg


def _build_backbone(model_id: str, runtime: Any) -> nn.Module:
    if model_id == "transolver":
        return PhoneDropTransolverOfficial()
    if model_id == "transolver_plus":
        return official.PhoneDropExternal8(
            official.external.TRANSOLVER_PLUS_MODEL_ID
        )
    if model_id == "transolver3":
        return official.PhoneDropExternal8(
            official.external.TRANSOLVER3_MODEL_ID
        )
    if model_id in {"geotransolver", "geots_flare"}:
        return instantiate(geo_config(model_id))
    if model_id in {"lts", "lts_ca", "lts_ca_fourier_tenv"}:
        return PhoneDropLTSOfficial(
            runtime.part_mapping.node_part_index,
            component_aware=model_id != "lts",
            fourier_tenv=model_id == "lts_ca_fourier_tenv",
        )
    raise ValueError(model_id)


def instantiate_model(model_id: str, seed: int, runtime: Any) -> nn.Module:
    official.formal.seed_model(seed)
    normalization = official.normalization_spec()
    conditioned = official.OfficialConditioningAdapter(
        _build_backbone(model_id, runtime)
    )
    model = official.DisplacementRMSModelAdapter(
        conditioned,
        displacement_rms=normalization["displacement_rms"],
        position_std=normalization["position_std"],
    )
    observed = sum(parameter.numel() for parameter in model.parameters())
    expected = EXPECTED_PARAMETER_COUNTS[model_id]
    if observed != expected:
        raise RuntimeError(
            f"{model_id} parameter count {observed} != frozen {expected}"
        )
    return model


def model_target(model_id: str) -> str:
    if model_id not in MODELS:
        raise ValueError(model_id)
    return (
        "examples.structural_mechanics.openradioss_dataset_gen."
        "droptest_official_backbones_20260826"
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
        "status": "READY_AFTER_8_MODEL_PREFLIGHT",
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
        "models": {model: model_identity(model) for model in MODELS},
        "parameter_counts": dict(EXPECTED_PARAMETER_COUNTS),
        "tasks": [
            task_mapping(index, output_root)
            for index in range(len(MODELS) * len(SEEDS))
        ],
        "gpu_packing": {
            "preflight": "8 models x 1 GPU on one 8-GPU H200 node",
            "formal": "one seed per 8-GPU H200 node; 3 seed jobs",
            "one_model_per_gpu": True,
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
