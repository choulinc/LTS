"""Official 192-run DropTest six-model time-conditional benchmark.

The module adapts the repository's audited 135-run timestep runner without
changing any baseline solver core.  It replaces only the dataset contract,
the official eight-field conditioning interface, train-only normalization,
paper Relative-L2 metrics, and the explicitly requested optimization policy.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import time
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as torch_dist
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from physicsnemo.models.transolver import Transolver
from physicsnemo.models.transolver.light_transolver import LightTransolver

from examples.structural_mechanics.openradioss_dataset_gen.drop_test.official_192_benchmark import (
    paper_metrics,
)
from examples.structural_mechanics.openradioss_dataset_gen.phone_drop_phase1.lts_motion_gatepart_deformation_expert.cclts import (
    CCLTS,
)
from examples.structural_mechanics.openradioss_dataset_gen.phone_drop_phase1.lts_motion_gatepart_deformation_expert.displacement_rms_models import (
    DisplacementRMSModelAdapter,
)
from examples.structural_mechanics.openradioss_dataset_gen.phone_drop_phase1.lts_motion_gatepart_deformation_expert import (
    run_displacement_rms_7model_formal as legacy,
)
from examples.structural_mechanics.openradioss_dataset_gen.phone_drop_phase1.lts_motion_gatepart_deformation_expert import (
    transolver_plus3_protocol as external,
)


REPOSITORY = Path("/work/u9845070/physicsnemo")
PACKAGE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = REPOSITORY / (
    "examples/structural_mechanics/openradioss_dataset_gen/drop_test/"
    "droptest_official_training_ready"
)
OUTPUT_ROOT = Path(
    "/work/u9845070/physicsnemo_outputs/drop_test/"
    "official192_six_model_timeconditional_20260825"
)
STATS_DIR = OUTPUT_ROOT / "normalization_stats_train132_v1"
PREFLIGHT_ROOT = OUTPUT_ROOT / "preflight"
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
FORMAT = "droptest_official192_six_model_timeconditional_v1"

GLOBAL_FEATURES = (
    "e_scale_mat1",
    "e_scale_mat4",
    "e_scale_mat5",
    "e_scale_mat8",
    "e_scale_mat9",
    "rwall_orientation_rx",
    "rwall_orientation_ry",
    "rwall_orientation_rz",
)
MATERIAL_FEATURES = GLOBAL_FEATURES[:5]
ORIENTATION_FEATURES = GLOBAL_FEATURES[5:]
MODELS = (
    "transolver",
    "geotransolver",
    "geots_flare",
    "transolver_plus",
    "transolver3",
    "cclts_fourier_tenv",
)
MODEL_LABELS = {
    "transolver": "Transolver",
    "geotransolver": "GeoTransolver",
    "geots_flare": "GeoTS-FLARE",
    "transolver_plus": "Transolver++",
    "transolver3": "Transolver-3",
    "cclts_fourier_tenv": "LTS-FTG",
}
SEEDS = (7, 17, 27)
TRAIN_TRAJECTORIES = 132
VALIDATION_TRAJECTORIES = 30
TEST_TRAJECTORIES = 30
PREDICTION_STEPS = 99
TRAIN_TIMESTEP_SAMPLES = TRAIN_TRAJECTORIES * PREDICTION_STEPS
MAX_EPOCHS = 25
GPU_MODE = "1gpu_batch1"
GPU_MODES = {
    GPU_MODE: {
        "world_size": 1,
        "micro_batch_per_gpu": 1,
        "gradient_accumulation": 1,
        "global_job_offset": 0,
    }
}
BEST_VALIDATION_CHECKPOINT = "best_validation_relative_l2.pt"
SELECTION_METRIC_NAME = "validation_clean_equal_field_relative_l2"
CHECKPOINT_SELECTION_DESCRIPTION = (
    "validation-only 0.5 * (physical displacement trajectory-macro Relative-L2 "
    "+ physical Von Mises stress trajectory-macro Relative-L2)"
)
FOURIER_BANDS = 10
TIME_ENVELOPE_WIDTH = 64
ARCH_SEED_OFFSET = 771013
OFFICIAL_TOPOLOGY_REFERENCE_VTU = DATASET_ROOT / "validation" / "run0031.vtu"
COMPONENT_REFERENCE_VTU = REPOSITORY / (
    "examples/structural_mechanics/openradioss_dataset_gen/phone_drop_phase1/"
    "final_dataset/validation/run_000006.vtu"
)
COMPONENT_MAPPING_EVIDENCE = OUTPUT_ROOT / "component_mapping_evidence.json"

shared = legacy.shared
formal = legacy.formal
phone = formal.phone
_ORIGINAL_PHONE_LOAD_CONFIG = phone.load_config
_ORIGINAL_SHARED_TRAIN = shared.train
_ORIGINAL_LOAD_PART_MAPPING = formal.load_part_mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def git_identity() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPOSITORY, text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def exact_paths() -> dict[str, str]:
    return {
        "train": str((DATASET_ROOT / "train").resolve()),
        "validation": str((DATASET_ROOT / "validation").resolve()),
        "test": str((DATASET_ROOT / "test").resolve()),
        "global_features": str((DATASET_ROOT / "global_features.json").resolve()),
        "split_manifest": str((DATASET_ROOT / "split_manifest.json").resolve()),
    }


def _split_ids() -> dict[str, list[str]]:
    return {
        split: [path.stem for path in sorted((DATASET_ROOT / split).glob("*.vtu"))]
        for split in ("train", "validation", "test")
    }


def validate_dataset_metadata(cfg: DictConfig | None = None) -> dict[str, Any]:
    manifest_path = DATASET_ROOT / "split_manifest.json"
    globals_path = DATASET_ROOT / "global_features.json"
    manifest = json.loads(manifest_path.read_text())
    globals_ = json.loads(globals_path.read_text())
    expected_counts = {
        "train": TRAIN_TRAJECTORIES,
        "validation": VALIDATION_TRAJECTORIES,
        "test": TEST_TRAJECTORIES,
    }
    if manifest.get("counts") != expected_counts:
        raise RuntimeError(f"Official split counts changed: {manifest.get('counts')}")
    if manifest.get("source_anomaly_audit_status") != "PASS":
        raise RuntimeError("Official anomaly audit is not PASS")
    if not bool(manifest.get("all_hard_links_verified_same_inode")):
        raise RuntimeError("Official hard-link identity gate is not PASS")
    run_to_split = manifest.get("run_to_split", {})
    if len(run_to_split) != 192 or len(globals_) != 192:
        raise RuntimeError("Official manifest/global record count is not 192")
    ids = _split_ids()
    for split, count in expected_counts.items():
        if len(ids[split]) != count or len(set(ids[split])) != count:
            raise RuntimeError(f"{split} file count/identity mismatch")
        expected = {run for run, role in run_to_split.items() if role == split}
        if set(ids[split]) != expected:
            raise RuntimeError(f"{split} files differ from immutable manifest")
    observed = set().union(*(set(values) for values in ids.values()))
    if observed != set(globals_) or observed != set(run_to_split):
        raise RuntimeError("VTU/global/manifest run IDs differ")
    for run, values in globals_.items():
        if list(values) != list(GLOBAL_FEATURES):
            raise RuntimeError(f"{run} official global feature order changed")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise FloatingPointError(f"{run} has non-finite conditioning")
        if any(float(values[name]) not in {0.8, 1.2} for name in MATERIAL_FEATURES):
            raise RuntimeError(f"{run} material scale is outside 0.8/1.2")
        orientation = tuple(float(values[name]) for name in ORIENTATION_FEATURES)
        if orientation not in {
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (-10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, -10.0, 0.0),
            (0.0, 0.0, 10.0),
        }:
            raise RuntimeError(f"{run} orientation is outside official design")
    if cfg is not None:
        resolved = {
            "train": str(Path(cfg.training.raw_data_dir).resolve()),
            "validation": str(Path(cfg.training.raw_data_dir_validation).resolve()),
            "test": str(Path(cfg.inference.raw_data_dir_test).resolve()),
            "global_features": str(
                Path(cfg.training.global_features_filepath).resolve()
            ),
        }
        expected = {key: value for key, value in exact_paths().items() if key != "split_manifest"}
        if resolved != expected:
            raise RuntimeError(f"Resolved paths differ from official paths: {resolved}")
        if list(cfg.datapipe.global_features) != list(GLOBAL_FEATURES):
            raise RuntimeError("Config does not use exactly the eight official globals")
    return {
        "split_ids": ids,
        "split_id_hashes": {key: sha256_json(value) for key, value in ids.items()},
        "split_manifest_sha256": sha256_file(manifest_path),
        "global_features_sha256": sha256_file(globals_path),
        "source_anomaly_audit_sha256": manifest["source_anomaly_audit_sha256"],
        "grouped_material_combinations": True,
    }


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def load_verified_part_mapping(reference_vtu: str | Path) -> Any:
    """Transfer PART_ID labels only after exact fixed-mesh topology checks.

    The official processed VTUs intentionally contain no PART_ID array.  Their
    fixed mesh is the same mesh as the established DropTest component reference;
    only the within-cell vertex ordering differs for a subset of cells.  Verify
    that invariant before using the old cell annotations to derive node labels.
    """

    import numpy as np
    import pyvista as pv

    official_path = Path(reference_vtu).resolve()
    if official_path != OFFICIAL_TOPOLOGY_REFERENCE_VTU.resolve():
        raise RuntimeError(f"Unexpected official topology reference: {official_path}")
    component_path = COMPONENT_REFERENCE_VTU.resolve()
    mapping = _ORIGINAL_LOAD_PART_MAPPING(component_path)
    official = pv.read(official_path)
    annotated = pv.read(component_path)
    if (official.n_points, official.n_cells) != (
        annotated.n_points,
        annotated.n_cells,
    ):
        raise RuntimeError("Official and annotated component mesh counts differ")
    if not np.array_equal(official.points, annotated.points):
        raise RuntimeError("Official and annotated component node order/coordinates differ")
    if not np.array_equal(official.celltypes, annotated.celltypes):
        raise RuntimeError("Official and annotated component cell types differ")
    if not np.array_equal(official.offset, annotated.offset):
        raise RuntimeError("Official and annotated component cell offsets differ")
    official_connectivity = np.asarray(official.cell_connectivity, dtype=np.int64)
    annotated_connectivity = np.asarray(annotated.cell_connectivity, dtype=np.int64)
    offsets = np.asarray(official.offset, dtype=np.int64)
    ordered_mismatch_cells = 0
    set_mismatch_cells = 0
    for lower, upper in zip(offsets[:-1], offsets[1:]):
        left = official_connectivity[lower:upper]
        right = annotated_connectivity[lower:upper]
        if np.array_equal(left, right):
            continue
        ordered_mismatch_cells += 1
        if not np.array_equal(np.sort(left), np.sort(right)):
            set_mismatch_cells += 1
    if set_mismatch_cells:
        raise RuntimeError(
            f"Official component topology has {set_mismatch_cells} cell-node-set mismatches"
        )
    if mapping.node_count != official.n_points or len(mapping.component_ids) != 17:
        raise RuntimeError("Verified component mapping is not 76065 nodes / 17 parts")
    evidence = {
        "status": "PASS",
        "purpose": "LTS-FTG component-aware attention and evaluator part audit only",
        "baseline_model_input_uses_components": False,
        "official_vtu": str(official_path),
        "annotated_reference_vtu": str(component_path),
        "official_has_part_id": "PART_ID" in official.cell_data,
        "annotated_has_part_id": "PART_ID" in annotated.cell_data,
        "node_count": int(official.n_points),
        "cell_count": int(official.n_cells),
        "component_ids": list(mapping.component_ids),
        "points_exact_and_same_order": True,
        "cell_types_exact_and_same_order": True,
        "cell_offsets_exact_and_same_order": True,
        "per_cell_node_sets_exact": True,
        "within_cell_order_mismatch_count": ordered_mismatch_cells,
        "cell_node_set_mismatch_count": set_mismatch_cells,
        "official_points_sha256": _array_sha256(official.points),
        "official_offsets_sha256": _array_sha256(official.offset),
        "official_celltypes_sha256": _array_sha256(official.celltypes),
        "official_connectivity_sha256": _array_sha256(official_connectivity),
        "annotated_connectivity_sha256": _array_sha256(annotated_connectivity),
        "node_part_index_sha256": _array_sha256(mapping.node_part_index.numpy()),
    }
    atomic_json(COMPONENT_MAPPING_EVIDENCE, evidence)
    return mapping


def stats_manifest() -> dict[str, Any]:
    path = STATS_DIR / "stats_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fresh train-only stats are missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("format") != "droptest_official192_train132_stats_v1":
        raise RuntimeError("Unexpected official-192 stats format")
    metadata = validate_dataset_metadata()
    if payload.get("train_split_id_hash") != metadata["split_id_hashes"]["train"]:
        raise RuntimeError("Stats were not computed from the immutable train split")
    if payload.get("validation_or_test_used") is not False:
        raise RuntimeError("Stats manifest does not exclude validation/test")
    return payload


def verify_stats(stats_dir: str | Path = STATS_DIR, *, require_read_only: bool) -> dict[str, str]:
    root = Path(stats_dir).resolve()
    if root != STATS_DIR.resolve():
        raise RuntimeError(f"Unexpected stats directory: {root}")
    payload = stats_manifest()
    observed: dict[str, str] = {}
    for filename, expected in payload["file_sha256"].items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Stats hash mismatch: {path}")
        if require_read_only and path.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise RuntimeError(f"Frozen stats file is writable: {path}")
        observed[filename] = expected
    return observed


def load_config(_: str = "official192") -> DictConfig:
    cfg = _ORIGINAL_PHONE_LOAD_CONFIG("phone_drop_tc_ddp_bf16_30ep_lts")
    paths = exact_paths()
    with open_dict(cfg):
        cfg.experiment_name = FORMAT
        cfg.dataset_contract.split_counts = {
            "train": TRAIN_TRAJECTORIES,
            "validation": VALIDATION_TRAJECTORIES,
            "test": TEST_TRAJECTORIES,
        }
        cfg.dataset_contract.total_runs = 192
        cfg.dataset_contract.global_dim = 8
        cfg.training.raw_data_dir = paths["train"]
        cfg.training.raw_data_dir_validation = paths["validation"]
        cfg.inference.raw_data_dir_test = paths["test"]
        cfg.training.global_features_filepath = paths["global_features"]
        cfg.training.num_training_samples = TRAIN_TRAJECTORIES
        cfg.training.num_validation_samples = VALIDATION_TRAJECTORIES
        cfg.training.epochs = MAX_EPOCHS
        cfg.training.scheduler_t_max = MAX_EPOCHS
        cfg.training.phone_ddp.world_size = 1
        cfg.training.phone_ddp.per_gpu_batch_size = 1
        cfg.training.phone_ddp.gradient_accumulation = 1
        cfg.training.phone_ddp.effective_global_batch = 1
        cfg.training.phone_ddp.effective_global_batch_size = 1
        cfg.training.phone_ddp.num_test_samples = TEST_TRAJECTORIES
        cfg.training.phone_ddp.normalization_stats_dir = str(STATS_DIR)
        cfg.training.phone_ddp.normalization_read_only = True
        cfg.datapipe.stats_dir = str(STATS_DIR)
        cfg.datapipe.global_features = list(GLOBAL_FEATURES)
        cfg.model.functional_dim = 4
        cfg.model.global_dim = 8
        cfg.model.out_dim = 4
    return cfg


def check_sample(
    sample: Any,
    *,
    role: str,
    expected_time: float | None,
    cfg: DictConfig,
) -> None:
    nodes = int(cfg.dataset_contract.nodes)
    expected_target = (nodes, 4) if role == "train" else (nodes, 99, 4)
    if tuple(sample.node_target.shape) != expected_target:
        raise RuntimeError(f"{role} target {tuple(sample.node_target.shape)} != {expected_target}")
    if tuple(sample.node_features["coords"].shape) != (nodes, 3):
        raise RuntimeError(f"{role} initial coordinate shape changed")
    if not bool(torch.isfinite(sample.node_target).all()):
        raise FloatingPointError(f"{role} target contains NaN/Inf")
    if sample.global_features is None or tuple(sample.global_features) != GLOBAL_FEATURES:
        raise RuntimeError(f"{role} global conditioning order changed")
    if expected_time is not None:
        observed = float(sample.node_features["time"].item())
        if not math.isclose(observed, expected_time, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(f"{role} target time {observed} != {expected_time}")


def _normalize_global(name: str, value: torch.Tensor) -> torch.Tensor:
    if name in MATERIAL_FEATURES:
        return (value - 1.0) / 0.2
    if name in ORIENTATION_FEATURES:
        return value / 10.0
    raise KeyError(name)


class OfficialConditioningAdapter(nn.Module):
    """Apply the fixed official preprocessing without changing a solver core."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, sample: Any, data_stats: dict[str, Any] | None = None) -> torch.Tensor:
        original = sample.global_features
        if original is None or tuple(original) != GLOBAL_FEATURES:
            raise RuntimeError("Official eight-field conditioning is missing")
        sample.global_features = {
            name: _normalize_global(name, original[name]) for name in GLOBAL_FEATURES
        }
        try:
            return self.model(sample=sample, data_stats=data_stats)
        finally:
            sample.global_features = original


class PhoneDropTransolver8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rollout_steps = 99
        self.backbone = Transolver(
            functional_dim=12,
            out_dim=4,
            embedding_dim=3,
            n_layers=6,
            n_hidden=256,
            dropout=0.0,
            n_head=8,
            act="gelu",
            mlp_ratio=4,
            slice_num=128,
            unified_pos=False,
            structured_shape=None,
            use_te=False,
            time_input=False,
            plus=False,
        )

    def _step(self, sample: Any) -> torch.Tensor:
        coords = sample.node_features["coords"]
        time_value = sample.node_features.get("time")
        if time_value is None or time_value.numel() != 1:
            raise RuntimeError("Transolver requires one normalized target time")
        globals_ = torch.stack([sample.global_features[name] for name in GLOBAL_FEATURES]).to(coords)
        raw = torch.cat(
            (
                coords,
                globals_.reshape(1, 8).expand(coords.shape[0], 8),
                time_value.to(coords).reshape(1, 1).expand(coords.shape[0], 1),
            ),
            dim=-1,
        )
        output = self.backbone(fx=raw.unsqueeze(0), embedding=coords.unsqueeze(0)).squeeze(0)
        return torch.cat((coords + output[:, :3], output[:, 3:4]), dim=-1)

    def forward(self, sample: Any, data_stats: dict[str, Any] | None = None) -> torch.Tensor:
        del data_stats
        if self.training:
            return self._step(sample)
        coords = sample.node_features["coords"]
        original = sample.node_features.get("time")
        had_time = "time" in sample.node_features
        outputs = []
        try:
            for index in range(99):
                sample.node_features["time"] = coords.new_tensor(index / 99)
                outputs.append(self._step(sample))
        finally:
            if had_time:
                sample.node_features["time"] = original
            else:
                sample.node_features.pop("time", None)
        return torch.stack(outputs, dim=1)


class PhoneDropExternal8(nn.Module):
    """Eight-field adapter around pristine Transolver++ or Transolver-3."""

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self.rollout_steps = 99
        official = external._load_official_module(model_id)  # noqa: SLF001
        if model_id == "transolver_plus_official":
            kwargs = {**external.TRANSOLVER_PLUS_KWARGS, "space_dim": 12}
            self.backbone = official.Model(**kwargs)
            self.evaluation_seed = 20260804
        elif model_id == "transolver3_official":
            kwargs = {**external.TRANSOLVER3_KWARGS, "space_dim": 12}
            self.backbone = official.Model(**kwargs)
            self.evaluation_seed = None
        else:
            raise ValueError(model_id)

    def _raw(self, sample: Any) -> tuple[torch.Tensor, torch.Tensor]:
        coords = sample.node_features["coords"]
        time_value = sample.node_features.get("time")
        if time_value is None or time_value.numel() != 1:
            raise RuntimeError("External baseline requires one normalized target time")
        globals_ = torch.stack([sample.global_features[name] for name in GLOBAL_FEATURES]).to(coords)
        raw = torch.cat(
            (
                coords,
                globals_.reshape(1, 8).expand(coords.shape[0], 8),
                time_value.to(coords).reshape(1, 1).expand(coords.shape[0], 1),
            ),
            dim=-1,
        )
        return coords, raw

    def _backbone_forward(self, raw: torch.Tensor) -> torch.Tensor:
        if self.model_id == "transolver_plus_official":
            if not torch_dist.is_initialized() or torch_dist.get_world_size() != 1:
                raise RuntimeError("Transolver++ requires initialized world-size-1")
            return self.backbone((raw, raw[..., :3], None))
        result = self.backbone([raw], use_checkpoint=self.training, input_list=True)
        if not isinstance(result, list) or len(result) != 1:
            raise RuntimeError("Transolver-3 output contract changed")
        return result[0]

    @contextmanager
    def _evaluation_rng(self, device: torch.device) -> Iterator[None]:
        if self.evaluation_seed is None:
            yield
            return
        devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.evaluation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(self.evaluation_seed)
            yield

    def _step(self, sample: Any) -> torch.Tensor:
        coords, raw = self._raw(sample)
        output = self._backbone_forward(raw.unsqueeze(0)).squeeze(0)
        return torch.cat((coords + output[:, :3], output[:, 3:4]), dim=-1)

    def forward(self, sample: Any, data_stats: dict[str, Any] | None = None) -> torch.Tensor:
        del data_stats
        if self.training:
            return self._step(sample)
        coords = sample.node_features["coords"]
        original = sample.node_features.get("time")
        had_time = "time" in sample.node_features
        outputs = []
        try:
            with self._evaluation_rng(coords.device):
                for index in range(99):
                    sample.node_features["time"] = coords.new_tensor(index / 99)
                    outputs.append(self._step(sample))
        finally:
            if had_time:
                sample.node_features["time"] = original
            else:
                sample.node_features.pop("time", None)
        return torch.stack(outputs, dim=1)


class PhoneDropCCLTS8(nn.Module):
    def __init__(self, node_part_index: torch.Tensor) -> None:
        super().__init__()
        self.rollout_steps = 99
        light = LightTransolver(
            functional_dim=12,
            out_dim=4,
            embedding_dim=3,
            n_layers=6,
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
        )
        self.backbone = CCLTS(light, node_part_index, num_parts=17)
        self.backbone.configure_training_pooling(max_reduction="amax", detach_source=False)
        self.backbone.configure_training_affine(implementation="contiguous_custom_backward")
        install_lts_ftg(light)

    @property
    def motion_expert(self) -> LightTransolver:
        return self.backbone.motion_expert

    def _step(self, sample: Any) -> torch.Tensor:
        coords = sample.node_features["coords"]
        time_value = sample.node_features.get("time")
        if time_value is None or time_value.numel() != 1:
            raise RuntimeError("LTS-FTG requires one normalized target time")
        globals_ = torch.stack([sample.global_features[name] for name in GLOBAL_FEATURES]).to(coords)
        raw = torch.cat(
            (
                coords,
                globals_.reshape(1, 8).expand(coords.shape[0], 8),
                time_value.to(coords).reshape(1, 1).expand(coords.shape[0], 1),
            ),
            dim=-1,
        )
        output = self.backbone(fx=raw.unsqueeze(0), embedding=coords.unsqueeze(0)).squeeze(0)
        return torch.cat((coords + output[:, :3], output[:, 3:4]), dim=-1)

    def forward(self, sample: Any, data_stats: dict[str, Any] | None = None) -> torch.Tensor:
        del data_stats
        if self.training:
            return self._step(sample)
        coords = sample.node_features["coords"]
        original = sample.node_features.get("time")
        had_time = "time" in sample.node_features
        outputs = []
        try:
            for index in range(99):
                sample.node_features["time"] = coords.new_tensor(index / 99)
                outputs.append(self._step(sample))
        finally:
            if had_time:
                sample.node_features["time"] = original
            else:
                sample.node_features.pop("time", None)
        return torch.stack(outputs, dim=1)


def _init_like_transolver(module: nn.Module) -> None:
    for item in module.modules():
        if isinstance(item, nn.Linear):
            nn.init.trunc_normal_(item.weight, std=0.02)
            if item.bias is not None:
                nn.init.zeros_(item.bias)


def _zero_last_linear(module: nn.Module) -> None:
    linears = [item for item in module.modules() if isinstance(item, nn.Linear)]
    nn.init.zeros_(linears[-1].weight)
    if linears[-1].bias is not None:
        nn.init.zeros_(linears[-1].bias)


def _fourier_expand(values: torch.Tensor, bands: int) -> torch.Tensor:
    frequencies = torch.pow(2.0, torch.arange(bands, device=values.device, dtype=values.dtype))
    scaled = values.unsqueeze(-1) * frequencies * math.pi
    return torch.cat((scaled.sin(), scaled.cos()), dim=-1).flatten(start_dim=-2)


def install_lts_features(
    lts: LightTransolver,
    *,
    use_fourier: bool,
    use_tenv: bool,
) -> None:
    """Install the existing fixed Fourier and/or TEnv LTS features.

    The full-model path intentionally retains the exact modules, initialization
    offsets, and hook locations used by the official-192 LTS-FTG baseline.  The
    boolean switches only make those two already-audited mechanisms independently
    selectable for factorial ablations.  BlockFiLM is not installed here.
    """

    if not use_fourier and not use_tenv:
        return

    context: dict[str, torch.Tensor] = {}

    def capture(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module
        raw = inputs[0]
        if raw.shape[-1] != 15:
            raise RuntimeError(f"LTS-FTG preprocess input must be 15-D, got {raw.shape}")
        context["raw"] = raw
        context["conditioning"] = raw[:, :1, 6:]

    lts.preprocess.register_forward_pre_hook(capture)
    if use_fourier:
        time_bands = FOURIER_BANDS - 2
        fourier_dimension = 3 * FOURIER_BANDS * 2 + time_bands * 2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + ARCH_SEED_OFFSET + 2)
            projector = nn.Sequential(
                nn.Linear(fourier_dimension, lts.n_hidden),
                nn.GELU(),
                nn.Linear(lts.n_hidden, lts.n_hidden),
            )
            _init_like_transolver(projector)
            _zero_last_linear(projector)
        lts.fourier_projector = projector

        def augment(
            module: nn.Module, inputs: Any, output: torch.Tensor
        ) -> torch.Tensor:
            del module, inputs
            raw = context["raw"]
            features = torch.cat(
                (
                    _fourier_expand(raw[..., :3], FOURIER_BANDS),
                    _fourier_expand(raw[..., -1:], time_bands),
                ),
                dim=-1,
            )
            return output + projector(features)

        lts.preprocess.register_forward_hook(augment)

    if use_tenv:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + ARCH_SEED_OFFSET + 4)
            envelope = nn.Sequential(
                nn.Linear(9, TIME_ENVELOPE_WIDTH),
                nn.GELU(),
                nn.Linear(TIME_ENVELOPE_WIDTH, 3),
            )
            _init_like_transolver(envelope)
            _zero_last_linear(envelope)
        lts.time_envelope = envelope

        def rescale(
            module: nn.Module, inputs: Any, output: torch.Tensor
        ) -> torch.Tensor:
            del module, inputs
            scale = 1.0 + envelope(context["conditioning"])
            return torch.cat((output[..., :3] * scale, output[..., 3:]), dim=-1)

        lts.output_head.register_forward_hook(rescale)


def install_lts_ftg(lts: LightTransolver) -> None:
    """Install the frozen Fourier-10 and TEnv-64 architecture for 8 globals."""

    install_lts_features(lts, use_fourier=True, use_tenv=True)


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
            "slice_num": 128,
            "n_layers": 6,
            "n_hidden": 256,
            "n_head": 8,
            "n_hidden_local": 32,
            "radii": [0.05, 0.25],
            "neighbors_in_radius": [8, 32],
            "attention_type": "GALE" if model_id == "geotransolver" else "GALE_FA",
            "use_te": False,
            "time_input": False,
            "include_local_features": True,
            "num_time_steps": 100,
            "activation_checkpointing": False,
        }
    )


def backbone_identity(model_id: str) -> dict[str, Any]:
    shared_scale = {
        "hidden_dim": 256,
        "layers": 6,
        "heads": 8,
        "physics_slices": 128,
        "mlp_ratio": 4,
        "dropout": 0.0,
    }
    if model_id == "transolver":
        return {"implementation": "physicsnemo.models.transolver.Transolver", **shared_scale, "functional_dim": 12}
    if model_id in {"geotransolver", "geots_flare"}:
        return {
            "implementation": "physicsnemo.experimental.models.geotransolver.GeoTransolver",
            "config": OmegaConf.to_container(geo_config(model_id), resolve=True),
        }
    if model_id == "transolver_plus":
        return {
            "implementation": str(external.external_source_path(external.TRANSOLVER_PLUS_MODEL_ID)),
            "upstream_commit": external.EXTERNAL_COMMITS[external.TRANSOLVER_PLUS_MODEL_ID],
            "native_config": {**external.TRANSOLVER_PLUS_KWARGS, "space_dim": 12},
        }
    if model_id == "transolver3":
        return {
            "implementation": str(external.external_source_path(external.TRANSOLVER3_MODEL_ID)),
            "upstream_commit": external.EXTERNAL_COMMITS[external.TRANSOLVER3_MODEL_ID],
            "native_config": {**external.TRANSOLVER3_KWARGS, "space_dim": 12},
            "native_training_activation_checkpointing": True,
        }
    if model_id == "cclts_fourier_tenv":
        return {
            "implementation": "LightTransolver + CCLTS + fixed Fourier + TEnv",
            **shared_scale,
            "functional_dim": 12,
            "component_aware": True,
            "fourier_bands": FOURIER_BANDS,
            "time_envelope_width": TIME_ENVELOPE_WIDTH,
            "block_film": False,
        }
    raise ValueError(model_id)


def normalization_spec() -> dict[str, Any]:
    payload = stats_manifest()
    node = json.loads((STATS_DIR / "node_stats.json").read_text())
    rms = json.loads((STATS_DIR / "displacement_rms_stats.json").read_text())
    return {
        "mode": "train132_displacement_rms_only_no_mean_and_stress_zscore",
        "stats_dir": str(STATS_DIR),
        "stats_hashes": verify_stats(STATS_DIR, require_read_only=True),
        "displacement_rms": rms["rms"],
        "position_std": node["pos_std"],
        "train_split_id_hash": payload["train_split_id_hash"],
        "validation_or_test_used": False,
    }


def model_identity(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "label": MODEL_LABELS[model_id],
        "backbone": backbone_identity(model_id),
        "conditioning": {
            "official_feature_order": list(GLOBAL_FEATURES),
            "material_preprocessing": "(E_scale - 1.0) / 0.2",
            "orientation_preprocessing": "angle_deg / 10.0",
            "time": "prediction_index / 99 for indices 0..98",
            "old_14d_padding": False,
        },
        "normalization": normalization_spec(),
        "output": "[D_x/RMS_x,D_y/RMS_y,D_z/RMS_z,Von_Mises_zscore] during training; canonical normalized position/stress for evaluation",
        "warm_start": False,
    }


def model_config_sha256(model_id: str) -> str:
    return sha256_json(model_identity(model_id))


def resolved_config(model_id: str, seed: int, gpu_mode: str, output_root: Path) -> DictConfig:
    if model_id not in MODELS or seed not in SEEDS or gpu_mode != GPU_MODE:
        raise ValueError((model_id, seed, gpu_mode))
    cfg = load_config()
    directory = run_directory(output_root, gpu_mode, model_id, seed)
    with open_dict(cfg):
        cfg.training.seed = seed
        cfg.training.epochs = MAX_EPOCHS
        cfg.training.scheduler_t_max = MAX_EPOCHS
        cfg.training.phone_ddp.config_name = FORMAT
        cfg.training.phone_ddp.run_name = model_id
        cfg.training.phone_ddp.model_id = model_id
        cfg.training.phone_ddp.model_label = MODEL_LABELS[model_id]
        cfg.training.phone_ddp.output_root = str(output_root)
        cfg.training.phone_ddp.output_dir = str(directory)
        cfg.training.phone_ddp.optimizer_updates_per_epoch = TRAIN_TIMESTEP_SAMPLES
        cfg.training.phone_ddp.checkpoint_selection = {
            "name": SELECTION_METRIC_NAME,
            "description": CHECKPOINT_SELECTION_DESCRIPTION,
            "uses_test": False,
        }
        cfg.model = OmegaConf.create(
            {
                "_target_": "official_192_benchmark.benchmark.instantiate_model",
                "model_id": model_id,
                "identity_sha256": model_config_sha256(model_id),
            }
        )
    return cfg


def _build_backbone(model_id: str, runtime: Any) -> nn.Module:
    if model_id == "transolver":
        return PhoneDropTransolver8()
    if model_id in {"geotransolver", "geots_flare"}:
        return instantiate(geo_config(model_id))
    if model_id == "transolver_plus":
        return PhoneDropExternal8(external.TRANSOLVER_PLUS_MODEL_ID)
    if model_id == "transolver3":
        return PhoneDropExternal8(external.TRANSOLVER3_MODEL_ID)
    if model_id == "cclts_fourier_tenv":
        return PhoneDropCCLTS8(runtime.part_mapping.node_part_index)
    raise ValueError(model_id)


def instantiate_model(model_id: str, seed: int, runtime: Any) -> nn.Module:
    formal.seed_model(seed)
    spec = normalization_spec()
    conditioned = OfficialConditioningAdapter(_build_backbone(model_id, runtime))
    return DisplacementRMSModelAdapter(
        conditioned,
        displacement_rms=spec["displacement_rms"],
        position_std=spec["position_std"],
    )


def run_directory(output_root: Path, gpu_mode: str, model_id: str, seed: int) -> Path:
    if gpu_mode != GPU_MODE:
        raise ValueError(gpu_mode)
    return Path(output_root) / "runs" / model_id / f"seed_{seed}"


def task_mapping(task_id: int, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if not 0 <= int(task_id) < 18:
        raise ValueError("Formal array task must be 0..17")
    model_id = MODELS[int(task_id) // 3]
    seed = SEEDS[int(task_id) % 3]
    return {
        "task_id": int(task_id),
        "model": model_id,
        "model_label": MODEL_LABELS[model_id],
        "seed": seed,
        "gpu_mode": GPU_MODE,
        "world_size": 1,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "effective_global_batch": 1,
        "training_samples_per_epoch": TRAIN_TIMESTEP_SAMPLES,
        "optimizer_updates_per_epoch": TRAIN_TIMESTEP_SAMPLES,
        "output": str(run_directory(output_root, GPU_MODE, model_id, seed)),
    }


def validation_checkpoint_selection(aggregate: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    del metrics
    displacement = float(aggregate["displacement_physical"]["relative_l2_macro"])
    stress = float(aggregate["stress_physical"]["relative_l2_macro"])
    value = 0.5 * (displacement + stress)
    if not all(math.isfinite(item) for item in (displacement, stress, value)):
        raise FloatingPointError("Validation Relative-L2 selector is non-finite")
    return {
        "name": SELECTION_METRIC_NAME,
        "value": value,
        "metrics": {
            SELECTION_METRIC_NAME: value,
            "validation_displacement_physical_relative_l2_macro": displacement,
            "validation_stress_physical_relative_l2_macro": stress,
            "validation_position_physical_relative_l2_macro": float(
                aggregate["position_physical"]["relative_l2_macro"]
            ),
        },
    }


def protocol_snapshot(cfg: DictConfig, *, model: str, gpu_mode: str) -> dict[str, Any]:
    if gpu_mode != GPU_MODE:
        raise RuntimeError("Formal benchmark requires 1gpu_batch1")
    metadata = validate_dataset_metadata(cfg)
    return {
        "format": FORMAT,
        "dataset_root": str(DATASET_ROOT),
        "split_counts": {"train": 132, "validation": 30, "test": 30},
        "split_ids": metadata["split_ids"],
        "split_id_hashes": metadata["split_id_hashes"],
        "split_manifest_sha256": metadata["split_manifest_sha256"],
        "global_features_sha256": metadata["global_features_sha256"],
        "source_anomaly_audit_sha256": metadata["source_anomaly_audit_sha256"],
        "model": model,
        "model_label": MODEL_LABELS[model],
        "model_identity": model_identity(model),
        "model_config_sha256": model_config_sha256(model),
        "seed": int(cfg.training.seed),
        "nodes": 76065,
        "timesteps": 100,
        "prediction_steps": 99,
        "global_feature_order": list(GLOBAL_FEATURES),
        "normalization_hashes": verify_stats(STATS_DIR, require_read_only=True),
        "training": {
            "sample": "one trajectory at one target timestep [N,4]",
            "gpu": "1x NVIDIA H200",
            "batch_size": 1,
            "gradient_accumulation": 1,
            "precision": "BF16 autocast, no GradScaler",
            "optimizer": "Adam",
            "start_lr": 1.0e-4,
            "end_lr": 3.0e-7,
            "scheduler": "CosineAnnealingLR(T_max=25)",
            "epochs": 25,
            "weight_decay": 0.0,
            "gradient_clipping": False,
            "warmup": False,
            "ema": False,
            "warm_start": False,
        },
        "checkpoint_selector": {
            "name": SELECTION_METRIC_NAME,
            "description": CHECKPOINT_SELECTION_DESCRIPTION,
            "validation_only": True,
        },
        "paper_metrics": {
            "position": "||P_pred-P_gt||_2/||P_gt||_2",
            "displacement": "||D_pred-D_gt||_2/||D_gt||_2",
            "stress": "||S_pred-S_gt||_2/||S_gt||_2 after inverse transform",
            "per_trajectory_aggregation": "all 76065 nodes x all 99 target steps; xyz joint",
        },
        "git": git_identity(),
    }


def model_target(model_id: str) -> str:
    if model_id not in MODELS:
        raise ValueError(model_id)
    return "official_192_benchmark.benchmark.DisplacementRMSModelAdapter"


def _rewrite_batch_metadata(directory: Path) -> None:
    for path in list(directory.glob("epoch_*.json")) + [
        directory / "training_summary.json",
        directory / "TRAINING_COMPLETED.json",
    ]:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if "epoch_row" in payload:
            payload["epoch_row"]["effective_global_batch"] = 1
        if "effective_global_batch" in payload:
            payload["effective_global_batch"] = 1
        atomic_json(path, payload)
    history = directory / "metrics_history.csv"
    if history.is_file():
        with history.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if "effective_global_batch" in row:
                row["effective_global_batch"] = "1"
        if rows:
            temporary = history.with_name(f".{history.name}.tmp.{os.getpid()}")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(history)


def train_wrapper(runtime: Any, model_id: str, seed: int, gpu_mode: str, directory: Path) -> dict[str, Any] | None:
    summary = _ORIGINAL_SHARED_TRAIN(runtime, model_id, seed, gpu_mode, directory)
    if runtime.dist.rank == 0 and summary is not None:
        _rewrite_batch_metadata(directory)
        summary["effective_global_batch"] = 1
    torch_dist.barrier()
    return summary


def configure_runner() -> None:
    phone.GLOBAL_FEATURES = GLOBAL_FEATURES
    phone.load_config = load_config
    phone.exact_paths = exact_paths
    phone.validate_phone_dataset_metadata = validate_dataset_metadata
    phone.verify_stats = verify_stats
    phone.check_sample = check_sample
    formal.WORLD_SIZE = 1
    formal.REFERENCE_VTU = OFFICIAL_TOPOLOGY_REFERENCE_VTU
    formal.load_part_mapping = load_verified_part_mapping
    formal.benchmark_metrics = paper_metrics

    shared.FORMAT = FORMAT
    shared.OUTPUT_ROOT = OUTPUT_ROOT
    shared.MODELS = MODELS
    shared.MODEL_LABELS = MODEL_LABELS
    shared.SEEDS = SEEDS
    shared.GPU_MODES = GPU_MODES
    shared.MAX_EPOCHS = MAX_EPOCHS
    shared.MIN_EPOCHS = MAX_EPOCHS
    shared.PATIENCE = MAX_EPOCHS
    shared.PREDICTION_STEPS = PREDICTION_STEPS
    shared.TRAIN_TRAJECTORIES = TRAIN_TRAJECTORIES
    shared.VALIDATION_TRAJECTORIES = VALIDATION_TRAJECTORIES
    shared.TEST_TRAJECTORIES = TEST_TRAJECTORIES
    shared.TRAIN_TIMESTEP_SAMPLES = TRAIN_TIMESTEP_SAMPLES
    shared.EFFECTIVE_GLOBAL_BATCH = 1
    shared.OPTIMIZER_UPDATES_PER_EPOCH = TRAIN_TIMESTEP_SAMPLES
    shared.EARLY_STOPPING_ENABLED = False
    shared.BEST_VALIDATION_CHECKPOINT = BEST_VALIDATION_CHECKPOINT
    shared.CHECKPOINT_MONITOR = "clean_equal_field_relative_l2"
    shared.CHECKPOINT_SELECTION_DESCRIPTION = CHECKPOINT_SELECTION_DESCRIPTION
    shared.SELECTION_METRIC_NAME = SELECTION_METRIC_NAME
    shared.OPTIMIZER_KIND = "adam"
    shared.OPTIMIZER_START_LR = 1.0e-4
    shared.OPTIMIZER_WEIGHT_DECAY = 0.0
    shared.TRAINING_OBJECTIVE_NAME = "channel_mean_mse_in_train_normalized_space"
    shared.model_target = model_target
    shared.model_identity = model_identity
    shared.model_config_sha256 = model_config_sha256
    shared.resolved_config = resolved_config
    shared.instantiate_model = instantiate_model
    shared.protocol_snapshot = protocol_snapshot
    shared.one_step_backward = legacy.one_step_backward
    shared.validation_checkpoint_selection = validation_checkpoint_selection
    shared.run_directory = run_directory
    shared.train = train_wrapper


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
    parser.add_argument("--prepare-stats", action="store_true")
    parser.add_argument("--preflight-all", action="store_true")
    args = parser.parse_args(argv)
    execution = args.prepare_stats or args.preflight_all or args.print_manifest or args.audit_task is not None
    if not execution and (args.model is None or args.seed is None):
        parser.error("formal execution requires --model and --seed")
    return args


def prepare_stats() -> dict[str, Any]:
    if STATS_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite stats directory {STATS_DIR}")
    validate_dataset_metadata()
    STATS_DIR.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    phone.prepare_phone_modules()
    data_cfg = phone.copy_datapipe_cfg(
        cfg,
        data_dir=str(DATASET_ROOT / "train"),
        num_samples=TRAIN_TRAJECTORIES,
        sample_type="all_time_steps",
    )
    dataset = instantiate(
        data_cfg,
        name="official192_train_stats",
        reader=instantiate(cfg.reader),
        split="train",
        logger=phone.PythonLogger("official192_train_stats"),
    )
    if len(dataset) != TRAIN_TRAJECTORIES:
        raise RuntimeError(f"Stats dataset length {len(dataset)} != 132")
    pos_std = torch.as_tensor(dataset.node_stats["pos_std"], dtype=torch.float64)
    sum_squares = torch.zeros(3, dtype=torch.float64)
    for positions in dataset.mesh_pos_seq:
        normalized_delta = positions[1:].double() - positions[0:1].double()
        physical_delta = normalized_delta * pos_std.reshape(1, 1, 3)
        sum_squares += torch.square(physical_delta).sum(dim=(0, 1))
    element_count = TRAIN_TRAJECTORIES * PREDICTION_STEPS * 76065
    rms = torch.sqrt(sum_squares / element_count)
    if not bool(torch.isfinite(rms).all()) or bool((rms <= 0).any()):
        raise FloatingPointError(f"Invalid train displacement RMS: {rms}")
    metadata = validate_dataset_metadata(cfg)
    displacement_payload = {
        "format": "droptest_official192_train132_displacement_rms_v1",
        "formula": "sqrt(sum_{train,t=1..99,node}(P_t-P_0)^2 / element_count)",
        "subtract_mean": False,
        "train_case_count": TRAIN_TRAJECTORIES,
        "prediction_steps": PREDICTION_STEPS,
        "nodes_per_case": 76065,
        "element_count_per_axis": element_count,
        "sum_squares": sum_squares.tolist(),
        "rms": rms.tolist(),
        "train_split_id_hash": metadata["split_id_hashes"]["train"],
        "validation_or_test_used": False,
    }
    atomic_json(STATS_DIR / "displacement_rms_stats.json", displacement_payload)
    files = (
        "node_stats.json",
        "feature_stats.json",
        "dynamic_target_stats.json",
        "displacement_rms_stats.json",
    )
    hashes = {name: sha256_file(STATS_DIR / name) for name in files}
    payload = {
        "format": "droptest_official192_train132_stats_v1",
        "status": "PASS",
        "dataset_root": str(DATASET_ROOT),
        "train_case_count": TRAIN_TRAJECTORIES,
        "train_case_ids": metadata["split_ids"]["train"],
        "train_split_id_hash": metadata["split_id_hashes"]["train"],
        "validation_or_test_used": False,
        "global_preprocessing_is_fixed_not_estimated": True,
        "file_sha256": hashes,
    }
    atomic_json(STATS_DIR / "stats_manifest.json", payload)
    for name in (*files, "stats_manifest.json"):
        (STATS_DIR / name).chmod(0o440)

    before = {name: sha256_file(STATS_DIR / name) for name in files}
    for role, count in (("validation", 1), ("test", 1)):
        dataset_check = phone.build_dataset(
            cfg,
            phone.PythonLogger(f"official192_{role}_stats_lock"),
            data_dir=str(DATASET_ROOT / role),
            num_samples=count,
            sample_type="all_time_steps",
            role=role,
        )
        if len(dataset_check) != count:
            raise RuntimeError(f"{role} stats-lock dataset failed")
    after = {name: sha256_file(STATS_DIR / name) for name in files}
    if before != after:
        raise RuntimeError("Validation/test changed frozen train statistics")
    payload["validation_test_stats_recompute_check"] = "PASS"
    (STATS_DIR / "stats_manifest.json").chmod(0o640)
    atomic_json(STATS_DIR / "stats_manifest.json", payload)
    (STATS_DIR / "stats_manifest.json").chmod(0o440)
    STATS_DIR.chmod(0o550)
    return payload


def _latency_summary(values_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(values_ms)
    return {
        "iterations": len(values_ms),
        "raw_ms": values_ms,
        "mean_ms": statistics.fmean(values_ms),
        "sample_std_ms": statistics.stdev(values_ms),
        "median_ms": statistics.median(values_ms),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def _cuda_measure(callable_: Any, *, warmups: int = 20, iterations: int = 100) -> list[float]:
    for _ in range(warmups):
        callable_()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def profile_model(model_id: str, seed: int, runtime: Any, directory: Path) -> dict[str, Any]:
    model = instantiate_model(model_id, seed, runtime).to(runtime.dist.device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1.0e-4, weight_decay=0.0, fused=True)
    train_sample = runtime.train_dataset[0].to(runtime.dist.device)
    check_sample(train_sample, role="train", expected_time=0.0, cfg=runtime.cfg)

    def train_step() -> None:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        legacy.one_step_backward(model, train_sample, loss_scale=1.0)
        optimizer.step()

    for _ in range(20):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_step()
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    train_ms = _cuda_measure(train_step, warmups=0, iterations=100)

    validation = phone.build_dataset(
        runtime.cfg,
        runtime.logger,
        data_dir=str(DATASET_ROOT / "validation"),
        num_samples=1,
        sample_type="all_time_steps",
        role="validation",
    )
    inference_sample = validation[0].to(runtime.dist.device)
    model.eval()

    def inference() -> None:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(sample=inference_sample, data_stats=runtime.data_stats)
        if tuple(output.shape) != (76065, 99, 4):
            raise RuntimeError(f"Profiler rollout shape changed: {output.shape}")
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("Profiler rollout contains NaN/Inf")

    inference_ms = _cuda_measure(inference)
    result = {
        "status": "PASS",
        "model": model_id,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(runtime.dist.device),
        "precision": "BF16",
        "batch_size": 1,
        "fixed_training_sample": metadata_train_id(runtime, 0),
        "fixed_training_timestep_index": 0,
        "fixed_inference_sample": runtime.protocol["split_ids"]["validation"][0],
        "dataloader_disk_io_excluded": True,
        "warmups": 20,
        "measured_iterations": 100,
        "peak_training_step_allocated_bytes": peak_allocated,
        "peak_training_step_reserved_bytes": peak_reserved,
        "training_step_definition": "zero_grad -> forward -> loss -> backward -> optimizer.step",
        "training_step_latency": _latency_summary(train_ms),
        "inference_definition": "one complete 99-step time-conditional trajectory, eval/no_grad",
        "inference_latency": _latency_summary(inference_ms),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
    }
    atomic_json(directory / "profiling.json", result)
    del model, optimizer, train_sample, inference_sample, validation
    shared.gc_collect_cuda()
    return result


def metadata_train_id(runtime: Any, flat_index: int) -> str:
    return runtime.protocol["split_ids"]["train"][flat_index // 99]


def run_preflight_all() -> dict[str, Any]:
    configure_runner()
    dist, gpu_names = shared.initialize_distributed(GPU_MODE)
    first_cfg = resolved_config(MODELS[0], 7, GPU_MODE, PREFLIGHT_ROOT)
    runtime = shared.FormalTrajectoryRuntime(first_cfg, dist, model=MODELS[0], gpu_mode=GPU_MODE)
    reports = {}
    for model_id in MODELS:
        cfg = resolved_config(model_id, 7, GPU_MODE, PREFLIGHT_ROOT)
        runtime.cfg = cfg
        runtime.protocol = protocol_snapshot(cfg, model=model_id, gpu_mode=GPU_MODE)
        directory = run_directory(PREFLIGHT_ROOT, GPU_MODE, model_id, 7)
        if dist.rank == 0:
            if directory.exists():
                raise RuntimeError(f"Refusing to overwrite preflight {directory}")
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(
                directory / "config_snapshot.json",
                {
                    "resolved_config": OmegaConf.to_container(cfg, resolve=True),
                    "protocol": runtime.protocol,
                    "model_identity": model_identity(model_id),
                },
            )
        torch_dist.barrier()
        report = shared.run_preflight(
            runtime,
            model_id,
            7,
            GPU_MODE,
            directory,
            optimizer_updates=2,
        )
        profile = profile_model(model_id, 7, runtime, directory)
        if dist.rank == 0:
            reports[model_id] = {"preflight": report, "profiling": profile}
    if dist.rank == 0:
        summary = {
            "status": "PASS",
            "format": FORMAT,
            "gpu_names": gpu_names,
            "models": reports,
            "all_six_models_passed": set(reports) == set(MODELS),
        }
        atomic_json(PREFLIGHT_ROOT / "PREFLIGHT_ALL_PASS.json", summary)
        return summary
    return {}


def launch_manifest(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    metadata = validate_dataset_metadata()
    tasks = [task_mapping(index, output_root) for index in range(18)]
    return {
        "format": FORMAT,
        "status": "READY_AFTER_PREFLIGHT",
        "dataset": {
            "paths": exact_paths(),
            "counts": {"train": 132, "validation": 30, "test": 30},
            "timesteps": 100,
            "prediction_steps": 99,
            "nodes": 76065,
            "split_grouped_by_complete_material_combinations": True,
            **metadata,
        },
        "git": git_identity(),
        "models": {model: model_identity(model) for model in MODELS},
        "parameter_counts": {},
        "tasks": tasks,
        "array": "0-17 (no concurrency throttle)",
        "training_protocol": protocol_snapshot(
            resolved_config(MODELS[0], 7, GPU_MODE, output_root),
            model=MODELS[0],
            gpu_mode=GPU_MODE,
        )["training"],
        "normalization": normalization_spec(),
        "checkpoint_selector": CHECKPOINT_SELECTION_DESCRIPTION,
        "evaluator": protocol_snapshot(
            resolved_config(MODELS[0], 7, GPU_MODE, output_root),
            model=MODELS[0],
            gpu_mode=GPU_MODE,
        )["paper_metrics"],
        "profiler": {
            "gpu": "1x H200",
            "precision": "BF16",
            "batch_size": 1,
            "warmups": 20,
            "measured_iterations": 100,
            "training_step": "complete optimizer step",
            "inference": "complete 99-step trajectory",
            "dataloader_disk_io_excluded": True,
        },
        "slurm": {"job_ids": [], "task_status": {}},
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_runner()
    if args.prepare_stats:
        print(json.dumps(prepare_stats(), indent=2, sort_keys=True))
        return
    if args.preflight_all:
        print(json.dumps(run_preflight_all(), indent=2, sort_keys=True))
        return
    if args.print_manifest:
        print(json.dumps(launch_manifest(args.output_root), indent=2, sort_keys=True))
        return
    if args.audit_task is not None:
        row = task_mapping(args.audit_task, args.output_root)
        if args.model is not None and args.model != row["model"]:
            raise RuntimeError("Array task/model mapping mismatch")
        if args.seed is not None and args.seed != row["seed"]:
            raise RuntimeError("Array task/seed mapping mismatch")
        if Path(row["output"]).exists():
            raise RuntimeError(f"Refusing to overwrite {row['output']}")
        print(json.dumps(row, indent=2, sort_keys=True))
        return
    shared.parse_args = lambda: args
    shared.main()


if __name__ == "__main__":
    main()
