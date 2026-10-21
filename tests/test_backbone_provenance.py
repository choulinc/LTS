from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "droptest" / "dtob6_0826"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_all_paper_models_record_setting_provenance() -> None:
    for name in (
        "transolver",
        "geotransolver",
        "geots_flare",
        "transolver_plus",
        "transolver3",
        "lts",
    ):
        config = _yaml(CONFIG_ROOT / f"{name}.yaml")
        assert config["provenance"]


def test_physicsnemo_volume_sources_are_unambiguous() -> None:
    expected_commit = "dc244e6747d2fba0bf42656422059349f4861fbc"
    for name in ("transolver", "geotransolver", "geots_flare"):
        provenance = _yaml(CONFIG_ROOT / f"{name}.yaml")["provenance"]
        assert provenance["commit"] == expected_commit
        assert "unified_external_aero_recipe/conf/model/" in provenance[
            "setting_source"
        ]


def test_author_released_settings_match_frozen_configs() -> None:
    plus = _yaml(CONFIG_ROOT / "transolver_plus.yaml")["model"]
    assert (plus["hidden_dim"], plus["layers"], plus["heads"]) == (256, 4, 8)
    assert (plus["physics_slices"], plus["mlp_ratio"], plus["dropout"]) == (
        32,
        2,
        0.1,
    )
    assert plus["world_size"] == 1
    assert plus["evaluation_rng_seed"] == 20260804

    transolver3 = _yaml(CONFIG_ROOT / "transolver3.yaml")["model"]
    assert (
        transolver3["hidden_dim"],
        transolver3["layers"],
        transolver3["heads"],
    ) == (256, 16, 8)
    assert (
        transolver3["physics_slices_per_head"],
        transolver3["mlp_ratio"],
        transolver3["dropout"],
    ) == (64, 2, 0.0)


def test_upstream_manifest_records_runtime_dependency() -> None:
    manifest = _yaml(ROOT / "metadata" / "upstream_backbones.yaml")
    runtime = manifest["physicsnemo_runtime"]
    assert runtime["distribution"] == "nvidia-physicsnemo"
    assert runtime["historical_commit"] == (
        "7676ad43105d11046c38dcdb73a2f2ecde40d4ed"
    )
    assert len(runtime["imported_files"]) == 4
