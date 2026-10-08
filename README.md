# LTS

This repository provides the official implementation of LTS from the following paper.

**LTS: A Lightweight Persistent-Latent Reformulation of Transolver for Transient Impact Simulation**  
Chia Feng Liao<sup>&#42;</sup>, Lin Chou<sup>&#42;</sup>, Shih-Chi Wang  
*Representations for the Physical Sciences Workshop at NeurIPS 2026*

<sup>&#42;</sup> Equal contribution.

LTS is a persistent-latent reformulation of Transolver for surrogate modeling
on large unstructured meshes. It performs mesh-to-latent routing once,
processes the resulting physics tokens through the latent stack, and decodes
to mesh nodes only at the output. The complete DropTest model additionally
uses Fourier input features, component-aware output adaptation, and a
trajectory-level displacement envelope.

This repository is the paper-facing source release. It separates reusable
model code from benchmark adapters, frozen experiment protocols, scheduler
files, and immutable paper results.

LTS is implemented on [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo),
not as a standalone reimplementation of the complete framework. The LTS
backbone is in `src/lts/models/light_transolver.py`; it reuses PhysicsNeMo's
module base classes, Transolver MLP, and physics-attention projection helpers.
The dependency and exact historical source boundary are documented in
[`docs/environment.md`](docs/environment.md).

## Repository status

The repository contains the frozen `DTOB6-0826` DropTest protocol, the
validation-only `DTOB6-0828` LTS selection, model identities, split hashes,
paper table, and the exact LTS model overlay used by the experiment. Raw
simulation data and checkpoints are not committed.
Additional benchmark adapters will be added without rewriting the frozen
`DTOB6-0826` files.

## Layout

```text
src/lts/                         reusable LTS modules
configs/droptest/dtob6_0826/    frozen paper configuration
configs/droptest/ablations/     versioned ablation configurations
experiments/droptest/dtob6_0826 benchmark adapter and provenance
experiments/droptest/ablations/ ablation registry and historical sources
protocols/                      human-readable experiment contracts
metadata/                       split IDs and checksums, never raw data
results/main_tables/            frozen CSV tables used by the paper
results/ablations/              frozen ablation aggregates and paper views
scripts/                        training, evaluation, aggregation entry points
slurm/                          cluster launch templates
tests/                          architecture and protocol invariants
```

## Installation

Create an environment containing the PhysicsNeMo version documented in
[`docs/environment.md`](docs/environment.md), then install this repository:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

The DropTest data are not redistributed. Point the runner to an independently
obtained dataset root instead of editing a checked-in YAML file:

```bash
export DROPTEST_ROOT=/path/to/droptest_official_training_ready
python scripts/validate_dataset.py \
  --config configs/droptest/dtob6_0826/common.yaml \
  --data-root "$DROPTEST_ROOT"
```

## Frozen paper result

`DTOB6-0826` denotes the official-backbone DropTest comparison trained for 25
epochs with seeds 7, 17, and 27. Five external baselines retain their
model-specific official backbone scales. The paper comparison reports the
validation-selected complete model (`LTS + CA + Fourier + TEnv`) as **LTS**.

- Current machine-readable table: [`results/main_tables/dtob6_0828_tuned_paper.csv`](results/main_tables/dtob6_0828_tuned_paper.csv)
- Frozen untuned table: [`results/main_tables/dtob6_0826_paper.csv`](results/main_tables/dtob6_0826_paper.csv)
- Raw three-seed aggregate: [`results/main_tables/dtob6_0826_raw.csv`](results/main_tables/dtob6_0826_raw.csv)
- Protocol: [`protocols/droptest_dtob6_0826.md`](protocols/droptest_dtob6_0826.md)
- LTS selection: [`protocols/droptest_dtob6_0828_tuning.md`](protocols/droptest_dtob6_0828_tuning.md)
- Accuracy--efficiency figure: [`results/figures/droptest_accuracy_efficiency_bottleneck_20260829/`](results/figures/droptest_accuracy_efficiency_bottleneck_20260829/)
- Ablations: [`experiments/droptest/ablations/README.md`](experiments/droptest/ablations/README.md)
- Backbone sources: [`docs/backbone_sources.md`](docs/backbone_sources.md)

The held-out test set is used once after validation-only checkpoint and LTS
configuration selection. No held-out trajectory is used for optimization or
model selection.

## Reproducibility policy

Every formal result is identified by:

1. an immutable protocol name;
2. model and resolved-config hashes;
3. ordered split-ID hashes and normalization-stat hashes;
4. seed and checkpoint-selection rule;
5. hardware and profiling definition.

Active configurations contain no machine-local paths, Slurm account names, or
credentials. Raw data, runtime caches, and checkpoints are excluded from Git.
The immutable historical sources under `experiments/**/reference/` retain old
cluster paths solely so their recorded source hashes remain auditable; those
paths are not used by the release API.

## License and attribution

The repository is released under Apache-2.0. Files derived from NVIDIA
PhysicsNeMo retain their original SPDX notices. Baseline implementations remain
subject to their respective upstream licenses; see
[`third_party/README.md`](third_party/README.md).
