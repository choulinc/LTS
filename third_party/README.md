# Third-party implementations

The paper compares LTS with Transolver, Transolver++, Transolver-3,
GeoTransolver, and GeoTS-FLARE. This repository does not relicense those
projects.

The frozen configuration records the upstream identity and model-specific
backbone scale used for every baseline. Baseline source should be obtained from
its official or author-released repository and used under its original license.
Files copied or adapted from NVIDIA PhysicsNeMo retain Apache-2.0 SPDX headers.

Exact repositories, commits, setting-source files, task-adapter changes, and
SHA-256 hashes are recorded in
[`docs/backbone_sources.md`](../docs/backbone_sources.md) and
[`metadata/upstream_backbones.yaml`](../metadata/upstream_backbones.yaml).
Those files supersede ambiguous labels such as `transolver_volume.yaml` without
a repository-relative path.
