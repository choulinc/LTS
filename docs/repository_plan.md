# Source-release plan

The repository is organized around immutable paper protocols rather than the
chronology of cluster jobs.

## Release 0.1: DropTest

- reusable persistent-latent LTS, component adaptation, Fourier features, and
  trajectory-envelope modules;
- frozen `DTOB6-0826` configs, split IDs, hashes, paper table, and provenance;
- model-identity, protocol, and result-schema tests;
- portable data validation and table rendering tools.

## Release 0.2: steady-state CFD

Add DrivAerML Surface, DrivAerML Volume, and PhysicsNeMo Datacenter as separate
protocol namespaces. Each benchmark receives its own data adapter,
normalization contract, model-specific YAML files, evaluation metrics, and
frozen result tables. Common model code must remain in `src/lts`.

## Release 0.3: structural benchmark and artifacts

Add Bumper Beam, checkpoint download manifests, ParaView exporters, and figure
reproduction scripts. Large weights and visualization bundles will be hosted
outside Git and verified through SHA-256 manifests.

## Freeze rules

- Never edit a frozen config or result in place; create a new protocol ID.
- Never label validation-tuned experiments as the fixed `DTOB6-0826` table.
- Keep raw aggregate output separate from paper-formatted metrics.
- Record whether memory is training/inference and allocated/reserved.
- Preserve model-specific official backbone identity in every main table.
