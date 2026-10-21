# Environment

The `DTOB6-0826` runs used Python 3.11, PyTorch with CUDA, BF16 autocast, and
one NVIDIA H200 per model/seed process. GradScaler was disabled.

## PhysicsNeMo dependency

The Python distribution is `nvidia-physicsnemo`; its import package is
`physicsnemo`. LTS directly imports:

- `physicsnemo.core.meta.ModelMetaData`;
- `physicsnemo.core.module.Module`;
- `physicsnemo.models.transolver.transolver._TransolverMlp`;
- `_project_input` and `_compute_slices_from_projections` from
  `physicsnemo.nn.module.physics_attention`.

PhysicsNeMo is therefore a required runtime dependency, not bundled source.
For ordinary development, install this repository with `pip install -e .`.
For source-exact experiment reconstruction, install the recorded PhysicsNeMo
checkout first:

```bash
git clone https://github.com/Russell512/physicsnemo-persistent-routing.git physicsnemo
git -C physicsnemo checkout 7676ad43105d11046c38dcdb73a2f2ecde40d4ed
python -m pip install -e ./physicsnemo
python -m pip install -e .
```

The historical run recorded that fork commit together with a dirty working
tree. The four PhysicsNeMo files imported by the released LTS model were clean
at the recorded commit; their hashes are frozen in
[`metadata/upstream_backbones.yaml`](../metadata/upstream_backbones.yaml). The
experiment-specific overlay is released under `src/` and
`experiments/droptest/dtob6_0826/reference/source_overlay/`.

The official volume backbone settings are a separate provenance layer. They
come from NVIDIA PhysicsNeMo commit
`dc244e6747d2fba0bf42656422059349f4861fbc`; see
[`backbone_sources.md`](backbone_sources.md) for exact files and task-adapter
changes.

Do not claim reproduction from the commit alone. A valid reproduction must
also pass the source, dataset-split, normalization, model-parameter-count, and
checkpoint-identity checks described by the frozen protocol.
