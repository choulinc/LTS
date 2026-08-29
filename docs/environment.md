# Environment

The `DTOB6-0826` runs used Python 3.11, PyTorch with CUDA, BF16 autocast, and
one NVIDIA H200 per model/seed process. GradScaler was disabled.

The historical run recorded PhysicsNeMo repository commit
`7676ad43105d11046c38dcdb73a2f2ecde40d4ed` together with a dirty source tree.
For that reason, the commit alone is insufficient provenance. The exact
experiment-relevant overlay files and their SHA-256 hashes are included in
this repository under `src/` and `experiments/droptest/dtob6_0826/`.

Do not claim reproduction from the commit alone. A valid reproduction must
also pass the source, dataset-split, normalization, model-parameter-count, and
checkpoint-identity checks described by the frozen protocol.
