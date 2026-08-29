# DTOB6-0826: frozen DropTest paper protocol

`DTOB6-0826` is the official-backbone DropTest main-table experiment completed
on 2026-08-26. It must not be confused with the earlier `DropTest-0813`
experiment. The paper LTS row uses the subsequent validation-only
[`DTOB6-0828`](droptest_dtob6_0828_tuning.md) configuration selection while
retaining this dataset and training protocol.

## Data

- 192 trajectories: 132 train, 30 validation, 30 held-out test.
- 76,065 nodes and 100 stored timesteps per trajectory.
- The model predicts `t001..t099`, giving 99 one-step training targets and one
  complete 99-step test trajectory per case.
- The immutable ordered IDs are stored in `metadata/splits/`.
- Split IDs, global-feature metadata, and the source anomaly audit are checked
  against SHA-256 hashes before a formal run.

The eight design variables, in order, are five material scale factors followed
by the three rigid-wall orientation values. The functional input is normalized
coordinates (3), design variables (8), and normalized target time (1). Initial
normalized coordinates (3) are also supplied as the geometry embedding.

The prediction target contains physical displacement in three axes and von
Mises stress. Displacement training uses train-only axis-wise RMS scaling with
no mean subtraction. Stress uses train-only standardization. Validation and
test outputs are inverse-transformed before physical Relative-L2 evaluation.

## Optimization and selection

Every model uses 25 epochs, Adam at `1e-4`, cosine decay to `3e-7`, BF16
autocast, batch size one, no GradScaler, no warmup, no clipping, and no warm
start. Formal seeds are 7, 17, and 27.

The saved checkpoint minimizes, on validation only,

```text
0.5 * trajectory-macro physical displacement Relative-L2
+ 0.5 * trajectory-macro physical stress Relative-L2.
```

Training still completes all 25 epochs. The selected checkpoint is frozen and
evaluated once on the common held-out test set. Test data never select an epoch
or configuration.

## Model identity

The common task adapter does not make the backbone architectures identical.
Transolver, Transolver++, Transolver-3, GeoTransolver, and GeoTS-FLARE retain
their official or author-released model-specific depth, slice count, and
checkpoint policy. LTS uses the Transolver volume scale (width 256, 12 layers,
8 heads, 256 slices) with one persistent routing cycle, 17-component output
adaptation, Fourier features, and a displacement-only trajectory envelope.
The selected paper configuration uses 16 Fourier bands, envelope width 128,
and detached component-pooling source features; all five external baselines
remain the frozen DTOB6-0826 runs.

## Metrics and efficiency

Position, displacement, and stress Relative-L2 are computed per held-out
trajectory across all 76,065 nodes and all 99 predicted steps, then macro
averaged across trajectories and summarized as mean plus sample standard
deviation across seeds. The compact main table reports displacement and stress;
position remains available in the raw result files.

Efficiency uses a single NVIDIA H200. Train-step latency includes BF16 forward,
the formal normalized loss, backward, and fused Adam update for one preloaded
sample, averaged over 50 steps after 20 warm-up steps. `Peak Train Alloc. Mem.`
means maximum allocated GPU memory within that training profile, not reserved
or inference memory. Inference latency covers one complete 99-step trajectory.
