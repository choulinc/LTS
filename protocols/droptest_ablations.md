# DropTest ablation protocol registry

## Formal paper-scale ablations

The `DTOB6-0826` ablations inherit every dataset and optimization setting from
[`droptest_dtob6_0826.md`](droptest_dtob6_0826.md). The component comparison
uses the following nested configurations:

1. plain persistent-latent LTS;
2. LTS with component-aware adaptation (CA);
3. LTS with CA, Fourier input encoding, and the trajectory-level displacement
   envelope (TEnv).

These ablations use the original 10-band, width-64, end-to-end component
pooling configuration, matching the released DTOB6-0826 main row.

The formal routing study fixes the complete LTS, total depth at 12, all
parameters, and shared routing weights. The only controlled change is routing
frequency `K`. The latent-block layouts are recorded in
[`routing_k.yaml`](../configs/droptest/ablations/dtob6_0826/routing_k.yaml).

## Released routing values

The formal study releases `K = {1, 2, 3, 4, 6, 12}`; the legacy eight-layer
study releases `K = {1, 2, 4}`. Configurations, aggregates, per-seed rows and
reference runners use these same sets. Results require traceable three-seed
inputs.
