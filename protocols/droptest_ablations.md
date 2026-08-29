# DropTest ablation protocol registry

## Formal paper-scale ablations

The `DTOB6-0826` ablations inherit every dataset and optimization setting from
[`droptest_dtob6_0826.md`](droptest_dtob6_0826.md). The component comparison
uses the following nested configurations:

1. plain persistent-latent LTS;
2. LTS with component-aware adaptation (CA);
3. LTS with CA, Fourier input encoding, and the trajectory-level displacement
   envelope (TEnv).

This is a nested component study, not a complete factorial: it does not by
itself identify isolated Fourier and TEnv main effects.

These ablations use the original 10-band, width-64, end-to-end component
pooling configuration. They are retained as controlled untuned ablations; the
paper main row is the later validation-selected 16-band, width-128, detached
configuration documented in
[`droptest_dtob6_0828_tuning.md`](droptest_dtob6_0828_tuning.md).

The formal routing study fixes the complete LTS, total depth at 12, all
parameters, and shared routing weights. The only controlled change is routing
frequency `K`. The latent-block layouts are recorded in
[`routing_k.yaml`](../configs/droptest/ablations/dtob6_0826/routing_k.yaml).

## Legacy complete factorial

The legacy 0813 experiment evaluates all eight combinations of CA, Fourier,
and TEnv at six blocks and 128 slices. Its raw result is released because it is
the only complete `2^3` factorial, but its scale differs from the formal
12-block/256-slice model. Any table or claim must identify it as the 0813
matched ablation and must not combine its rows with DTOB6-0826.

## Exclusions

- The incomplete `droptest_official_backbones_k1248_20260827` directory is not
  authoritative; the completed `k12346812` run supersedes it.
- Validation-only candidates from the 0828 tuning search are not reported as
  held-out results; only the locked winner has a held-out evaluation.
- Results lacking a frozen aggregate and traceable three-seed source are not
  included.
