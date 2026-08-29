# DropTest accuracy--efficiency figure (2026-08-29)

This package reproduces the paper figure with the five frozen DTOB6-0826
external baselines and the validation-selected DTOB6-0828 LTS configuration.
It supersedes the 2026-08-28 figure, whose LTS point used the untuned
configuration and its earlier matched profile.

## Formal data used

- Models: Transolver, Transolver++, Transolver-3, GeoTransolver,
  GeoTS-FLARE, and LTS. Plain LTS and CA-only ablations are omitted.
- Seeds: 7, 17, and 27.
- Accuracy axis: per-seed
  `0.5 * (displacement physical Relative-L2 + stress physical Relative-L2)`,
  followed by the arithmetic mean across the three seeds and conversion to
  percent. This uses the same two accuracy fields as the compact paper table.
- External-baseline accuracy: frozen DTOB6-0826 held-out-test results.
- LTS accuracy: DTOB6-0828 validation-selected configuration evaluated on the
  same held-out test split. Selection did not use test results.
- Peak train allocated memory: maximum CUDA allocated memory from the matched
  H200 training-step profiler; this is neither reserved nor inference memory.
- Train-step latency: one H200, BF16, batch size one, 20 warm-up and 50 timed
  steps.

The timed region uses a preloaded CUDA sample and includes BF16 forward,
formal normalized loss, backward, and fused Adam `optimizer.step`. Dataloader,
host-to-device transfer, `zero_grad`, validation, checkpoint I/O, and first-use
initialization are outside the timed region.

## Right-hand breakdown

The right panel uses matched component profiles for Transolver and the tuned
LTS. Non-overlapping forward ranges identify slice-weight computation,
mesh-to-slice, latent processing, slice-to-mesh, mesh FFN, and component
adaptation. Backward CUDA kernels are correlated with their forward operators
through PyTorch autograd sequence numbers. Remaining instrumented kernels are
`Other instrumented`; matched wall time not explained by instrumented CUDA
kernels is `Unattributable`.

Full source paths, SHA-256 hashes, plotted values, and protocol definitions are
embedded in `plot_data.json`.

## Files

- `droptest_accuracy_efficiency_bottleneck_20260829.png`: 976 x 624 PNG
- `droptest_accuracy_efficiency_bottleneck_20260829_2x.png`: 2x PNG
- `droptest_accuracy_efficiency_bottleneck_20260829.pdf`: vector PDF
- `droptest_accuracy_efficiency_bottleneck_20260829.svg`: editable vector SVG
- `plot_data.csv`, `plot_data.json`: plotted values and provenance
- `plot_accuracy_efficiency_bottleneck.py`: deterministic renderer
- `SHA256SUMS`: package checksums

Regenerate with:

```bash
python plot_accuracy_efficiency_bottleneck.py
```
