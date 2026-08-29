# DropTest ablations

This directory is the registry for completed, provenance-backed DropTest
ablations. Results from different architecture scales are deliberately kept in
separate namespaces.

## DTOB6-0826 official-backbone scale

The paper-scale LTS uses 12 latent blocks, width 256, eight heads, and 256
physics slices. It shares the data, normalization, optimization, validation
selector, seeds, held-out test, and H200 measurement definitions in
[`protocols/droptest_dtob6_0826.md`](../../../protocols/droptest_dtob6_0826.md).

- Component stack: plain LTS, LTS + CA, and complete LTS.
- Routing frequency: `K = {1, 2, 3, 4, 6, 8, 12}` with 12 total latent blocks
  and shared route weights. Only the number and placement of re-slicing cycles
  changes.
- `K=8` is retained in the machine-readable result even though seed 27 became
  unstable. It must not be silently discarded when reporting the study.

Results are under
[`results/ablations/droptest/`](../../../results/ablations/droptest/).

## Legacy 0813 scale

The complete `2^3` CA/Fourier/TEnv factorial used six blocks and 128 slices.
Its separate routing study used eight blocks and 128 slices. These experiments
use the same official-192 split and evaluation protocol, but they do **not**
use the DTOB6-0826 backbone scale. They are useful as matched internal
ablations and must not be presented as rows from the DTOB6-0826 main table.

## Result conventions

- Relative L2 columns in `*_paper.csv` are percentages.
- Relative L2 columns in `*_raw.csv` are unitless ratios.
- CSV files are LF-normalized on import; original and repository checksums are
  both recorded in `results/ablations/droptest/source_provenance.json`.
- Dispersion is the sample standard deviation over seeds 7, 17, and 27.
- Inference latency covers one complete 99-step trajectory.
- Peak memory is peak allocated training GPU memory, not reserved memory.
- Held-out test metrics are evaluated after validation-only checkpoint
  selection.

The unfinished 0828 tuning search is excluded: it has validation selections
but no frozen three-seed held-out result.
