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
- Routing frequency: `K = {1, 2, 3, 4, 6, 12}` with 12 total latent blocks
  and shared route weights. Only the number and placement of re-slicing cycles
  changes.

Results are under
[`results/ablations/droptest/`](../../../results/ablations/droptest/).

## Legacy 0813 scale

The separate legacy routing study uses eight blocks and 128 slices with
K = {1, 2, 4}. It shares the official-192 split and evaluation protocol, but
its architecture differs from DTOB6-0826 and its rows remain separate.

## Result conventions

- Relative L2 columns in `*_paper.csv` are percentages.
- Relative L2 columns in `*_raw.csv` are unitless ratios.
- CSV files are LF-normalized. Original import hashes and current repository
  checksums are recorded in
  `results/ablations/droptest/source_provenance.json`.
- Dispersion is the sample standard deviation over seeds 7, 17, and 27.
- Inference latency covers one complete 99-step trajectory.
- Peak memory is peak allocated training GPU memory, not reserved memory.
- Held-out test metrics are evaluated after validation-only checkpoint
  selection.
