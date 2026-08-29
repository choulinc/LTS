# DTOB6-0828: validation-only LTS configuration selection

This search changes only method-specific LTS settings. The DropTest split,
normalization, task adapter, 12-layer/256-hidden/256-slice backbone, optimizer,
learning-rate schedule, epoch budget, seeds, loss, and checkpoint selector are
identical to [`DTOB6-0826`](droptest_dtob6_0826.md). Routing is fixed at
`K=1` to preserve the persistent-latent main-model definition.

## Search space

The 12 functionally distinct configurations are the Cartesian product of:

- Fourier bands: `{8, 10, 16}`;
- temporal-envelope width: `{64, 128}`;
- component-pooling source: `{end-to-end, detached}`.

Each configuration is trained from scratch for 25 epochs with seeds 7, 17,
and 27. Configurations are ranked by the mean validation checkpoint-selection
metric across the three seeds. The held-out test set is not read during this
search. Two equivalent PyTorch max-reduction APIs were also exercised as an
implementation check; because they define the same forward model and produced
identical results, they are not counted as distinct hyperparameters.

## Locked selection and test policy

The selected configuration uses 16 Fourier bands, temporal-envelope width
128, detached component-pooling source features, `amax` pooling, and `K=1`.
Its validation-best epochs are 25, 25, and 24 for seeds 7, 17, and 27. Only
after locking this selection are those three frozen checkpoints evaluated once
on the common held-out test set.

The complete validation ranking is stored in
[`results/tuning/dtob6_0828_validation_search.csv`](../results/tuning/dtob6_0828_validation_search.csv).
The locked model configuration is stored in
[`configs/droptest/dtob6_0828_tuned/lts.yaml`](../configs/droptest/dtob6_0828_tuned/lts.yaml),
and the current paper view is
[`results/main_tables/dtob6_0828_tuned_paper.csv`](../results/main_tables/dtob6_0828_tuned_paper.csv).
The held-out aggregate and per-seed records are stored in
[`results/main_tables/dtob6_0828_tuned_raw.csv`](../results/main_tables/dtob6_0828_tuned_raw.csv)
and
[`results/main_tables/dtob6_0828_tuned_per_seed_raw.csv`](../results/main_tables/dtob6_0828_tuned_per_seed_raw.csv).
