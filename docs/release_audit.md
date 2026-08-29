# Release audit

The initial source release was audited against the historical `DTOB6-0826`
implementation before publication. The paper-facing default was subsequently
updated to the validation-selected `DTOB6-0828` configuration.

## Model identity

At seed 7, the historical complete LTS constructor and the initial release
constructor with 10 Fourier bands, envelope width 64, and end-to-end component
pooling produced:

- identical ordered `state_dict` keys: 169 versus 169;
- zero missing or additional tensors;
- byte-equal values for every tensor;
- identical parameter count: 8,335,055.

The comparison used the same fixed 17-component topology and reset the PyTorch
seed immediately before each constructor. This checks not only parameter count
but module construction order, initialization, Fourier projection, component
adaptation, and trajectory-envelope state.

The current `build_lts(LTSConfig())` default is the locked paper model: 16
Fourier bands, envelope width 128, detached component-pooling source features,
one routing cycle, and 8,348,175 parameters. The historical 8,335,055-parameter
identity remains constructible through explicit `LTSConfig` fields and is
covered by an automated test.

## Functional smoke test

A reduced-width LTS with component adaptation, Fourier features, and trajectory
envelope completed a CPU forward pass with output shape `[1, 6, 4]`; every
output was finite.

## Automated checks

- unit tests cover frozen protocol fields, held-out-test isolation, paper-table
  schema, matched-profile semantics, tuned model identity, and preservation of
  the untuned identity;
- split files reproduce the frozen ordered-ID SHA-256 values;
- no committed file exceeds 1 MiB;
- no private-key or common credential pattern is present;
- no broken symlink is present.

Historical reference sources intentionally retain old absolute cluster paths
to preserve their source hashes. Active configs and scripts do not use them.
