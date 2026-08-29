# DTOB6-0826 experiment sources

The files in `reference/` preserve the historical runner sources used on the
cluster. They intentionally retain historical import names and paths and are
included for provenance, not as the public CLI.

The reusable implementation lives in `src/lts/`; the portable dataset check is
`scripts/validate_dataset.py`. The release factory reproduces the exact model
parameter identities:

| Variant | Parameters |
|---|---:|
| plain LTS | 8,018,444 |
| LTS + CA | 8,248,716 |
| LTS + CA + Fourier + TEnv (untuned ablation) | 8,335,055 |
| LTS + CA + Fourier + TEnv (validation-selected paper model) | 8,348,175 |

The historical experiment source was composed across two repositories and the
recorded Git tree was dirty. Therefore the reference files and source hashes,
not the historical Git commit alone, are authoritative.

The historical files in this directory preserve the untuned 0826 run. The
release factory default and paper table use the later validation-selected
configuration documented in
[`protocols/droptest_dtob6_0828_tuning.md`](../../../protocols/droptest_dtob6_0828_tuning.md).
