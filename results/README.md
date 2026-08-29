# Results

`dtob6_0826_raw.csv` is the immutable aggregate emitted by the historical
experiment. Its Relative-L2 values are ratios, not percentages, and its
memory column belongs to the original aggregate path.

`dtob6_0826_paper.csv` is the six-row paper view. It removes the two internal
LTS ablation rows, renames the full `LTS + CA + Fourier + TEnv` row to `LTS`,
multiplies Relative-L2 ratios by 100, and uses the matched component profiler
for peak allocated training memory and inference latency.

Do not mix memory columns between these files without reading the profiling
definition.

Completed DropTest ablations are in [`ablations/droptest/`](ablations/droptest/):

- `dtob6_0826_components.csv`: paper-scale plain/CA/full LTS comparison;
- `dtob6_0826_routing_k_paper.csv`: percentage-unit routing-frequency view;
- `dtob6_0826_routing_k_raw.csv` and `*_per_seed_raw.csv`: ratio-unit formal
  aggregates;
- `legacy_0813_*`: the separate 128-slice factorial and routing study.

The formal and legacy namespaces are not interchangeable. See
[`protocols/droptest_ablations.md`](../protocols/droptest_ablations.md).
