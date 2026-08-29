# Results

`dtob6_0826_raw.csv` is the immutable aggregate emitted by the historical
experiment. Its Relative-L2 values are ratios, not percentages, and its
memory column belongs to the original aggregate path.

`dtob6_0826_paper.csv` and `dtob6_0826_main_table.tex` preserve the original
untuned six-row paper view. `dtob6_0828_tuned_paper.csv` and
`dtob6_0828_tuned_main_table.tex` retain the five frozen external baselines and
replace only the LTS row with the validation-selected DTOB6-0828 result.
Relative-L2 ratios are converted to percentages. In the tuned view,
train-step latency and peak allocated training memory come from the matched
H200 profiler; inference latency covers a complete 99-step trajectory.

The original untuned aggregate remains immutable in `dtob6_0826_raw.csv` and
`dtob6_0826_per_seed_raw.csv`. The selected LTS raw records are stored
separately as `dtob6_0828_tuned_raw.csv` and
`dtob6_0828_tuned_per_seed_raw.csv`. The aggregate-path memory field in those
raw files is not the matched train-profile memory used by the paper table.

Do not mix memory columns between these files without reading the profiling
definition.

The current accuracy--efficiency figure and its complete provenance are in
[`figures/droptest_accuracy_efficiency_bottleneck_20260829/`](figures/droptest_accuracy_efficiency_bottleneck_20260829/).

Completed DropTest ablations are in [`ablations/droptest/`](ablations/droptest/):

- `dtob6_0826_components.csv`: paper-scale plain/CA/full LTS comparison;
- `dtob6_0826_routing_k_paper.csv`: percentage-unit routing-frequency view;
- `dtob6_0826_routing_k_raw.csv` and `*_per_seed_raw.csv`: ratio-unit formal
  aggregates;
- `legacy_0813_*`: the separate 128-slice factorial and routing study.

The formal and legacy namespaces are not interchangeable. See
[`protocols/droptest_ablations.md`](../protocols/droptest_ablations.md).
