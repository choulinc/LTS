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
