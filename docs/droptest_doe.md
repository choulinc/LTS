# OpenRadioss DropTest192 DOE and regeneration recipe

The release now supplies the complete design manifest and the pinned deck
mutator behind DropTest. This is a simulation-input regeneration recipe.
Matching its design variables does not establish equality of historical dynamic
targets, mesh ordering, or normalization statistics used by `DTOB6-0826`.

## Inputs, variables, and deterministic ordering

The unmodified NVIDIA PhysicsNeMo generator is pinned to commit
[`9303b6580fe852595e235898688a317f9c04c7ba`](https://github.com/NVIDIA/physicsnemo/tree/9303b6580fe852595e235898688a317f9c04c7ba/examples/structural_mechanics/openradioss_dataset_gen).
Its source, license, source URLs and individual SHA-256 values are preserved in
[`third_party/physicsnemo_doe`](../third_party/physicsnemo_doe/).
The canonical case-ID manifest is
[`planned_doe.csv`](../metadata/doe/droptest192/planned_doe.csv), with the
hierarchical schema in
[`planned_summary.json`](../metadata/doe/droptest192/planned_summary.json).
Each row has `run_id`, a null `timestamp` (planned input, not execution), and
`parameters.e_scales` plus `parameters.rwall_orientation_deg`.

| Variable | Unit | Base template value | Discrete levels / constraint |
| --- | --- | --- | --- |
| E scale, material 1, polymer (PLAS_TAB) | dimensionless | E = 2000 MPa | 0.8, 1.2 |
| E scale, material 4, battery (ELAST) | dimensionless | E = 2000 MPa | 0.8, 1.2 |
| E scale, material 5, glass (ELAST) | dimensionless | E = 70000 MPa | 0.8, 1.2 |
| E scale, material 8, PCB (ELAST) | dimensionless | E = 15000 MPa | 0.8, 1.2 |
| E scale, material 9, composites (ELAST) | dimensionless | E = 2000 MPa | 0.8, 1.2 |
| Wall rx, ry, rz | degrees | original plane normal | ordered tuples below |

The six orientations, in order, are `(0,0,0)`, `(10,0,0)`, `(-10,0,0)`,
`(0,10,0)`, `(0,-10,0)`, `(0,0,10)`. They rotate the original wall normal
`M1 - M` about M, applying X then Y then Z, using the pinned generator's
`_rotate_vector_xyz`. They do not rotate the phone or change its initial velocity.
Only these six tuples are allowed; angles are not independent random variables.
Each of the five E factors varies independently in the full factorial design.
There is no random sampling distribution, randomization algorithm, or DOE seed:
this is exhaustive enumeration, with equal representation of each level and
orientation. All other physical parameters remain fixed.

Ordering is `itertools.product` over material IDs `[1,4,5,8,9]`, each in
`[0.8,1.2]`, then `product(material_combinations, orientations)`, numbered from
one as `run0001..run0192`. Thus the orientation varies fastest and material 9
varies fastest among E factors. The count is `2^5 * 6 = 192`.

## Obtain templates separately

Obtain the Cell Phone Drop example starter and engine from the
[OpenRadioss sample model](https://openradioss.atlassian.net/wiki/spaces/OPENRADIOSS/pages/11141156/Cell+Phone+Drop+Test)
or an independently retained copy. The original attachment was unavailable
when our reconstruction was prepared; this release does not promise its current
availability or redistribute the decks. Place these two files in a local
`/path/to/templates` directory, preserving their original bytes:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| Cell_Phone_Drop_0000.rad | 42378237 | `435499d744ad560a843d9a21210483bbe74c9fc6e2775da4a232ce609cf097e2` |
| Cell_Phone_Drop_0001.rad | 576 | `1a090204d2913c1759429c45a65d8b67852a1889ed8cac73d6e5eb38cb6d98ff` |

The starter was corroborated byte-for-byte between two public mirrors; the
engine was supplied independently by the authors. The complete original sample
ZIP was not recovered or authenticated. A mirror's edited engine is not an
acceptable substitute. Hash mismatches fail before case preparation.

The template unit system is Mg, mm, s; stress and Young's modulus are MPa.
The mesh has 257099 nodes, 109600 TETRA10 and 20618 BRICK elements (130218
solids); the union of TETRA10 corner and BRICK nodes has 76065 nodes. No mesh
scaling, remeshing, node renumbering or external include is used. All material
laws, densities, Poisson ratios, plasticity tables, contacts, tied interfaces,
groups, gravity cards and constraints are inherited unchanged from the hashed
starter. Initial translation `/INIVEL/TRA/1` on group 1 is
`(-3200.900476,-3511.241739,-2618.285239)` mm/s. Gravity scale factors in X/Y/Z
are `(-5788.233566,-6349.428058,-4734.6822)` mm/s², using the unchanged function
and group in the template. The planar rigid wall has origin
`(-26.944114,-49.826285,3.587149)` mm, endpoint
`(-26.35408,-49.179045,4.069787)` mm, friction 0.3 and search distance
32.578377 mm. The template hash specifies the remaining fixed inputs exactly.

## Generate metadata or a representative case

Python 3.11+ and its standard library suffice. From the repository root:

```bash
# Metadata only; no deck, solver, model dependencies or network required.
python scripts/prepare_droptest_doe.py --output /tmp/droptest-metadata

# One representative case, with independently obtained original decks.
python scripts/prepare_droptest_doe.py \
  --templates /path/to/templates --case run0001 --output /tmp/droptest-one

# All 192 decks (allow approximately 8 GiB before simulation outputs).
python scripts/prepare_droptest_doe.py \
  --templates /path/to/templates --case all --output /data/droptest192
```

Use a new output directory outside this repository. Existing output directories
are refused. All modes regenerate and verify the full manifest and global
features against committed hashes. The global-feature SHA-256 is exactly the
value frozen in `metadata/datasets/dtob6_0826.json`:
`33cd8cc4bb3f716100101a28cbbe07e51fdd3c55cb1f0b70d37dc6710c004b99`.
The actual decks are independently reread to verify every scaled E and rotated
wall normal. The engine is copied byte-for-byte. Each prepared case includes
its canonical `runNNNN.json` and a `preparation_receipt.json` containing the
actual command, wrapper/source/template/output hashes, parsed timing and
physical-input checks. Failed partial outputs are retained; no successful
receipt is written on failure. The regenerated run0001 starter was checked
against the independently recorded reconstruction hash:
`96749f1b19ca167b2cbf49f27abf7b84b37bf7bd66f354dc54d1abc07e0e43d2`.

## Solver version and commands

The reconstruction used Linux x86-64 OpenRadioss stable `latest-20260728`,
release commit `a62b27e6baa555d222a580d6218867d0be4d70b5`, SMP/OpenMP double
precision, eight threads per case. Binary hashes and provenance limits are in
[`reconstruction_provenance.json`](../metadata/doe/droptest192/reconstruction_provenance.json).
The engine's `/VERS/2022` selects the deck format, not the solver build.
No MPI or GPU is used. With the matching installation, run from the case folder:

```bash
export OPENRADIOSS_PATH=/path/to/OpenRadioss
export RAD_CFG_PATH="$OPENRADIOSS_PATH/hm_cfg_files"
export OMP_NUM_THREADS=8
export OMP_STACKSIZE=400m
export LD_LIBRARY_PATH="$OPENRADIOSS_PATH/extlib/hm_reader/linux64:$OPENRADIOSS_PATH/extlib/h3d/lib/linux64:${LD_LIBRARY_PATH:-}"
cd /tmp/droptest-one/run0001
"$OPENRADIOSS_PATH/exec/starter_linux64_gf" \
  -i Cell_Phone_Drop_0000.rad -nt 8 > starter.log 2>&1 &&
"$OPENRADIOSS_PATH/exec/engine_linux64_gf" \
  -i Cell_Phone_Drop_0001.rad > engine.log 2>&1
# Execute only after reviewing normal solver termination and absence of errors.
for anim in Cell_Phone_DropA[0-9][0-9][0-9]; do
  "$OPENRADIOSS_PATH/exec/anim_to_vtk_linux64_gf" "$anim" > "$anim.vtk" || break
done
```

Preserve commands, environment, executable hashes, return codes, logs and
listings for each attempt. Solver zero exit status alone is insufficient.
Keep `/STOP`, `/NEGVOL/STOP`, `/DT1TET10`, timestep controls and all output
requests unchanged. End time is `1e-4` s, animation starts at zero with a
requested interval `1e-6` s. Native explicit integration can cross a requested
threshold: retain actual timestamps, not invented uniform times. Preserve all
100 or 101 native frames; never interpolate, truncate or add a frame to force
cohort membership. A 101-frame result is a separate reconstruction branch, not
a 100-frame frozen benchmark case.

## ETL, failures, exclusions, and the frozen split

Use the historical PhysicsNeMo-Curator DropTest ETL at commit
[`73e64e7e5b1453c9a3a080da4db46d723817f86c`](https://github.com/NVIDIA/physicsnemo-curator/tree/73e64e7e5b1453c9a3a080da4db46d723817f86c/examples/structural_mechanics/drop_test),
including its dependencies and serialization config. From that example folder:

```bash
python run_etl.py --config-name=drop_test_etl \
  etl.source.input_dir=/data/droptest192 \
  serialization_format=vtu \
  serialization_format.sink.output_dir=/data/droptest-vtu \
  etl.processing.num_processes=4
```

Follow that pinned source's reference-frame coordinates, displacement/stress
formulas, cell connectivity and wall filtering (`wall_threshold=1e-5`). Before
ETL, independently reject missing/inconsistent/nonfinite nodal GPS stress
components: the historical source can warn and emit zeros when they are absent.
Do not substitute a different stress formula silently. Preserve retained node
IDs, their order, cell mapping, real times and source/target hashes. VTU
compression is a storage choice; the ETL itself transforms physical fields.

Reject missing/changed engine or template, implicit includes, silent E-mutation
no-ops, duplicate IDs/designs, invalid rotations, abnormal/early solver stops,
missing or noncontiguous animations, invalid/truncated VTK, missing TIME,
nonfinite coordinates/targets, or inconsistent node/field counts. Retain failed
attempts and reasons; rerun the same canonical case without changing physics.
No replacement samples, favorable-result exclusions or target-based split
selection are permitted. Record native frame counts separately from physical
simulation validity. The reconstruction's first wrapper incorrectly required
uniform native times and an endpoint snapshot; that check was revised without
changing the engine. Later native 101-frame cases were retained separately.
All 192 reconstructed cases ultimately passed the reconstruction audit; this
is not a claim that every first attempt passed or that historical anomaly-audit
identity was recovered.

The frozen ordered IDs in `metadata/splits/dtob6_0826_{train,validation,test}.txt`
are authoritative. Reuse them exactly: 132/30/30 cases, with 22/5/5 complete
six-orientation material groups. Split generation is not rerun or reseeded;
its original randomization implementation/seed is not established by the
recovered evidence. The training seeds 7/17/27 do not define the DOE or split.
The preparation tool checks ordered split hashes, disjointness, full DOE
coverage and group isolation. Assign each processed case by its canonical run
ID. New simulations must separately establish mesh/target/normalization and
anomaly-audit compatibility before they can be used as `DTOB6-0826`; run
`scripts/validate_dataset.py` against the frozen config. This release does not
regenerate the missing historical normalization or claim a drop-in replacement.

## Licensing and verification

LTS-authored scripts, artificial fixtures and metadata use this repository's
Apache-2.0 license. Vendored NVIDIA source retains its Apache-2.0 notices and
license. The original Altair sample headers state Copyright 2022 Altair
Engineering Inc., CC BY-NC 4.0. Decks, generated deck copies, simulation fields,
checkpoints and normalization arrays are excluded from this contribution; the
repository's Apache license does not relicense those materials. Obtain the
inputs separately under their own terms and assess the terms applicable to
outputs before redistribution.

```bash
python -m pytest -q tests/test_droptest_doe.py
```

These tests exercise the actual pinned mutator with an independently authored,
nonphysical tiny fixture, full-factorial bounds/order, exact metadata hashes,
split overlap/leakage, source/template tampering, silent mutation no-ops and
refusal to overwrite outputs. No solver or restricted raw data is needed.
