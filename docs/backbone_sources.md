# Backbone setting sources

This document identifies where every `DTOB6-0826` backbone setting came from.
“Official backbone” means that the model-specific depth, width, attention-head
count, slice count, MLP ratio, dropout, and native checkpointing policy were
taken from the cited NVIDIA or author-released task profile. It does **not**
mean that the original dataset-specific input/output dimensions were retained:
those dimensions must be adapted to the common DropTest task.

## Source summary

| Model | Setting profile | Frozen source |
|---|---|---|
| Transolver | PhysicsNeMo unified external-aerodynamics volume | [`transolver_volume.yaml`](https://github.com/NVIDIA/physicsnemo/blob/dc244e6747d2fba0bf42656422059349f4861fbc/examples/cfd/external_aerodynamics/unified_external_aero_recipe/conf/model/transolver_volume.yaml) |
| GeoTransolver | PhysicsNeMo unified DrivAerML volume | [`geotransolver_volume.yaml`](https://github.com/NVIDIA/physicsnemo/blob/dc244e6747d2fba0bf42656422059349f4861fbc/examples/cfd/external_aerodynamics/unified_external_aero_recipe/conf/model/geotransolver_volume.yaml) |
| GeoTS-FLARE | GeoTransolver volume scale with `GALE_FA` attention | same PhysicsNeMo GeoTransolver volume profile plus the recorded attention override |
| Transolver++ | author AirCraft profile | [`main_airplane.py`](https://github.com/thuml/Transolver_plus/blob/d5a23bc734a0ebac56384cf72049a26af9673452/main_airplane.py) |
| Transolver-3 | author DrivAerML volume profile | [`main_drivaerml_volume.py`](https://github.com/thuml/Transolver-3/blob/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37/main_drivaerml_volume.py) |
| LTS | matched to the Transolver volume scale | Transolver profile above plus the released LTS implementation |

The complete machine-readable record, including source hashes, is
[`metadata/upstream_backbones.yaml`](../metadata/upstream_backbones.yaml).

## Retained backbone settings

| Model | Width | Layers | Heads | Slices | MLP ratio | Dropout |
|---|---:|---:|---:|---:|---:|---:|
| Transolver | 256 | 12 | 8 | 256 | 4 | 0.0 |
| GeoTransolver | 256 | 12 | 8 | 256 | 4 | 0.0 |
| GeoTS-FLARE | 256 | 12 | 8 | 256 | 4 | 0.0 |
| Transolver++ | 256 | 4 | 8 | 32 | 2 | 0.1 |
| Transolver-3 | 256 | 16 | 8 | 64 per head | 2 | 0.0 |
| LTS | 256 | 12 | 8 | 256 | 4 | 0.0 |

Transolver-3 retains the author implementation's training activation-
checkpoint policy. Transolver++ retains its world-size-one internal routing
behavior and uses a fixed evaluation RNG for stochastic Gumbel routing.

## DropTest task adaptation

All models receive the same DropTest information and predict the same four
fields. Consequently, native task I/O is replaced while the solver core and
the model-specific scale above are preserved:

- input: normalized coordinates, eight global design variables, and target
  time;
- output: three normalized displacement components and normalized von Mises
  stress;
- unchanged common protocol: split, normalization, optimizer, epochs, seeds,
  checkpoint selection, and held-out evaluation.

For Transolver and GeoTransolver this changes only dataset-facing dimensions.
For Transolver++ the AirCraft `space_dim`/output head is adapted. For
Transolver-3 the DrivAerML-volume `space_dim` is adapted. These are task
adapters, not claims of reproducing the source datasets' original training
recipes.

## GeoTS-FLARE naming boundary

The reported GeoTS-FLARE row instantiates PhysicsNeMo GeoTransolver with
`attention_type="GALE_FA"` at the GeoTransolver volume scale. It is not the
standalone `physicsnemo.experimental.models.flare.FLARE` model and has no
separate author-released YAML. The configuration and paper should keep this
distinction explicit.

## PhysicsNeMo code boundary

The LTS package does not copy the entire PhysicsNeMo framework. It declares
`nvidia-physicsnemo` as a runtime dependency and imports the exact symbols
listed in [`environment.md`](environment.md). The historical runner used the
PhysicsNeMo fork commit recorded there; official volume *setting profiles*
are cited separately at NVIDIA commit
`dc244e6747d2fba0bf42656422059349f4861fbc`.
