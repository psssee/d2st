# Trajectory Geometry Experiment

This branch implements a D2ST-native trajectory-geometry prototype inspired by
GeoTRAM (arXiv:2610.38482). It is not a full GeoTRAM reproduction: the branch
does not add DINOv2 seed sampling, CoTracker, monocular depth, or an external
trajectory cache.

The matcher pools the existing CLIP patch grid to a sparse point grid, performs
soft feature-based correspondence between consecutive frames, and represents
each step with feature change, displacement, and local relative displacement.
The resulting trajectory descriptors are matched with bidirectional Bi-MHM and
fused through an independently learned residual gate.

## Controlled runs

Run the trajectory-only control first:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TRAJECTORY_GEOMETRY.yaml
```

Run the matched two-module control in a separate job:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_TRAJECTORY_CONTROL.yaml
```

Then run the proposed combination:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_TRAJECTORY_GEOMETRY.yaml
```

The matched control and combination use the same SSv2 sampling, episode seed,
schedule, and output protocol. They differ only in whether
`TRAJECTORY_GEOMETRY` is enabled; each has a separate output directory.

The validation/test log reports the trajectory gate, correspondence confidence,
and paired `full_acc` versus `without_trajectory_geometry_acc` on identical
episodes. A positive paired delta is necessary but not sufficient; compare the
independently trained trajectory-only control and the existing dual-module
control before claiming a gain.
