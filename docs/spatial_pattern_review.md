# Spatial Pattern Review: 2026-09-28

## Decision

Keep the spatial branch as a candidate, but do not claim a successful combined
module yet. Fix the frozen gate before tuning learning rates, grid size, or
training duration. This is an implementation correction, not a new architecture.

## Observed results

SSv2-Full accuracy (%), as recorded in the local logs:

| Run | 1-shot | 5-shot |
| --- | ---: | ---: |
| Historical D2ST baseline (SSv2_Full_v2) | 65.190 | 80.650 |
| D2ST + SPATIAL_PATTERN | 66.460 | 81.316 |
| D2ST + TASK_MATCH + MULTI_VELOCITY | 67.426 | 82.338 |
| D2ST + TASK_MATCH + MULTI_VELOCITY + SPATIAL_PATTERN | 67.278 | 82.056 |

Source locations:

- Historical baseline: `output/SSv2_Full_v2/train.log:753` (1-shot validation),
  `output/SSv2_Full_v2/test.log:901` (5-shot).
- Spatial: `output/SSv2_Full_SPATIAL_PATTERN/test.log:404` and `:873`.
- Two modules: `output/SSv2_Full_TASK_MATCH_MULTI_VELOCITY/test.log:418` and `:901`.
- Three modules: `output/SSv2_Full_TASK_MATCH_MULTI_VELOCITY_SPATIAL_PATTERN/test.log:425`
  and `:915`.

The spatial-only run exceeds the historical scores by 1.270 / 0.666 percentage
points. The combined run trails the existing two-module run by 0.148 / 0.282
points. These are observed score differences, not statistical significance.
The historical baseline log does not record the newer `TEST.EPISODE_SEED: 2026`,
and its 1-shot value comes from training-time validation. It is not a matched
episode comparison with the new spatial run.

At 10k / 20k / 30k episodes, validation accuracy is:

- Spatial-only: 59.672 / 65.544 / 66.418.
- Combined spatial: 60.514 / 66.440 / 67.450.

The recorded validation checkpoints are still improving. A train-validation gap
alone does not establish that longer training caused the combination's decline.
Do not shorten or extend the existing 30k schedule on that basis.

## Confirmed implementation error

`runs/train_net_few_shot.py` freezes all parameters outside a name whitelist.
The spatial matcher and `spatial_pattern_alpha` were absent from that whitelist.
The gate was therefore frozen even though it is declared as `nn.Parameter` and
was included in the optimizer during construction.

Both new training logs show `spatial_pattern=0.017986` at every evaluation,
which equals sigmoid(-4). The matcher itself has no learned parameters and
`DETACH_INPUT: true` blocks direct spatial gradients into the features.
These runs used a fixed spatial metric with a fixed fusion coefficient, not
the intended learned spatial coefficient.

The branch was not completely inactive: its fixed logits changed predictions
and the cross-entropy gradient received by the other branches. Conversely,
unfreezing the scalar alone is not evidence that the combined score will improve.

## Changes

- Add the spatial names to the training whitelist.
- Check gate trainability and optimizer membership before training.
- Check that the gate receives a finite gradient on the first backward pass.
- At validation and test time, report the same checkpoint with and without the
  spatial residual, without a second encoder pass.
- Report spatial-only accuracy, helped/hurt/changed predictions, and the paired
  accuracy difference with an approximate episode-level 95% interval.
- Preserve the original configs and result directories. `GATE_FIX` configs keep
  all existing hyperparameters and use fresh output directories. `AUDIT` configs
  load the existing best checkpoint and run evaluation only.

The paired interval captures sampled-episode variation for one checkpoint, not
variation across training seeds. Turning a branch off at inference also does not
replace a separately trained ablation: it cannot undo its effect during training.
The old `logit_delta` was an unweighted difference from only the last episode;
class-independent logit shifts can make it large without changing predictions.

## Run order

Run from the project root on the training server, where the saved checkpoints
and SSv2 videos exist. Start with the combined checkpoint audit:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_SPATIAL_PATTERN_AUDIT.yaml
```

For the spatial-only checkpoint:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_SPATIAL_PATTERN_AUDIT.yaml
```

The line `Spatial pattern paired evaluation` reports `full_acc`,
`without_spatial_acc`, `spatial_only_acc`, `paired_delta_pp`,
`paired_ci95_halfwidth_pp`, `helped_predictions`, and `hurt_predictions`.

For the controlled training correction, run the combined configuration first:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_SPATIAL_PATTERN_GATE_FIX.yaml
```

The standalone counterpart is
`config/ssv2_full/ViT_SSv2_full_SPATIAL_PATTERN_GATE_FIX.yaml`.
Both keep `GRID_SIZE: 2`, `DISTANCE_SCALE: 1.0`, `ALPHA_INIT: -4.0`,
`DETACH_INPUT: true`, the existing optimizer, and the 10k evaluation interval.
Do not automatically resume the old frozen-gate run as the controlled comparison.

Use an independent validation split for architecture and hyperparameter
selection, then evaluate held-out test classes. The existing training pipeline
loads `test_few_shot.txt` for checkpoint selection; the historical final scores
should not be treated as an untouched test-set estimate. The audit configs
preserve this pipeline for diagnosis; they do not fix that experimental protocol.

## Whether to keep the module

1. Confirm the corrected gate is trainable and updates. A tiny gate alone does
   not prove redundancy: score scale and classification margins also matter.
2. Inspect the paired effect and separately trained ablation on validation tasks.
   If the gate suppresses the branch and there is no repeatable validation gain,
   remove it from the combined model rather than forcing a larger coefficient.
3. If gains appear, verify across training seeds with matched evaluation episodes
   before expanding to more combinations. One favorable run is insufficient.

The fixed 2x2 grid pools each 7x7 patch region into one vector; it may dilute
small action-relevant objects. Its shot-wise regional averaging also assumes
rough correspondence before matching. These are structural hypotheses, not
causes demonstrated by the current logs. Do not change pooling, detach behavior,
and gate initialization simultaneously with the training correction.

DiST (Qu et al., TPAMI 2026, arXiv:2602.18043) motivates spatial prototypes but
uses knowledge-guided adaptive patch aggregation. This fixed-grid matcher does
not reproduce that mechanism, and DiST's reported gains do not validate it.
Reference: https://arxiv.org/abs/2602.18043

## Local verification

CPU regression tests cover regional pooling, matching axes and scale, 1/5-shot
handling, frozen-gate prevention, optimizer updates, detached/attached feature
gradients, combined TASK_MATCH/MULTI_VELOCITY fusion, paired statistics, and
configuration parity. Tests isolate production definitions with supplied encoder
features; they do not download CLIP or exercise video decoding/GPU training.

```sh
python -m unittest discover -s tests -v
```
