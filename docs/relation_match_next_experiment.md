# Relation Match: SSv2 Follow-up (2026-10-19)

## Observed Results

The completed run is the last training session in
`output/SSv2_Full_RELATION_MATCH/train.log`, starting at 10/16 16:15:23.
The file also contains three earlier starts; do not combine their curves.
Testing loaded `checkpoints/checkpoint_best.pyth` from this experiment.

| Measurement | 1-shot (%) | 5-shot (%) |
| --- | ---: | ---: |
| Historical D2ST baseline supplied by the user | 65.190 | 80.650 |
| D2ST + RELATION_MATCH | 66.260 | 81.452 |
| Same relation-trained checkpoint, relation disabled at inference | 66.268 | 81.438 |
| Historical TASK_MATCH + MULTI_VELOCITY | 67.426 | 82.338 |

The apparent gain over the user-supplied baseline is +1.070/+0.802
percentage points. This is not yet a controlled estimate of module benefit:
historical baseline logs contain multiple runs and changed data-loading code.
The same-checkpoint inference contribution is -0.008/+0.014 points.
Across 50,000 queries, the relation branch helped/hurt 93/97 (1-shot) and
52/45 (5-shot) predictions. This does not show a substantial direct gain.
It does not rule out a training-time effect on D2ST features either.

Validation at 10k/20k/30k episodes was 59.910/65.406/66.344.
The relation gate was 0.034744/0.028485/0.027307; its initialization was
sigmoid(-4), approximately 0.017986. There is no basis here for increasing
the gate or extending training beyond 30k episodes.

## Next Comparison

Two full configurations preserve the common dual-module settings:

| Configuration suffix | Modules | GPU | TCP port |
| --- | --- | --- | --- |
| TASK_MATCH_MULTI_VELOCITY_RELATION_CONTROL | TASK_MATCH + MULTI_VELOCITY | 0 | 1114 |
| TASK_MATCH_MULTI_VELOCITY_RELATION_MATCH | TASK_MATCH + MULTI_VELOCITY + RELATION_MATCH | 1 | 1115 |

They differ only in RELATION_MATCH.ENABLE and the GPU/port/output directory.
Both start from pretrained CLIP with newly trained adapters, not the old
relation-only or dual-module checkpoint. Each runs 30k training episodes,
validates every 10k episodes, and evaluates 10k episodes for each shot.
The previous experiment configurations and outputs remain unchanged.

The worker count (8), pin memory (true), and GPU augmentation (false) match
the completed server log. A previous explanation incorrectly inferred GPU
augmentation from config/run.yaml alone: config/OTAM_base.yaml overrides
USE_GPU to false, which is also confirmed by the run's logged configuration.

Use the same server code revision, video files, annotations, dependencies,
worker settings, and checkpoint-selection rule for both runs. The finished
relation run still contains object-type errors and episode retries after
20k episodes. Treat it as exploratory evidence until these errors are
resolved on the server; do not silently treat such errors as corrupt videos.
If loading settings must change, change both configurations together.
The repeated training seeds do not guarantee identical training episodes;
training randomness and GPU differences require replication.

## Decision Rule

1. First compare the triple-module result with its new dual-module control.
   The historical 67.426/82.338 result is a reference, not a replacement for
   this control. Report same-checkpoint paired deltas as a separate measure.
2. If gains are consistent, repeat both configurations with at least three
   training seeds, then add TASK_MATCH + RELATION_MATCH and MULTI_VELOCITY +
   RELATION_MATCH to identify which interaction contributes the gain.
3. If the paired delta stays near zero but independent retraining reliably
   improves accuracy, investigate training-time effects rather than claiming
   the relation distance substantially improves inference matching.
4. If the triple model does not improve on the control, defer further module
   combinations; inspect the two-way ablations before adjusting parameters.

The current meter's changed_predictions counts correctness changes, not all
class-label changes; helped/hurt and the accuracy deltas are usable. It does
not log episode-level confidence intervals. Do not infer significance from
the historical count alone. The current training code also uses the `test`
loader for checkpoint validation; confirm an independent validation split
before making a final paper-level test claim or tuning additional settings.

Only static inspection was performed locally, as requested. No training,
inference, dependency changes, or local Python tests were run for this update.
