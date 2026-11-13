# Class-Conditional Semantic-Motion Alignment

## Motivation

The previous TASK_MATCH + MULTI_VELOCITY fusion adds two independent score
residuals to the base classifier. TASK_MATCH supplies an episode-level frame
alignment prior shared across classes, while MULTI_VELOCITY uses globally
shared velocity weights. Neither branch conditions velocity selection on the
query and candidate class.

The new matcher uses the TASK_MATCH frame prior to weight a class-specific
affinity matrix. It mixes class-specific appearance affinity with motion
affinity, where a small router predicts velocity weights from each query and
candidate class context. Its output is the joint matcher score minus the
TASK_MATCH score, so it contributes an interaction correction rather than a
third standalone classifier score.

## Configuration

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_CLASS_CONDITIONAL_MOTION.yaml
```

The experiment files spell out the full SSv2 dual-module protocol. They differ
only in the CLASS_CONDITIONAL_MOTION switch, runtime port/GPU, and output
directory. The legacy TASK_MATCH and MULTI_VELOCITY parameters are identical
in both files.

## Evaluation

The validation/test logs report the same-checkpoint paired comparison with and
without the interaction correction, including paired delta, changed, helped,
and hurt query counts. They also report the learned interaction gate and
motion mixing coefficient. Compare the separately trained configuration with
a dual-module control trained from the same code revision, data settings,
seed, and checkpoint-selection protocol. The historical 67.426/82.338 scores
are references only, not a matched control.

Treat this as a mechanism test, not a guaranteed accuracy improvement. Expand
to additional datasets and training seeds only if the matched SSv2 control
shows consistent gains in both 1-shot and 5-shot evaluation.
