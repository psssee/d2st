# Motion-Guided Action Token Selection

## Method

The visual backbone computes temporal changes for every spatial patch token,
learns a soft action-relevance distribution, and pools the selected patch
content into one action token per frame. A zero-initialized residual projection
adds this token to the existing CLS frame feature. `TASK_MATCH` and
`MULTI_VELOCITY` then operate on the enhanced frame sequence without any new
classifier or matcher branch.

The previous layerwise spatial/temporal router is explicitly disabled in both
configs, so this experiment tests one visual representation change on top of
the existing two-module baseline.

## Run order

Run the matched control first:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_MOTION_TOKEN_CONTROL.yaml
```

Then run the selector:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_MOTION_TOKEN.yaml
```

Both configurations train for 30,000 episodes and evaluate 1-shot/5-shot on
the same SSv2 protocol. Training logs report the residual gate, motion bias,
selection entropy, and mean patch motion energy.

## Interpretation

The selector is useful only if it improves over the matched control while its
selection entropy remains finite and does not collapse immediately to one
patch. A null or negative result is evidence that patch-level motion weighting
does not complement the current D2ST frame matchers and should not be rescued
by stacking another matcher on top.
