# Motion-Guided Action Token Selection

## Target

Add a motion-guided soft patch-token selector to the existing ViT D2ST visual
backbone. The change must preserve the original CLS feature when disabled and
must not add a standalone classification or matcher logit branch.

## Inputs and Outputs

- Input: the existing CLIP token sequence entering each `ViT_D2ST_Adapter`.
- Configuration: `MOTION_TOKEN_SELECTOR` in the full SSv2 experiment files.
- Output: the same `(B,T,D)` frame feature shape as the existing backbone;
  training logs report selector gate and entropy diagnostics.

## Success Check

`python -m py_compile models/base/adapter.py runs/train_net_few_shot.py runs/test_net_few_shot.py tests/test_motion_token_selector.py`

## Base Commit

`417ebb4`

## Scope Cuts

- No backbone replacement or new video dataset.
- No new matcher/logit branch; TASK_MATCH and MULTI_VELOCITY remain unchanged.
- GPU training is deferred to the server.
