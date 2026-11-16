# Layerwise Spatiotemporal Adapter

## Target

Add a depth-aware spatial/temporal routing mechanism inside the existing ViT
D2ST adapter. The change must preserve the original equal-weight spatial and
temporal average when disabled and must not add a standalone classification
logit branch.

## Inputs and Outputs

- Input: the existing CLIP token sequence entering each `ViT_D2ST_Adapter`.
- Configuration: `LAYERWISE_ST_ADAPTER` in the full SSv2 experiment files.
- Output: the same token shape as the existing adapter; training validation
  logs report per-layer spatial/temporal routing weights.

## Success Check

`python -m py_compile models/base/adapter.py runs/train_net_few_shot.py runs/test_net_few_shot.py tests/test_layerwise_adapter.py`

## Base Commit

`417ebb4`

## Scope Cuts

- No backbone replacement or new video dataset.
- No new matcher/logit branch; TASK_MATCH and MULTI_VELOCITY remain unchanged.
- GPU training is deferred to the server.
