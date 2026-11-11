# Adaptive Fusion Prototype

## Target

Add query-wise uncertainty-aware fusion for the existing TASK_MATCH and
MULTI_VELOCITY branches.

## Inputs

The existing base, task-match, and multi-velocity class logits for one episode.

## Outputs

The final class logits plus evaluation-only paired logits without adaptive
fusion and diagnostics for the learned gate.

## Success command

`python -m py_compile models/base/adapter.py runs/train_net_few_shot.py runs/test_net_few_shot.py tests/test_spatial_pattern.py`

## Base commit

`417ebb4f4caa61137715685161787d0ebbf25abc`

## Scope cuts

This prototype does not change either matcher, backbone training policy, data
sampling, or the trajectory geometry branch. GPU training and PyTorch tests are
deferred to the server environment.
