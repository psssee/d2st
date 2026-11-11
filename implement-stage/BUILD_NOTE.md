# Build Note - Adaptive Fusion

| Rung | Feature | Acceptance check (ONE command) | Tier | Status |
|------|---------|-------------------------------|------|--------|
| F0 | branch and implementation ledger | `git branch --show-current` | MUST | ✅ |
| F1 | query-wise adaptive fusion module and config | `python -m py_compile models/base/adapter.py runs/train_net_few_shot.py runs/test_net_few_shot.py` | MUST | ✅ |
| F2 | CPU shape/gradient and paired-output regression tests | `python -m unittest tests.test_spatial_pattern -v` | MUST | 🚧 |
| F3 | server SSv2 training/test run | `python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_ADAPTIVE_FUSION.yaml` | SHOULD | ⬜ |

## Run record

- F0: created `research/d2st-adaptive-fusion` from `417ebb4`.
- F1: Python syntax compilation and YAML parse passed; the config inherits the
  two-module SSv2 protocol and enables adaptive fusion only.
- F2: blocked locally because this Python environment has no PyTorch (`No module
  named 'torch'`); test cases are added for the server/runtime environment.

## Deferred

- F3 requires the server GPU environment and is not run locally.

## Blockers

- Local environment does not provide PyTorch, so F2 may be unavailable locally.
