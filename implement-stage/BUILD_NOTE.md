# Build Note - Layerwise Spatiotemporal Adapter

| Rung | Feature | Acceptance check (ONE command) | Tier | Status |
|------|---------|-------------------------------|------|--------|
| F0 | branch/spec/assumption ledger | `git branch --show-current` | MUST | ✅ |
| F1 | depth-aware spatial/temporal adapter routing | `python -m py_compile models/base/adapter.py runs/train_net_few_shot.py runs/test_net_few_shot.py tests/test_layerwise_adapter.py` | MUST | ✅ |
| F2 | CPU shape, initialization, and router-gradient regression checks | `python -m unittest tests.test_layerwise_adapter -v` | MUST | 🚧 |
| F3 | SSv2 training and test on server | `python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_LAYERWISE_ST_ADAPTER.yaml` | SHOULD | ⬜ |

## Run record

- F0: created `research/d2st-layerwise-spatiotemporal-adapter` from `417ebb4`.
- F1: `python -m py_compile models/base/adapter.py runs/train_net_few_shot.py runs/test_net_few_shot.py tests/test_layerwise_adapter.py` -> exit 0.
- F1: YAML pair check -> exit 0; routed/control configs match on all training and
  evaluation fields except the intended router/runtime fields, and both use
  `TRAIN.ENABLE: true` with 30,000 training tasks and `[1, 5]` test shots.
- F1: `git diff --check` -> exit 0.
- F2: `python -m unittest tests.test_layerwise_adapter -v` -> not run; local
  environment has no `torch` installation.

## Deferred

- F3 requires the server GPU and video files.

## Blockers

- Local environment may not contain PyTorch; runtime tests will be recorded as
  unavailable if the dependency is missing.
