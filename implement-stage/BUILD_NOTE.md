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
- F2: `python -m unittest tests.test_layerwise_adapter -v` -> exit 1 before test
  execution because the local environment has no `torch` installation
  (`ModuleNotFoundError: No module named 'torch'`).
- Sweep round 1: reviewer returned `verdict: gaps` with four semantic and one
  interface finding; all were added to the ledger, and SPEC was corrected.
- Sweep round 2: reviewer returned `verdict: gaps` with no undeclared rows; one
  stale wording in A-007 was corrected to reflect the existing test-list loader.

## Deferred

- F3 requires the server GPU and video files.

## Blockers

- F2 runtime checks remain blocked until run in an environment with PyTorch.
