# Build Note - Motion-Guided Action Token Selection

| Rung | Feature | Acceptance check (ONE command) | Tier | Status |
|------|---------|-------------------------------|------|--------|
| F0 | branch/spec/assumption ledger | `git branch --show-current` | MUST | ✅ |
| F1 | motion-guided soft patch-token selector | `python -m py_compile models/base/adapter.py runs/train_net_few_shot.py runs/test_net_few_shot.py tests/test_motion_token_selector.py` | MUST | ✅ |
| F2 | CPU shape, baseline-preservation, gradient, and config-pair checks | `python -m unittest tests.test_motion_token_selector -v` | MUST | 🚧 |
| F3 | SSv2 matched-control and selector training/test on server | `python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_MOTION_TOKEN.yaml` | SHOULD | ⬜ |

## Run record

- F0: created `research/d2st-motion-guided-token-selection` from the latest
  remote layerwise experiment branch.
- F1: Python static compilation -> exit 0.
- F1: YAML pair check -> exit 0; TASK_MATCH and MULTI_VELOCITY are enabled in
  both files, while the old layerwise router is disabled in both.
- F1: `git diff --check` -> exit 0.
- F2: not executable locally because the environment has no PyTorch; run the
  test on the training server.
- Post-run audit: remote selector logs showed gate=`0.017986`, motion bias=`0`,
  and entropy=`5.278115` at all checkpoints. The module had been omitted from
  the training script's trainable-parameter allowlist and was frozen. The
  allowlist and a startup optimizer/trainability check are now fixed.

## Deferred

- F3 requires the server GPU and video files.

## Blockers

- Runtime checks remain blocked locally until run in an environment with PyTorch.
