# Query-Adaptive Fusion Experiment

This branch tests whether per-query calibrated fusion can exploit the existing
TASK_MATCH and MULTI_VELOCITY branches more effectively than their learned
global scalar residual gates. It does not enable or depend on trajectory
geometry.

The gate uses base-logit margin and entropy plus each branch's calibrated
margin, normalized logit disagreement, and entropy. Branch logits are centered
and rescaled to the base-logit scale per query before residual fusion. A
query-wise multiplier scales each learned global branch gate; it is initialized
to one, so the adaptive and static fusion outputs are identical before training.
The evaluation compares adaptive fusion against both calibrated static gates
and the legacy raw-logit global gates on the same episodes, with episode-clustered
confidence intervals for each paired delta.

Run from the repository root on the SSv2 server:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_ADAPTIVE_FUSION.yaml
```

The base configuration inherits the existing TASK_MATCH + MULTI_VELOCITY
settings and episode protocol. It starts a fresh training run, uses a dedicated
output directory, and enables only `ADAPTIVE_FUSION` as the experimental
change. Compare 1-shot and 5-shot with the existing two-module run; do not
interpret a positive paired delta alone as a generalization claim.
