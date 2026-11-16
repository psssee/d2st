# Assumption Ledger - Layerwise Spatiotemporal Adapter
<!-- ASK mode: never -->

| ID | Under-determined by the request | Chosen | Class | Source |
|----|--------------------------------|--------|-------|--------|
| A-001 | Which existing model is the control? | The existing SSv2 TASK_MATCH + MULTI_VELOCITY configuration at base commit `417ebb4`. | interface | default |
| A-002 | How should depth affect routing? | Use a configurable transition layer; initialize shallow layers spatial-heavy and deep layers temporal-heavy, then learn a two-way softmax per layer. | semantic | default |
| A-003 | What is the training/evaluation protocol? | Complete 30,000-episode training followed by 1-shot/5-shot testing, with a separately trained disabled-router control. | semantic | default |
| A-004 | How is the original path preserved? | The disabled flag bypasses router parameters and retains `(spatial + temporal) / 2` exactly. | interface | default |
| A-005 | How are depth indices and initialization strength interpreted? | Use zero-based layer indices; normalize depth by `num_layers - 1`; apply opposing spatial/temporal logits with `INIT_STRENGTH` and place the configured transition layer at a 0.5/0.5 prior. | semantic | sweep |
| A-006 | What tensors does the router mix? | Mix the residual-inclusive spatial and temporal outputs (`x + attention_delta`) before GELU and up projection. | semantic | sweep |
| A-007 | Which checkpoint is evaluated after training? | Keep the checkpoint with the highest accuracy from the loader built with `build_loader(cfg, "test")` at each `VAL_FRE_ITER` checkpoint and evaluate `checkpoint_best.pyth`, rather than the final training state. In the current Few-shot dataset path this loader uses the test-list data. | semantic | sweep |
| A-008 | Which seeds, episode count, and split define evaluation? | Use `RANDOM_SEED: 18`, `EPISODE_SEED: 2026`, 10,000 test episodes, and the dataset test-list loader; the current `TEST_SET: val` field is not consulted by this execution path. | semantic | sweep |
| A-009 | Where are routing diagnostics emitted? | Emit per-layer routing weights from training validation checkpoints; standalone test execution reports matcher meters but does not call the training fusion-state logger. | interface | sweep |

## Notes

- The transition layer is a hypothesis to test, not a claim that layer 6 is
  optimal. The first run keeps it fixed to avoid a sweep confounded with the
  architectural comparison.
