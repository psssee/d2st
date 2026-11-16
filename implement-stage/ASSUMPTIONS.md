# Assumption Ledger - Layerwise Spatiotemporal Adapter
<!-- ASK mode: never -->

| ID | Under-determined by the request | Chosen | Class | Source |
|----|--------------------------------|--------|-------|--------|
| A-001 | Which existing model is the control? | The existing SSv2 TASK_MATCH + MULTI_VELOCITY configuration at base commit `417ebb4`. | interface | default |
| A-002 | How should depth affect routing? | Use a configurable transition layer; initialize shallow layers spatial-heavy and deep layers temporal-heavy, then learn a two-way softmax per layer. | semantic | default |
| A-003 | What is the training/evaluation protocol? | Complete 30,000-episode training followed by 1-shot/5-shot testing, with a separately trained disabled-router control. | semantic | default |
| A-004 | How is the original path preserved? | The disabled flag bypasses router parameters and retains `(spatial + temporal) / 2` exactly. | interface | default |

## Notes

- The transition layer is a hypothesis to test, not a claim that layer 6 is
  optimal. The first run keeps it fixed to avoid a sweep confounded with the
  architectural comparison.
