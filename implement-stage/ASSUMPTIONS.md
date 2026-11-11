# Assumption Ledger - Adaptive Fusion
<!-- ASK mode: never -->

| ID | Under-determined by the request | Chosen | Class | Source |
|----|--------------------------------|--------|-------|--------|
| A-001 | Which existing branches are fused? | Base, TASK_MATCH, and MULTI_VELOCITY only; trajectory geometry remains disabled for this trial. | interface | default |
| A-002 | What signal should control a query gate? | Per-query base margin, entropy, branch disagreement, and branch margin features. | semantic | default |
| A-003 | How should the new fusion be trained? | Train only the small fusion head and existing optional matcher parameters under the repo freeze policy. | interface | default |
| A-004 | How is the contribution measured? | On identical episodes, compare adaptive fusion with calibrated static global gates and with the legacy raw-logit global gates; keep 1-shot/5-shot and the existing SSv2 protocol. | semantic | default |

## Notes

- A-002 chooses a compact gate so the experiment tests complementarity rather
  than adding another large feature extractor.
- A-004 preserves paired episode evaluation and separates query-wise gating
  effects from per-query logit calibration effects.
