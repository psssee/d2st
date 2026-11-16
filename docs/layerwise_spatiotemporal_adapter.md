# Layerwise Spatiotemporal Adapter

## Motivation

The existing ViT D2ST adapter applies the same spatial deformable attention and
temporal deformable attention at every Transformer depth, then averages the two
paths equally. This experiment tests a visual-backbone change rather than a
third classifier branch: each depth learns a two-way spatial/temporal mixture.

The initialization is spatial-heavy in shallow layers and temporal-heavy in
deep layers, with a transition at layer 6 for the 12-layer ViT-B/16. The two
paths and all downstream TASK_MATCH/MULTI_VELOCITY code remain unchanged.

## Run order

Run the matched control first:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_LAYERWISE_ST_ADAPTER_CONTROL.yaml
```

Then run the routed model:

```sh
python runs/run.py --cfg config/ssv2_full/ViT_SSv2_full_LAYERWISE_ST_ADAPTER.yaml
```

Both files are complete configurations and train from scratch for 30,000
episodes before 1-shot/5-shot evaluation. They differ only in the router
switch, runtime port/GPU, and output directory.

The training log reports per-layer spatial/temporal weights. A useful result
requires the routed model to improve over its matched control, and the learned
weights should show a meaningful depth trend rather than collapse to 0.5/0.5
or to one branch at every layer.
