# Demonstration guide

## Architecture

1. A frozen model and representations are fitted on the training partition; correctness calibration uses a separate historical partition.
2. Each monitoring batch exposes features to the online detector and impact estimator. The scoring interface has no labels.
3. The runner writes prediction artifacts before the label broker releases due labels. Release delay is 0, 5, or 20 batches.
4. Released labels verify previous predictions. The offline evaluator joins these outcomes for analysis; the evidence view filters them to the selected as-of clock.
5. Feature rankings and clusters are descriptive. The concern score and plug-in interval are not certified population-risk estimates.

## Representative trace

Open `reports/phase3/final_v1/evidence_view.html`. Select seed 100, condition C3, delay 5, and clock 16. Explain the four aligned channels: adjusted input-test evidence, feature distances, estimated accuracy/concern, and scheduled releases. Batches 1-11 may show verification because their five-batch delays elapsed; batches 12-16 have not yet released labels. At later clocks, verified outcomes appear only after release. The evaluator-only accuracy trace in static scientific reports is retrospective and is not live dashboard evidence.

## Reproduction

From the repository root in PowerShell, use the exact full rerun commands in `docs/phase3/23I-2581_23I-2604_Phase3_Reproduction.md`. The v1 final run takes roughly ten minutes according to the historical protocol receipt. V2 and adaptation each have separate reproduction scripts and should use fresh output directories. Never target the submitted archive or existing final evidence paths.
