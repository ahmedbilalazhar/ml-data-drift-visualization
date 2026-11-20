# Superseded exploratory execution

The first Day 2 execution is retained under `results/phase3/external_uci224_day2_v1/`. A split-indexing implementation bug allocated 30% training, 30% calibration, and 40% reference rows instead of the locked 60/20/20. The result and its diagnostics are not used in the Day 2/Day 3 reports. Corrected `external_uci224_day2_v2/` is the sole primary external result. This correction changes no model, metric, threshold, or cohort selection.
