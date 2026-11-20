# Day 3 sensitivity question (exploratory)

The locked Day 2 results show severe underestimation of error. Before computing this sensitivity, fix the question: **How much of the cohort accuracy variation is associated with changing class proportions, and does source calibration confidence transfer to the later batches?**

Use the already frozen model, predictions, and batch-1 split. Do not refit or choose a new model or threshold. For each cohort, report class-specific accuracy and a common-class standardized accuracy weighted by batch-1 reference proportions renormalized over classes present in that cohort. Report the difference from aggregate accuracy as a descriptive composition-associated contrast; it is not a causal decomposition and excludes classes absent in a cohort. Compare calibration-set, batch-1 reference, and later cohort confidence versus correctness, with denominators. Treat this as outcome-informed exploratory diagnosis of a failure observed in the locked primary run.
