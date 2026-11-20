import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/"results/phase3/noaa_training_uncertainty_20261117_v2"
REPORT=ROOT/"reports/phase3/noaa_training_uncertainty_20261117_v3"

def test_training_bootstrap_refits_and_keeps_evaluation_separate():
    receipt=json.loads((RUN/"audit_receipt.json").read_text(encoding="utf-8"))
    draws=pd.read_csv(RUN/"training_bootstrap_draws.csv")
    meta=pd.read_csv(RUN/"replicate_metadata.csv")
    summary=pd.read_csv(RUN/"training_uncertainty_summary.csv")
    assert receipt["replicates"]==200 and receipt["training_block_length_rows"]==30
    assert receipt["calibration_reference_and_monitor"].startswith("fixed original")
    assert len(draws)==200*21 and len(meta)==200 and len(summary)==21
    assert draws.groupby("replicate").batch.nunique().eq(21).all()
    assert draws.groupby("batch").replicate.nunique().eq(200).all()
    assert meta.resampling_attempts.min()>=1
    assert (meta.class_0_rows>0).all() and (meta.class_1_rows>0).all()
    locked=pd.read_csv(ROOT/"results/phase3/noaa_weather_1949_1999_v1/evaluator/batch_metrics.csv")
    assert np.allclose(summary.verified_accuracy.to_numpy(),locked.accuracy.to_numpy())
    assert np.allclose(summary.verified_accuracy.to_numpy(),draws.groupby("batch").baseline_verified_accuracy.first().to_numpy())
    assert summary.replicate_estimator_mae.between(0,1).all()
    assert int(summary.accuracy_interval_covers_original_estimator.sum())>=18
    assert np.isfinite(draws[["accuracy_estimate","concern","reference_accuracy"]].to_numpy()).all()
    for name,digest in receipt["output_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest

def test_training_uncertainty_report_artifacts_exist():
    for name in ("NOAA_TRAINING_UNCERTAINTY.tex","NOAA_TRAINING_UNCERTAINTY.pdf","training_uncertainty.png","training_uncertainty.svg","training_uncertainty.pdf"):
        assert (REPORT/name).is_file()
    tex=(REPORT/"NOAA_TRAINING_UNCERTAINTY.tex").read_text(encoding="utf-8")
    assert "does not provide a generalization guarantee" in tex
    assert "original frozen model accuracy estimate" in tex
    assert "true or future accuracy" in tex
