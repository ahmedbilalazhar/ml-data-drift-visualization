import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_selective_label_audit import select_from_confidence


def test_selection_policy_uses_only_confidence_and_reports_inclusion_probabilities():
    confidence = np.linspace(.5, .99, 500)
    rng = np.random.default_rng(17)
    selected, pi = select_from_confidence(confidence, rng, policy="stratified_random_audit")
    assert len(selected) >= 50
    assert np.all((pi == 1.0) | (pi == .10))
    assert np.array_equal(selected[pi == 1.0], np.arange(50))
    assert len(selected[pi == 1.0]) == 50
    assert len(selected[pi == .10]) > 0


def test_targeted_only_selection_does_not_claim_population_estimability():
    run = ROOT / "results/phase3/selective_label_audit_20261117_v2"
    receipt = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(run / "batch_estimates.csv")
    selected = pd.read_csv(run / "selected_label_log.csv", nrows=200)
    assert receipt["status"] == "complete"
    assert receipt["selection_feature_only"] is True
    assert receipt["targeted_only_population_estimability"] is False
    targeted = frame[frame.policy == "targeted_10pct"]
    assert targeted.selected_rows.eq(50).all()
    assert targeted.horvitz_thompson_accuracy.isna().all()
    assert selected.inclusion_probability.between(.1, 1).all()
    assert frame.seed.nunique() == 30
    assert len(frame) == 30 * 2 * 2 * 96 * 3
    assert frame.groupby(["task", "mechanism", "seed", "batch"]).policy.nunique().eq(3).all()
    for rel, expected in receipt["output_sha256"].items():
        digest = hashlib.sha256()
        with (ROOT / rel).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected
