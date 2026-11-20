"""P0 detector + pipeline checks: JS bounds/determinism, build_pipelines contract."""
import pandas as pd
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.models.frozen_models import build_pipelines, fit_frozen
from drift_monitoring.detectors.drift_detectors import js_window_score


def _toy_frames(seed=0):
    df, _ = generate_stream(ScenarioSpec(template_id="S1", seed=seed,
                                         n_train=800, n_calib=400, n_test=1200))
    num = [c for c in df.columns if c.startswith("x")]
    ref = df[df.phase == "train"].sample(400, random_state=0)
    cur = df[df.phase == "test"].iloc[:400]
    return ref, cur, num, df


def test_js_bounded_and_deterministic():
    ref, cur, num, _ = _toy_frames()
    a = js_window_score(ref, cur, num)
    b = js_window_score(ref, cur, num)
    assert 0.0 <= a["global_JS"] <= 1.0
    assert a == b
    assert set(a["per_feature"]) == set(num)


def test_js_zero_on_identical():
    ref, _, num, _ = _toy_frames()
    out = js_window_score(ref, ref, num)
    assert out["global_JS"] == 0.0


def test_build_pipelines_unfitted_contract():
    _, _, num, df = _toy_frames()
    pipes = build_pipelines(num, random_state=1)
    assert set(pipes) >= {"tree", "logreg", "numeric"}
    X = df[num + ["cat"]].iloc[:50]
    import pytest
    with pytest.raises(Exception):
        pipes["tree"].predict(X)  # unfitted must fail, not silently pass
    m = fit_frozen(df[df.phase == "train"])
    pred = m["tree"].predict(X)
    assert len(pred) == 50
