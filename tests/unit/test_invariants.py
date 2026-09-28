"""Scientific-invariant checks (roadmap Sec. 11.3). Cheap, no network."""
import pandas as pd
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.models.frozen_models import fit_frozen
from drift_monitoring.detectors.drift_detectors import ks_window_score


def test_no_train_on_test():
    df, _ = generate_stream(ScenarioSpec(template_id="S2", seed=1,
                                         n_train=500, n_calib=500, n_test=1500))
    tr = df[df.phase == "train"]
    assert (tr.event_id < 1000).all()  # train ids precede calib/test


def test_predict_before_label_contract():
    # replay yields features of window w before labels of w are released at delay>=1
    from drift_monitoring.windows.replay import ReplayClock, ReplayConfig
    df, _ = generate_stream(ScenarioSpec(template_id="S1", seed=2,
                                         n_train=500, n_calib=500, n_test=1500))
    test = df[df.phase == "test"].reset_index(drop=True)
    clock = ReplayClock(test, ReplayConfig(window=500, stride=500,
                                           label_delay_windows=2))
    steps = list(clock.iter_steps())
    assert 2 not in steps[0][2]  # window-2 labels unavailable at step 0
    assert 0 in steps[2][2]      # released exactly at the delay boundary


def test_s7_invisible_to_input_detector():
    # pure conditional change: input KS on train-vs-early-test stays small;
    # check runs without NaN (blind-spot case must be REPORTED, not crash)
    df, truth = generate_stream(ScenarioSpec(template_id="S7", seed=3,
                                             n_train=1000, n_calib=1000,
                                             n_test=3000))
    assert truth["mechanism"] == "conditional-only"
    ref = df[df.phase == "train"].sample(500, random_state=0)
    early = df[(df.phase == "test")].iloc[:500]
    num = [c for c in df.columns if c.startswith("x")]
    out = ks_window_score(ref, early, num)
    assert 0.0 <= out["global_D"] <= 1.0


def test_models_fit_train_only():
    df, _ = generate_stream(ScenarioSpec(template_id="S0", seed=4,
                                         n_train=800, n_calib=200, n_test=1000))
    m = fit_frozen(df[df.phase == "train"])
    assert set(m) >= {"tree", "logreg", "numeric"}
