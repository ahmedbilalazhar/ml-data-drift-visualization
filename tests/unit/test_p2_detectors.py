"""P2 detector + DDE checks: ADWIN/MMD bounds, DDE flat-vs-lift behavior."""
import pandas as pd
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.detectors.drift_detectors import (
    adwin_window_score, mmd_window_score, joint_window_score)
from drift_monitoring.explanations.localization import (
    fit_dde_curve, dde_scores)


def _frames(seed=0):
    df, _ = generate_stream(ScenarioSpec(template_id="S4", seed=seed,
                                         n_train=800, n_calib=400, n_test=1200))
    num = [c for c in df.columns if c.startswith("x")]
    ref = df[df.phase == "train"].sample(400, random_state=0)
    cur = df[df.phase == "test"].iloc[:400]
    return ref, cur, num


def test_adwin_zero_identical_bounded_deterministic():
    ref, cur, num = _frames()
    z = adwin_window_score(ref, ref, num)
    assert z["global_ADWIN"] == 0.0
    a = adwin_window_score(ref, cur, num)
    b = adwin_window_score(ref, cur, num)
    assert a == b and a["global_ADWIN"] >= 0.0


def test_mmd_zero_identical_nonneg_symmetric():
    ref, cur, num = _frames()
    z = mmd_window_score(ref, ref, num)
    assert z["global_MMD"] == 0.0
    a = mmd_window_score(ref, cur, num)
    b = mmd_window_score(cur, ref, num)
    assert a["global_MMD"] >= 0.0
    assert abs(a["global_MMD"] - b["global_MMD"]) < 1e-9


def test_dde_flat_vs_lift():
    import pandas as pd
    feats = ["a", "b", "c", "d"]
    # independent drift: uniform correlation -> DDE(z) constant wherever pairs
    # qualify (flat lift ~1.0); Covertype-like failure = no upward slope
    corr_flat = pd.DataFrame(0.9, index=feats, columns=feats)
    for f in feats:
        corr_flat.loc[f, f] = 1.0
    fit = fit_dde_curve(corr_flat, {"a", "b"}, feats)
    lifts = list(fit["lift"].values())
    assert max(lifts) - min(lifts) < 1e-9
    assert fit["status"] == "inferred-dde"
    # correlated block: a,b highly correlated and both drifted -> upward lift
    corr_hi = pd.DataFrame(0.05, index=feats, columns=feats)
    for f in feats:
        corr_hi.loc[f, f] = 1.0
    corr_hi.loc["a", "b"] = corr_hi.loc["b", "a"] = 0.95
    fit2 = fit_dde_curve(corr_hi, {"a", "b"}, feats)
    assert fit2["lift"][0.9] > fit2["lift"][0.0]
    sc = dde_scores(fit2["curve"], corr_hi, {"a", "b"}, ["c", "d"])
    assert sc["status"] == "inferred-dde"
    assert set(sc["scores"]) == {"c", "d"}


def test_joint_auc_range_and_separation():
    ref, cur, num = _frames()
    same = joint_window_score(ref, ref, num)
    assert 0.4 <= same["auc"] <= 0.6  # indistinguishable
    sep = joint_window_score(ref, cur, num)
    assert 0.5 <= sep["auc"] <= 1.0
    again = joint_window_score(ref, cur, num)
    assert sep["auc"] == again["auc"]
