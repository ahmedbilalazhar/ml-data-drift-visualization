"""DDE end-to-end demo on S4 correlated group (P2-3 verification).

Monitored = top-6 KS; drifted = KS D > 0.15 (transparent demo rule);
fit lift curve on reference |Pearson rho|; score the rest via Eq.25
averaging. Writes results/aggregates/dde_demo_S4.csv + prints lift.
x2-style KS false positives propagate honestly (circularity caveat).
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.detectors.drift_detectors import ks_window_score
from drift_monitoring.explanations.localization import (
    rank_features, fit_dde_curve, dde_scores)

OUT = ROOT / "results" / "aggregates"


def main(template="S4", seed=0):
    df, truth = generate_stream(ScenarioSpec(
        template_id=template, seed=seed, n_train=2000, n_calib=2000, n_test=6000))
    num = [c for c in df.columns if c.startswith("x")]
    ref = df[df.phase == "train"]
    corr = ref[num].corr().abs()
    seg = df[df.phase == "test"].iloc[3000:3500]
    ks = ks_window_score(ref.sample(1000, random_state=0), seg, num)
    rk = rank_features(ks["per_feature"], k=12)
    monitored = rk["ranking"][:6]
    drifted = {f for f in monitored if rk["scores"][f] > 0.15}
    rest = [f for f in num if f not in monitored]
    fit = fit_dde_curve(corr, drifted, monitored)
    sc = dde_scores(fit["curve"], corr, drifted, rest)
    print("truth:", sorted(truth.get("affected_original_vars", [])))
    print("monitored:", monitored)
    print("drifted:", sorted(drifted))
    print("lift:", {k: round(v, 3) for k, v in fit["lift"].items()})
    rows = [{"feature": f, "dde_score": round(s, 4),
             "in_truth": f in set(truth.get("affected_original_vars", [])),
             "status": sc["status"]} for f, s in sc["scores"].items()]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "dde_demo_S4.csv", index=False)
    print(f"-> {OUT / 'dde_demo_S4.csv'}")


if __name__ == "__main__":
    main()
