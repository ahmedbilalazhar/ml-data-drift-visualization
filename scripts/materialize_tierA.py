"""Materialize synthetic TierA (configs/datasets/synthetic_tierA.yaml).

10k train / 10k calib / 30k test per template S0-S7, seed 0 (pilot scale-up
from 2k/2k/6k). Writes data/processed/tierA/*.parquet + tierA_manifest.json
with row counts + generator code version. Laptop-OK (numpy generator).
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
import yaml

OUT = ROOT / "data" / "processed" / "tierA"
MAN = ROOT / "data" / "manifests"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "datasets" / "synthetic_tierA.yaml").read_text())
    d = cfg["dataset"]
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    t0 = time.time()
    for tid in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        df, truth = generate_stream(ScenarioSpec(
            template_id=tid, seed=0, n_train=d["n_train"],
            n_calib=d["n_calib"], n_test=d["n_test"]))
        p = OUT / f"tierA-{tid}-s0.parquet"
        df.to_parquet(p, index=False)
        recs.append({"template": tid, "rows": len(df), "file": str(p.relative_to(ROOT)),
                     "truth_mechanism": truth.get("mechanism")})
        print(f"{tid}: {len(df)} rows -> {p.name}")
    man = {"tier": "A", "config": d, "runs": recs,
           "total_rows": sum(r["rows"] for r in recs),
           "seconds": round(time.time() - t0, 1),
           "note": "seed 0 only; multiseed held-out seeds via run_multiseed.py"}
    (MAN / "tierA_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"-> {MAN / 'tierA_manifest.json'} ({man['total_rows']} rows)")


if __name__ == "__main__":
    main()
