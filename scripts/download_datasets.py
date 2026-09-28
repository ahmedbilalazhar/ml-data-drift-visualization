"""Download Elec2 + Covertype raw sources (roadmap Sec. 7.2 audit gate).

- Elec2: OpenML 'electricity' v1 (normalized NSW market, ~45k rows, binary).
- Covertype: sklearn fetch_covtype (UCI mirror, 581k x 54+1, 7 classes).
- Stores under data/raw/ (immutable, gitignored), writes manifests with
  source URL, date, file bytes, sha256, rows/cols, license notes.
- Subsets: Elec2 full chronological; Covertype stratified 100k (seed 0) with
  recorded row IDs -> data/processed/. Row-order manifests included.
Laptop-OK (downloads ~100MB once); heavy replays stay Colab-gated.
"""
import hashlib, json, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAN = ROOT / "data" / "manifests"
PROC = ROOT / "data" / "processed"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from sklearn.datasets import fetch_covtype, fetch_openml
    RAW.mkdir(parents=True, exist_ok=True)
    MAN.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    # ---- Covertype (UCI direct; sklearn figshare mirror returned 403, see
    # docs/incidents/001_covtype_figshare_403.md) ----
    import urllib.request
    cov_dir = RAW / "covtype"
    cov_dir.mkdir(parents=True, exist_ok=True)
    gz = cov_dir / "covtype.data.gz"
    if not gz.exists():
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", gz)
    import gzip
    names = (["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
              "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
              "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
              "Horizontal_Distance_To_Fire_Points"]
             + [f"Wilderness_Area{i}" for i in range(1, 5)]
             + [f"Soil_Type{i}" for i in range(1, 41)] + ["Cover_Type"])
    import pandas as pd
    with gzip.open(gz, "rt") as f:
        cov_df = pd.read_csv(f, header=None, names=names)
    cov_man = {"dataset": "covertype", "source": "UCI direct download (figshare mirror 403)",
               "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
               "acquired": today, "license": "CC-BY-4.0 (UCI record terms apply)",
               "n_rows": len(cov_df), "n_features": 54,
               "classes": sorted(map(int, cov_df.Cover_Type.unique().tolist())),
               "files": [{"path": str(gz.relative_to(ROOT)), "bytes": gz.stat().st_size,
                          "sha256": sha256(gz)}]}
    (MAN / "covtype_manifest.json").write_text(json.dumps(cov_man, indent=1))
    print(f"covtype: {cov_df.shape}, classes {cov_man['classes']}")

    # stratified 100k subset, fixed seed, recorded ids
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    y = cov_df.Cover_Type.to_numpy()
    keep = []
    per_class = 100_000 // 7
    for c in sorted(set(y.tolist())):
        idx = np.where(y == c)[0]
        k = min(per_class, len(idx))
        keep += rng.choice(idx, k, replace=False).tolist()
    keep = np.array(sorted(keep))
    sub = cov_df.iloc[keep].reset_index(drop=True)
    sub = sub.rename(columns={"Cover_Type": "label"})
    sub.insert(0, "event_id", np.arange(len(sub)))
    sub.to_parquet(PROC / "covtype_subset100k.parquet", index=False)
    (MAN / "covtype_subset_manifest.json").write_text(json.dumps(
        {"seed": 0, "n": len(sub), "strategy": "stratified-balanced ~14.3k/class",
         "replay_order": "event_id ascending = subsample-sorted source order (EXPLICITLY constructed, not natural time)",
         "class_counts": sub.label.value_counts().sort_index().to_dict(),
         "file": "data/processed/covtype_subset100k.parquet"}, indent=1))
    print(f"covtype subset: {sub.shape}")

    # ---- Elec2 / electricity ----
    from sklearn.datasets import fetch_openml
    el = fetch_openml(name="electricity", version=1, data_home=str(RAW),
                      parser="auto")
    import pandas as pd
    X = el.data.copy()
    X.columns = [str(c) for c in X.columns]
    X["label"] = (el.target.astype(str).str.upper().isin(["UP", "1"])).astype(int)
    X.insert(0, "event_id", np.arange(len(X)))
    el_files = sorted((RAW / "openml").rglob("*")) 
    el_files = [p for p in el_files if p.is_file()]
    el_man = {"dataset": "elec2", "source": "OpenML 'electricity' v1 (NSW market, normalized)",
              "url": "https://www.openml.org/d/151", "acquired": today,
              "license": "OpenML terms (original: Harvill et al. electricity pricing)",
              "n_rows": len(X), "columns": list(X.columns),
              "class_counts": X.label.value_counts().sort_index().to_dict(),
              "time_semantics": "file order = chronological market intervals (MUST hold for replay)",
              "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                         "sha256": sha256(p)} for p in el_files]}
    (MAN / "elec2_manifest.json").write_text(json.dumps(el_man, indent=1))
    X.to_parquet(PROC / "elec2_ordered.parquet", index=False)
    print(f"elec2: {X.shape}, classes {el_man['class_counts']}")


if __name__ == "__main__":
    main()
