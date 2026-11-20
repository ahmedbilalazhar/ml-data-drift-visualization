"""Rebuild readable, checksummed Elec2 and Covertype processed data in v2 paths.

The previous processed Parquet files are left untouched. Inputs are the exact
raw files already recorded in data/manifests; this script does not acquire or
silently change datasets.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/verified_v2"
MAN = ROOT / "data/manifests"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        el_manifest_path = MAN / "elec2_verified_v2_manifest.json"
        cov_manifest_path = MAN / "covtype_subset_verified_v2_manifest.json"
        if not all(p.exists() for p in (OUT / "elec2_ordered.parquet",
                                        OUT / "covtype_subset100k.parquet",
                                        el_manifest_path, cov_manifest_path)):
            raise FileExistsError(f"refusing to overwrite incomplete verified output directory {OUT}")
        el_manifest = json.loads(el_manifest_path.read_text(encoding="utf-8"))
        cov_manifest = json.loads(cov_manifest_path.read_text(encoding="utf-8"))
        el_path, cov_path = OUT / "elec2_ordered.parquet", OUT / "covtype_subset100k.parquet"
        if (sha256(el_path) != el_manifest["processed_file"]["sha256"]
                or sha256(cov_path) != cov_manifest["sha256"]
                or len(pd.read_parquet(el_path)) != 45_312
                or len(pd.read_parquet(cov_path)) != int(cov_manifest["n"])):
            raise ValueError("existing verified_v2 outputs do not match manifests or fail read-back")
        print(json.dumps({"status": "already_built_and_verified", "output": str(OUT),
                          "elec2_sha256": sha256(el_path), "covtype_sha256": sha256(cov_path)}, indent=2))
        return
    OUT.mkdir(parents=True)
    el_src = ROOT / "data/raw/openml/openml.org/data/v1/download/2419/electricity.arff.gz"
    el_manifest = json.loads((MAN / "elec2_manifest.json").read_text(encoding="utf-8"))
    if not el_src.exists() or sha256(el_src) not in {f["sha256"] for f in el_manifest["files"]}:
        raise ValueError("Elec2 ARFF is missing or does not match its recorded source manifest")
    with gzip.open(el_src, "rt", encoding="utf-8") as stream:
        array, metadata = arff.loadarff(stream)
    el = pd.DataFrame()
    for name in metadata.names():
        values = array[name]
        if values.dtype.kind == "S":
            decoded = [x.decode("utf-8") for x in values]
            if metadata[name][0] == "nominal":
                el[name] = pd.Categorical(decoded, categories=list(metadata[name][1]))
            else:
                el[name] = decoded
        else:
            el[name] = values
    el = el.rename(columns={"class": "label"})
    el["label"] = (el.label.astype(str).str.upper().isin(["UP", "1"])).astype("int8")
    el.insert(0, "event_id", np.arange(len(el), dtype=np.int64))
    el_path = OUT / "elec2_ordered.parquet"
    el.to_parquet(el_path, index=False, compression="zstd")
    el_read = pd.read_parquet(el_path)
    if len(el_read) != 45_312 or el_read.event_id.tolist() != list(range(45_312)):
        raise ValueError("Elec2 row-count or event-order read-back failed")
    el_out_manifest = {**el_manifest, "dataset_id": "elec2-openml-151-verified-v2",
                       "processed_file": {"path": el_path.relative_to(ROOT).as_posix(),
                                          "sha256": sha256(el_path), "rows": len(el_read),
                                          "columns": list(el_read.columns),
                                          "class_counts": el_read.label.value_counts().sort_index().to_dict(),
                                          "readback_verified": True},
                       "processing_note": "Rebuilt from byte-identical source ARFF recorded in the preserved v1 manifest; file-order event_id retained.",
                       "supersedes_for_new_runs": "unreadable legacy data/processed/elec2_ordered.parquet; legacy file preserved"}
    (MAN / "elec2_verified_v2_manifest.json").write_text(json.dumps(el_out_manifest, indent=2, default=str) + "\n", encoding="utf-8")

    cov_src = ROOT / "data/raw/covtype/covtype.data.gz"
    cov_manifest = json.loads((MAN / "covtype_manifest.json").read_text(encoding="utf-8"))
    if not cov_src.exists() or sha256(cov_src) not in {f["sha256"] for f in cov_manifest["files"]}:
        raise ValueError("Covertype source is missing or does not match its recorded source manifest")
    names = (["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
              "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
              "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
              "Horizontal_Distance_To_Fire_Points"]
             + [f"Wilderness_Area{i}" for i in range(1, 5)]
             + [f"Soil_Type{i}" for i in range(1, 41)] + ["Cover_Type"])
    with gzip.open(cov_src, "rt") as stream:
        cov = pd.read_csv(stream, header=None, names=names)
    rng = np.random.default_rng(0)
    labels = cov.Cover_Type.to_numpy()
    keep = []
    per_class = 100_000 // 7
    for cls in sorted(set(labels.tolist())):
        indices = np.where(labels == cls)[0]
        keep.extend(rng.choice(indices, min(per_class, len(indices)), replace=False).tolist())
    keep = np.array(sorted(keep), dtype=np.int64)
    subset = cov.iloc[keep].reset_index(drop=True).rename(columns={"Cover_Type": "label"})
    subset.insert(0, "event_id", np.arange(len(subset), dtype=np.int64))
    cov_path = OUT / "covtype_subset100k.parquet"
    subset.to_parquet(cov_path, index=False, compression="zstd")
    source_ids_path = OUT / "covtype_source_row_ids.csv"
    pd.DataFrame({"event_id": np.arange(len(keep), dtype=np.int64),
                  "source_row_id_zero_based": keep}).to_csv(source_ids_path, index=False)
    cov_read = pd.read_parquet(cov_path)
    if len(cov_read) != len(keep) or cov_read.event_id.tolist() != list(range(len(keep))):
        raise ValueError("Covertype row-count or event-order read-back failed")
    cov_out_manifest = {**json.loads((MAN / "covtype_subset_manifest.json").read_text(encoding="utf-8")),
                        "dataset_id": "uci-covertype-stratified-verified-v2",
                        "n": len(cov_read), "source_row_ids_path": source_ids_path.relative_to(ROOT).as_posix(),
                        "source_row_ids_sha256": sha256(source_ids_path),
                        "file": cov_path.relative_to(ROOT).as_posix(), "sha256": sha256(cov_path),
                        "readback_verified": True,
                        "supersedes_for_new_runs": "unreadable legacy data/processed/covtype_subset100k.parquet; legacy file preserved"}
    (MAN / "covtype_subset_verified_v2_manifest.json").write_text(json.dumps(cov_out_manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "elec2": {"rows": len(el_read), "columns": list(el_read.columns),
                      "sha256": sha256(el_path)}, "covtype_subset": {"rows": len(cov_read),
                      "sha256": sha256(cov_path), "source_ids_sha256": sha256(source_ids_path)},
                      "outputs": str(OUT)}, indent=2, default=str))


if __name__ == "__main__":
    main()
