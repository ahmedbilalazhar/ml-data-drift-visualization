"""Run registry + resumable orchestration (roadmap Sec. 11.2, 13.5).

Every run: immutable run id, config hash, status, timing, resource peaks,
artifact checksums. Resume restores next window; never duplicates windows.
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import pandas as pd


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            pd.DataFrame(columns=["run_id", "config_hash", "status",
                                  "n_windows_done"]).to_csv(self.path, index=False)

    def mark(self, run_id: str, chash: str, status: str, n_done: int):
        df = pd.read_csv(self.path)
        df = df[df.run_id != run_id]
        df.loc[len(df)] = [run_id, chash, status, n_done]
        df.to_csv(self.path, index=False)


def run_stream_experiment(run_id: str, df: pd.DataFrame, truth: dict,
                          models: dict, outdir: Path,
                          window: int = 500, seed: int = 0) -> Path:
    """E0 core loop: reference -> per-window KS/Wasserstein + frozen preds.

    Evidence records carry run/dataset/scenario/model/ref versions, window
    ids, observation time, score+units, threshold=NULL (calibrated later),
    group counts, measured status, compute cost (roadmap Sec. 6.2).
    """
    from ..windows.replay import ReplayConfig, ReplayClock, make_windows
    from ..detectors.drift_detectors import ks_window_score, wasserstein_window_score
    from ..evaluation.metrics import window_macro_f1

    t0 = time.time()
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"{run_id}.checkpoint.json"
    evid_path = outdir / f"{run_id}.evidence.csv"
    start_wid = 0
    if ckpt.exists():  # resume: continue from recorded next window
        start_wid = int(json.loads(ckpt.read_text())["next_window"])
    if evid_path.exists():
        if start_wid == 0:
            evid_path.unlink()  # fresh start: never append to a stale file
        else:
            # crash may have landed between record-append and checkpoint-write:
            # resume past whatever is already persisted (no duplicates, no gaps)
            have = pd.read_csv(evid_path)
            if len(have):
                start_wid = max(start_wid, int(have.window_id.max()) + 1)

    train = df[df.phase == "train"]
    test = df[df.phase == "test"].reset_index(drop=True)
    ref = train.sample(min(5000, len(train)), random_state=seed)
    num = models["numeric"]
    clock = ReplayClock(test, ReplayConfig(window=window, stride=window,
                                           label_delay_windows=2))
    n_new = 0
    for w, seg, released in clock.iter_steps():
        if w["window_id"] < start_wid:
            continue
        ks = ks_window_score(ref, seg, num)
        ws = wasserstein_window_score(ref, seg, num)
        Xw = seg[num + ["cat"]]
        pred = models["tree"].predict(Xw)
        f1 = window_macro_f1(seg["label"].to_numpy(), pred)
        # label availability marker: current window labels NOT yet released
        rec = pd.DataFrame([{
            "run_id": run_id, "window_id": w["window_id"],
            "start": w["start"], "end": w["end"],
            "ks_global_D": ks["global_D"], "w_global_norm": ws["global_W_norm"],
            "macro_f1": f1, "label_status": "delayed-2w",
            "n": len(seg), "evidence_status": "measured",
            "template": truth["template"], "seed": truth["seed"],
        }])
        # incremental persist BEFORE checkpoint: a kill loses at most one window,
        # and resume never duplicates (start_wid reconciled with file on entry)
        rec.to_csv(evid_path, mode="a", header=not evid_path.exists(), index=False)
        ckpt.write_text(json.dumps({"next_window": w["window_id"] + 1}))
        n_new += 1
    evid = pd.read_csv(evid_path)
    (outdir / f"{run_id}.truth.json").write_text(json.dumps(truth, indent=1, default=str))
    (outdir / f"{run_id}.meta.json").write_text(json.dumps({
        "run_id": run_id, "seconds": time.time() - t0,
        "n_windows": len(evid), "code_version": "e0-v2-incremental"}, indent=1))
    return evid_path
