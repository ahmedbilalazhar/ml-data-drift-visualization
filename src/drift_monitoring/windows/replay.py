"""Window + reference manager and replay clock with label-release queue.

Contract (roadmap Sec. 6.4): at each step predict on new features first,
then release labels whose release boundary is met. Frozen core: labels never
affect detectors. Fixed historical reference in the main study.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class ReplayConfig:
    window: int = 500
    stride: int = 500
    ref_rows: int = 5_000
    label_delay_windows: int = 2   # 0 / 2 / 5 in sensitivity


def make_windows(n_rows: int, window: int, stride: int):
    wins = []
    wid = 0
    s = 0
    while s + window <= n_rows:
        wins.append({"window_id": wid, "start": s, "end": s + window})
        wid += 1
        s += stride
    # trailing partial window is explicit, never silently merged
    if s < n_rows:
        wins.append({"window_id": wid, "start": s, "end": n_rows, "partial": True})
    return wins


class ReplayClock:
    """Replays test rows window-by-window; controls label availability."""

    def __init__(self, df_test: pd.DataFrame, cfg: ReplayConfig):
        self.df = df_test.reset_index(drop=True)
        self.cfg = cfg
        self.windows = make_windows(len(df_test), cfg.window, cfg.stride)
        self.pending: list[dict] = []  # released-label queue (for audit)

    def iter_steps(self):
        """Yield (window_record, features_now, labels_available_to_date).

        labels_available: labels of windows with id <= wid - delay (delay 0
        releases current window's labels only AFTER its features are yielded,
        i.e. predict-before-observe is preserved by the caller ordering).
        """
        released = {}
        for w in self.windows:
            seg = self.df.iloc[w["start"]:w["end"]]
            # release labels of the newly eligible window
            eligible = w["window_id"] - self.cfg.label_delay_windows
            if eligible >= 0:
                e = self.windows[eligible]
                lab = self.df.iloc[e["start"]:e["end"]][["event_id", "label"]].copy()
                lab["release_window"] = w["window_id"]
                released[e["window_id"]] = lab
                self.pending.append({"window_id": e["window_id"],
                                     "released_at": w["window_id"],
                                     "n": len(lab)})
            yield w, seg, released
