"""Released-label error change detector for delayed verification experiments.

This is a small-window, exact split-scan ADWIN-style implementation: it uses
Hoeffding bounds over released binary errors and trims the older window after a
detected change. It deliberately omits ADWIN's exponential bucket compression;
results must not be described as a River/official ADWIN reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class ReleasedErrorADWINStyle:
    delta: float = 0.002
    min_window_batches: int = 2
    window: list[tuple[int, int]] = field(default_factory=list)
    updates: int = 0

    def __post_init__(self):
        if not math.isfinite(self.delta) or not 0 < self.delta < 1:
            raise ValueError("delta must be finite and in (0,1)")
        if (not isinstance(self.min_window_batches, int) or isinstance(self.min_window_batches, bool)
                or self.min_window_batches < 1):
            raise ValueError("minimum window must be a positive integer")

    def update(self, errors: int, n: int) -> dict:
        """Consume one cohort only when its labels are available.

        Returns alarm state, trimmed window size, and maximum tested difference.
        The input is an error count and denominator, not a predicted score.
        """
        if (not isinstance(errors, int) or isinstance(errors, bool)
                or not isinstance(n, int) or isinstance(n, bool)
                or n < 1 or not 0 <= errors <= n):
            raise ValueError("released errors must be integer counts in [0,n]")
        self.updates += 1
        self.window.append((errors, n))
        alarm = False
        max_difference = 0.0
        cuts_tested = 0
        while len(self.window) >= 2 * self.min_window_batches:
            candidate = None
            cut_count = len(self.window) - 2 * self.min_window_batches + 1
            for cut in range(self.min_window_batches,
                             len(self.window) - self.min_window_batches + 1):
                left = self.window[:cut]
                right = self.window[cut:]
                left_n = sum(item[1] for item in left)
                right_n = sum(item[1] for item in right)
                left_mean = sum(item[0] for item in left) / left_n
                right_mean = sum(item[0] for item in right) / right_n
                difference = abs(left_mean - right_mean)
                # Union-bound over tested cuts and updates (bounded by current clock).
                delta_cut = self.delta / max(1, cut_count * self.updates)
                bound = math.sqrt(.5 * math.log(4.0 / delta_cut) *
                                  (1.0 / left_n + 1.0 / right_n))
                cuts_tested += 1
                max_difference = max(max_difference, difference)
                if difference > bound:
                    candidate = cut
                    break
            if candidate is None:
                break
            alarm = True
            del self.window[:candidate]
        return {"alarm": alarm, "window_batches": len(self.window),
                "max_mean_difference": max_difference, "cuts_tested": cuts_tested,
                "updates": self.updates}
