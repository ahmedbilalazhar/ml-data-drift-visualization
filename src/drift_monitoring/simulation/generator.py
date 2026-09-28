"""Transparent tabular generator with S0-S7 drift templates.

Design (roadmap Sec. 7.1 Tier A, 7.3, 7.6):
- relevant / irrelevant / correlated / independent numeric features
- one categorical feature (valid codes preserved)
- binary label from a fixed logistic rule (covariate-only cases keep the rule;
  S7 changes the rule while holding the input process fixed)
- every scenario card records onset/transition/affected vars/mechanism/seed
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class ScenarioSpec:
    template_id: str = "S0"          # S0..S7
    n_features: int = 12
    n_corr_group: int = 3            # first k relevant features are correlated
    corr: float = 0.7
    seed: int = 0
    n_train: int = 10_000
    n_calib: int = 10_000
    n_test: int = 30_000
    test_onset: float = 0.4          # fraction into test where drift starts
    transition: int = 2_000          # rows for gradual transition (S2)
    shift: float = 1.0               # mean shift in std units
    affected: list = field(default_factory=lambda: [0, 1])
    conditional_flip: bool = False   # S7: flip sign of w[0], w[1]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_inputs(rng: np.random.Generator, n: int, d: int,
                   corr: float, k_corr: int) -> np.ndarray:
    X = rng.normal(0, 1, size=(n, d))
    # correlated group: x_j = corr * x_0 + sqrt(1-corr^2) * noise
    for j in range(1, min(k_corr, d)):
        X[:, j] = corr * X[:, 0] + np.sqrt(max(1 - corr ** 2, 1e-6)) * rng.normal(0, 1, n)
    return X


def _label_rule(X: np.ndarray, w: np.ndarray, b: float,
                rng: np.random.Generator) -> np.ndarray:
    logits = X[:, :len(w)] @ w + b
    p = 1.0 / (1.0 + np.exp(-logits))
    return (rng.uniform(0, 1, len(X)) < p).astype(int)


BASE_W = np.array([1.2, -0.9, 0.6, 0.0, 0.0, 0.0])


def generate_stream(spec: ScenarioSpec) -> tuple[pd.DataFrame, dict]:
    """Return (df with event_idx, features, label, phase; truth dict)."""
    rng = _rng(spec.seed)
    d = spec.n_features
    w = np.zeros(d)
    w[:len(BASE_W)] = BASE_W[:min(d, len(BASE_W))]
    b = -0.2
    truth: dict = {
        "template": spec.template_id, "seed": spec.seed,
        "affected_original_vars": [f"x{i}" for i in spec.affected],
        "onset_test_row": None, "transition_rows": 0,
        "mechanism": "stationary",
    }

    def block(n, phase):
        X = _sample_inputs(rng, n, d, spec.corr, spec.n_corr_group)
        cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
        y = _label_rule(X, w[:d] if d <= len(w) else np.pad(w, (0, d - len(w))), b, rng)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
        df["cat"] = cat
        df["label"] = y
        df["phase"] = phase
        return df

    train = block(spec.n_train, "train")
    calib = block(spec.n_calib, "calib")

    # ---- test block with intervention ----
    n = spec.n_test
    X = _sample_inputs(rng, n, d, spec.corr, spec.n_corr_group)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    onset = int(n * spec.test_onset)
    w_test = w.copy()
    ramp = np.zeros(n)

    tid = spec.template_id
    if tid == "S0":
        pass
    elif tid == "S1":  # abrupt shift in IRRELEVANT input (x_{d-1} unused by w)
        truth.update(mechanism="abrupt-irrelevant", onset_test_row=onset)
        X[onset:, d - 1] += spec.shift * 2.0
        truth["affected_original_vars"] = [f"x{d-1}"]
    elif tid == "S2":  # gradual shift in relevant inputs, fixed rule
        truth.update(mechanism="gradual-relevant",
                     onset_test_row=onset, transition_rows=spec.transition)
        end = min(n, onset + spec.transition)
        ramp[onset:end] = np.linspace(0, 1, end - onset)
        ramp[end:] = 1.0
        for j in spec.affected:
            X[:, j] += ramp * spec.shift * 1.5
    elif tid == "S3":  # A -> B -> A recurrence
        mid = onset + n // 4
        truth.update(mechanism="recurrent", onset_test_row=onset,
                     transition_rows=mid - onset)
        X[onset:mid, spec.affected] += spec.shift * 1.5
    elif tid == "S4":  # correlated-group intervention on x0 (proxy x1,x2 move)
        truth.update(mechanism="correlated-group", onset_test_row=onset)
        # intervene on the latent driver: shift x0, regenerate proxies to keep dependency spec
        X[onset:, 0] += spec.shift * 1.5
        for j in range(1, min(spec.n_corr_group, d)):
            noise = rng.normal(0, 1, n - onset)
            X[onset:, j] = (spec.corr * X[onset:, 0]
                             + np.sqrt(max(1 - spec.corr ** 2, 1e-6)) * noise)
    elif tid == "S5":  # isolated shift (independent feature, outside monitored subset)
        truth.update(mechanism="isolated", onset_test_row=onset)
        iso = d - 2  # independent by construction (outside corr group)
        X[onset:, iso] += spec.shift * 2.0
        truth["affected_original_vars"] = [f"x{iso}"]
        truth["monitored_subset"] = [f"x{i}" for i in range(d - 3)]
    elif tid == "S6":  # dependence change, unchanged univariate marginals
        truth.update(mechanism="dependence-change", onset_test_row=onset)
        # rotate (x0,x1): preserves N(0,1) marginals, changes joint/correlation
        rot = np.array([[np.cos(0.9), -np.sin(0.9)], [np.sin(0.9), np.cos(0.9)]])
        X[onset:, 0:2] = X[onset:, 0:2] @ rot.T
        truth["affected_original_vars"] = ["x0", "x1 (joint)"]
    elif tid == "S7":  # pure conditional change: inputs fixed, rule flips
        truth.update(mechanism="conditional-only", onset_test_row=onset)
        w_test = w.copy()
        w_test[0] = -w[0]
        w_test[1] = -w[1]
    else:
        raise ValueError(f"unknown template {tid}")

    # labels: same fixed rule except S7 switches rule at onset
    y = np.empty(n, dtype=int)
    y_pre = _label_rule(X[:onset], w[:d] if d <= len(w) else w, b, rng) if onset else np.empty(0, int)
    w_post = w_test[:d] if d <= len(w_test) else np.pad(w_test, (0, d - len(w_test)))
    w_pre = w[:d] if d <= len(w) else np.pad(w, (0, d - len(w)))
    y[:onset] = _label_rule(X[:onset], w_pre, b, rng) if onset else []
    y[onset:] = _label_rule(X[onset:], w_post, b, rng)
    _ = y_pre  # kept for clarity of pre/post rule split

    test = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
    test["cat"] = cat
    test["label"] = y
    test["phase"] = "test"

    df = pd.concat([train, calib, test], ignore_index=True)
    df.insert(0, "event_id", np.arange(len(df)))
    truth.update(n_train=spec.n_train, n_calib=spec.n_calib, n_test=spec.n_test)
    return df, truth
