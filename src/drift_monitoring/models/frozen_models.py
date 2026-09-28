"""Frozen predictive models (roadmap Sec. 6.1, 8.1).

Two modest models, fit ONLY on the authorized train split, frozen for the
principal monitoring runs: depth-capped tree (inspectable) + regularized
logistic (family sensitivity). No retraining inside core replay.
"""
from __future__ import annotations
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def feature_lists(df: pd.DataFrame):
    num = [c for c in df.columns if c.startswith("x")]
    return num, ["cat"]


def build_pipelines(random_state: int = 0):
    pre = ColumnTransformer([
        ("num", StandardScaler(), None),  # columns set at fit time via wrapper
    ], remainder="drop")
    # NOTE: ColumnTransformer needs concrete columns; we construct full
    # pipelines in fit_frozen() below instead. Kept simple on purpose.
    raise NotImplementedError


def _make_pre(num_cols: list[str]):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["cat"]),
    ])


def fit_frozen(df_train: pd.DataFrame, random_state: int = 0) -> dict:
    num, _ = feature_lists(df_train)
    X = df_train[num + ["cat"]]
    y = df_train["label"].to_numpy()
    tree = Pipeline([("pre", _make_pre(num)),
                     ("clf", DecisionTreeClassifier(max_depth=6,
                                                    random_state=random_state))])
    logr = Pipeline([("pre", _make_pre(num)),
                     ("clf", LogisticRegression(max_iter=1000, C=1.0))])
    tree.fit(X, y)
    logr.fit(X, y)
    return {"tree": tree, "logreg": logr,
            "numeric": num, "model_versions": {"tree": "dt-d6-v1",
                                              "logreg": "lr-C1-v1"}}
