"""Fit-only-on-train tabular preprocessing (roadmap Sec. 6.1).

Encoders/scalers fit on train; reused frozen for calib/test. Maps encoded
columns back to ORIGINAL variables (group one-hot back to source var).
"""
from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FrozenTabular:
    def __init__(self, numeric: list[str], cat_col: str = "cat"):
        self.numeric = numeric
        self.cat_col = cat_col
        self.scaler = StandardScaler()
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._fit = False

    def fit(self, df_train: pd.DataFrame) -> "FrozenTabular":
        self.scaler.fit(df_train[self.numeric].to_numpy(float))
        self.enc.fit(df_train[[self.cat_col]])
        self._fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fit, "preprocessing used before fit (leakage guard)"
        out = pd.DataFrame(self.scaler.transform(df[self.numeric].to_numpy(float)),
                           columns=self.numeric, index=df.index)
        out[self.cat_col] = df[self.cat_col].astype(str)  # codes stay valid
        return out
