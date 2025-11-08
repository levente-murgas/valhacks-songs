from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

CATEGORICAL_COLS = [
    "explicit",  # bool/string
    "mode",  # 0/1
    "key",  # -1..11 (we treat as categorical bucket including -1)
    "time_signature",  # 3..7
    "track_genre",  # string category
]

META_COLS = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop unnamed index column if present
    if df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def build_preprocessor(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> ColumnTransformer:
    exclude_set = set(exclude or [])
    # Filter numeric & categorical excluding requested columns
    numeric = [c for c in NUMERIC_COLS if c in df.columns and c not in exclude_set]
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns and c not in exclude_set]

    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(with_mean=True, with_std=True), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))

    if not transformers:
        raise ValueError("No features left after applying excludes; choose fewer columns to exclude.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def fit_transform_features(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> Tuple[np.ndarray, ColumnTransformer, List[str]]:
    pre = build_preprocessor(df, exclude=exclude)
    # Determine used columns (order matters for reproducibility)
    exclude_set = set(exclude or [])
    used_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df.columns and c not in exclude_set]
    X = pre.fit_transform(df[used_cols])
    return X.astype(np.float32), pre, used_cols


def transform_features(df: pd.DataFrame, pre: ColumnTransformer) -> np.ndarray:
    return pre.transform(df[[*pre.feature_names_in_]]).astype(np.float32)


def get_meta(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in META_COLS if c in df.columns]
    return df[cols].copy()
