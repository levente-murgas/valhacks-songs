from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

NUMERIC_FEATURES: Sequence[str] = (
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
)


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        msg = f"Dataset not found at {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    summary: dict[str, object] = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "numeric_describe": df[list(NUMERIC_FEATURES)].describe().to_dict(),
        "unique_counts": df.select_dtypes(include="object")
        .nunique(dropna=False)
        .to_dict(),
    }
    return summary


def normalize_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    numeric_values = df[list(NUMERIC_FEATURES)]
    scaled = scaler.fit_transform(numeric_values)

    normalized_df = df.copy()
    for feature, values in zip(NUMERIC_FEATURES, scaled.T, strict=True):
        normalized_df[f"{feature}_zscore"] = values

    return normalized_df, scaler


def persist_outputs(
    clean_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    summary: dict,
    scaler: StandardScaler,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = PROCESSED_DIR / "dataset_clean.csv"
    normalized_path = PROCESSED_DIR / "dataset_normalized.csv"
    summary_path = REPORTS_DIR / "data_summary.json"
    scaler_path = PROCESSED_DIR / "numeric_scaler.joblib"

    clean_df.to_csv(clean_path, index=False)
    normalized_df.to_csv(normalized_path, index=False)

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    joblib.dump(scaler, scaler_path)


def main() -> None:
    df = load_raw_dataset()
    summary = compute_summary(df)
    normalized_df, scaler = normalize_numeric_features(df)
    persist_outputs(df, normalized_df, summary, scaler)


if __name__ == "__main__":
    main()

