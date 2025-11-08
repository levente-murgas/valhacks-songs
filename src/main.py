from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset() -> pd.DataFrame:
    """
    Load the Spotify track dataset into a pandas DataFrame.

    Returns:
        DataFrame containing all columns from data/dataset.csv.
    """
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"
    if not dataset_path.exists():
        msg = f"Dataset not found at {dataset_path}"
        raise FileNotFoundError(msg)

    return pd.read_csv(dataset_path)


def main() -> None:
    df = load_dataset()
    print(df.head())


if __name__ == "__main__":
    main()

