from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "figures"

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

CATEGORICAL_FEATURES: Sequence[str] = (
    "explicit",
    "mode",
    "time_signature",
    "track_genre",
)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        msg = f"Dataset not found at {DATA_PATH}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    numeric_df = df[list(NUMERIC_FEATURES)].copy()

    # Convert duration to minutes for readability
    numeric_df["duration_minutes"] = numeric_df["duration_ms"] / 60000
    numeric_df = numeric_df.drop(columns=["duration_ms"])

    features = numeric_df.columns.tolist()
    n_features = len(features)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.histplot(
            numeric_df[feature].dropna(),
            ax=ax,
            kde=True,
            bins=40,
            color="#1db954",
        )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("")

    # hide unused axes
    for ax in axes[n_features:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "numeric_distributions.png", dpi=150)
    plt.close(fig)


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(data=df, x="explicit", ax=axes[0], color="#1db954")
    axes[0].set_title("Explicit Flag Distribution")

    sns.countplot(data=df, x="mode", ax=axes[1], color="#1db954")
    axes[1].set_title("Mode Distribution")

    sns.countplot(data=df, x="time_signature", ax=axes[2], color="#1db954")
    axes[2].set_title("Time Signature Distribution")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "categorical_distributions.png", dpi=150)
    plt.close(fig)

    top_genres = (
        df["track_genre"]
        .value_counts()
        .head(20)
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x=top_genres.values,
        y=top_genres.index,
        color="#1db954",
        orient="h",
    )
    plt.title("Top 20 Track Genres")
    plt.xlabel("Count")
    plt.ylabel("Track Genre")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "top_track_genres.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_df = df[list(NUMERIC_FEATURES)]
    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_correlation_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = load_dataset()
    plot_numeric_distributions(df)
    plot_categorical_distributions(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()

