from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_normalized.csv"

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

Z_SCORE_FEATURES: Sequence[str] = tuple(f"{feature}_zscore" for feature in NUMERIC_FEATURES)


@dataclass(frozen=True)
class SeedProfile:
    present_track_ids: list[str]
    missing_track_ids: list[str]
    weighted_numeric_means: dict[str, float]
    weighted_zscore_centroid: dict[str, float]
    genre_distribution: dict[str, float]
    artist_set: set[str]
    explicit_rate: float
    major_mode_rate: float
    recent_track_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "present_track_ids": self.present_track_ids,
            "missing_track_ids": self.missing_track_ids,
            "weighted_numeric_means": self.weighted_numeric_means,
            "weighted_zscore_centroid": self.weighted_zscore_centroid,
            "genre_distribution": self.genre_distribution,
            "artist_list": sorted(self.artist_set),
            "explicit_rate": self.explicit_rate,
            "major_mode_rate": self.major_mode_rate,
            "recent_track_ids": self.recent_track_ids,
        }


def load_normalized_dataset(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        msg = (
            "Normalized dataset not found. "
            "Run `python src/data_ingestion.py` before building seed profiles."
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)
    return df


def _deduplicate_by_track_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="track_id", keep="first").set_index("track_id")


def _compute_weights(n_tracks: int) -> np.ndarray:
    if n_tracks == 0:
        return np.array([], dtype=float)

    # Exponentially increase the weight toward the most recent tracks
    weights = np.exp(np.linspace(-1.0, 0.0, n_tracks))
    normalized = weights / weights.sum()
    return normalized


def _extract_artist_set(series: pd.Series) -> set[str]:
    artists: set[str] = set()
    for entry in series.dropna():
        for artist in str(entry).split(";"):
            artist_name = artist.strip()
            if artist_name:
                artists.add(artist_name)
    return artists


def _compute_explicit_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0

    lower = series.astype(str).str.lower()
    explicit_match = lower.isin({"true", "1", "yes"})
    if explicit_match.any():
        return float(explicit_match.mean())

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float(numeric.mean())

    return 0.0


def _compute_major_mode_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float(numeric.mean())

    lower = series.astype(str).str.lower()
    if lower.isin({"1", "major"}).any():
        return float(lower.isin({"1", "major"}).mean())

    return 0.0


def build_seed_profile(
    track_ids: Sequence[str],
    dataset: pd.DataFrame | None = None,
) -> SeedProfile:
    if dataset is None:
        dataset = load_normalized_dataset()

    if "track_id" not in dataset.columns:
        raise KeyError("Dataset must contain a 'track_id' column.")

    deduped = _deduplicate_by_track_id(dataset)

    present_track_ids = [track_id for track_id in track_ids if track_id in deduped.index]
    missing_track_ids = [track_id for track_id in track_ids if track_id not in deduped.index]

    if not present_track_ids:
        msg = "None of the provided track IDs were found in the dataset."
        raise ValueError(msg)

    seed_df = deduped.loc[present_track_ids].copy()

    weights = _compute_weights(len(seed_df))
    weight_matrix = np.expand_dims(weights, axis=1)

    numeric_matrix = seed_df[list(NUMERIC_FEATURES)].to_numpy(dtype=float)
    weighted_numeric = (numeric_matrix * weight_matrix).sum(axis=0)
    weighted_numeric_means = dict(zip(NUMERIC_FEATURES, weighted_numeric, strict=True))

    zscore_matrix = seed_df[list(Z_SCORE_FEATURES)].to_numpy(dtype=float)
    weighted_zscores = (zscore_matrix * weight_matrix).sum(axis=0)
    weighted_zscore_centroid = dict(zip(Z_SCORE_FEATURES, weighted_zscores, strict=True))

    genre_distribution = (
        seed_df["track_genre"]
        .dropna()
        .value_counts(normalize=True)
        .sort_values(ascending=False)
        .to_dict()
    )

    artist_set = _extract_artist_set(seed_df["artists"])
    explicit_rate = _compute_explicit_rate(seed_df["explicit"])
    major_mode_rate = _compute_major_mode_rate(seed_df["mode"])

    recent_track_ids = present_track_ids[-3:] if len(present_track_ids) >= 3 else present_track_ids

    return SeedProfile(
        present_track_ids=present_track_ids,
        missing_track_ids=missing_track_ids,
        weighted_numeric_means=weighted_numeric_means,
        weighted_zscore_centroid=weighted_zscore_centroid,
        genre_distribution=genre_distribution,
        artist_set=artist_set,
        explicit_rate=explicit_rate,
        major_mode_rate=major_mode_rate,
        recent_track_ids=recent_track_ids,
    )

def _cli(track_ids: Sequence[str] | None = None) -> None:
    dataset = load_normalized_dataset()
    if not track_ids:
        track_ids = dataset["track_id"].head(5).tolist()

    seed_profile = build_seed_profile(track_ids, dataset=dataset)
    print(json.dumps(seed_profile.to_dict(), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a seed playlist profile from track IDs.")
    parser.add_argument(
        "--track-ids",
        nargs="+",
        help="Spotify track IDs representing the seed playlist (in playlist order).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _cli(args.track_ids)


if __name__ == "__main__":
    main()

