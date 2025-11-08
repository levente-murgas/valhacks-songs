from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from seed_profile import SeedProfile, build_seed_profile, load_normalized_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_normalized.csv"
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_PATH = MODELS_DIR / "candidate_index.joblib"

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
class CandidateIndex:
    model: NearestNeighbors
    track_ids: np.ndarray
    feature_matrix: np.ndarray


def _deduplicate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="track_id", keep="first").set_index("track_id")


def load_feature_matrix(path: Path = PROCESSED_DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    df = load_normalized_dataset(path)
    deduped = _deduplicate_tracks(df)
    track_ids = deduped.index.to_numpy()
    feature_matrix = deduped[list(Z_SCORE_FEATURES)].to_numpy(dtype=np.float32)
    return track_ids, feature_matrix


def fit_candidate_index(feature_matrix: np.ndarray, n_trees: int | None = None) -> NearestNeighbors:
    # Using brute-force cosine distance keeps behaviour deterministic for evaluation-scale data
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(feature_matrix)
    return model


def build_candidate_index(path: Path = PROCESSED_DATA_PATH) -> CandidateIndex:
    track_ids, feature_matrix = load_feature_matrix(path)
    model = fit_candidate_index(feature_matrix)
    return CandidateIndex(model=model, track_ids=track_ids, feature_matrix=feature_matrix)


def persist_candidate_index(index: CandidateIndex, destination: Path = INDEX_PATH) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": index.model,
            "track_ids": index.track_ids,
            "feature_matrix": index.feature_matrix,
        },
        destination,
    )


def load_candidate_index(source: Path = INDEX_PATH) -> CandidateIndex:
    if not source.exists():
        msg = "Candidate index not found. Run `python src/candidate_generation.py --build-index` first."
        raise FileNotFoundError(msg)

    payload = joblib.load(source)
    return CandidateIndex(
        model=payload["model"],
        track_ids=payload["track_ids"],
        feature_matrix=payload["feature_matrix"],
    )


def candidate_query_vector(seed: SeedProfile) -> np.ndarray:
    values = [seed.weighted_zscore_centroid.get(feature, 0.0) for feature in Z_SCORE_FEATURES]
    vector = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return vector


def generate_candidate_track_ids(
    seed: SeedProfile,
    index: CandidateIndex,
    n_candidates: int = 300,
    exclude_track_ids: Iterable[str] | None = None,
) -> list[str]:
    query_vec = candidate_query_vector(seed)
    distances, indices = index.model.kneighbors(query_vec, n_neighbors=min(n_candidates * 2, len(index.track_ids)))

    exclude_set = set(seed.present_track_ids)
    if exclude_track_ids:
        exclude_set.update(exclude_track_ids)

    ranked_candidates: list[str] = []
    for idx in indices[0]:
        track_id = str(index.track_ids[idx])
        if track_id in exclude_set:
            continue
        ranked_candidates.append(track_id)
        if len(ranked_candidates) >= n_candidates:
            break

    return ranked_candidates


def _cli_build_index() -> None:
    index = build_candidate_index()
    persist_candidate_index(index)
    print(f"Candidate index built and saved to {INDEX_PATH}")


def _cli_generate_candidates(track_ids: Sequence[str] | None) -> None:
    dataset = load_normalized_dataset()
    if not track_ids:
        track_ids = dataset["track_id"].head(5).tolist()

    seed = build_seed_profile(track_ids, dataset=dataset)

    try:
        index = load_candidate_index()
    except FileNotFoundError:
        index = build_candidate_index()
        persist_candidate_index(index)

    candidates = generate_candidate_track_ids(seed, index)
    result = {
        "seed_track_ids": seed.present_track_ids,
        "missing_track_ids": seed.missing_track_ids,
        "candidate_track_ids": candidates[:50],
    }
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate generation utilities.")
    parser.add_argument("--build-index", action="store_true", help="Fit and persist the nearest-neighbor index.")
    parser.add_argument(
        "--track-ids",
        nargs="+",
        help="Track IDs to seed candidate generation (falls back to head of dataset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_index:
        _cli_build_index()
    else:
        _cli_generate_candidates(args.track_ids)


if __name__ == "__main__":
    main()

