from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from candidate_generation import CandidateIndex, generate_candidate_track_ids, load_candidate_index
from seed_profile import SeedProfile, build_seed_profile, load_normalized_dataset

logger = logging.getLogger(__name__)

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


@dataclass
class CandidateFeatures:
    track_id: str
    base_similarity: float
    short_term_similarity: float
    genre_weight: float
    artist_overlap: float
    target_artist_boost: float
    explicit_penalty: float
    mode_penalty: float
    novelty_bonus: float
    variance_penalty: float
    final_score: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "track_id": self.track_id,
            "base_similarity": self.base_similarity,
            "short_term_similarity": self.short_term_similarity,
            "genre_weight": self.genre_weight,
            "artist_overlap": self.artist_overlap,
            "target_artist_boost": self.target_artist_boost,
            "explicit_penalty": self.explicit_penalty,
            "mode_penalty": self.mode_penalty,
            "novelty_bonus": self.novelty_bonus,
            "variance_penalty": self.variance_penalty,
            "final_score": self.final_score,
        }


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def compute_variance_tolerance(seed: SeedProfile, dataset: pd.DataFrame) -> dict[str, float]:
    deduped = dataset.drop_duplicates(subset="track_id", keep="first").set_index("track_id")
    seed_tracks = deduped.reindex(seed.present_track_ids).dropna(how="all")
    tolerances: dict[str, float] = {}
    for feature in NUMERIC_FEATURES:
        std = float(seed_tracks[feature].std(ddof=0))
        if np.isnan(std) or std == 0:
            std = 0.01
        tolerances[feature] = std
    return tolerances


def compute_candidate_features(
    candidate_ids: Sequence[str],
    seed: SeedProfile,
    dataset: pd.DataFrame,
    target_artist: set[str],
) -> list[CandidateFeatures]:
    deduped = dataset.drop_duplicates(subset="track_id", keep="first").set_index("track_id")
    seed_centroid = np.array(
        [seed.weighted_zscore_centroid.get(feature, 0.0) for feature in Z_SCORE_FEATURES],
        dtype=float,
    )

    recent_ids = seed.recent_track_ids[-3:]
    recent_vectors = []
    for track_id in recent_ids:
        if track_id in deduped.index:
            vector = deduped.loc[track_id, list(Z_SCORE_FEATURES)].to_numpy(dtype=float)
            recent_vectors.append(vector)
    recent_weights = np.array([0.5, 0.3, 0.2], dtype=float)[: len(recent_vectors)]
    if recent_weights.sum() > 0:
        recent_weights /= recent_weights.sum()

    tolerances = compute_variance_tolerance(seed, dataset)

    global_popularity_mean = float(dataset["popularity"].mean())
    global_popularity_std = float(dataset["popularity"].std(ddof=0))

    results: list[CandidateFeatures] = []

    for track_id in candidate_ids:
        if track_id not in deduped.index:
            continue
        row = deduped.loc[track_id]

        candidate_vector = row[list(Z_SCORE_FEATURES)].to_numpy(dtype=float)
        base_similarity = cosine_similarity(seed_centroid, candidate_vector)

        short_term_similarity = 0.0
        if recent_vectors:
            similarities = [
                cosine_similarity(candidate_vector, recent_vector) for recent_vector in recent_vectors
            ]
            short_term_similarity = float(np.dot(similarities, recent_weights))

        genre_weight = seed.genre_distribution.get(row["track_genre"], 0.0)
        candidate_artists = _split_artists(row["artists"])
        artist_overlap = 1.0 if seed.artist_set.intersection(candidate_artists) else 0.0
        target_artist_boost = 1.0 if target_artist and target_artist.intersection(candidate_artists) else 0.0

        explicit_penalty = 0.0
        if seed.explicit_rate < 0.2:
            explicit_penalty = 0.2 if str(row["explicit"]).lower() in {"true", "1"} else 0.0

        mode_penalty = 0.0
        seed_major = seed.major_mode_rate >= 0.5
        candidate_major = _is_major_mode(row["mode"])
        if seed_major and not candidate_major:
            mode_penalty = 0.1
        elif not seed_major and candidate_major:
            mode_penalty = 0.05

        popularity = float(row["popularity"])
        if global_popularity_std == 0:
            novelty_bonus = 0.0
        else:
            novelty_bonus = float((popularity - global_popularity_mean) / global_popularity_std)
            novelty_bonus = -0.05 * novelty_bonus

        variance_penalty = 0.0
        for feature in ["tempo", "energy", "valence"]:
            tolerance = max(tolerances.get(feature, 0.1), 0.05)
            diff = abs(float(row[feature]) - seed.weighted_numeric_means.get(feature, 0.0))
            variance_penalty += max(0.0, diff - 2 * tolerance) * 0.01

        final_score = (
            0.45 * base_similarity
            + 0.35 * short_term_similarity
            + 0.15 * genre_weight
            + 0.1 * artist_overlap
            + 0.2 * target_artist_boost
            + novelty_bonus
            - explicit_penalty
            - mode_penalty
            - variance_penalty
        )

        features = CandidateFeatures(
            track_id=track_id,
            base_similarity=base_similarity,
            short_term_similarity=short_term_similarity,
            genre_weight=genre_weight,
            artist_overlap=artist_overlap,
            target_artist_boost=target_artist_boost,
            explicit_penalty=explicit_penalty,
            mode_penalty=mode_penalty,
            novelty_bonus=novelty_bonus,
            variance_penalty=variance_penalty,
            final_score=final_score,
        )
        results.append(features)

    return results


def _split_artists(artists_str: str) -> set[str]:
    if not isinstance(artists_str, str):
        return set()
    return {artist.strip() for artist in artists_str.split(";") if artist.strip()}


def _is_major_mode(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return int(value) == 1
    text = str(value).strip().lower()
    if not text:
        return False
    if text in {"1", "true", "major"}:
        return True
    if text in {"0", "false", "minor"}:
        return False
    try:
        return int(float(text)) == 1
    except ValueError:
        return False


def rank_candidates(
    seed: SeedProfile,
    candidate_index: CandidateIndex,
    dataset: pd.DataFrame,
    target_artist: set[str],
    n_recommendations: int,
    exclude_track_ids: Iterable[str] | None = None,
) -> list[CandidateFeatures]:
    deduped = dataset.drop_duplicates(subset="track_id", keep="first").set_index("track_id")
    candidate_ids = generate_candidate_track_ids(
        seed=seed,
        index=candidate_index,
        n_candidates=500,
        exclude_track_ids=exclude_track_ids,
    )
    features = compute_candidate_features(candidate_ids, seed, dataset, target_artist)

    features.sort(key=lambda item: item.final_score, reverse=True)

    ranked: list[CandidateFeatures] = []
    artist_counts: dict[str, int] = {}

    for feature in features:
        if feature.track_id not in deduped.index:
            continue
        row = deduped.loc[feature.track_id]
        artists = _split_artists(row["artists"])
        is_target_artist = bool(target_artist.intersection(artists))

        if not is_target_artist:
            exceeded = any(artist_counts.get(artist, 0) >= 1 for artist in artists)
            if exceeded:
                continue

        ranked.append(feature)
        for artist in artists:
            artist_counts[artist] = artist_counts.get(artist, 0) + 1

        if len(ranked) >= n_recommendations:
            break

    if len(ranked) < n_recommendations:
        logger.debug(
            "Not enough unique-artist candidates; backfilling to reach %d recommendations.",
            n_recommendations,
        )
        for feature in features:
            if feature in ranked:
                continue
            ranked.append(feature)
            if len(ranked) >= n_recommendations:
                break

    return ranked[:n_recommendations]


def recommend_tracks(
    input_track_ids: Sequence[str],
    n_recommendations: int,
    target_artist: set[str] | None = None,
    dataset_path: Path = PROCESSED_DATA_PATH,
) -> tuple[list[str], list[dict[str, float | str]]]:
    return recommend_tracks_with_resources(
        input_track_ids=input_track_ids,
        n_recommendations=n_recommendations,
        target_artist=target_artist,
        dataset_path=dataset_path,
    )


def recommend_tracks_with_resources(
    input_track_ids: Sequence[str],
    n_recommendations: int,
    target_artist: set[str] | None = None,
    dataset_path: Path = PROCESSED_DATA_PATH,
    dataset: pd.DataFrame | None = None,
    candidate_index: CandidateIndex | None = None,
) -> tuple[list[str], list[dict[str, float | str]]]:
    if dataset is None:
        dataset = load_normalized_dataset(dataset_path)
    seed = build_seed_profile(input_track_ids, dataset=dataset)
    try:
        candidate_index = candidate_index or load_candidate_index()
    except FileNotFoundError:
        candidate_index = generate_and_store_index(dataset_path)

    target_artist = target_artist or set()

    ranked_features = rank_candidates(
        seed=seed,
        candidate_index=candidate_index,
        dataset=dataset,
        target_artist=target_artist,
        n_recommendations=n_recommendations,
        exclude_track_ids=input_track_ids,
    )

    explanations = []
    for feature in ranked_features:
        explanations.append(feature.to_dict())
        logger.debug("Recommendation %s details: %s", feature.track_id, feature.to_dict())

    recommended_track_ids = [feature.track_id for feature in ranked_features]
    return recommended_track_ids, explanations


def generate_and_store_index(dataset_path: Path = PROCESSED_DATA_PATH) -> CandidateIndex:
    from candidate_generation import build_candidate_index, persist_candidate_index

    index = build_candidate_index(dataset_path)
    persist_candidate_index(index)
    return index

