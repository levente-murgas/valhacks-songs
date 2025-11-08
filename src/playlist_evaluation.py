from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from data.recommender import Recommender
from seed_profile import load_normalized_dataset

logger = logging.getLogger(__name__)


@dataclass
class PlaylistEvaluationResult:
    playlist_id: str
    seed_size: int
    holdout_size: int
    hit_rate: float
    average_precision: float
    recommended_track_ids: list[str]
    target_track_ids: list[str]
    explanations: list[dict[str, float | str]]


def compute_hit_rate(recommended: Sequence[str], targets: Sequence[str]) -> float:
    if not targets:
        return 0.0
    target_set = set(targets)
    hits = sum(1 for track_id in recommended if track_id in target_set)
    return hits / len(targets)


def compute_average_precision(recommended: Sequence[str], targets: Sequence[str]) -> float:
    if not targets:
        return 0.0

    target_set = set(targets)
    score = 0.0
    hits = 0

    for idx, track_id in enumerate(recommended, start=1):
        if track_id in target_set:
            hits += 1
            score += hits / idx

    return score / len(targets)


def get_target_artist_hint(track_ids: Iterable[str], track_metadata: pd.DataFrame) -> set[str]:
    hint: set[str] = set()
    for track_id in track_ids:
        if track_id not in track_metadata.index:
            continue
        artists_value = track_metadata.at[track_id, "artists"]
        if isinstance(artists_value, str):
            for artist in artists_value.split(";"):
                artist_name = artist.strip()
                if artist_name:
                    hint.add(artist_name)
    return hint


def evaluate_playlists(
    playlists: pd.DataFrame,
    recommender: Recommender | None = None,
    *,
    playlist_col: str = "playlist_id",
    track_col: str = "track_id",
    position_col: str | None = "position",
    holdout_size: int = 5,
    n_recommendations: int = 5,
    dataset_path: Path | None = None,
) -> list[PlaylistEvaluationResult]:
    """
    Evaluate recommender performance on real playlists by holding out the last N tracks.

    Args:
        playlists: DataFrame containing at least playlist and track identifier columns.
        recommender: Optional shared recommender instance. If omitted, a new one is constructed.
        playlist_col: Column representing playlist identifiers.
        track_col: Column representing track identifiers.
        position_col: Column determining order within the playlist (ascending). If None, existing order is used.
        holdout_size: Number of final tracks to remove per playlist for evaluation.
        n_recommendations: Number of tracks to recommend.
        dataset_path: Optional custom path to normalized dataset.

    Returns:
        A list of per-playlist evaluation results, including recommendation explanations.
    """
    if recommender is None:
        recommender = Recommender()

    dataset = load_normalized_dataset(dataset_path) if dataset_path else load_normalized_dataset()
    track_metadata = dataset.drop_duplicates(subset="track_id", keep="first").set_index("track_id")

    results: list[PlaylistEvaluationResult] = []
    skipped: list[str] = []

    for playlist_id, group in playlists.groupby(playlist_col):
        ordered_group = group
        if position_col and position_col in group.columns:
            ordered_group = group.sort_values(position_col, ascending=True)

        track_ids = ordered_group[track_col].dropna().astype(str).tolist()
        if len(track_ids) <= holdout_size:
            skipped.append(str(playlist_id))
            continue

        seed_track_ids = track_ids[:-holdout_size]
        target_track_ids = track_ids[-holdout_size:]

        target_artist_hint = get_target_artist_hint(target_track_ids, track_metadata)

        try:
            recommended_ids, explanations = recommender.get_recommendations_with_details(
                input_track_ids=seed_track_ids,
                n_recommendations=n_recommendations,
                target_artist=target_artist_hint,
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            logger.exception("Failed to evaluate playlist %s: %s", playlist_id, exc)
            continue

        hit_rate = compute_hit_rate(recommended_ids, target_track_ids)
        avg_precision = compute_average_precision(recommended_ids, target_track_ids)

        results.append(
            PlaylistEvaluationResult(
                playlist_id=str(playlist_id),
                seed_size=len(seed_track_ids),
                holdout_size=len(target_track_ids),
                hit_rate=hit_rate,
                average_precision=avg_precision,
                recommended_track_ids=recommended_ids,
                target_track_ids=target_track_ids,
                explanations=explanations,
            )
        )

        logger.info(
            "Playlist %s — Hit@%d: %.2f, MAP@%d: %.2f",
            playlist_id,
            n_recommendations,
            hit_rate,
            n_recommendations,
            avg_precision,
        )

    if skipped:
        logger.warning("Skipped %d playlists (too short): %s", len(skipped), ", ".join(skipped[:5]))

    return results


def summarise_results(results: Sequence[PlaylistEvaluationResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["metric", "value"])

    summary = {
        "playlist_count": len(results),
        "mean_hit_rate": float(pd.Series(result.hit_rate for result in results).mean()),
        "mean_map": float(pd.Series(result.average_precision for result in results).mean()),
        "median_hit_rate": float(pd.Series(result.hit_rate for result in results).median()),
        "median_map": float(pd.Series(result.average_precision for result in results).median()),
    }
    return pd.DataFrame(list(summary.items()), columns=["metric", "value"])

