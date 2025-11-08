from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd

try:
    from evaluation import evaluate
except ImportError:  # pragma: no cover - optional external dependency
    evaluate = None  # type: ignore[assignment]

from ranking import CandidateIndex, recommend_tracks_with_resources
from seed_profile import load_normalized_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Recommender:

    def __init__(self) -> None:
        self._dataset: Optional[pd.DataFrame] = None
        self._candidate_index: Optional[CandidateIndex] = None

    def _ensure_resources(self) -> None:
        if self._dataset is None:
            logger.info("Loading normalized dataset for recommender.")
            self._dataset = load_normalized_dataset()
        if self._candidate_index is None:
            logger.info("Loading candidate index for recommender.")
            from candidate_generation import build_candidate_index, load_candidate_index, persist_candidate_index

            try:
                self._candidate_index = load_candidate_index()
            except FileNotFoundError:
                logger.info("Candidate index not found. Building a new one.")
                self._candidate_index = build_candidate_index()
                persist_candidate_index(self._candidate_index)

    def get_recommendations(
        self,
        input_track_ids: Sequence[str],
        n_recommendations: int,
        target_artist: set[str],
    ) -> list[str]:
        """
        Generate playlist continuation recommendations using seed-based heuristics.
        """
        recommendations, _ = self.get_recommendations_with_details(
            input_track_ids=input_track_ids,
            n_recommendations=n_recommendations,
            target_artist=target_artist,
        )
        return recommendations

    def get_recommendations_with_details(
        self,
        input_track_ids: Sequence[str],
        n_recommendations: int,
        target_artist: set[str],
    ) -> tuple[list[str], list[dict[str, float | str]]]:
        """
        Return recommendations along with detailed scoring contributions.
        """
        if not input_track_ids:
            raise ValueError("At least one input track ID is required.")
        if n_recommendations <= 0:
            raise ValueError("Number of recommendations must be positive.")

        self._ensure_resources()
        assert self._dataset is not None
        assert self._candidate_index is not None

        recommended_ids, explanations = recommend_tracks_with_resources(
            input_track_ids=list(input_track_ids),
            n_recommendations=n_recommendations,
            target_artist=target_artist,
            dataset=self._dataset,
            candidate_index=self._candidate_index,
        )

        for idx, info in enumerate(explanations, start=1):
            logger.info("Rec %d: %s", idx, info)

        return recommended_ids, explanations


if __name__ == "__main__":
    if evaluate is not None:
        recommender = Recommender()
        results = evaluate(recommender)
        print(results)
    else:
        logger.warning("evaluation package not available; skipping automatic evaluation run.")