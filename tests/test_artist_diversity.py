from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import ranking


def _make_feature(track_id: str, score: float) -> ranking.CandidateFeatures:
    return ranking.CandidateFeatures(
        track_id=track_id,
        base_similarity=score,
        short_term_similarity=0.0,
        genre_weight=0.0,
        artist_overlap=0.0,
        target_artist_boost=0.0,
        explicit_penalty=0.0,
        mode_penalty=0.0,
        novelty_bonus=0.0,
        variance_penalty=0.0,
        final_score=score,
        artist_diversity_penalty=0.0,
        adjusted_score=score,
    )


class ArtistDiversityTests(unittest.TestCase):
    def setUp(self) -> None:
        data = [
            {"track_id": "seed_a1", "track_name": "Seed A1", "artists": "Artist A", "track_genre": "rock"},
            {"track_id": "seed_a2", "track_name": "Seed A2", "artists": "Artist A", "track_genre": "rock"},
            {"track_id": "seed_a3", "track_name": "Seed A3", "artists": "Artist A", "track_genre": "rock"},
            {"track_id": "seed_b1", "track_name": "Seed B1", "artists": "Artist B", "track_genre": "rock"},
            {"track_id": "seed_b2", "track_name": "Seed B2", "artists": "Artist B", "track_genre": "rock"},
            {"track_id": "seed_c1", "track_name": "Seed C1", "artists": "Artist C", "track_genre": "rock"},
            {"track_id": "track_a2", "track_name": "Track A2", "artists": "Artist A", "track_genre": "rock"},
            {"track_id": "track_a3", "track_name": "Track A3", "artists": "Artist A", "track_genre": "rock"},
            {"track_id": "track_b1", "track_name": "Track B1", "artists": "Artist B", "track_genre": "rock"},
            {
                "track_id": "track_a_feat_c",
                "track_name": "Track A Feat C",
                "artists": "Artist A; Artist C",
                "track_genre": "rock",
            },
        ]
        self.dataset = pd.DataFrame(data)
        self.deduped = self.dataset.drop_duplicates(subset="track_id", keep="first").set_index("track_id")

    def test_penalty_discourages_recent_repeat_when_alternatives_exist(self) -> None:
        playlist_context = ranking._build_playlist_artist_context(["seed_a1"], self.deduped)
        features = [
            _make_feature("track_a2", 0.9),
            _make_feature("track_b1", 0.88),
        ]

        ranking._apply_artist_diversity_adjustments(features, self.deduped, playlist_context, target_artist=set())

        scores = {feature.track_id: feature.adjusted_score for feature in features}
        self.assertLess(scores["track_a2"], scores["track_b1"])
        self.assertGreaterEqual(features[0].artist_diversity_penalty, 0.0)

    def test_penalty_relaxes_when_only_single_artist_available(self) -> None:
        playlist_context = ranking._build_playlist_artist_context(["seed_a1"], self.deduped)
        features = [_make_feature("track_a2", 0.9)]

        ranking._apply_artist_diversity_adjustments(features, self.deduped, playlist_context, target_artist=set())

        self.assertAlmostEqual(features[0].artist_diversity_penalty, 0.0, places=6)
        self.assertAlmostEqual(features[0].adjusted_score, features[0].final_score, places=6)

    def test_reordering_breaks_adjacent_artist_duplicates(self) -> None:
        last_artists = {"Artist A"}
        features = [
            _make_feature("track_a2", 0.9),
            _make_feature("track_b1", 0.88),
            _make_feature("track_a3", 0.87),
        ]

        reordered = ranking._avoid_adjacent_artist_repeats(features, self.deduped, last_artists)
        self.assertTrue(reordered)
        first_track = reordered[0].track_id
        first_artists = ranking._split_artists(self.deduped.loc[first_track]["artists"])
        self.assertFalse(first_artists.intersection(last_artists))

    def test_collaboration_penalty_is_lower_than_single_artist_repeat(self) -> None:
        playlist_context = ranking._build_playlist_artist_context(
            ["seed_a1", "seed_a2", "seed_a3", "seed_b1", "seed_c1"],
            self.deduped,
        )
        features = [
            _make_feature("track_a2", 0.9),
            _make_feature("track_a_feat_c", 0.89),
        ]

        ranking._apply_artist_diversity_adjustments(features, self.deduped, playlist_context, target_artist=set())

        penalties = {feature.track_id: feature.artist_diversity_penalty for feature in features}
        self.assertLess(penalties["track_a_feat_c"], penalties["track_a2"])
        self.assertLessEqual(penalties["track_a_feat_c"], 0.1)

    def test_target_weighting_makes_target_candidates_dominate(self) -> None:
        playlist_context = ranking._build_playlist_artist_context(
            ["seed_b1", "seed_c1", "seed_b2"],
            self.deduped,
        )
        features = [
            _make_feature("track_b1", 0.84),
            _make_feature("track_a2", 0.81),
        ]

        ranking._apply_target_artist_weighting(features, self.deduped, playlist_context, target_artist={"Artist A"})
        ranking._apply_artist_diversity_adjustments(features, self.deduped, playlist_context, target_artist={"Artist A"})

        adjustments = {feature.track_id: feature.adjusted_score for feature in features}
        self.assertGreater(adjustments["track_a2"], adjustments["track_b1"])
        self.assertGreater(features[1].final_score, 0.81)


if __name__ == "__main__":
    unittest.main()

