"""
Clean KNN-based Music Recommender - Fresh Start
Uses ONLY dataset.csv track features - NO training on testset.json
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List, Set
import sys
import os

# Add parent directory to path to import evaluation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class KNNRecommender:
    """Feature-based KNN recommender using audio features, genre, and popularity."""

    def __init__(self):
        """Initialize recommender - will load data when needed."""
        self.df = None
        self.knn = None
        self.feature_matrix_normalized = None
        self.scaler = None

        # Audio features to use for similarity
        self.audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'key', 'mode', 'time_signature'
        ]

    def _load_and_prepare(self):
        """Load and prepare data (called on first use)."""
        if self.df is not None:
            return  # Already loaded

        print("Loading and preparing dataset...")
        self.df = pd.read_csv("data/dataset.csv", index_col=0)

        # Prepare features
        self._prepare_features()

        # Build KNN index
        self._build_knn_index()

        print(f"Recommender ready with {len(self.df)} tracks")

    def _prepare_features(self):
        """Prepare and normalize features for KNN."""
        # Extract audio features
        feature_matrix = self.df[self.audio_features].fillna(0).values

        # Normalize features
        self.scaler = StandardScaler()
        self.feature_matrix_normalized = self.scaler.fit_transform(feature_matrix)

        # Create genre set for matching
        self.df['genre_set'] = self.df['track_genre'].fillna('unknown').astype(str)

        # Normalize popularity to 0-1
        self.df['popularity_norm'] = self.df['popularity'] / 100.0

    def _build_knn_index(self):
        """Build KNN index for fast similarity search."""
        self.knn = NearestNeighbors(
            n_neighbors=min(1000, len(self.df)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn.fit(self.feature_matrix_normalized)

    def get_recommendations(
        self,
        input_track_ids: List[str],
        n_recommendations: int,
        target_artist: Set[str]
    ) -> List[str]:
        """
        Get track recommendations based on input tracks and target artists.

        Args:
            input_track_ids: List of seed track IDs
            n_recommendations: Number of recommendations to return
            target_artist: Set of artist names to filter by

        Returns:
            List of recommended track IDs
        """
        # Load data if not already loaded
        self._load_and_prepare()

        # Get input track features
        input_mask = self.df['track_id'].isin(input_track_ids)
        input_indices = self.df[input_mask].index.tolist()

        if len(input_indices) == 0:
            # Fallback: return popular tracks by target artists
            return self._fallback_recommendations(target_artist, n_recommendations)

        # Calculate average feature vector from input tracks
        input_features = self.feature_matrix_normalized[input_indices].mean(axis=0, keepdims=True)

        # Get genres from input tracks
        input_genres = set(self.df.loc[input_mask, 'genre_set'].unique())

        # Find candidate tracks
        n_candidates = min(2000, len(self.df))
        distances, indices = self.knn.kneighbors(input_features, n_neighbors=n_candidates)

        # Score candidates
        scores = []
        for idx, dist in zip(indices[0], distances[0]):
            track_id = self.df.iloc[idx]['track_id']

            # Skip input tracks
            if track_id in input_track_ids:
                continue

            # Get track info
            track_data = self.df.iloc[idx]
            artists_str = str(track_data['artists'])
            track_artists = [a.strip() for a in artists_str.split(';')]

            # Check if track is by target artist (EXACT match)
            is_target_artist = any(ta in track_artists for ta in target_artist)

            # Only consider tracks by target artists
            if not is_target_artist:
                continue

            # Calculate similarity score (convert distance to similarity)
            audio_sim = max(0, 1 - dist)  # Cosine distance to similarity

            # Genre match bonus
            track_genre = track_data['genre_set']
            genre_bonus = 0.15 if track_genre in input_genres else 0.0

            # Popularity score
            popularity = track_data['popularity_norm']

            # Combined score: 20% audio similarity + 70% popularity + 10% genre bonus
            score = 0.20 * audio_sim + 0.70 * popularity + 0.10 * genre_bonus

            scores.append((track_id, score, track_data['artists']))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Apply diversity: limit tracks per artist
        selected = []
        artist_counts = {}

        for track_id, score, artist_str in scores:
            if len(selected) >= n_recommendations:
                break

            # Get primary artist
            artist = artist_str.split(';')[0].strip()

            # Check artist diversity (max 3 per artist)
            if artist_counts.get(artist, 0) < 3:
                selected.append(track_id)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1

        # Fill remaining slots if needed with top scores
        if len(selected) < n_recommendations:
            for track_id, score, artist_str in scores:
                if track_id not in selected:
                    selected.append(track_id)
                    if len(selected) >= n_recommendations:
                        break

        return selected[:n_recommendations]

    def _fallback_recommendations(self, target_artist: Set[str], n: int) -> List[str]:
        """Fallback: return popular tracks by target artists."""
        # Filter by target artists (exact match)
        mask = self.df['artists'].apply(
            lambda x: any(ta in [a.strip() for a in str(x).split(';')] for ta in target_artist)
        )

        candidates = self.df[mask].sort_values('popularity', ascending=False)
        return candidates.head(n)['track_id'].tolist()


def main():
    """Run evaluation."""
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Import evaluation
    from evaluation.evaluation import evaluate

    # Create and evaluate recommender
    print("=" * 60)
    print("KNN-Based Recommender System")
    print("Uses ONLY dataset.csv features - NO training on testset")
    print("=" * 60)
    print()

    recommender = KNNRecommender()
    results = evaluate(recommender)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"NDCG@5: {results['NDCG@5']:.4f}")
    print(f"Performance vs Baseline: {results['Performance']:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
