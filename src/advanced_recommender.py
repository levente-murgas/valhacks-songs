"""
Advanced Music Recommender - Multiple Algorithms
Testing different approaches to beat 0.1563 NDCG@5
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import List, Set
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class PCARecommender:
    """KNN with PCA dimensionality reduction for denoising."""

    def __init__(self, n_components=50):
        self.df = None
        self.n_components = n_components
        self.pca = None
        self.knn = None
        self.features_pca = None

        self.audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'key', 'mode', 'time_signature'
        ]

    def _load_and_prepare(self):
        if self.df is not None:
            return

        print(f"Loading dataset for PCA recommender (n_components={self.n_components})...")
        self.df = pd.read_csv("data/dataset.csv", index_col=0)

        # Extract features
        feature_matrix = self.df[self.audio_features].fillna(0).values

        # Normalize
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(feature_matrix)

        # PCA for dimensionality reduction
        self.pca = PCA(n_components=self.n_components)
        self.features_pca = self.pca.fit_transform(features_normalized)

        # Build KNN on PCA features
        self.knn = NearestNeighbors(n_neighbors=min(1000, len(self.df)), metric='cosine', algorithm='brute')
        self.knn.fit(self.features_pca)

        # Prepare other fields
        self.df['genre_set'] = self.df['track_genre'].fillna('unknown').astype(str)
        self.df['popularity_norm'] = self.df['popularity'] / 100.0

        print(f"PCA explains {self.pca.explained_variance_ratio_.sum():.2%} of variance")

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        self._load_and_prepare()

        input_mask = self.df['track_id'].isin(input_track_ids)
        input_indices = self.df[input_mask].index.tolist()

        if len(input_indices) == 0:
            return self._fallback(target_artist, n_recommendations)

        # Average PCA features
        input_features = self.features_pca[input_indices].mean(axis=0, keepdims=True)
        input_genres = set(self.df.loc[input_mask, 'genre_set'].unique())

        # Find candidates
        distances, indices = self.knn.kneighbors(input_features, n_neighbors=min(2000, len(self.df)))

        scores = []
        for idx, dist in zip(indices[0], distances[0]):
            track_id = self.df.iloc[idx]['track_id']
            if track_id in input_track_ids:
                continue

            track_data = self.df.iloc[idx]
            artists_str = str(track_data['artists'])
            track_artists = [a.strip() for a in artists_str.split(';')]

            if not any(ta in track_artists for ta in target_artist):
                continue

            audio_sim = max(0, 1 - dist)
            genre_bonus = 0.15 if track_data['genre_set'] in input_genres else 0.0
            popularity = track_data['popularity_norm']

            # Score: 25% similarity + 65% popularity + 10% genre
            score = 0.25 * audio_sim + 0.65 * popularity + 0.10 * genre_bonus
            scores.append((track_id, score, track_data['artists']))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Diversity
        selected = []
        artist_counts = {}
        for track_id, score, artist_str in scores:
            if len(selected) >= n_recommendations:
                break
            artist = artist_str.split(';')[0].strip()
            if artist_counts.get(artist, 0) < 3:
                selected.append(track_id)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1

        return selected[:n_recommendations]

    def _fallback(self, target_artist: Set[str], n: int) -> List[str]:
        mask = self.df['artists'].apply(
            lambda x: any(ta in [a.strip() for a in str(x).split(';')] for ta in target_artist)
        )
        return self.df[mask].sort_values('popularity', ascending=False).head(n)['track_id'].tolist()


class PopularityFirstRecommender:
    """Pure popularity-first with light audio filtering."""

    def __init__(self):
        self.df = None
        self.knn = None
        self.features_normalized = None

        self.audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'key', 'mode', 'time_signature'
        ]

    def _load_and_prepare(self):
        if self.df is not None:
            return

        print("Loading dataset for popularity-first recommender...")
        self.df = pd.read_csv("data/dataset.csv", index_col=0)

        # Normalize features
        feature_matrix = self.df[self.audio_features].fillna(0).values
        scaler = StandardScaler()
        self.features_normalized = scaler.fit_transform(feature_matrix)

        # Build KNN
        self.knn = NearestNeighbors(n_neighbors=min(500, len(self.df)), metric='cosine', algorithm='brute')
        self.knn.fit(self.features_normalized)

        self.df['genre_set'] = self.df['track_genre'].fillna('unknown').astype(str)
        self.df['popularity_norm'] = self.df['popularity'] / 100.0

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        self._load_and_prepare()

        # Get all target artist tracks
        mask = self.df['artists'].apply(
            lambda x: any(ta in [a.strip() for a in str(x).split(';')] for ta in target_artist)
        )
        candidates = self.df[mask].copy()

        # Remove input tracks
        candidates = candidates[~candidates['track_id'].isin(input_track_ids)]

        if len(candidates) == 0:
            return []

        # Get input track features for similarity filtering
        input_mask = self.df['track_id'].isin(input_track_ids)
        input_indices = self.df[input_mask].index.tolist()

        if len(input_indices) > 0:
            input_features = self.features_normalized[input_indices].mean(axis=0, keepdims=True)
            input_genres = set(self.df.loc[input_mask, 'genre_set'].unique())

            # Calculate similarity for all candidates
            candidate_indices = candidates.index.tolist()
            candidate_features = self.features_normalized[candidate_indices]

            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(input_features, candidate_features)[0]

            candidates['audio_sim'] = similarities
            candidates['genre_match'] = candidates['genre_set'].apply(lambda x: 1.0 if x in input_genres else 0.0)
        else:
            candidates['audio_sim'] = 0.5
            candidates['genre_match'] = 0.0

        # Score: 90% popularity + 8% audio sim + 2% genre
        candidates['score'] = (
            0.90 * candidates['popularity_norm'] +
            0.08 * candidates['audio_sim'] +
            0.02 * candidates['genre_match']
        )

        # Sort by score
        candidates = candidates.sort_values('score', ascending=False)

        # Apply diversity
        selected = []
        artist_counts = {}

        for _, row in candidates.iterrows():
            if len(selected) >= n_recommendations:
                break

            artist = str(row['artists']).split(';')[0].strip()
            if artist_counts.get(artist, 0) < 3:
                selected.append(row['track_id'])
                artist_counts[artist] = artist_counts.get(artist, 0) + 1

        return selected[:n_recommendations]


class EnsembleRecommender:
    """Ensemble of multiple recommenders."""

    def __init__(self):
        self.recommenders = [
            ('popularity', PopularityFirstRecommender(), 0.7),
            ('pca8', PCARecommender(n_components=8), 0.15),
            ('pca10', PCARecommender(n_components=10), 0.15),
        ]

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        # Get recommendations from each
        all_recs = {}

        for name, rec, weight in self.recommenders:
            recs = rec.get_recommendations(input_track_ids, n_recommendations * 3, target_artist)
            for rank, track_id in enumerate(recs):
                score = weight * (1.0 / (rank + 1))  # Weighted reciprocal rank
                all_recs[track_id] = all_recs.get(track_id, 0) + score

        # Sort by combined score
        ranked = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)

        # Apply diversity
        selected = []
        artist_counts = {}

        # Need to load data to get artist info
        df = pd.read_csv("data/dataset.csv", index_col=0)

        for track_id, score in ranked:
            if len(selected) >= n_recommendations:
                break

            track_data = df[df['track_id'] == track_id]
            if len(track_data) == 0:
                continue

            artist = str(track_data.iloc[0]['artists']).split(';')[0].strip()
            if artist_counts.get(artist, 0) < 3:
                selected.append(track_id)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1

        return selected[:n_recommendations]


def test_recommender(name, recommender_class, **kwargs):
    """Test a recommender and return NDCG@5."""
    from evaluation.evaluation import evaluate

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    if kwargs:
        recommender = recommender_class(**kwargs)
    else:
        recommender = recommender_class()

    results = evaluate(recommender)

    print(f"\nNDCG@5: {results['NDCG@5']:.4f}")
    print(f"Performance: {results['Performance']:.2f}x")

    return results['NDCG@5']


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = {}

    # Test different approaches
    print("\n" + "="*60)
    print("TESTING ADVANCED ALGORITHMS")
    print("Baseline KNN: 0.1563 NDCG@5")
    print("="*60)

    # 1. Popularity-first
    results['Popularity-First'] = test_recommender('Popularity-First (90% pop)', PopularityFirstRecommender)

    # 2. PCA with different components (max 13 since we have 13 features)
    results['PCA-8'] = test_recommender('PCA-8 components', PCARecommender, n_components=8)
    results['PCA-10'] = test_recommender('PCA-10 components', PCARecommender, n_components=10)

    # 3. Ensemble
    results['Ensemble'] = test_recommender('Ensemble', EnsembleRecommender)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - NDCG@5 Results")
    print("="*60)
    print(f"{'Baseline KNN:':<30} 0.1563")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        improvement = ((score - 0.1563) / 0.1563) * 100
        print(f"{name:<30} {score:.4f} ({improvement:+.1f}%)")

    best = max(results.items(), key=lambda x: x[1])
    print(f"\n🏆 BEST: {best[0]} with {best[1]:.4f} NDCG@5")


if __name__ == "__main__":
    main()
