from __future__ import annotations

from typing import Iterable, Sequence

from sklearn.neighbors import NearestNeighbors

from typing import List, Set, Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    # Load the dataset
    df = pd.read_csv("dataset.csv", index_col=0)
    df.drop_duplicates(subset=['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                               'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                               'duration_ms', 'popularity',
                               'artists', 'track_name', 'time_signature'],
                       inplace=True)  # There are duplicates that have different track_id, genre and album. There are duplicates in other dimensions (eg. popularity and duration) but these are taken

    # One-hot encode genres
    genre_dummies = pd.get_dummies(df['track_genre'], prefix='genre')
    time_signature_dummies = pd.get_dummies(df['time_signature'], prefix='time_signature')

    # Combine with original dataframe
    df_with_genres = pd.concat([df, genre_dummies, time_signature_dummies], axis=1)

    # Define columns to aggregate
    group_cols = [
                     'track_id', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                     'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
                     'popularity',
                     'artists', 'album_name', 'track_name'
                 ] + list(time_signature_dummies.columns)

    # Aggregation dictionary
    agg_dict = {col: 'max' for col in genre_dummies.columns}
    for col in group_cols:
        agg_dict[col] = 'first'

    # Merge duplicates
    df_merged = df_with_genres.groupby('track_id', as_index=False).agg(agg_dict)

    # Final feature columns
    feature_columns = [
                          'danceability', 'energy', 'key', 'loudness', 'mode',
                          'speechiness', 'acousticness', 'instrumentalness',
                          'liveness', 'valence', 'tempo', 'duration_ms', 'popularity',
                          'explicit'
                      ] + list(genre_dummies.columns) + list(time_signature_dummies.columns)

    # Drop missing values
    df_clean = df_merged.dropna(subset=feature_columns).copy()

    # Create feature matrix
    X = df_clean[feature_columns].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open('./testset.json', 'r') as f:
        testset = json.load(f)

    return df_clean, X_scaled, testset

class BaselineRecommender:
    def __init__(self, df, features_scaled):
        """
        Initialize the recommender system

        Args:
            df: DataFrame with song information including track_id
            features_scaled: Normalized feature matrix (after PCA)
        """
        self.df = df.reset_index(drop=True)
        self.features = features_scaled
        self.track_id_to_idx = {track_id: idx for idx, track_id in enumerate(self.df['track_id'])}

    def get_recommendations(self, input_track_ids, n_recommendations, target_artist):
        """
        Get recommendations based on multiple input songs

        Args:
            input_track_ids: List of track IDs to base recommendations on
            n_recommendations: Number of songs to recommend
            target_artist: If provided, only recommend songs from this artist

        Returns:
            List of recommended track IDs
        """
        valid_indices = [self.track_id_to_idx[item] for item in input_track_ids if item in self.track_id_to_idx]

        if len(valid_indices) == 0:
            # If no valid input tracks, return random recommendations
            artist_songs = self.df[self.df['artists'].isin(target_artist)]
            if len(artist_songs) >= n_recommendations:
                return artist_songs.sample(n_recommendations)['track_id'].tolist()
            return self.df.sample(n_recommendations)['track_id'].tolist()

        # Get features of input tracks and compute average profile
        input_features = self.features[valid_indices]
        avg_profile = np.mean(input_features, axis=0).reshape(1, -1)

        # Calculate similarity with all songs
        similarities = cosine_similarity(avg_profile, self.features)[0]

        # If target_artist is specified, filter to only songs from those artists
        artist_mask = self.df['artists'].isin(target_artist)
        # Set similarity to -inf for songs not by the target artists
        similarities = np.where(artist_mask, similarities, -np.inf)

        # Get indices of most similar songs
        similar_indices = np.argsort(similarities)[::-1]

        # Filter out input songs if requested
        similar_indices = [idx for idx in similar_indices if idx not in valid_indices]

        # Get top n recommendations
        recommended_indices = similar_indices[:n_recommendations]
        recommended_track_ids = self.df.iloc[recommended_indices]['track_id'].tolist()

        return recommended_track_ids

def recommender_metrics(recommender, testset, n_recommendations=5):
    """
    Evaluate the recommender system using the testset

    Args:
        recommender: ContentBasedRecommender instance
        testset: Dict of playlist_name -> [input_tracks, target_tracks] pairs
                 where each track is a [track_id, artist] pair
        n_recommendations: Number of recommendations from the system

    Returns:
        Dictionary with evaluation metrics
    """
    total_ndcg = 0

    for playlist_name, (input_tracks, target_tracks) in tqdm(testset.items()):
        # Extract track IDs from tracks
        # _tracks is a list of [track_id, artist] pairs
        input_track_ids = [track[0] for track in input_tracks]
        target_track_ids = [track[0] for track in target_tracks]
        target_set = set(target_track_ids)

        # Build a pool of candidate songs: only songs by artists in target_tracks
        target_artists = set([track[1] for track in target_tracks])

        # Get recommendations filtered by artists
        predictions = recommender.get_recommendations(
            input_track_ids,
            n_recommendations=n_recommendations,
            target_artist=target_artists
        )

        # NDCG@K: Normalized Discounted Cumulative Gain
        # Binary relevance: 1 if the song is in target_tracks, 0 otherwise
        dcg = 0.0
        idcg = 0.0

        # Calculate DCG for predictions
        for rank, track_id in enumerate(predictions, start=1):
            relevance = 1 if track_id in target_set else 0
            dcg += relevance / np.log2(rank + 1)

        # Calculate IDCG (ideal DCG) - assumes all relevant items at top positions
        n_relevant = min(len(target_track_ids), n_recommendations)
        for rank in range(1, n_relevant + 1):
            idcg += 1.0 / np.log2(rank + 1)

        # Normalize DCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg

    n_playlists = len(testset)

    metrics = {
        'NDCG@5': total_ndcg.item() / n_playlists
    }

    return metrics

def evaluate(recommender, n_recommendations=5):
    df_clean, X_scaled, testset = load_data()
    baseline = BaselineRecommender(df_clean, X_scaled)
    t0 = time()
    print('Testing recommender quality...')
    metrics = recommender_metrics(recommender, testset, n_recommendations)
    t1 = time()
    print('Testing recommender performance...')
    recommender_metrics(baseline, testset, n_recommendations)
    t2 = time()

    metrics['Performance'] = (t1 - t0) / (t2 - t1)

    return metrics

def _safe_path(base: Path, maybe_abs: str) -> Path:
    candidate = Path(maybe_abs)
    return candidate if candidate.is_absolute() else base / maybe_abs.lstrip("/")

class Recommender:
    """
    KNN-based recommender that learns song similarity directly from audio features:
      * builds a cosine-nearest-neighbors index on standardized feature vectors
      * creates a listener profile from the most recent seed tracks
      * boosts tracks that match requested artists and popularity
      * falls back to globally popular titles when no seeds match
    Only the CSV dataset is consumed; no auxiliary metadata sources are required.
    """

    BASE_NUMERIC_FEATURES: Sequence[str] = (
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "popularity",
        "key",
        "mode",
        "time_signature",
    )

    RECENCY_WINDOW = 40
    RECENCY_DECAY = 0.82
    CANDIDATE_MULTIPLIER = 12
    MIN_CANDIDATES = 128
    POP_WEIGHT = 0.15
    ARTIST_BONUS = 0.12

    def __init__(self, dataset_file: str = "dataset.csv") -> None:
        data_root = Path("data").resolve()
        dataset_path = _safe_path(data_root, dataset_file)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = (
            df.dropna(subset=["track_id"])
            .drop_duplicates("track_id", keep="first")
            .reset_index(drop=True)
        )

        df["artists_set"] = (
            df["artists"]
            .fillna("")
            .apply(lambda s: {a.strip() for a in str(s).split(";") if a.strip()})
        )

        numeric_features = [c for c in self.BASE_NUMERIC_FEATURES if c in df.columns]
        if not numeric_features:
            raise ValueError("No supported numeric features found in dataset.")

        feature_frame = df[numeric_features].copy()
        for col in numeric_features:
            feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")
        medians = feature_frame.median(axis=0, skipna=True)
        feature_frame = feature_frame.fillna(medians)

        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_frame.to_numpy(dtype=np.float32)).astype(
            np.float32
        )

        nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        nn_model.fit(feature_matrix)

        self.df = df.reset_index(drop=True)
        self.features = feature_matrix
        self.scaler = scaler
        self.numeric_features = numeric_features
        self.nn_model = nn_model
        self.track_ids = self.df["track_id"].to_numpy(dtype=object)
        self.id_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(self.track_ids)}
        self.artist_sets: List[Set[str]] = self.df["artists_set"].tolist()
        self.artist_sets_lower: List[Set[str]] = [{a.lower() for a in aset} for aset in self.artist_sets]

        if "popularity" in self.df.columns:
            pop = (
                pd.to_numeric(self.df["popularity"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        else:
            pop = np.zeros(len(self.df), dtype=np.float32)
        pop = np.clip(pop, 0.0, 100.0)
        self.popularity = (pop + 1.0) / 101.0
        self.popularity_sorted_idx = np.argsort(-self.popularity, kind="mergesort")

        self._recency_cache: Dict[int, np.ndarray] = {}

    def _recency_weights(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=np.float32)
        cached = self._recency_cache.get(length)
        if cached is not None:
            return cached
        order = np.arange(length, dtype=np.float32)
        # newest samples receive highest weights via exponential decay
        weights = np.power(self.RECENCY_DECAY, length - 1 - order).astype(np.float32)
        total = weights.sum()
        weights = weights / total if total else np.full(length, 1.0 / length, dtype=np.float32)
        self._recency_cache[length] = weights
        return weights

    def _popularity_fallback(
        self, seen: Set[str], needed: int, target_artists: Optional[Iterable[str]]
    ) -> List[str]:
        out: List[str] = []
        if needed <= 0:
            return out
        normalized_targets = {a.lower() for a in (target_artists or set()) if a}
        # try to honor artist hints before generic popularity
        if normalized_targets:
            prioritized = [
                idx
                for idx, artists in enumerate(self.artist_sets_lower)
                if artists & normalized_targets
            ]
            for idx in prioritized:
                tid = self.track_ids[idx]
                if tid in seen:
                    continue
                out.append(tid)
                seen.add(tid)
                if len(out) >= needed:
                    return out
        for idx in self.popularity_sorted_idx:
            tid = self.track_ids[idx]
            if tid in seen:
                continue
            out.append(tid)
            seen.add(tid)
            if len(out) >= needed:
                break
        return out

    def _user_profile(self, idxs: Sequence[int]) -> np.ndarray:
        recent = idxs[-self.RECENCY_WINDOW :]
        weights = self._recency_weights(len(recent))
        profile = np.average(self.features[recent], axis=0, weights=weights).astype(np.float32)
        norm = np.linalg.norm(profile)
        return profile / (norm if norm else 1.0)

    def get_recommendations(
        self,
        input_track_ids: Sequence[str],
        n_recommendations: int,
        target_artist: Optional[Set[str]] = None,
    ) -> List[str]:
        if n_recommendations <= 0:
            return []
        normalized_target = {a.lower() for a in (target_artist or set()) if a}

        valid_indices = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]
        seen: Set[str] = set(input_track_ids)

        if not valid_indices:
            return self._popularity_fallback(seen, n_recommendations, normalized_target)

        profile = self._user_profile(valid_indices)
        candidate_pool = min(
            len(self.df),
            max(self.MIN_CANDIDATES, n_recommendations * self.CANDIDATE_MULTIPLIER),
        )
        distances, neighbor_idxs = self.nn_model.kneighbors(profile.reshape(1, -1), candidate_pool)
        distances = distances[0]
        neighbor_idxs = neighbor_idxs[0]

        candidate_scores: Dict[str, float] = {}
        for rank, (idx, dist) in enumerate(zip(neighbor_idxs, distances)):
            tid = self.track_ids[idx]
            if tid in seen:
                continue
            similarity = 1.0 - dist  # cosine distance -> similarity
            score = similarity + self.popularity[idx] * self.POP_WEIGHT
            if normalized_target and (self.artist_sets_lower[idx] & normalized_target):
                score += self.ARTIST_BONUS
            # small rank-based tie-breaker keeps closer neighbors first
            score -= rank * 1e-4
            # keep the best score if duplicate candidate surfaces
            prev = candidate_scores.get(tid)
            if prev is None or score > prev:
                candidate_scores[tid] = score

        ordered = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
        recommendations = [tid for tid, _ in ordered[:n_recommendations]]

        if len(recommendations) < n_recommendations:
            missing = n_recommendations - len(recommendations)
            recommendations.extend(
                self._popularity_fallback(seen | set(recommendations), missing, normalized_target)
            )

        return recommendations


recommender = Recommender()

results = evaluate(recommender)
print(results)