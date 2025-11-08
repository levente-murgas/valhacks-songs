from typing import List, Set, Dict, Optional, Mapping, ClassVar, Tuple
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
    p = Path(maybe_abs)
    return p if p.is_absolute() else base / maybe_abs.lstrip("/")

class Recommender:
    """
    Content-based recommender (centroid + cosine) with:
      - De-duplication by track_id
      - Exclusion by track_id (not just row index)
      - Unique outputs guaranteed
      - Fast vectorized scoring + argpartition
    """

    BASE_NUMERIC_FEATURES = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
        "popularity", "key", "mode", "time_signature"
    ]

    DEFAULT_WEIGHTS: ClassVar[Dict[str, float]] = {
        "energy":         2.20,  # up: separates hard/classic/party from soft/grunge
        "danceability":   1.80,  # up: party/pop/gym cohesion
        "valence":        1.80,  # up: pop/party vs. grunge/alt split
        "loudness":       1.25,  # up: hard/classic presence

        "acousticness":   1.20,  # up: soft rock / “Always” / Oldies
        "instrumentalness": 0.50,# slightly down: most sets are vocal
        "liveness":       0.35,  # slightly down: live vs studio less critical overall
        "speechiness":    0.40,  # down: rap/monologue shouldn't dominate general use

        "tempo":          1.40,  # down from 2.40: still useful, not dominant
        "duration_ms":    0.35,  # down: song length shouldn’t drive matches
        "time_signature": 0.10,  # down: not discriminative across your buckets
        "key":            0.40,
        "mode":           0.25,
        "popularity":     0.00,
    }

    RECENCY_WINDOW: ClassVar[int] = 25
    RECENCY_DECAY: ClassVar[float] = 0.72
    LAST_TRACK_FOCUS: ClassVar[float] = 0.35
    GENRE_HISTORY_WINDOW: ClassVar[int] = 15
    GENRE_TOP_K: ClassVar[int] = 3
    GENRE_MATCH_BONUS: ClassVar[float] = 0.12
    ARTIST_BASE_BONUS: ClassVar[float] = 0.25
    ARTIST_SIMILARITY_BONUS: ClassVar[float] = 0.35

    def __init__(
            self,
            dataset_file: str = "dataset.csv",
            feature_weights: Optional[Mapping[str, float]] = None,
    ):
        # ---------- Resolve paths ----------
        data_root = Path("data").resolve()
        dataset_path = _safe_path(data_root, dataset_file)

        # ---------- Load data ----------
        df = pd.read_csv(dataset_path)

        # De-dupe by track_id (keep most popular if available)
        if "popularity" in df.columns:
            df["__pop__"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(-1)
            df = (df.sort_values("__pop__", ascending=False, kind="mergesort")
                  .drop_duplicates("track_id", keep="first")
                  .drop(columns="__pop__")
                  .reset_index(drop=True))
        else:
            df = df.drop_duplicates("track_id", keep="first").reset_index(drop=True)

        df = df.dropna(subset=["track_id"]).copy()

        # Parse artists into sets
        df["artists_set"] = (
            df["artists"].fillna("")
            .apply(lambda s: {a.strip() for a in str(s).split(";") if a.strip()})
        )

        # Features present in this CSV
        numeric_features = [c for c in self.BASE_NUMERIC_FEATURES if c in df.columns]
        if not numeric_features:
            raise ValueError("No numeric features from BASE_NUMERIC_FEATURES found in dataset.")

        # Coerce numerics + median impute
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[numeric_features].median(axis=0, skipna=True)
        df[numeric_features] = df[numeric_features].fillna(med)

        weights = self._resolve_weights(numeric_features, feature_weights)

        # ---------- Build standardized, weighted, unit-normalized matrix ----------
        X = df[numeric_features].to_numpy(dtype=np.float32)

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xz = scaler.fit_transform(X).astype(np.float32)  # store to allow weight updates later

        wvec = np.array([weights[f] for f in numeric_features], dtype=np.float32)
        Xw = Xz * wvec  # apply weights

        # Row-normalize to unit length (cosine -> dot product)
        norms = np.linalg.norm(Xw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xw_unit = (Xw / norms).astype(np.float32)

        # Popularity multipliers (vectorized)
        if "popularity" in numeric_features:
            pop = df["popularity"].to_numpy(dtype=np.float32)
            pop = np.clip(pop, 0.0, 100.0)
            pop_mult = (0.95 + (pop / 100.0) * 0.15).astype(np.float32)
        else:
            pop_mult = np.ones(len(df), dtype=np.float32)

        # Track genre helper structures
        if "track_genre" in df.columns:
            df["track_genre"] = df["track_genre"].fillna("").astype(str)
            track_genres = df["track_genre"].tolist()
        else:
            track_genres = None

        genre_to_idx: Dict[str, np.ndarray] = {}
        if track_genres:
            for i, genre in enumerate(track_genres):
                if not genre:
                    continue
                genre_to_idx.setdefault(genre, []).append(i)
            for genre, idxs in list(genre_to_idx.items()):
                genre_to_idx[genre] = np.fromiter(idxs, dtype=np.int32)

        # Artist inverted index: artist -> np.array(row_idx)
        artist_to_idx: Dict[str, np.ndarray] = {}
        for i, aset in enumerate(df["artists_set"].values):
            for a in aset:
                artist_to_idx.setdefault(a, []).append(i)
        for a, idxs in list(artist_to_idx.items()):
            artist_to_idx[a] = np.fromiter(idxs, dtype=np.int32)

        # Save state
        self.df = df.reset_index(drop=True)
        self.numeric_features = numeric_features
        self.weights = weights
        self.scaler = scaler
        self.Xz = Xz
        self.wvec = wvec
        self.Xw_unit = Xw_unit
        self.pop_mult = pop_mult
        self.artist_to_idx = artist_to_idx
        self.track_genres = np.array(track_genres, dtype=object) if track_genres else None
        self.genre_to_idx = genre_to_idx
        self.id_to_idx = {tid: i for i, tid in enumerate(self.df["track_id"])}
        self.track_ids = self.df["track_id"].to_numpy()
        if "popularity" in self.df.columns:
            pop_arr = self.df["popularity"].to_numpy(dtype=np.float32)
        else:
            pop_arr = np.zeros(len(self.df), dtype=np.float32)
        self.popularity = pop_arr
        self.popularity_sorted_idx = np.argsort(-pop_arr, kind="mergesort")
        self._recency_cache: Dict[int, np.ndarray] = {}

    def _resolve_weights(self, numeric_features: List[str], override: Optional[Mapping[str, float]]) -> Dict[str, float]:
        """Combine class-level defaults + per-feature constants + per-call overrides."""
        w = {f: 1.0 for f in numeric_features}

        # 1) Class-level mapping DEFAULT_WEIGHTS
        for k, v in getattr(self, "DEFAULT_WEIGHTS", {}).items():
            if k in w:
                w[k] = float(v)

        # 2) Class-level per-feature constants WEIGHT_<FEATURE>
        for f in numeric_features:
            attr = f"WEIGHT_{f.upper()}"
            if hasattr(self, attr):
                w[f] = float(getattr(self, attr))

        # 3) Per-call override
        if override:
            for k, v in override.items():
                if k in w:
                    w[k] = float(v)

        return w

    def set_weights(self, new_weights: Mapping[str, float]) -> None:
        """Update weights after initialization and rebuild the weighted/unit-norm matrix."""
        any_change = False
        for k, v in new_weights.items():
            if k in self.weights:
                old = self.weights[k]
                nv = float(v)
                if nv != old:
                    self.weights[k] = nv
                    any_change = True
        if not any_change:
            return
        self.wvec = np.array([self.weights[f] for f in self.numeric_features], dtype=np.float32)
        Xw = self.Xz * self.wvec
        norms = np.linalg.norm(Xw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.Xw_unit = (Xw / norms).astype(np.float32)

    def _recency_weights(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=np.float32)
        cached = self._recency_cache.get(length)
        if cached is not None:
            return cached
        positions = np.arange(length, dtype=np.float32)
        exponents = length - 1 - positions
        weights = np.power(self.RECENCY_DECAY, exponents).astype(np.float32)
        total = float(weights.sum())
        if total == 0.0:
            weights = np.full(length, 1.0 / length, dtype=np.float32)
        else:
            weights /= total
        self._recency_cache[length] = weights
        return weights

    def _genre_preferences(self, idxs: List[int], weights: np.ndarray) -> Dict[str, float]:
        if self.track_genres is None or not idxs:
            return {}
        scores: Dict[str, float] = {}
        for idx, w in zip(idxs[-self.GENRE_HISTORY_WINDOW:], weights[-self.GENRE_HISTORY_WINDOW:]):
            genre = self.track_genres[idx]
            if not genre:
                continue
            scores[genre] = scores.get(genre, 0.0) + float(w)
        if not scores:
            return {}
        total = sum(scores.values())
        if total == 0.0:
            return {}
        normed = {g: v / total for g, v in scores.items()}
        top_items = sorted(normed.items(), key=lambda kv: kv[1], reverse=True)[:self.GENRE_TOP_K]
        return dict(top_items)

    def _apply_genre_bonus(self, scores: np.ndarray, prefs: Dict[str, float]) -> None:
        if not prefs or not self.genre_to_idx:
            return
        for genre, weight in prefs.items():
            idx = self.genre_to_idx.get(genre)
            if idx is None or idx.size == 0:
                continue
            scores[idx] += self.GENRE_MATCH_BONUS * float(weight)

    def _fallback_from_popularity(self, seen: Set[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        out: List[str] = []
        if n_recommendations <= 0:
            return out
        artist_priority: Set[int] = set()
        if target_artist:
            artist_indices = self._artist_boost_indices(self.artist_to_idx, target_artist)
            if artist_indices.size > 0:
                artist_priority = set(map(int, artist_indices.tolist()))
        for idx in self.popularity_sorted_idx:
            if idx in artist_priority:
                tid = self.track_ids[idx]
                if tid in seen:
                    continue
                out.append(tid)
                seen.add(tid)
                if len(out) >= n_recommendations:
                    return out
        for idx in self.popularity_sorted_idx:
            tid = self.track_ids[idx]
            if tid in seen:
                continue
            out.append(tid)
            seen.add(tid)
            if len(out) >= n_recommendations:
                break
        return out

    @staticmethod
    def _artist_boost_indices(artist_to_idx: Dict[str, np.ndarray], target_artist: Set[str]) -> np.ndarray:
        if not target_artist:
            return np.empty(0, dtype=np.int32)
        chunks = [artist_to_idx[a] for a in target_artist if a in artist_to_idx]
        if not chunks:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(chunks))

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        valid_pairs: List[Tuple[str, int]] = [
            (tid, self.id_to_idx[tid]) for tid in input_track_ids if tid in self.id_to_idx
        ]

        if not valid_pairs:
            seen = set(input_track_ids)
            return self._fallback_from_popularity(seen, n_recommendations, target_artist)

        tail_pairs = valid_pairs[-self.RECENCY_WINDOW:]
        tail_indices = [idx for _, idx in tail_pairs]
        weights = self._recency_weights(len(tail_indices))

        centroid = np.average(self.Xw_unit[tail_indices], axis=0, weights=weights).astype(np.float32)
        last_idx = tail_indices[-1]
        if self.LAST_TRACK_FOCUS > 0.0:
            centroid = (
                centroid * (1.0 - self.LAST_TRACK_FOCUS)
                + self.Xw_unit[last_idx] * self.LAST_TRACK_FOCUS
            )
        cn = np.linalg.norm(centroid)
        centroid_unit = centroid / (cn if cn != 0 else 1.0)

        scores = (self.Xw_unit @ centroid_unit).astype(np.float32)
        scores *= self.pop_mult

        genre_pref = self._genre_preferences(tail_indices, weights)
        self._apply_genre_bonus(scores, genre_pref)

        if target_artist:
            idx = self._artist_boost_indices(self.artist_to_idx, target_artist)
            if idx.size > 0:
                scores[idx] += self.ARTIST_BASE_BONUS
                if last_idx is not None and self.ARTIST_SIMILARITY_BONUS > 0.0:
                    sim = (self.Xw_unit[idx] @ self.Xw_unit[last_idx]).astype(np.float32)
                    scores[idx] += np.clip(sim, -1.0, 1.0) * self.ARTIST_SIMILARITY_BONUS

        seed_mask = self.df["track_id"].isin(input_track_ids).to_numpy()
        scores[seed_mask] = -np.inf

        N = scores.size
        k = min(max(n_recommendations * 3, n_recommendations), N)
        part_idx = np.argpartition(-scores, k - 1)[:k]
        local_sorted = part_idx[np.argsort(-scores[part_idx], kind="mergesort")]

        rec_ids: List[str] = []
        seen: Set[str] = set(input_track_ids)
        for j in local_sorted:
            if not np.isfinite(scores[j]):
                continue
            tid = self.track_ids[j]
            if tid in seen:
                continue
            rec_ids.append(tid)
            seen.add(tid)
            if len(rec_ids) >= n_recommendations:
                break

        if len(rec_ids) < n_recommendations:
            full_order = np.argsort(-scores, kind="mergesort")
            for j in full_order:
                if not np.isfinite(scores[j]):
                    continue
                tid = self.track_ids[j]
                if tid in seen:
                    continue
                rec_ids.append(tid)
                seen.add(tid)
                if len(rec_ids) >= n_recommendations:
                    break

        return rec_ids

recommender = Recommender()

results = evaluate(recommender)
print(results)