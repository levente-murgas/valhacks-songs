from typing import List, Set, Dict, Optional, Mapping, ClassVar, Tuple
from pathlib import Path
import json
from tqdm import tqdm
from time import time
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Data loading (unchanged)
# -----------------------------
def load_data():
    # Load the dataset
    df = pd.read_csv("dataset.csv", index_col=0)
    df.drop_duplicates(subset=['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                               'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                               'duration_ms', 'popularity',
                               'artists', 'track_name', 'time_signature'],
                       inplace=True)

    # One-hot encode genres and time_signature
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


# -----------------------------
# Baseline (tidied artist mask)
# -----------------------------
class BaselineRecommender:
    def __init__(self, df, features_scaled):
        self.df = df.reset_index(drop=True)
        self.features = features_scaled
        self.track_id_to_idx = {track_id: idx for idx, track_id in enumerate(self.df['track_id'])}

    @staticmethod
    def _artist_mask(series: pd.Series, target_artist: Set[str]) -> pd.Series:
        if not target_artist:
            return pd.Series(True, index=series.index)
        return series.fillna("").apply(
            lambda s: any(a.strip() in target_artist for a in re.split(r"[;,]", str(s)))
        )

    def get_recommendations(self, input_track_ids, n_recommendations, target_artist):
        valid_indices = [self.track_id_to_idx[item] for item in input_track_ids if item in self.track_id_to_idx]

        if len(valid_indices) == 0:
            artist_mask = self._artist_mask(self.df['artists'], set(target_artist or []))
            pool = self.df[artist_mask] if artist_mask.any() else self.df
            if 'popularity' in pool.columns:
                pool = pool.sort_values('popularity', ascending=False)
            return pool.head(n_recommendations)['track_id'].tolist()

        input_features = self.features[valid_indices]
        avg_profile = np.mean(input_features, axis=0).reshape(1, -1)

        similarities = cosine_similarity(avg_profile, self.features)[0]
        artist_mask = self._artist_mask(self.df['artists'], set(target_artist or [])).to_numpy()
        similarities = np.where(artist_mask, similarities, -np.inf)

        similarities[valid_indices] = -np.inf
        similar_indices = np.argsort(similarities)[::-1]
        recommended_indices = similar_indices[:n_recommendations]
        return self.df.iloc[recommended_indices]['track_id'].tolist()


# -----------------------------
# Evaluator (unchanged)
# -----------------------------
def recommender_metrics(recommender, testset, n_recommendations=5):
    total_ndcg = 0.0

    for playlist_name, (input_tracks, target_tracks) in tqdm(testset.items()):
        input_track_ids = [track[0] for track in input_tracks]
        target_track_ids = [track[0] for track in target_tracks]
        target_set = set(target_track_ids)

        target_artists = set([track[1] for track in target_tracks])

        predictions = recommender.get_recommendations(
            input_track_ids,
            n_recommendations=n_recommendations,
            target_artist=target_artists
        )

        dcg = 0.0
        idcg = 0.0

        for rank, track_id in enumerate(predictions, start=1):
            relevance = 1 if track_id in target_set else 0
            dcg += relevance / np.log2(rank + 1)

        n_relevant = min(len(target_track_ids), n_recommendations)
        for rank in range(1, n_relevant + 1):
            idcg += 1.0 / np.log2(rank + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg

    n_playlists = len(testset)
    return {'NDCG@5': float(total_ndcg) / n_playlists}


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


# -----------------------------
# QUALITY-FOCUSED Recommender
# -----------------------------
class Recommender:
    """
    Content-based recommender (centroid + cosine) with quality upgrades:
      - Seed-adaptive feature weighting (no retrain)
      - Harmonic key proximity (circular)
      - Tempo alignment with half/double-time equivalence
      - Album continuity bonus
      - STRICT artist candidate set (to match evaluator intent)
    """

    BASE_NUMERIC_FEATURES = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
        "popularity", "key", "mode", "time_signature"
    ]

    # Base per-feature weights (you can tweak, but adaptive weighting will also modulate per playlist)
    DEFAULT_WEIGHTS: ClassVar[Dict[str, float]] = {
        "energy": 2.10,
        "danceability": 1.95,
        "valence": 1.85,
        "loudness": 1.20,

        "acousticness": 1.10,
        "instrumentalness": 0.60,
        "liveness": 0.35,
        "speechiness": 0.45,

        "tempo": 1.30,
        "duration_ms": 0.30,
        "time_signature": 0.10,
        "key": 0.30,
        "mode": 0.20,
        "popularity": 0.00,
    }

    # Recency / focus
    RECENCY_WINDOW: ClassVar[int] = 25
    RECENCY_DECAY: ClassVar[float] = 0.72
    LAST_TRACK_FOCUS: ClassVar[float] = 0.35

    # Genre shaping
    GENRE_HISTORY_WINDOW: ClassVar[int] = 15
    GENRE_TOP_K: ClassVar[int] = 3
    GENRE_MATCH_BONUS: ClassVar[float] = 0.12

    # Music-aware re-rank bonuses
    KEY_MAX_BONUS: ClassVar[float] = 0.08  # peak bonus when same key (+mode if desired)
    KEY_SIGMA_SEMITONES: ClassVar[float] = 2.0  # broader = more tolerant
    TEMPO_MAX_BONUS: ClassVar[float] = 0.12  # peak bonus when tempo aligned
    TEMPO_REL_SIGMA: ClassVar[float] = 0.06  # ~6% relative bpm tolerance
    ALBUM_BONUS: ClassVar[float] = 0.08  # small album continuity bump

    # Seed-adaptive feature emphasis
    ADAPTIVE_STRENGTH: ClassVar[float] = 0.65  # 0..1; multiplies importance ← 1/(eps+seed_std)
    ADAPTIVE_EPS: ClassVar[float] = 1e-3

    def __init__(
            self,
            dataset_file: str = "dataset.csv",
            feature_weights: Optional[Mapping[str, float]] = None,
    ):
        # ---------- Resolve paths (prefer given, then ./data/) ----------
        p = Path(dataset_file)
        if not p.exists():
            alt = Path("data") / dataset_file
            if alt.exists():
                p = alt
        if not p.exists():
            raise FileNotFoundError(f"Could not find dataset at '{dataset_file}' or 'data/{dataset_file}'")

        # ---------- Load data ----------
        df = pd.read_csv(p)

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

        # Parse artists into sets (handle semicolons/commas)
        df["artists_set"] = (
            df["artists"].fillna("")
            .apply(lambda s: {a.strip() for a in re.split(r"[;,]", str(s)) if a.strip()})
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
        Xz = scaler.fit_transform(X).astype(np.float32)  # keep to enable seed-adaptive weighting

        wvec = np.array([weights[f] for f in numeric_features], dtype=np.float32)
        Xw = Xz * wvec  # apply global base weights

        # Row-normalize to unit length (cosine -> dot product)
        norms = np.linalg.norm(Xw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xw_unit = (Xw / norms).astype(np.float32)

        # Popularity multipliers (mild)
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

        # Cache music-aware fields
        keys = df["key"].to_numpy(dtype=np.float32) if "key" in df.columns else np.full(len(df), -1, np.float32)
        modes = df["mode"].to_numpy(dtype=np.float32) if "mode" in df.columns else np.full(len(df), -1, np.float32)
        tempos = df["tempo"].to_numpy(dtype=np.float32) if "tempo" in df.columns else np.full(len(df), -1, np.float32)
        album_names = df["album_name"].fillna("").astype(str).to_numpy() if "album_name" in df.columns else np.array(
            [""] * len(df))

        # Save state
        self.df = df.reset_index(drop=True)
        self.numeric_features = numeric_features
        self.weights = weights
        self.scaler = scaler
        self.Xz = Xz  # standardized features
        self.wvec = wvec  # global base weights
        self.Xw_unit = Xw_unit  # row-normalized weighted features
        self.pop_mult = pop_mult
        self.artist_to_idx = artist_to_idx
        self.track_genres = np.array(track_genres, dtype=object) if track_genres else None
        self.genre_to_idx = genre_to_idx
        self.id_to_idx = {tid: i for i, tid in enumerate(self.df["track_id"])}
        self.track_ids = self.df["track_id"].to_numpy()
        self.popularity = df["popularity"].to_numpy(dtype=np.float32) if "popularity" in df.columns else np.zeros(
            len(df), np.float32)
        self.popularity_sorted_idx = np.argsort(-self.popularity, kind="mergesort")
        self.keys = keys
        self.modes = modes
        self.tempos = tempos
        self.albums = album_names
        self._recency_cache: Dict[int, np.ndarray] = {}

    # ---------- Weight resolution ----------
    def _resolve_weights(self, numeric_features: List[str], override: Optional[Mapping[str, float]]) -> Dict[
        str, float]:
        w = {f: 1.0 for f in numeric_features}
        for k, v in getattr(self, "DEFAULT_WEIGHTS", {}).items():
            if k in w:
                w[k] = float(v)
        for f in numeric_features:
            attr = f"WEIGHT_{f.upper()}"
            if hasattr(self, attr):
                w[f] = float(getattr(self, attr))
        if override:
            for k, v in override.items():
                if k in w:
                    w[k] = float(v)
        return w

    # ---------- Seed-adaptive query vector ----------
    def _adaptive_query_vector(self, tail_idx: np.ndarray, last_idx: int) -> np.ndarray:
        """
        Build a query vector in the SAME space as Xw_unit,
        but adaptively up-weight features that are tight within the seed set.
        """
        Z = self.Xz[tail_idx]  # standardized space
        mu = Z.mean(axis=0).astype(np.float32)
        seed_std = Z.std(axis=0).astype(np.float32)
        # Importance: inverse of seed spread (bounded)
        importance = 1.0 / (self.ADAPTIVE_EPS + seed_std)
        # Normalize importance to mean=1 to avoid exploding norms
        importance /= max(importance.mean(), 1e-6)
        # Interpolate with neutral (1.0) by ADAPTIVE_STRENGTH
        importance = 1.0 + self.ADAPTIVE_STRENGTH * (importance - 1.0)

        q = mu * (self.wvec * importance)  # apply base weights + adaptive gains
        # blend with last track focus (already in the weighted/unit space)
        q_unit = q / (np.linalg.norm(q) or 1.0)
        if self.LAST_TRACK_FOCUS > 0.0:
            last_vec = self.Xw_unit[last_idx]
            q_unit = (1.0 - self.LAST_TRACK_FOCUS) * q_unit + self.LAST_TRACK_FOCUS * last_vec
            q_unit = q_unit / (np.linalg.norm(q_unit) or 1.0)
        return q_unit.astype(np.float32)

    # ---------- Recency weights ----------
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

    # ---------- Genre preferences ----------
    def _genre_preferences(self, idxs: List[int], weights: np.ndarray) -> Dict[str, float]:
        if self.track_genres is None or not idxs:
            return {}
        scores: Dict[str, float] = {}
        for idx, w in zip(idxs[-15:], weights[-15:]):
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

    # ---------- Music-aware bonuses ----------
    def _harmonic_bonus(self, cand_idx: np.ndarray, last_idx: int) -> np.ndarray:
        if self.keys[last_idx] < 0 or self.keys[cand_idx].size == 0:
            return np.zeros(cand_idx.size, dtype=np.float32)
        k0 = self.keys[last_idx] % 12
        k = self.keys[cand_idx] % 12
        # circular semitone distance (0..6)
        d = np.abs(k - k0)
        d = np.minimum(d, 12 - d)
        # optional mode agreement can be added (small addition)
        same_mode = (self.modes[cand_idx] == self.modes[last_idx]).astype(np.float32)
        gauss = np.exp(-0.5 * (d / self.KEY_SIGMA_SEMITONES) ** 2).astype(np.float32)
        return self.KEY_MAX_BONUS * (0.8 * gauss + 0.2 * same_mode)

    def _tempo_bonus(self, cand_idx: np.ndarray, last_idx: int) -> np.ndarray:
        if self.tempos[last_idx] <= 0 or np.all(self.tempos[cand_idx] <= 0):
            return np.zeros(cand_idx.size, dtype=np.float32)
        t0 = float(self.tempos[last_idx])
        t = self.tempos[cand_idx]
        # consider half/double equivalence
        ratios = np.vstack([
            np.abs(t - t0) / t0,
            np.abs(t - (2.0 * t0)) / (2.0 * t0),
            np.abs(t - (0.5 * t0)) / (0.5 * t0),
        ]).min(axis=0)
        gauss = np.exp(-0.5 * (ratios / self.TEMPO_REL_SIGMA) ** 2).astype(np.float32)
        return self.TEMPO_MAX_BONUS * gauss

    def _album_bonus(self, cand_idx: np.ndarray, last_idx: int) -> np.ndarray:
        if self.albums.size == 0:
            return np.zeros(cand_idx.size, dtype=np.float32)
        same = (self.albums[cand_idx] == self.albums[last_idx]).astype(np.float32)
        return self.ALBUM_BONUS * same

    # ---------- Helpers ----------
    @staticmethod
    def _artist_indices(artist_to_idx: Dict[str, np.ndarray], target_artist: Set[str]) -> np.ndarray:
        if not target_artist:
            return np.empty(0, dtype=np.int32)
        chunks = [artist_to_idx[a] for a in target_artist if a in artist_to_idx]
        if not chunks:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(chunks))

    def _fallback_from_popularity(self, seen: Set[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        out: List[str] = []
        if n_recommendations <= 0:
            return out
        artist_priority: Set[int] = set()
        if target_artist:
            artist_indices = self._artist_indices(self.artist_to_idx, target_artist)
            if artist_indices.size > 0:
                artist_priority = set(map(int, artist_indices.tolist()))
        for idx in self.popularity_sorted_idx:
            if artist_priority and idx not in artist_priority:
                continue
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

    # ---------- Main API ----------
    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[
        str]:
        # Map seeds to indices
        seed_idx = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]
        if not seed_idx:
            return self._fallback_from_popularity(set(input_track_ids), n_recommendations, target_artist)

        tail_idx = np.asarray(seed_idx[-self.RECENCY_WINDOW:], dtype=np.int32)
        last_idx = int(tail_idx[-1])

        # Build an adaptive, music-aware query vector
        q_unit = self._adaptive_query_vector(tail_idx, last_idx)

        # Candidate set: STRICT filter to target artists (matches evaluator's relevance pool)
        if target_artist:
            cand_idx = self._artist_indices(self.artist_to_idx, target_artist)
            if cand_idx.size == 0:
                return self._fallback_from_popularity(set(input_track_ids), n_recommendations, target_artist)
        else:
            cand_idx = np.arange(self.Xw_unit.shape[0], dtype=np.int32)

        # Exclude seeds early
        if seed_idx:
            cand_idx = cand_idx[~np.isin(cand_idx, np.asarray(seed_idx, dtype=np.int32))]
            if cand_idx.size == 0:
                return self._fallback_from_popularity(set(input_track_ids), n_recommendations, target_artist)

        # Core similarity (dot since rows are unit-norm)
        scores = np.full(self.Xw_unit.shape[0], -np.inf, dtype=np.float32)
        base = (self.Xw_unit[cand_idx] @ q_unit).astype(np.float32)

        # Popularity multiplier (mild)
        base *= self.pop_mult[cand_idx]

        # Write base before bonuses
        scores[cand_idx] = base

        # Seed history weights (for genre prefs)
        rw = self._recency_weights(len(tail_idx))
        genre_pref = self._genre_preferences(tail_idx.tolist(), rw)
        self._apply_genre_bonus(scores, genre_pref)

        # Music-aware bonuses (vectorized over candidates)
        scores[cand_idx] += self._harmonic_bonus(cand_idx, last_idx)
        scores[cand_idx] += self._tempo_bonus(cand_idx, last_idx)
        scores[cand_idx] += self._album_bonus(cand_idx, last_idx)

        # Top-K over sparse vector
        k = int(min(max(n_recommendations * 3, n_recommendations), cand_idx.size))
        if k <= 0:
            return []
        part = np.argpartition(-scores, k - 1)[:k]
        ordered = part[np.argsort(-scores[part], kind="mergesort")]

        rec_ids: List[str] = []
        for j in ordered:
            if not np.isfinite(scores[j]):
                continue
            rec_ids.append(self.track_ids[j])
            if len(rec_ids) >= n_recommendations:
                break

        # Fill if necessary (very constrained cases)
        if len(rec_ids) < n_recommendations:
            chosen = set(rec_ids)
            for j in cand_idx[np.argsort(-self.popularity[cand_idx], kind="mergesort")]:
                tid = self.track_ids[j]
                if tid in chosen:
                    continue
                rec_ids.append(tid)
                chosen.add(tid)
                if len(rec_ids) >= n_recommendations:
                    break

        return rec_ids


# -----------------------------
# Run
# -----------------------------
recommender = Recommender(dataset_file="dataset.csv")
results = evaluate(recommender)
print(results)
