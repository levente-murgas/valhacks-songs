from __future__ import annotations

from typing import Iterable, Sequence, List, Set, Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity  # used by BaselineRecommender


def load_data():
    # Load the dataset
    df = pd.read_csv("dataset.csv", index_col=0)
    df.drop_duplicates(
        subset=[
            "explicit",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "popularity",
            "artists",
            "track_name",
            "time_signature",
        ],
        inplace=True,
    )

    # One-hot encode genres and time_signature for the baseline features
    genre_dummies = pd.get_dummies(df["track_genre"], prefix="genre")
    time_signature_dummies = pd.get_dummies(df["time_signature"], prefix="time_signature")

    # Combine with original dataframe
    df_with_genres = pd.concat([df, genre_dummies, time_signature_dummies], axis=1)

    # Define columns to aggregate
    group_cols = [
        "track_id",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "popularity",
        "artists",
        "album_name",
        "track_name",
    ] + list(time_signature_dummies.columns)

    # Aggregation dictionary
    agg_dict = {col: "max" for col in genre_dummies.columns}
    for col in group_cols:
        agg_dict[col] = "first"

    # Merge duplicates
    df_merged = df_with_genres.groupby("track_id", as_index=False).agg(agg_dict)

    # Final feature columns for baseline
    feature_columns = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "popularity",
        "explicit",
    ] + list(genre_dummies.columns) + list(time_signature_dummies.columns)

    # Drop missing values
    df_clean = df_merged.dropna(subset=feature_columns).copy()

    # Create feature matrix
    X = df_clean[feature_columns].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open("./testset.json", "r") as f:
        testset = json.load(f)

    return df_clean, X_scaled, testset


class BaselineRecommender:
    def __init__(self, df, features_scaled):
        self.df = df.reset_index(drop=True)
        self.features = features_scaled
        self.track_id_to_idx = {track_id: idx for idx, track_id in enumerate(self.df["track_id"])}

    def get_recommendations(self, input_track_ids, n_recommendations, target_artist):
        valid_indices = [self.track_id_to_idx[item] for item in input_track_ids if item in self.track_id_to_idx]

        if len(valid_indices) == 0:
            # If no valid input tracks, return random recommendations (preferring target artists if present)
            artist_songs = self.df[self.df["artists"].isin(target_artist)]
            if len(artist_songs) >= n_recommendations:
                return artist_songs.sample(n_recommendations)["track_id"].tolist()
            return self.df.sample(n_recommendations)["track_id"].tolist()

        # Average profile of seeds
        input_features = self.features[valid_indices]
        avg_profile = np.mean(input_features, axis=0).reshape(1, -1)

        # Cosine to all
        similarities = cosine_similarity(avg_profile, self.features)[0]

        # Keep only target artists if provided
        artist_mask = self.df["artists"].isin(target_artist)
        similarities = np.where(artist_mask, similarities, -np.inf)

        # Sort, remove seeds, take top-k
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = [idx for idx in similar_indices if idx not in valid_indices]
        recommended_indices = similar_indices[:n_recommendations]
        return self.df.iloc[recommended_indices]["track_id"].tolist()


def recommender_metrics(recommender, testset, n_recommendations=5):
    total_ndcg = 0.0
    # Precompute discount denominators once
    discounts = 1.0 / np.log2(np.arange(2, n_recommendations + 2, dtype=np.float32))

    for _, (input_tracks, target_tracks) in tqdm(testset.items()):
        input_track_ids = [t[0] for t in input_tracks]
        target_track_ids = [t[0] for t in target_tracks]
        target_set = set(target_track_ids)

        target_artists = set([t[1] for t in target_tracks])
        predictions = recommender.get_recommendations(
            input_track_ids,
            n_recommendations=n_recommendations,
            target_artist=target_artists,
        )

        rel = np.fromiter((1 if pid in target_set else 0 for pid in predictions), dtype=np.float32, count=n_recommendations)
        dcg = float((rel * discounts).sum())

        n_rel = min(len(target_set), n_recommendations)
        idcg = float(discounts[:n_rel].sum()) if n_rel > 0 else 0.0
        ndcg = (dcg / idcg) if idcg > 0.0 else 0.0
        total_ndcg += ndcg

    n_playlists = max(1, len(testset))
    return {"NDCG@{}".format(n_recommendations): total_ndcg / n_playlists}


def evaluate(recommender, n_recommendations=5):
    df_clean, X_scaled, testset = load_data()
    baseline = BaselineRecommender(df_clean, X_scaled)

    t0 = time()
    print("Testing recommender quality...")
    metrics = recommender_metrics(recommender, testset, n_recommendations)
    t1 = time()

    print("Testing recommender performance (baseline)...")
    recommender_metrics(baseline, testset, n_recommendations)
    t2 = time()

    metrics["Performance"] = (t1 - t0) / max(1e-9, (t2 - t1))
    return metrics


def _safe_path(base: Path, maybe_abs: str) -> Path:
    candidate = Path(maybe_abs)
    return candidate if candidate.is_absolute() else base / maybe_abs.lstrip("/")


class Recommender:
    """
    Fast content-based recommender:
      * Standardize numeric features, then L2-normalize rows (cosine -> dot product).
      * Single BLAS matvec per query + np.argpartition top-K (no sklearn kneighbors).
      * If target artists are provided, only score those rows.
      * Popularity and artist bonus applied vectorially.
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
    POP_WEIGHT = 0.15
    ARTIST_BONUS = 0.12

    def __init__(self, dataset_file: str = "dataset.csv") -> None:
        # Be flexible: look in CWD and in ./data
        candidates = [Path(dataset_file), _safe_path(Path("data").resolve(), dataset_file)]
        dataset_path = None
        for c in candidates:
            if Path(c).exists():
                dataset_path = Path(c)
                break
        if dataset_path is None:
            raise FileNotFoundError(f"Dataset not found. Tried: {candidates}")

        df = pd.read_csv(dataset_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = (
            df.dropna(subset=["track_id"])
            .drop_duplicates("track_id", keep="first")
            .reset_index(drop=True)
        )

        # Parse artists into sets (accept ";" or "," separated)
        def _split_artists(s: str) -> Set[str]:
            s = str(s or "")
            raw = []
            if ";" in s and "," in s:
                # handle mixed separators
                for part in s.split(";"):
                    raw.extend(part.split(","))
            elif ";" in s:
                raw = s.split(";")
            else:
                raw = s.split(",")
            return {a.strip() for a in raw if a.strip()}

        df["artists_set"] = df["artists"].fillna("").apply(_split_artists)
        df["artists_set_lower"] = df["artists_set"].apply(lambda st: {a.lower() for a in st})

        # Numeric feature frame
        numeric_features = [c for c in self.BASE_NUMERIC_FEATURES if c in df.columns]
        if not numeric_features:
            raise ValueError("No supported numeric features found in dataset.")

        feat = df[numeric_features].apply(pd.to_numeric, errors="coerce")
        feat = feat.fillna(feat.median(axis=0, skipna=True))

        # Standardize => L2 normalize rows so cosine == dot
        scaler = StandardScaler()
        X = scaler.fit_transform(feat.to_numpy(np.float32)).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X_unit = X / norms

        # Popularity 0..1
        if "popularity" in df.columns:
            pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        else:
            pop = np.zeros(len(df), dtype=np.float32)
        pop = np.clip(pop, 0.0, 100.0)
        pop = (pop + 1.0) / 101.0

        # Artist → indices inverted index
        artist_to_indices: Dict[str, np.ndarray] = {}
        for i, aset in enumerate(df["artists_set_lower"]):
            for a in aset:
                artist_to_indices.setdefault(a, []).append(i)
        for a, lst in artist_to_indices.items():
            artist_to_indices[a] = np.fromiter(lst, dtype=np.int32)

        # Expose members
        self.df = df.reset_index(drop=True)
        self.track_ids = self.df["track_id"].to_numpy(object)
        self.id_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(self.track_ids)}
        self.features_unit = X_unit.astype(np.float32, copy=False)
        self.popularity = pop.astype(np.float32, copy=False)
        self.artist_to_indices = artist_to_indices
        self._recency_cache: Dict[int, np.ndarray] = {}

    # ---- internals ----
    def _recency_weights(self, length: int) -> np.ndarray:
        cached = self._recency_cache.get(length)
        if cached is not None:
            return cached
        if length <= 0:
            w = np.empty(0, dtype=np.float32)
        else:
            order = np.arange(length, dtype=np.float32)
            w = np.power(self.RECENCY_DECAY, length - 1 - order).astype(np.float32)
            w /= w.sum()
        self._recency_cache[length] = w
        return w

    def _user_profile(self, idxs: Sequence[int]) -> np.ndarray:
        recent = idxs[-self.RECENCY_WINDOW:]
        w = self._recency_weights(len(recent))
        prof = np.average(self.features_unit[recent], axis=0, weights=w).astype(np.float32)
        n = np.linalg.norm(prof)
        if n:
            prof /= n
        return prof

    def _artist_candidate_indices(self, targets_lower: Set[str]) -> Optional[np.ndarray]:
        if not targets_lower:
            return None
        buckets = [self.artist_to_indices[a] for a in targets_lower if a in self.artist_to_indices]
        if not buckets:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(buckets))

    def _popularity_fallback(self, seen: Set[str], needed: int, targets_lower: Optional[Iterable[str]]) -> List[str]:
        out: List[str] = []
        if needed <= 0:
            return out
        tset = {a for a in (targets_lower or set()) if a}

        # Prefer target artists first
        if tset:
            idxs = self._artist_candidate_indices(tset)
            if idxs is not None and idxs.size:
                for i in idxs:
                    tid = self.track_ids[int(i)]
                    if tid in seen:
                        continue
                    out.append(tid)
                    seen.add(tid)
                    if len(out) >= needed:
                        return out

        # Then globally popular
        for i in np.argsort(-self.popularity, kind="mergesort"):
            tid = self.track_ids[int(i)]
            if tid in seen:
                continue
            out.append(tid)
            seen.add(tid)
            if len(out) >= needed:
                break
        return out

    # ---- main API ----
    def get_recommendations(
        self,
        input_track_ids: Sequence[str],
        n_recommendations: int,
        target_artist: Optional[Set[str]] = None,
    ) -> List[str]:
        if n_recommendations <= 0:
            return []

        targets_lower = {a.lower() for a in (target_artist or set()) if a}
        valid_indices = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]
        seen_ids: Set[str] = set(input_track_ids)

        # Cold start → popularity (respect artist hint if present)
        if not valid_indices:
            return self._popularity_fallback(seen_ids, n_recommendations, targets_lower)

        profile = self._user_profile(valid_indices)

        # Candidate restriction by artist if given
        cand_idxs = self._artist_candidate_indices(targets_lower)
        if cand_idxs is None or cand_idxs.size == 0:
            sims = self.features_unit @ profile  # (N,)
            pop = self.popularity
            artist_bonus = 0.0
            # mask seen
            seen_mask = np.zeros_like(sims, dtype=bool)
            for tid in seen_ids:
                idx = self.id_to_idx.get(tid)
                if idx is not None:
                    seen_mask[idx] = True
            sims = np.where(seen_mask, -np.inf, sims)
        else:
            sims = self.features_unit[cand_idxs] @ profile  # (C,)
            pop = self.popularity[cand_idxs]
            artist_bonus = self.ARTIST_BONUS
            # mask seen
            seen_mask = np.isin(cand_idxs, [self.id_to_idx[tid] for tid in seen_ids if tid in self.id_to_idx])
            sims = np.where(seen_mask, -np.inf, sims)

        scores = sims + pop * self.POP_WEIGHT + artist_bonus

        # Top-K without full sort
        k = min(n_recommendations, scores.size if scores.size else 0)
        recs: List[str] = []
        if k > 0:
            part = np.argpartition(scores, -k)[-k:]
            order = np.argsort(-scores[part], kind="mergesort")
            top_local = part[order]
            if cand_idxs is None or cand_idxs.size == 0:
                top_global = top_local
            else:
                top_global = cand_idxs[top_local]
            recs = [self.track_ids[int(i)] for i in top_global]

        # Fill if not enough candidates
        if len(recs) < n_recommendations:
            missing = n_recommendations - len(recs)
            recs.extend(self._popularity_fallback(set(seen_ids) | set(recs), missing, targets_lower))

        return recs


recommender = Recommender()
results = evaluate(recommender)
print(results)
