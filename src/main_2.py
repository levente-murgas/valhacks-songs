import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Set, Dict, Iterable

# -------------------------
# Small utilities
# -------------------------

def _to_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)

class Recommender:
    """
    Fast, content-based recommender (centroid + cosine) with vectorized reranking.

    Pipeline (fit-time / __init__):
      - Read CSV
      - Clean + impute numerics (median)
      - Standardize each column (z-score)
      - Apply per-feature weights
      - Row-normalize to unit vectors (cosine -> dot)
      - Precompute:
          * popularity multipliers
          * artist->row inverted index
          * id->row index map

    Query (get_recommendations):
      - Build centroid of seed rows in the same space
      - Normalize centroid to unit vector
      - Scores = X_unit @ centroid_unit           (cosine similarity)
      - scores *= pop_mult                        (vectorized)
      - if target_artist: scores[artist_rows] *= 1.10
      - Exclude seeds (score = -inf)
      - Top-K via argpartition (O(N)) then small sort on K
    """

    # Default feature list; adjust to your schema
    BASE_NUMERIC_FEATURES = [
        "danceability","energy","loudness","speechiness","acousticness",
        "instrumentalness","liveness","valence","tempo","duration_ms",
        "popularity", "key", "mode", "time_signature"
    ]

    def __init__(self, dataset_file: str = "dataset.csv", weights_file: str = "feature_weights.json"):
        # ---------- Resolve paths ----------
        data_root = Path("src/data").resolve()
        dataset_path = data_root.joinpath(dataset_file)
        weights_path = data_root.joinpath(weights_file)

        print(Path("src/data").resolve().joinpath(dataset_file))

        # ---------- Load data ----------
        df = pd.read_csv(dataset_path)
        df = df.dropna(subset=["track_id"]).copy()

        # Artists as sets for readability + to build an inverted index
        df["artists_set"] = (
            df["artists"]
            .fillna("")
            .apply(lambda s: {a.strip() for a in str(s).split(";") if a.strip()})
        )

        # Keep only features that actually exist
        numeric_features = [c for c in self.BASE_NUMERIC_FEATURES if c in df.columns]
        if not numeric_features:
            raise ValueError("No numeric features from BASE_NUMERIC_FEATURES found in dataset.")

        # ---------- Coerce + Impute ----------
        # Coerce to numeric, compute medians, fill NaN with median
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        medians = df[numeric_features].median(axis=0, skipna=True)
        df[numeric_features] = df[numeric_features].fillna(medians)

        # ---------- Load weights ----------
        weights: Dict[str, float] = {f: 1.0 for f in numeric_features}
        if weights_path.exists():
            try:
                user_w = json.loads(weights_path.read_text())
                for k, v in user_w.items():
                    if k in weights:
                        weights[k] = float(v)
            except Exception:
                # Keep defaults if parsing fails
                pass

        # ---------- Build feature matrix (float32) ----------
        X = _to_float32(df[numeric_features].to_numpy())

        # Standardize with simple numpy (faster than allocating sklearn objects repeatedly)
        mean = _to_float32(np.mean(X, axis=0))
        std = _to_float32(np.std(X, axis=0))
        std[std == 0] = 1.0  # avoid divide-by-zero if a column is constant
        Xz = (X - mean) / std

        # Apply weights by column (no need to build a diagonal matrix)
        wvec = _to_float32(np.array([weights[f] for f in numeric_features], dtype=np.float32))
        Xw = Xz * wvec  # broadcasting

        # Row-normalize to unit length so cosine similarity = dot product
        norms = np.linalg.norm(Xw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xw_unit = Xw / norms

        # ---------- Precompute popularity multipliers (vector) ----------
        if "popularity" in numeric_features:
            pop = _to_float32(df["popularity"].to_numpy())
            # clip to [0,100] just in case, then map linearly to ~[0.95, 1.10]
            pop = np.clip(pop, 0.0, 100.0)
            pop_mult = _to_float32(0.95 + (pop / 100.0) * 0.15)
        else:
            pop_mult = _to_float32(np.ones(len(df), dtype=np.float32))

        # ---------- Build artist inverted index ----------
        # artist -> np.array(row_indices)
        artist_to_idx: Dict[str, np.ndarray] = {}
        for i, aset in enumerate(df["artists_set"].values):
            for a in aset:
                if a not in artist_to_idx:
                    artist_to_idx[a] = [i]
                else:
                    artist_to_idx[a].append(i)
        for a in list(artist_to_idx.keys()):
            artist_to_idx[a] = np.fromiter(artist_to_idx[a], dtype=np.int32)

        # ---------- Save state ----------
        self.df = df
        self.numeric_features = numeric_features
        self.weights = weights
        self.mean = mean
        self.std = std
        self.wvec = wvec
        self.Xw_unit = Xw_unit
        self.pop_mult = pop_mult
        self.artist_to_idx = artist_to_idx
        self.id_to_idx = {tid: i for i, tid in enumerate(df["track_id"])}

        # Pre-store numpy view for quick access
        self.track_ids = df["track_id"].to_numpy()

    # ----- Small helpers for multipliers -----
    @staticmethod
    def _artist_boost_indices(artist_to_idx: Dict[str, np.ndarray], target_artist: Set[str]) -> np.ndarray:
        """
        Return a deduplicated array of row indices that contain ANY artist in target_artist.
        """
        if not target_artist:
            return np.empty(0, dtype=np.int32)
        gathered: Iterable[np.ndarray] = (artist_to_idx.get(a, None) for a in target_artist)
        idx_list = [arr for arr in gathered if arr is not None]
        if not idx_list:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(idx_list))

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        # ---- Map seeds to indices ----
        seed_idx = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]

        # Fallback if no valid seeds: top by popularity (already preloaded in df)
        if not seed_idx:
            if "popularity" in self.df.columns:
                return (
                    self.df.drop_duplicates("track_id")
                           .sort_values("popularity", ascending=False)["track_id"]
                           .head(n_recommendations)
                           .tolist()
                )
            # If no popularity column, just return any top-N distinct
            return self.df.drop_duplicates("track_id")["track_id"].head(n_recommendations).tolist()

        # ---- Centroid in unit space ----
        # Average the unit vectors of the seed rows; then re-normalize to unit length
        centroid = np.mean(self.Xw_unit[seed_idx], axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm == 0:
            # degenerate seed; fall back to unweighted averaging in case of odd input
            centroid = np.mean(self.Xw_unit, axis=0)
            c_norm = np.linalg.norm(centroid)
        centroid_unit = centroid / (c_norm if c_norm != 0 else 1.0)

        # ---- Cosine similarity = dot product ----
        scores = self.Xw_unit @ centroid_unit  # shape: (N,)

        # ---- Popularity multiplier (vectorized) ----
        scores *= self.pop_mult

        # ---- Artist boost (vectorized) ----
        if target_artist:
            idx = self._artist_boost_indices(self.artist_to_idx, target_artist)
            if idx.size > 0:
                scores[idx] *= 1.10  # small, controlled boost

        # ---- Exclude seeds ----
        scores[np.asarray(seed_idx, dtype=np.int32)] = -np.inf

        # ---- Top-K via argpartition for speed ----
        k = min(n_recommendations, scores.size)
        # Get K best indices in arbitrary order, then sort just those K
        part_idx = np.argpartition(-scores, k - 1)[:k]
        top_sorted_local = np.argsort(-scores[part_idx])
        top_idx = part_idx[top_sorted_local]

        # ---- Map to track_ids (preserve order) ----
        # No duplicates in track_id expected; if present, drop_duplicates in init is an option.
        rec_ids = [str(self.track_ids[i]) for i in top_idx[:n_recommendations] if np.isfinite(scores[i])]
        return rec_ids

if __name__ == "__main__":
    r = Recommender()
    print("Columns used:", r.numeric_features)
    print(
        r.get_recommendations(
            ["5SuOikwiRyPMVoIQDJUgSV", "0wihfILRNOwE2156Shezc8", "63bmIgH9sS6sX5Sc7MetGq", "3wpZTp7HM8Dv25oExNgCC6",
             "3QAE1arPJAMVKt3NUqjikE"],  # seed track_ids
            10,  # how many to return
            {"Gen Hoshino", "Mariah Angeliq"}  # optional artist hint
        )
    )
