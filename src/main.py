import numpy as np
import json
import pandas as pd
from typing import List, Set, Dict, Final
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ARTIST_BOOST: Final[float] = 1.50

def _safe_path(base: Path, maybe_abs: str) -> Path:
    p = Path(maybe_abs)
    if p.is_absolute():
        return p
    return base / maybe_abs.lstrip("/")

class Recommender:
    """
    Content-based recommender (centroid + cosine) with:
      - De-duplication by track_id
      - Exclusion by track_id (not just row index)
      - Unique outputs guaranteed
      - Fast vectorized scoring + argpartition
    """

    BASE_NUMERIC_FEATURES = [
        "danceability","energy","loudness","speechiness","acousticness",
        "instrumentalness","liveness","valence","tempo","duration_ms",
        "popularity","key","mode","time_signature"
    ]

    def __init__(self, dataset_file: str = "dataset.csv", weights_file: str = "feature_weights.json"):
        # ---------- Resolve paths ----------
        data_root = Path("src/data").resolve()
        dataset_path = _safe_path(data_root, dataset_file)
        weights_path = _safe_path(data_root, weights_file)

        # ---------- Load data ----------
        df = pd.read_csv(dataset_path)
        # ---- de-duplicate by track_id BEFORE anything else ----
        if "popularity" in df.columns:
            # Keep the most popular row for each track_id (stable sort for determinism)
            df["__pop__"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(-1)
            df = (df.sort_values("__pop__", ascending=False, kind="mergesort")
                    .drop_duplicates("track_id", keep="first")
                    .drop(columns="__pop__")
                    .reset_index(drop=True))
        else:
            df = df.drop_duplicates("track_id", keep="first").reset_index(drop=True)

        # Drop rows without a usable id
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

        # ---------- Load weights ----------
        weights: Dict[str, float] = {f: 1.0 for f in numeric_features}
        if weights_path.exists():
            try:
                user_w = json.loads(weights_path.read_text())
                for k, v in user_w.items():
                    if k in weights:
                        weights[k] = float(v)
            except Exception:
                pass

        # ---------- Build standardized, weighted, unit-normalized matrix ----------
        X = df[numeric_features].to_numpy(dtype=np.float32)

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xz = scaler.fit_transform(X).astype(np.float32)

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
        self.wvec = wvec
        self.Xw_unit = Xw_unit
        self.pop_mult = pop_mult
        self.artist_to_idx = artist_to_idx
        self.id_to_idx = {tid: i for i, tid in enumerate(self.df["track_id"])}
        self.track_ids = self.df["track_id"].to_numpy()

    @staticmethod
    def _artist_boost_indices(artist_to_idx: Dict[str, np.ndarray], target_artist: Set[str]) -> np.ndarray:
        if not target_artist:
            return np.empty(0, dtype=np.int32)
        chunks = [artist_to_idx[a] for a in target_artist if a in artist_to_idx]
        if not chunks:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(chunks))

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        # Map seeds → indices
        seed_idx = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]

        # Fallback if no valid seeds
        if not seed_idx:
            base = self.df
            if "popularity" in self.df.columns:
                base = base.sort_values("popularity", ascending=False, kind="mergesort")
            out = []
            seen = set(input_track_ids)
            for tid in base["track_id"]:
                if tid not in seen:
                    out.append(tid); seen.add(tid)
                if len(out) >= n_recommendations:
                    break
            return out

        # Centroid (mean of seed unit vectors), renormalized to unit
        centroid = np.mean(self.Xw_unit[seed_idx], axis=0)
        cn = np.linalg.norm(centroid)
        centroid_unit = centroid / (cn if cn != 0 else 1.0)

        # Cosine = dot product with pre-normalized rows
        scores = (self.Xw_unit @ centroid_unit).astype(np.float32)

        # Popularity multiplier (vectorized)
        scores *= self.pop_mult

        # Artist boost (vectorized)
        if target_artist:
            idx = self._artist_boost_indices(self.artist_to_idx, target_artist)
            if idx.size > 0:
                scores[idx] *= ARTIST_BOOST

        # ---- exclude *all rows* whose track_id is one of the seeds ----
        seed_mask = self.df["track_id"].isin(input_track_ids).to_numpy()
        scores[seed_mask] = -np.inf

        # Top-K using argpartition with oversampling to survive uniqueness filtering
        N = scores.size
        k = min(max(n_recommendations * 3, n_recommendations), N)
        part_idx = np.argpartition(-scores, k - 1)[:k]

        # Stable final order within the slice for determinism
        local_sorted = part_idx[np.argsort(-scores[part_idx], kind="mergesort")]

        # Collect unique IDs, preserving score order
        rec_ids: List[str] = []
        seen: Set[str] = set(input_track_ids)  # also prevents seed IDs sneaking back
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

        # If oversampling wasn't enough (rare), fall back to full sort once
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

if __name__ == "__main__":
    r = Recommender()
    print(
        r.get_recommendations(
            [
                "7o2CTH4ctstm8TNelqjb51",
                "2zYzyRzz6pRmhPzyfMEC8s",
                "08mG3Y1vljYA6bvDt4Wqkj",
                "0bVtevEgtDIeRjCJbK3Lmv",
                "3YBZIN3rekqsKxbJc9FZko",
                "57bgtoPSgt236HzfBOd8kj",
                "7LRMbd3LEoV5wZJvXT1Lwb",
                "2SiXAy7TuUkycRVbbWDEpo",
                "0C80GCp0mMuBzLf3EAXqxv"
            ],
            2,
            {"AC/DC", "Europe", "Guns N' Roses"}
        )
    )
