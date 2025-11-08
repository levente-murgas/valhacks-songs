import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Set
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    """
    Content-based baseline using Spotify-style audio features.
    - Standardizes numeric features
    - Applies user-defined weights to each feature
    - Builds a centroid from the input tracks and ranks by cosine similarity
    - Light popularity prior and optional artist boosting using `target_artist`
    """

    def __init__(self, dataset_file: str = "/dataset.csv", weights_file: str = "/feature_weights.json"):
        data_path = os.path.abspath(os.path.dirname("src/data/"))

        self.df = pd.read_csv(data_path + dataset_file)
        self.df = self.df.dropna(subset=["track_id"]).copy()
        self.df["artists_set"] = self.df["artists"].fillna("").apply(lambda s: set([a.strip() for a in str(s).split(";") if a.strip()]))

        self.numeric_features = [
            "danceability","energy","loudness","speechiness","acousticness",
            "instrumentalness","liveness","valence","tempo","duration_ms"
        ]

        # Optional extras if present
        for opt in ["popularity","key","mode","time_signature"]:
            if opt in self.df.columns:
                self.numeric_features.append(opt)

        # Fill missing numeric values
        for col in self.numeric_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(self.df[col].median())

        # Load weights (or default to 1.0)
        self.weights = {f: 1.0 for f in self.numeric_features}
        weights_fp = Path(data_path + weights_file)
        if weights_fp.exists():
            try:
                user_w = json.loads(weights_fp.read_text())
                for k, v in user_w.items():
                    if k in self.weights:
                        self.weights[k] = float(v)
            except Exception:
                pass

        # Build feature matrix
        self.X = self.df[self.numeric_features].to_numpy().astype(float)

        # Standardize then apply weights
        self.scaler = StandardScaler()
        Xz = self.scaler.fit_transform(self.X)
        W = np.diag([self.weights[f] for f in self.numeric_features])
        self.Xw = Xz @ W

        # Map for fast id->row
        self.id_to_idx = {tid: i for i, tid in enumerate(self.df["track_id"])}

    def _artist_boost(self, row_artists: set, target_artist: Set[str]) -> float:
        if not target_artist:
            return 1.0
        # If any overlap, give a small multiplier
        return 1.10 if row_artists & target_artist else 1.0

    def _popularity_prior(self, popularity: float) -> float:
        if np.isnan(popularity):
            return 1.0
        # Smooth prior: map [0,100] -> [0.95, 1.10]
        return 0.95 + (float(popularity) / 100.0) * 0.15

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        # Gather valid indices
        seed_idx = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]
        if not seed_idx:
            # Fall back to most popular tracks if no seeds are found
            fallback = self.df.drop_duplicates("track_id").sort_values("popularity", ascending=False)["track_id"].head(n_recommendations).tolist()
            return fallback

        # Build centroid of seeds
        centroid = self.Xw[seed_idx].mean(axis=0, keepdims=True)

        # Cosine similarity
        sims = cosine_similarity(self.Xw, centroid).ravel()

        # Base score = similarity
        scores = sims.copy()

        # Apply artist boost and popularity prior
        has_pop = "popularity" in self.df.columns
        for i in range(len(scores)):
            if i in seed_idx:
                scores[i] = -np.inf  # exclude seeds
                continue
            boost = self._artist_boost(self.df.iloc[i]["artists_set"], target_artist)
            popp = self._popularity_prior(self.df.iloc[i]["popularity"]) if has_pop else 1.0
            scores[i] *= (boost * popp)

        # Rank
        order = np.argsort(-scores)
        rec_ids = []
        for j in order:
            tid = self.df.iloc[j]["track_id"]
            if tid not in input_track_ids and np.isfinite(scores[j]):
                rec_ids.append(tid)
            if len(rec_ids) >= n_recommendations:
                break
        return rec_ids


if __name__ == "__main__":
    r = Recommender()
    print("Columns used:", r.numeric_features)
    print(r.get_recommendations(
        ["5SuOikwiRyPMVoIQDJUgSV", "0wihfILRNOwE2156Shezc8"],
        10,
        {"Gen Hoshino"}
    ))

