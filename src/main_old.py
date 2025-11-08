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

    High-level idea:
      1) Load tracks with numeric audio features.
      2) Standardize features so each column has mean 0 and stdev 1.
      3) Apply user-provided weights to each feature (your "pondering" knobs).
      4) Compute a centroid (average) vector of the seed tracks in this space.
      5) Rank all tracks by cosine similarity to that centroid.
      6) Apply tiny multipliers for artist overlap and (optionally) popularity.
      7) Return the top-N track IDs.

    Why this works:
      - Standardization makes features comparable.
      - Weights reflect what you care about more/less (danceability, energy...).
      - Cosine similarity captures "directional" similarity of the feature vector.
    """

    def __init__(self, dataset_file: str = "/dataset.csv", weights_file: str = "/feature_weights.json"):
        """
        Args:
          dataset_file: path (relative to src/data/) to the CSV with tracks.
          weights_file: path (relative to src/data/) to a JSON dict of {feature: weight}.

        Note:
          We resolve a base folder "src/data/" and then append dataset_file
          and weights_file. If you pass absolute paths here, you're effectively
          concatenating strings. Prefer passing relative names like "/dataset.csv"
          as in this default, or switch to Path-joining for robustness.
        """

        # Resolve a base folder where data is expected to live: "src/data/"
        # os.path.dirname("src/data/") -> "src/data"
        # abspath(...) -> absolute path to that folder based on current working dir.
        data_path = os.path.abspath(os.path.dirname("src/data/"))

        # Read the CSV into a DataFrame. We concatenate the base folder + filename.
        # Example result: "<abs>/src/data/" + "/dataset.csv" -> "<abs>/src/data//dataset.csv"
        # (double slash is harmless on POSIX)
        self.df = pd.read_csv(data_path + dataset_file)

        # Drop rows without a track_id, because we need IDs to map and to return
        self.df = self.df.dropna(subset=["track_id"]).copy()

        # Turn the "artists" column (e.g., "A; B; C") into a Python set for fast overlap checks
        # - Fill missing with empty string
        # - Split by ';'
        # - Strip whitespace
        # - Build a set of non-empty artist names
        self.df["artists_set"] = self.df["artists"].fillna("").apply(
            lambda s: set([a.strip() for a in str(s).split(";") if a.strip()])
        )

        # Core numeric audio features used for similarity. Adjust/extend as needed.
        self.numeric_features = [
            "danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
            "popularity", "key", "mode", "time_signature"
        ]

        # Ensure every chosen feature is numeric and has no NaNs:
        # - to_numeric(..., errors="coerce") converts invalid strings to NaN
        for col in self.numeric_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # -----------------------------
        # Load user-defined feature weights
        # -----------------------------
        # Default every feature to weight 1.0 (neutral).
        self.weights = {f: 1.0 for f in self.numeric_features}

        # Build a Path to the weights file beside the dataset (same base folder).
        weights_fp = Path(data_path + weights_file)

        # If a JSON file exists, parse it and override defaults.
        # Expected format: {"danceability": 1.2, "energy": 0.8, ...}
        if weights_fp.exists():
            try:
                user_w = json.loads(weights_fp.read_text())
                for k, v in user_w.items():
                    if k in self.weights:
                        # Safely cast to float; ignore unknown keys
                        self.weights[k] = float(v)
            except Exception:
                # If parsing fails, silently keep defaults.
                # (You may want to log a warning in a production system.)
                pass

        # -----------------------------
        # Build the feature matrix
        # -----------------------------
        # X: shape (num_tracks, num_features), dtype float
        self.X = self.df[self.numeric_features].to_numpy().astype(float)

        # Standardize columns to mean=0, stdev=1 so units don't dominate similarity.
        self.scaler = StandardScaler()
        Xz = self.scaler.fit_transform(self.X)

        # Create a diagonal matrix W with your per-feature weights on the diagonal
        # Then compute Xw = Xz @ W to get the weighted standardized features.
        W = np.diag([self.weights[f] for f in self.numeric_features])
        self.Xw = Xz @ W

        # Map track_id -> row index for O(1) lookup of seed rows later
        self.id_to_idx = {tid: i for i, tid in enumerate(self.df["track_id"])}

    def _artist_boost(self, row_artists: set, target_artist: Set[str]) -> float:
        """
        Small multiplicative boost if the candidate's artists overlap with the hint.

        Args:
          row_artists: set of artists for the candidate row.
          target_artist: set of one or more artist names provided as a hint.

        Returns:
          A multiplier (>= 1.0). Default 1.10 if there's ANY overlap; 1.0 otherwise.
        """
        if not target_artist:
            return 1.0
        return 1.10 if row_artists & target_artist else 1.0

    def _popularity_prior(self, popularity: float) -> float:
        """
        Gentle multiplier based on 'popularity' (if available), so more popular
        tracks get a small nudge but don't dominate.

        Mapping (linear):
          popularity in [0, 100] -> multiplier in ~[0.95, 1.10]

        Args:
          popularity: numeric popularity (NaN-safe).

        Returns:
          A multiplier in [~0.95, ~1.10]. Returns 1.0 if popularity is NaN.
        """
        if np.isnan(popularity):
            return 1.0
        return 0.95 + (float(popularity) / 100.0) * 0.15

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[
        str]:
        """
        Produce N recommended track_ids given one or more seed tracks and an optional artist hint.

        Steps:
          1) Find row indices for all valid seed track_ids.
          2) If none found, fall back to the most popular tracks.
          3) Compute the centroid (mean vector) of the seeds in weighted space.
          4) Compute cosine similarity of every track to that centroid.
          5) Exclude seeds themselves (score = -inf).
          6) Multiply each candidate score by:
             - artist boost (if an overlap with target_artist),
             - popularity prior (if the 'popularity' column exists).
          7) Sort by final score and return the top-N track_ids.
        """

        # Convert seed IDs to indices (skip any that aren't in the dataset)
        seed_idx = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]

        # If no seeds are valid/found, return top-N by popularity as a reasonable default
        if not seed_idx:
            fallback = (
                self.df.drop_duplicates("track_id")
                .sort_values("popularity", ascending=False)["track_id"]
                .head(n_recommendations)
                .tolist()
            )
            return fallback

        # Compute centroid (average vector) of the seeds in the weighted feature space
        centroid = self.Xw[seed_idx].mean(axis=0, keepdims=True)

        # Cosine similarity between every track and the centroid -> base relevance
        sims = cosine_similarity(self.Xw, centroid).ravel()

        # Start with similarity as the score
        scores = sims.copy()

        # Whether the dataset has 'popularity' column
        has_pop = "popularity" in self.df.columns

        # Adjust scores:
        # - Exclude seeds by setting their score to -inf
        # - Apply artist boost and popularity prior to candidates
        for i in range(len(scores)):
            if i in seed_idx:
                scores[i] = -np.inf  # exclude seeds from being recommended
                continue

            # Small multiplier if the artist overlaps the hint set
            boost = self._artist_boost(self.df.iloc[i]["artists_set"], target_artist)

            # Gentle popularity multiplier if present; otherwise 1.0 (no change)
            popp = self._popularity_prior(self.df.iloc[i]["popularity"]) if has_pop else 1.0

            # Multiply in-place (keeps cosine similarity as the main signal)
            scores[i] *= (boost * popp)

        # Rank by descending score (np.argsort sorts ascending; negate for descending)
        order = np.argsort(-scores)

        # Collect top-N track_ids, skipping seeds and any non-finite scores
        rec_ids = []
        for j in order:
            tid = self.df.iloc[j]["track_id"]
            if tid not in input_track_ids and np.isfinite(scores[j]):
                rec_ids.append(tid)
            if len(rec_ids) >= n_recommendations:
                break

        return rec_ids


if __name__ == "__main__":
    # Minimal smoke test: build the recommender and query with two seeds & an artist hint
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
