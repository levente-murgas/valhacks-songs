# Recommendation Pipeline Overview

This document explains the end-to-end flow for generating playlist continuation recommendations in the `valhacks-songs` project. It covers the data artifacts, feature engineering, candidate generation, heuristic ranking, evaluation, and playlist building utilities that together power the recommender.

## 1. Data Foundations

- **Raw catalog**  
  - Source file: `data/dataset.csv`  
  - Contains Spotify track metadata and audio features (`danceability`, `energy`, `valence`, etc.), plus identifiers (`track_id`, `artists`, `track_genre`).  
  - Ingestion script: `src/data_ingestion.py` loads the CSV, drops unnamed index columns, and produces two key assets:
    - `data/processed/dataset_clean.csv` – cleaned copy of the raw catalog.
    - `data/processed/dataset_normalized.csv` – same catalog with z-scored versions of each numeric feature appended.  
    - `data/processed/numeric_scaler.joblib` – fitted `StandardScaler` for reuse.

- **Exploratory analysis**  
  - `src/exploration.py` visualises distributions and correlations, but does not modify the pipeline.  

- **Reusable loading helpers**  
  - `seed_profile.load_normalized_dataset` reads the processed catalog and is shared by ranking, evaluation, and playlist utilities.

## 2. Seed Profile Construction

File: `src/seed_profile.py`

Purpose: summarise a playlist seed (one or more track IDs) into a feature-rich representation used downstream.

Steps:

1. **Deduplicate catalog by `track_id`** to avoid repeated rows.
2. **Match requested track IDs** and record:
   - `present_track_ids` – successfully matched IDs.
   - `missing_track_ids` – IDs not found (used for diagnostics).
3. **Exponential weighting**  
   - Use an exponentially increasing weight (newest tracks weighted more) to mimic playlist flow.
4. **Aggregate features**  
   - Weighted averages over original numeric features (`popularity`, `tempo`, etc.).
   - Weighted averages over z-scored features (for cosine similarity).
5. **Categorical signals**  
   - Genre distribution (probabilities per `track_genre`).
   - Artist set (unique artists gleaned from semicolon-delimited list).
   - Explicit rate (share of tracks marked explicit).
   - Major-mode rate (share of tracks with `mode == 1`).
   - `recent_track_ids` (the last three IDs for short-term similarity).

Output: a `SeedProfile` dataclass consumed by ranking logic.

## 3. Candidate Generation

File: `src/candidate_generation.py`

Goal: retrieve a large set of acoustically similar tracks using approximate nearest neighbours.

Pipeline:

1. Load `dataset_normalized.csv`, deduplicate, and extract the z-score matrix.
2. Fit a cosine-distance `NearestNeighbors` model (brute-force) to build a retrieval index:
   - Persisted at `models/candidate_index.joblib` (includes model, track IDs, feature matrix).
3. For a given seed profile:
   - Build a query vector from the weighted z-score centroid.
   - Run `kneighbors` to fetch top matches (typically 500).
   - Filter out seed tracks (and optionally any exclusion list).
   - Return ordered candidate IDs for ranking.

## 4. Heuristic Ranking

File: `src/ranking.py`

This is the core scoring logic turning candidates into final recommendations.

### 4.1 Feature computation

For each candidate track:

- **Base similarity**: cosine similarity between candidate z-score vector and seed centroid.
- **Short-term similarity**: weighted cosine similarity to the seed's most recent tracks (weights: 0.5, 0.3, 0.2).
- **Genre weight**: probability of the candidate's genre in the seed profile.
- **Artist overlap**: 1 if candidate artist appears in seed artist set, else 0.
- **Target artist boost**: 1 if candidate artist matches `target_artist` hint.
- **Explicit and mode alignment**: penalties when seed preference conflicts with candidate (e.g. explicit mismatch, major/minor mismatch).
- **Novelty bonus**: scaled by z-score of popularity to slightly prefer less mainstream tracks while avoiding extreme negatives.
- **Variance penalty**: dampens candidates that differ dramatically in tempo, energy, or valence beyond seed tolerances.

### 4.2 Score aggregation

Final score = weighted combination:

```
0.45 * base_similarity
+ 0.35 * short_term_similarity
+ 0.15 * genre_weight
+ 0.10 * artist_overlap
+ 0.20 * target_artist_boost
+ novelty_bonus
- explicit_penalty
- mode_penalty
- variance_penalty
```

### 4.3 Diversity rules

- After sorting by final score, enforce artist diversity (max one track per artist) unless the candidate belongs to the target-artist hint set.
- If diversity filter depletes too many tracks, fall back to the original ranking to fill remaining slots.

### 4.4 Public API

- `recommend_tracks` – main entry, given track IDs and optional target artist set.
- `recommend_tracks_with_resources` – same but allows passing preloaded dataset/index (used by `Recommender`).
- `generate_and_store_index` – convenience wrapper to build and save the ANN index.

## 5. Recommender Class

File: `data/recommender.py`

Responsibilities:

- Lazy-loads normalized dataset and candidate index (builds one if missing).
- Provides `get_recommendations` (IDs only) and `get_recommendations_with_details` (IDs plus detailed feature breakdown).
- Integrates with external evaluation harness when available.

Usage example:

```python
from data.recommender import Recommender

recommender = Recommender()
ids, details = recommender.get_recommendations_with_details(
    input_track_ids=["4l62h4tiuUwn7eD6hxMlnt"],
    n_recommendations=5,
    target_artist=set()
)
```

## 6. Evaluation Tools

File: `src/playlist_evaluation.py`

Functionality:

- `evaluate_playlists`: Given a DataFrame of historical playlists, hold out the last `k` tracks, run the recommender on the observed prefix, and compute:
  - Hit rate @ k
  - Mean Average Precision @ k
- Outputs `PlaylistEvaluationResult` entries with recommendation explanations.
- `summarise_results`: aggregates metrics across playlists.

Assumes playlists reference tracks present in our catalog. (Feature-only evaluation paths were removed to keep the pipeline focused on in-catalog testing.)

## 7. Playlist Builder CLI

File: `src/playlist_builder.py`

Purpose: user-facing script to seed a playlist (even with a single track) and iteratively extend it.

Features:

- Random seed selection if none provided.
- Iterative growth:
  - `--total-tracks`: desired output length.
  - `--step-size`: how many new tracks to request per iteration (default 5).
  - Repeatedly call the recommender, appending recommendations until reaching total length.
- Output options:
  - `--json`: print playlist payload to stdout.
  - `--output <path>`: save playlist + explanations as JSON file.
  - `--text-output <path>`: save plain-text lines formatted as `Artist - Title`.

Example:

```bash
python src/playlist_builder.py \
  --total-tracks 20 \
  --step-size 3 \
  --seed-track-id 59mrqUmhpmcfUns8BKkV30 \
  --target-artist "Marc Anthony" \
  --output data/my_playlist.json \
  --text-output data/my_playlist.txt
```

## 8. Workflow Summary

1. **Ingest catalog** (`data_ingestion.py`) -> normalized dataset & scaler.
2. **Build ANN index** (`candidate_generation.py --build-index`) once.
3. **For each recommendation request**:
   - Construct a `SeedProfile` from seed track IDs.
   - Query candidate index to obtain similar tracks.
   - Compute heuristic features and scores.
   - Apply diversity and target-artist logic to select top N.
4. **Evaluate** (optional) using `playlist_evaluation.py`.
5. **Generate playlists** interactively via `playlist_builder.py`.

This modular design keeps the heavy preprocessing (scaling, ANN index) off the request path, while the heuristic ranker provides explainable scoring and quick iteration for tuning weights.

