# Recommender (Weighted Content-Based)

## Procedure

1. **Load and deduplicate** `src/data/dataset.csv`, preferring the most popular copy per `track_id`, and parse the `artists` column into per-track sets for later filtering/boosting.
2. **Select numeric audio features** from `BASE_NUMERIC_FEATURES`, coerce them to floats, median-impute any gaps, standardize them with `StandardScaler`, and apply the configured `DEFAULT_WEIGHTS` (or caller overrides) before normalizing each row to unit length.
3. **Precompute auxiliaries** including popularity multipliers, genre → track indices, artist → track indices, and a fast `track_id → row index` map to accelerate inference.
4. **Handle recency and genre history** by precomputing geometric decay weights and tracking the most recent genres from the input seeds to bias scores toward matching styles.
5. **Serve recommendations** by forming a recency-weighted centroid of the seed vectors, optionally nudging toward the last track, applying cosine similarity, genre/artist bonuses, de-duplication, and popularity-based fallbacks if the requested count is not met.

## Code Walkthrough

- `Recommender.__init__` resolves file paths, ingests the dataset, deduplicates, and prepares the numeric feature matrix (`Xw_unit`), popularity multipliers, genre/artist inverted indices, and caching helpers used during inference.
- `_resolve_weights` merges the default class-level weights with any per-instance overrides, allowing quick experimentation without retraining.
- `_recency_weights`, `_genre_preferences`, and `_apply_genre_bonus` collectively implement the playlist-awareness within this content-only model. They compute a decayed weight per seed position, infer the most recent genres, and boost candidates matching those genres.
- `_artist_boost_indices` and the artist-related constants (`ARTIST_BASE_BONUS`, `ARTIST_SIMILARITY_BONUS`) make it possible to incorporate the `target_artist` hint from the evaluation harness by selectively raising candidate scores for matching artists.
- `_fallback_from_popularity` guarantees deterministic output when none of the input seeds exist in the dataset: it iterates over the popularity-sorted list, prioritizing tracks by target artists, and fills the remainder with top-popularity items.
- `get_recommendations` is the main entry point. It filters seeds that exist in the dataset, builds a centroid with recency weighting, calculates cosine similarities (`scores`), applies popularity multipliers and genre/artist boosts, excludes already seen tracks, and returns the highest-scoring unique track IDs. If insufficient unique results are found, it expands the search and finally falls back to popularity.

### How to Use

```bash
python src/recommender_weight_based.py
```

Initialize `Recommender(dataset_file=..., feature_weights=...)` to point at different datasets or tweak the audio feature weights. The class exposes `set_weights` to re-weight features after initialization, which is useful for quick tuning when running offline evaluations like `mpd_eval.py`.
