# Recommender (Playlist-Aware)

## Procedure

1. **Load metadata** from `src/data/dataset.csv`, clean duplicate tracks, parse artist strings, and build a normalized numeric feature matrix (standardized + per-feature weights + cosine-ready vectors).
2. **Discover curated playlists** automatically under `src/data/human_curated_playlists/` (JSON or JSONL). If none exist, synthesize surrogate playlists grouped by genre/album to keep the pipeline functional.
3. **Build a co-occurrence graph** by iterating through each curated playlist, deduplicating per playlist, and accumulating windowed co-play counts (distance-weighted) plus track frequency statistics. Store the top-N neighbors per track to keep inference fast.
4. **Prepare popularity priors** from the dataset and an artist inverted index to support fallbacks and artist hint boosts.
5. **Serve recommendations** by blending the playlist graph signal, content centroid similarity, popularity priors, and target-artist hints, while ensuring candidates are unique and exclude seen seeds.

## Code Walkthrough

- `Recommender.__init__` wires the whole pipeline: loads the dataset, builds the feature space (`_build_feature_space`), caches popularity, and materializes curated playlists via `_load_curated_playlists` (falling back to `_build_surrogate_playlists`). It then calls `_build_playlist_graph` to populate `playlist_neighbors`, `playlist_freq`, and normalization factors.
- `_build_feature_space` converts the numeric columns to floats, median-imputes missing values, standardizes them using `StandardScaler`, applies optional per-feature weights, and normalizes every track vector to unit length so cosine similarity reduces to a dot product.
- `_load_curated_playlists` accepts either a folder of JSON/JSONL slices (like MPD) or a single file. `_parse_playlist_file` and `_extract_single_playlist` normalize multiple schemas by calling `_parse_track_id` on each track entry, enforcing playlist length bounds.
- `_build_playlist_graph` iterates through deduplicated playlists, counts occurrence frequency, and, for each track, stores the strongest neighbors within a sliding window. It also computes frequency-normalized weights so popular tracks do not dominate.
- `_playlist_signal`, `_content_scores`, and `_popularity_fallback` are the three scoring branches used during inference. `_playlist_signal` projects the seed indices onto the co-occurrence graph, `_content_scores` computes cosine similarity from the recency-weighted centroid, and `_popularity_fallback` ensures coverage even when no data overlaps.
- `get_recommendations` orchestrates inference: filters valid seed indices, computes recency weights, obtains playlist/content signals, builds a candidate pool, mixes the weighted scores (playlist > content > popularity) plus `ARTIST_HINT_BONUS`, and finally yields a deduplicated, length-controlled list of track IDs.

### How to Use

```bash
python src/recommender_trained.py          # quick sanity run
```

The constructor automatically picks up curated playlists under `src/data/human_curated_playlists/`. Pass `curated_source` explicitly if you want a different folder/file, or `feature_weights` if you need custom numeric weighting. For bulk evaluation, point `mpd_eval.py` at this recommender by updating the import target.
