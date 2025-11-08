# VAE-based Spotify Track Recommender

This project trains a simple Variational Autoencoder (VAE) on Spotify track audio/meta features and uses latent space cosine similarity to recommend songs.

## Features Used
Numeric: popularity, duration_ms, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo.
Categorical: explicit, mode, key, time_signature, track_genre.

High-cardinality text fields (artists, album_name, track_name) are excluded from the VAE and retained only for metadata and optional artist boosting.

## Train the VAE
```bash
python -m vae.train_vae --csv data/dataset.csv --epochs 15 --latent-dim 32 --hidden 256,128 --beta 1.0
```

Artifacts saved to `vae/artifacts/`:
* `vae_model.pt` – model state dict and config
* `preprocessor.joblib`, `used_cols.joblib` – sklearn pipeline and column list
* `meta.parquet` – metadata for mapping and display
* `latents.npy` – mean latent vectors for all tracks

## Use the Recommender
```python
from data.recommender import recommender
recommendations = recommender.get_recommendations([
    "5SuOikwiRyPMVoIQDJUgSV",  # example track IDs
    "4qPNDBW1i3p13qLCt0Ki3A"
], n_recommendations=10, target_artist=set())
print(recommendations)
```

Optionally boost certain artists:
```python
recommendations = recommender.get_recommendations([
    "5SuOikwiRyPMVoIQDJUgSV"
], 10, {"Jason Mraz"})
```

## Recommendation Logic
1. Load precomputed latent vectors (mean of posterior) for all tracks.
2. Average latent vectors for input seed tracks to form a user/profile embedding.
3. Compute cosine similarity between profile and all track latents.
4. Apply a small similarity boost (+0.05) for tracks whose artist matches any in `target_artist`.
5. Return top-k excluding provided seed tracks.

## Extending
- Replace average pooling with a weighted scheme (e.g., popularity weighting).
- Incorporate artist or genre embeddings directly into the VAE input.
- Use ANN libraries (e.g., FAISS) for faster large-scale similarity search.
- Add diversification (e.g., penalize very similar tracks to broaden results).

## Dependencies
See `requirements.txt`. Create a virtual environment and install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes
If the evaluation harness `evaluation.py` is not present, the recommender prints a manual usage hint. Integrate with your scoring script by importing the `recommender` instance.

## License
Provided dataset terms belong to Spotify; model code here is MIT (add license file as needed).
