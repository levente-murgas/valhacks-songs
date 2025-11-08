from __future__ import annotations

from pathlib import Path
import argparse
from typing import List, Set

import numpy as np
import pandas as pd
import torch
import joblib
import wandb



# Lazy import for optional evaluation harness
try:
    from evaluation import evaluate  # type: ignore
except Exception:  # pragma: no cover - local dev fallback
    evaluate = None  # type: ignore


class Recommender:
    """VAE-based content recommender using song feature embeddings.

    Artifacts expected under vae/artifacts:
      - vae_model.pt
      - preprocessor.joblib
      - used_cols.joblib
      - latents.npy
      - meta.parquet
    """

    def __init__(self, artifact_dir: str) -> None:
        self._loaded = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.artifact_dir = artifact_dir

    def _load(self):
        if self._loaded:
            return
        art = Path(self.artifact_dir)
        if not art.exists():
            raise FileNotFoundError("Artifacts not found. Please run: python -m vae.train_vae")

        self.latents = np.load(art / "latents.npy")  # [N, D]
        self.preprocessor = joblib.load(art / "preprocessor.joblib")
        self.used_cols = joblib.load(art / "used_cols.joblib")
        self.meta = pd.read_parquet(art / "meta.parquet")

        # Build id->index mapping
        self.id_to_idx = {tid: i for i, tid in enumerate(self.meta["track_id"].tolist())}

        # Normalize latents for cosine similarity
        norms = np.linalg.norm(self.latents, axis=1, keepdims=True) + 1e-10
        self.latents_norm = self.latents / norms

        self._loaded = True

    def _profile_from_ids(self, input_track_ids: List[str]) -> np.ndarray:
        idxs = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]
        if not idxs:
            raise ValueError("None of the input track IDs were found in the dataset.")
        vecs = self.latents_norm[idxs]
        prof = vecs.mean(axis=0)
        prof = prof / (np.linalg.norm(prof) + 1e-10)
        return prof

    def get_recommendations(self, input_track_ids: List[str], n_recommendations: int, target_artist: Set[str]) -> List[str]:
        """Return top-N recommended track IDs based on latent cosine similarity.

        Args:
            input_track_ids: Seed track IDs to build user/profile embedding.
            n_recommendations: Number of tracks to return.
            target_artist: Optional set of artist names to lightly boost.
        """
        self._load()
        profile = self._profile_from_ids(input_track_ids)
        sims = (self.latents_norm @ profile)

        if target_artist:
            artists_series = self.meta["artists"].fillna("")
            boost = artists_series.apply(lambda a: any(t in a.split(";") for t in target_artist))
            sims = sims + boost.to_numpy(dtype=float) * 0.05

        input_set = set(input_track_ids)
        mask = np.array([tid not in input_set for tid in self.meta["track_id"]], dtype=bool)
        sims_filtered = sims.copy()
        sims_filtered[~mask] = -1e9

        k = min(n_recommendations * 3, len(sims_filtered) - 1)
        topk_idx = np.argpartition(-sims_filtered, kth=k)[:k]
        topk_idx = topk_idx[np.argsort(-sims_filtered[topk_idx])]
        seen = set()
        dedup_ids: List[str] = []
        for tid in self.meta.iloc[topk_idx]["track_id"].tolist():
            if tid not in seen:
                seen.add(tid)
                dedup_ids.append(tid)
            if len(dedup_ids) >= n_recommendations:
                break
        return dedup_ids[:n_recommendations]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE-based recommender CLI")
    parser.add_argument("--seed", nargs="*", default=None, help="Seed track IDs (space-separated)")
    parser.add_argument("--n", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--target-artists", type=str, default="", help="Comma-separated artist names for slight boost")
    parser.add_argument("--print-meta", action="store_true", help="Print artists and track names alongside IDs")
    parser.add_argument("--artifact", type=str, default="", help="Optional local artifact directory; if omitted use wandb artifact")
    parser.add_argument("--wandb-artifact", type=str, default="valhacks-syntax-terror/valhacks/artifact:v0", help="WandB artifact reference")
    args = parser.parse_args()

    if args.artifact:
        artifact_dir = args.artifact
    else:
        run = wandb.init()
        artifact = run.use_artifact(args.wandb_artifact, type='vae_model')
        artifact_dir = artifact.download('./live_artifact')

    recommender = Recommender(artifact_dir)

    if args.seed:
        target = set([a.strip() for a in args.target_artists.split(",") if a.strip()])
        try:
            recs = recommender.get_recommendations(args.seed, args.n, target)
            if args.print_meta:
                df = recommender.meta.set_index("track_id").loc[recs]
                for tid in recs:
                    row = df.loc[tid]
                    print(f"{tid}\t{row.get('artists','?')}\t{row.get('track_name','?')}")
            else:
                print(recs)
        except ValueError as e:
            print(f"Error: {e}")
    else:
        if evaluate is not None:
            results = evaluate(recommender)
            print(results)
        else:
            print("No --seed provided and evaluation harness not found.")
            print("Usage: python data/recommender.py --seed <id1> <id2> --n 10 --target-artists 'Artist A,Artist B'")