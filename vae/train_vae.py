"""Training script for tabular VAE on Spotify track features.

Usage (after installing requirements):
    python -m vae.train_vae --csv data/dataset.csv --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from .model import MLPVAE, VAEConfig
from .data_utils import load_dataset, fit_transform_features, get_meta, transform_features
import wandb



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/dataset.csv", help="Path to dataset.csv")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden", type=str, default="256,128", help="Comma-separated hidden dims")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0, help="KL weight")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="vae/artifacts")
    p.add_argument("--exclude", type=str, default="", help="Comma-separated list of feature columns to exclude (e.g. popularity,track_genre)")
    p.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction (0 to disable validation)")
    return p.parse_args()


def build_dataloader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(X)
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train(model: MLPVAE, train_loader: DataLoader, val_loader: DataLoader | None, epochs: int, lr: float, device: str, run: wandb.Run) -> dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = model.loss_fn(batch, recon, mu, logvar)
            loss.backward()
            opt.step()
            running += loss.item() * batch.size(0)
        train_epoch_loss = running / len(train_loader.dataset)

        run.log({"train_loss": train_epoch_loss}, step=epoch)

        if val_loader is not None:
            # Validation
            model.eval()
            with torch.no_grad():
                val_running = 0.0
                for (vbatch,) in val_loader:
                    vbatch = vbatch.to(device)
                    vrecon, vmu, vlogvar = model(vbatch)
                    vloss = model.loss_fn(vbatch, vrecon, vmu, vlogvar)
                    val_running += vloss.item() * vbatch.size(0)
            val_epoch_loss = val_running / len(val_loader.dataset)

            history["train_loss"].append(train_epoch_loss)
            history["val_loss"].append(val_epoch_loss)
            run.log({"val_loss": val_epoch_loss}, step=epoch)
            print(f"Epoch {epoch}/{epochs} train_loss={train_epoch_loss:.4f} val_loss={val_epoch_loss:.4f}")
            model.train()
        else:
            history["train_loss"].append(train_epoch_loss)
            print(f"Epoch {epoch}/{epochs} train_loss={train_epoch_loss:.4f}")
    return history


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="valhacks-syntax-terror",
        # Set the wandb project where this run will be logged.
        project="valhacks",
        # Track hyperparameters and run metadata.
        config=vars(args),
    )

    print("Loading dataset...")
    df = load_dataset(args.csv)
    # Parse excludes
    raw_excludes = [c.strip() for c in args.exclude.split(",") if c.strip()]
    
    meta = get_meta(df)
    if args.val_split and args.val_split > 0:
        # Train/val split indices
        indices = np.arange(len(df))
        train_idx, val_idx = train_test_split(indices, test_size=args.val_split, random_state=42, shuffle=True)
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        # Fit preprocessor on train only to avoid leakage
        X_train, preprocessor, used_cols = fit_transform_features(df_train, exclude=raw_excludes)
        X_val = transform_features(df_val, preprocessor)
        print(f"Train/Val shapes: {X_train.shape} / {X_val.shape}")
    else:
        # No validation: fit on full dataset
        X_train, preprocessor, used_cols = fit_transform_features(df, exclude=raw_excludes)
        X_val = None
        print(f"Train shape: {X_train.shape} (no validation split)")

    hidden_dims = [int(x) for x in args.hidden.split(",") if x.strip()]
    cfg = VAEConfig(input_dim=X_train.shape[1], latent_dim=args.latent_dim, hidden_dims=hidden_dims, beta=args.beta)
    model = MLPVAE(cfg).to(args.device)

    train_loader = build_dataloader(X_train, args.batch_size, shuffle=True)
    val_loader = build_dataloader(X_val, args.batch_size, shuffle=False) if X_val is not None else None
    history = train(model, train_loader, val_loader, args.epochs, args.lr, args.device, run)

    # Save artifacts
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, out_dir / "vae_model.pt")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    joblib.dump(used_cols, out_dir / "used_cols.joblib")
    joblib.dump(raw_excludes, out_dir / "excluded_cols.joblib")
    meta.to_parquet(out_dir / "meta.parquet")

    # Store latent representations (mean vectors) for all tracks for fast similarity search
    model.eval()
    with torch.no_grad():
        # Compute features for ALL rows using train-fitted preprocessor
        X_all = transform_features(df, preprocessor)
        all_tensor = torch.from_numpy(X_all).to(args.device)
        mu, logvar = model.encode(all_tensor)
        latents = mu.cpu().numpy().astype(np.float32)
    np.save(out_dir / "latents.npy", latents)
    print("Saved artifacts to", out_dir)
    if "val_loss" in history:
        print("Final epoch train/val:", history["train_loss"][-1], history["val_loss"][-1])
    else:
        print("Final epoch train:", history["train_loss"][-1])

    logged_artifact = run.log_artifact(
    out_dir,
    "artifact",
    type="vae_model"
    )
    run.link_artifact(
    artifact=logged_artifact,
    target_path="murgaslevi/model-registry/vae"
    )
    run.finish()


if __name__ == "__main__":
    main()
