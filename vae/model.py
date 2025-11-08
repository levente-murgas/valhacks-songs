from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Iterable[int] | None = (256, 128)
    dropout: float = 0.0
    beta: float = 1.0  # weight for KL term


class MLPVAE(nn.Module):
    """
    A simple MLP-based Variational Autoencoder for tabular data.

    Reconstruction loss uses MSE (suited for standardized continuous features and one-hot categoricals).
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        hidden_dims = list(cfg.hidden_dims or [])
        dims = [cfg.input_dim] + hidden_dims

        enc_layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.ReLU())
            if cfg.dropout and cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
        self.encoder = nn.Sequential(*enc_layers) if enc_layers else nn.Identity()

        last_dim = dims[-1] if enc_layers else cfg.input_dim
        self.fc_mu = nn.Linear(last_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(last_dim, cfg.latent_dim)

        # Decoder
        dec_dims = [cfg.latent_dim] + hidden_dims[::-1] + [cfg.input_dim]
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_dims) - 2):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            dec_layers.append(nn.ReLU())
            if cfg.dropout and cfg.dropout > 0:
                dec_layers.append(nn.Dropout(cfg.dropout))
        dec_layers.append(nn.Linear(dec_dims[-2], dec_dims[-1]))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_fn(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # MSE reconstruction
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.cfg.beta * kl
