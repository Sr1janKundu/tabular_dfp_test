"""
dfp_pipeline/model/autoencoder.py
----------------------------------
TabularAutoencoder — symmetric MLP autoencoder.

  Encoder: input_dim → hidden_dims        → latent_dim
  Decoder: latent_dim → hidden_dims (rev) → input_dim

Reconstruction loss (MSE per sample) is the anomaly score at inference:
higher loss ⟹ more anomalous.  This mirrors DFP's approach.

Design choices:
  - LayerNorm instead of BatchNorm: stable for small batches and
    behaves identically at train vs. eval (no running stats).
  - GELU: smooth activation, works well for mixed numeric/embedding inputs.
  - Dropout applied only in hidden layers, not on the bottleneck or output.
"""

import torch
import torch.nn as nn


class TabularAutoencoder(nn.Module):
    """
    Parameters
    ----------
    input_dim   : total flattened feature dimension (from TabularDFPDataset).
    latent_dim  : bottleneck width — controls compression ratio.
    hidden_dims : layer widths for the encoder side; decoder mirrors them.
    dropout     : dropout probability for hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self.encoder = self._build_encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = self._build_decoder(latent_dim, hidden_dims, input_dim, dropout)

    # ── builders ──────────────────────────────────────────────────────────

    @staticmethod
    def _block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_encoder(self, input_dim, hidden_dims, latent_dim, dropout):
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(self._block(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))  # no activation on bottleneck
        return nn.Sequential(*layers)

    def _build_decoder(self, latent_dim, hidden_dims, input_dim, dropout):
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers.append(self._block(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, input_dim))   # no activation on output
        return nn.Sequential(*layers)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation only (used at inference)."""
        return self.encoder(x)

    # ── convenience ───────────────────────────────────────────────────────

    def config(self) -> dict:
        """Serialisable constructor kwargs — saved inside checkpoints."""
        return {
            "input_dim":   self.input_dim,
            "latent_dim":  self.latent_dim,
            "hidden_dims": self.hidden_dims,
        }
