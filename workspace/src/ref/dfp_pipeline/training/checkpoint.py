"""
dfp_pipeline/training/checkpoint.py
-------------------------------------
save_checkpoint() / load_checkpoint() following PyTorch 2.x conventions:

  • Save: state_dict pattern — never pickle the entire model object.
          Saves model + optimiser + metadata in a single .pth file.
  • Load: torch.load(..., weights_only=True) — the safe default since
          PyTorch 2.0 that avoids arbitrary pickle execution.
          model_config is stored inside the checkpoint so the architecture
          can be reconstructed without any extra book-keeping.
"""

import os
import torch
import torch.optim as optim

from ..model.autoencoder import TabularAutoencoder


def save_checkpoint(
    model: TabularAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
) -> None:
    """
    Save a general checkpoint suitable for both resuming training
    and inference.

    Saved keys
    ----------
    epoch                 : last completed epoch number
    model_state_dict      : learned weights
    optimizer_state_dict  : optimiser state (momentum, adaptive LR buffers, ...)
    val_loss              : validation loss at save time
    model_config          : constructor kwargs to rebuild TabularAutoencoder

    Convention: .pth for single-model checkpoints, .tar for multi-component.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             val_loss,
            "model_config":         model.config(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> tuple[TabularAutoencoder, torch.optim.Optimizer, dict]:
    """
    Load a checkpoint saved by save_checkpoint().

    Uses weights_only=True (PyTorch 2.x recommendation) to prevent
    arbitrary code execution from malicious checkpoint files.

    Returns
    -------
    model      : TabularAutoencoder with weights loaded, set to eval() mode.
    optimizer  : AdamW with saved state (ready to resume training).
    meta       : dict — epoch, val_loss, model_config.

    Notes
    -----
    Call model.train() before resuming training.
    Call model.eval()  before inference (already set here by default).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    cfg   = checkpoint["model_config"]
    model = TabularAutoencoder(**cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # sets dropout / layernorm to inference behaviour

    optimizer = optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    meta = {
        "epoch":        checkpoint["epoch"],
        "val_loss":     checkpoint["val_loss"],
        "model_config": cfg,
    }
    return model, optimizer, meta
