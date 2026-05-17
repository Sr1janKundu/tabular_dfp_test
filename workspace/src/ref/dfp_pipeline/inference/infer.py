"""
dfp_pipeline/inference/infer.py
---------------------------------
infer() — runs a trained autoencoder over a dataset and returns:

  reconstruction_errors : per-sample MSE  (N,)  — the DFP anomaly score
  latent_vectors        : encoder output  (N, latent_dim)
  anomaly_flags         : bool tensor     (N,)  — only if threshold is given

Threshold guidance
------------------
Train a threshold on your *normal* validation set, e.g.:

    results = infer(model, val_dataset, device=device)
    threshold = float(torch.quantile(results["reconstruction_errors"], 0.99))

Then pass that threshold at production inference time:

    results = infer(model, new_dataset, device=device, threshold=threshold)
    flagged  = results["anomaly_flags"]
"""

import torch
from torch.utils.data import DataLoader

from ..model.autoencoder import TabularAutoencoder
from ..data.dataset import TabularDFPDataset


@torch.no_grad()
def infer(
    model: TabularAutoencoder,
    dataset: TabularDFPDataset,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
    threshold: float = None,
) -> dict:
    """
    Parameters
    ----------
    model      : loaded TabularAutoencoder (already in eval() mode).
    dataset    : TabularDFPDataset built with fit=False + fitted_state.
    batch_size : inference batch size (no gradient memory, so can be large).
    device     : inference device.
    threshold  : optional anomaly score cut-off (MSE per sample).

    Returns
    -------
    dict with keys:
      'reconstruction_errors'  : FloatTensor (N,)
      'latent_vectors'         : FloatTensor (N, latent_dim)
      'anomaly_flags'          : BoolTensor  (N,)  — only if threshold given
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_errors  = []
    all_latents = []

    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        z     = model.encode(x)
        x_hat = model.decoder(z)

        # per-sample MSE: mean over feature dimension → shape (batch,)
        errors = ((x - x_hat) ** 2).mean(dim=1)

        all_errors.append(errors.cpu())
        all_latents.append(z.cpu())

    reconstruction_errors = torch.cat(all_errors)   # (N,)
    latent_vectors        = torch.cat(all_latents)  # (N, latent_dim)

    result = {
        "reconstruction_errors": reconstruction_errors,
        "latent_vectors":        latent_vectors,
    }

    if threshold is not None:
        result["anomaly_flags"] = reconstruction_errors > threshold

    return result
