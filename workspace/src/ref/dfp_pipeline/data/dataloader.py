"""
dfp_pipeline/data/dataloader.py
--------------------------------
build_dataloaders() — splits a TabularDFPDataset into train/val DataLoaders.

In a DFP/anomaly-detection context the dataset contains only *normal*
behaviour.  A val split is still important: it lets you monitor
reconstruction loss to detect overfitting or under-training without
ever touching labelled anomaly data.
"""

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import TabularDFPDataset


def build_dataloaders(
    dataset: TabularDFPDataset,
    val_fraction: float = 0.15,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Parameters
    ----------
    dataset      : a fully initialised TabularDFPDataset (fit=True).
    val_fraction : fraction of samples held out for validation.
    batch_size   : samples per batch.
    num_workers  : DataLoader worker processes (set 0 on Windows / notebooks).
    seed         : controls the train/val split reproducibly.

    Returns
    -------
    (train_loader, val_loader)
    """
    n_val   = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader
