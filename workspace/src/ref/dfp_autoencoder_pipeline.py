"""
DFP-style Tabular Autoencoder Pipeline
=======================================
A PyTorch refresher covering:
  - Dataset class for mixed tabular columns (numerical, ordinal, nominal, text)
  - DataLoader setup
  - Autoencoder model
  - Train / validation loop
  - Checkpoint save / load
  - Inference

Inspired by NVIDIA Morpheus DFP (Digital Fingerprinting) pipeline structure,
where per-user behavioral tabular data is compressed to a latent embedding,
and reconstruction error flags anomalies.

Conventions: PyTorch 2.x (weights_only=True, state_dict checkpointing).
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sentence_transformers import SentenceTransformer


# =============================================================================
# 1. DATASET
# =============================================================================

class TabularDFPDataset(Dataset):
    """
    Dataset for mixed-type tabular data in a DFP-style pipeline.

    Column roles are declared explicitly via column_config, e.g.:

        column_config = {
            "numerical":  ["bytes_sent", "bytes_recv", "session_duration"],
            "ordinal":    ["severity_level"],          # low < med < high
            "nominal":    ["protocol", "country_code"],
            "text":       ["user_agent_string"],        # sentence-embedded
        }

    Processing:
      - numerical  → StandardScaler  → float32
      - ordinal    → OrdinalEncoder  → float32
      - nominal    → OneHotEncoder   → float32
      - text       → SentenceTransformer → float32 (frozen, pre-computed)

    All processed tensors are concatenated into a single flat vector,
    which becomes the autoencoder's input/target.

    Parameters
    ----------
    df             : raw pandas DataFrame
    column_config  : dict mapping column roles to lists of column names
    fit            : if True, fit all scalers/encoders on this data (training set).
                     if False, use already-fitted transforms (val/test/inference).
    fitted_state   : dict returned by a previous fit=True call, for reuse.
    text_model_name: any sentence-transformers model name.
    device         : device for embedding pre-computation ('cpu' or 'cuda').
    """

    def __init__(
        self,
        df: pd.DataFrame,
        column_config: dict,
        fit: bool = True,
        fitted_state: dict = None,
        text_model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.column_config = column_config
        self.device = device

        # ── numerical ──────────────────────────────────────────────────────
        num_cols = column_config.get("numerical", [])
        if num_cols:
            if fit:
                self.num_scaler = StandardScaler()
                num_arr = self.num_scaler.fit_transform(df[num_cols].values.astype(np.float32))
            else:
                self.num_scaler = fitted_state["num_scaler"]
                num_arr = self.num_scaler.transform(df[num_cols].values.astype(np.float32))
        else:
            num_arr = np.empty((len(df), 0), dtype=np.float32)
            self.num_scaler = None

        # ── ordinal ────────────────────────────────────────────────────────
        ord_cols = column_config.get("ordinal", [])
        if ord_cols:
            if fit:
                self.ord_encoder = OrdinalEncoder()
                ord_arr = self.ord_encoder.fit_transform(df[ord_cols].values).astype(np.float32)
            else:
                self.ord_encoder = fitted_state["ord_encoder"]
                ord_arr = self.ord_encoder.transform(df[ord_cols].values).astype(np.float32)
        else:
            ord_arr = np.empty((len(df), 0), dtype=np.float32)
            self.ord_encoder = None

        # ── nominal (one-hot) ──────────────────────────────────────────────
        nom_cols = column_config.get("nominal", [])
        if nom_cols:
            if fit:
                self.nom_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                nom_arr = self.nom_encoder.fit_transform(df[nom_cols].values).astype(np.float32)
            else:
                self.nom_encoder = fitted_state["nom_encoder"]
                nom_arr = self.nom_encoder.transform(df[nom_cols].values).astype(np.float32)
        else:
            nom_arr = np.empty((len(df), 0), dtype=np.float32)
            self.nom_encoder = None

        # ── text (sentence embeddings, pre-computed once) ──────────────────
        txt_cols = column_config.get("text", [])
        if txt_cols:
            # Concatenate all text columns into a single string per row
            sentences = df[txt_cols].fillna("").apply(
                lambda row: " | ".join(row.values.astype(str)), axis=1
            ).tolist()
            st_model = SentenceTransformer(text_model_name, device=device)
            with torch.no_grad():
                txt_arr = st_model.encode(
                    sentences,
                    batch_size=256,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
            # Store embed dim for later reference
            self.text_embed_dim = txt_arr.shape[1]
            del st_model  # free VRAM after pre-computation
        else:
            txt_arr = np.empty((len(df), 0), dtype=np.float32)
            self.text_embed_dim = 0

        # ── concatenate all parts ──────────────────────────────────────────
        self.data = torch.from_numpy(
            np.concatenate([num_arr, ord_arr, nom_arr, txt_arr], axis=1)
        )  # shape: (N, input_dim)

        self.input_dim = self.data.shape[1]

        # ── pack fitted state for reuse on val/test ────────────────────────
        self.fitted_state = {
            "num_scaler":   self.num_scaler,
            "ord_encoder":  self.ord_encoder,
            "nom_encoder":  self.nom_encoder,
            "input_dim":    self.input_dim,
            "text_embed_dim": self.text_embed_dim,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Autoencoder: input == target
        x = self.data[idx]
        return x, x


# =============================================================================
# 2. DATALOADER FACTORY
# =============================================================================

def build_dataloaders(
    dataset: TabularDFPDataset,
    val_fraction: float = 0.15,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Split a dataset into train/val and return DataLoaders.

    For anomaly detection autoencoders (DFP style) the dataset is assumed to
    contain only *normal* behaviour — val split is still useful to monitor
    reconstruction loss and detect overfitting / under-training.
    """
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# =============================================================================
# 3. MODEL
# =============================================================================

class TabularAutoencoder(nn.Module):
    """
    Symmetric MLP autoencoder for tabular data.

    Encoder: input_dim → hidden_dims → latent_dim
    Decoder: latent_dim → hidden_dims (reversed) → input_dim

    Reconstruction loss (MSE) is the anomaly score at inference time —
    higher loss ⟹ more anomalous, matching DFP's approach.

    Parameters
    ----------
    input_dim   : total flattened feature dimension from the dataset
    latent_dim  : bottleneck size (controls compression / expressiveness)
    hidden_dims : list of layer widths for one side of the autoencoder
    dropout     : dropout probability applied in each hidden layer
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        # ── encoder ───────────────────────────────────────────────────────
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── decoder ───────────────────────────────────────────────────────
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        """Return latent representation only."""
        return self.encoder(x)


# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Single training epoch. scaler is a torch.cuda.amp.GradScaler for AMP."""
    model.train()
    total_loss = 0.0

    for x, target in loader:
        x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # more efficient than zero_grad()

        if scaler is not None:  # mixed precision
            with torch.autocast(device_type=device.type):
                out  = model(x)
                loss = criterion(out, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(x)
            loss = criterion(out, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Validation pass — no gradient tracking."""
    model.eval()
    total_loss = 0.0

    for x, target in loader:
        x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
        out  = model(x)
        loss = criterion(out, target)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def fit(
    model,
    train_loader,
    val_loader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 7,
    checkpoint_path: str = "best_autoencoder.pth",
    device: torch.device = torch.device("cpu"),
    use_amp: bool = False,
):
    """
    Full train + validation loop with:
      - AdamW optimiser
      - ReduceLROnPlateau scheduler
      - Early stopping on val loss
      - Best-checkpoint saving (state_dict pattern)
      - Mixed precision (optional)
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    amp_scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    best_val_loss = float("inf")
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        val_loss   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # ── checkpoint on improvement ──────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"  ✓ checkpoint saved (val_loss={val_loss:.6f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    return history


# =============================================================================
# 5. SAVE / LOAD  (latest PyTorch conventions)
# =============================================================================

def save_checkpoint(model, optimizer, epoch, loss, path: str):
    """
    Save a general checkpoint — model + optimizer + metadata.
    Convention: .pth for single model weights, .tar for multi-component.
    Here we use .pth since there's one model + one optimiser.
    """
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             loss,
            # store constructor kwargs so you can rebuild the model later
            "model_config": {
                "input_dim":   model.input_dim,
                "latent_dim":  model.latent_dim,
            },
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device = torch.device("cpu")):
    """
    Load a checkpoint saved with save_checkpoint().

    Returns
    -------
    model      : TabularAutoencoder with weights loaded, set to eval mode
    optimizer  : AdamW with state loaded (ready to resume training)
    meta       : dict with epoch, val_loss, model_config
    """
    # weights_only=True is the current safe default (PyTorch 2.x)
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    cfg   = checkpoint["model_config"]
    model = TabularAutoencoder(
        input_dim=cfg["input_dim"],
        latent_dim=cfg["latent_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # important: sets dropout/batchnorm to eval behaviour

    optimizer = optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    meta = {
        "epoch":        checkpoint["epoch"],
        "val_loss":     checkpoint["val_loss"],
        "model_config": cfg,
    }
    return model, optimizer, meta


# =============================================================================
# 6. INFERENCE
# =============================================================================

@torch.no_grad()
def infer(
    model: TabularAutoencoder,
    dataset: TabularDFPDataset,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
    threshold: float = None,
):
    """
    Run inference over a dataset, returning:
      - reconstruction_errors : per-sample MSE  (shape: N,)
      - latent_vectors        : encoder output  (shape: N, latent_dim)
      - anomaly_flags         : bool tensor if threshold is provided

    In DFP style, you'd compute threshold on a validation set
    (e.g. 99th percentile of reconstruction error during training).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_errors  = []
    all_latents = []

    for x, _ in loader:
        x = x.to(device)
        z     = model.encode(x)
        x_hat = model.decoder(z)

        # per-sample mean squared error
        errors = ((x - x_hat) ** 2).mean(dim=1)
        all_errors.append(errors.cpu())
        all_latents.append(z.cpu())

    reconstruction_errors = torch.cat(all_errors)    # (N,)
    latent_vectors        = torch.cat(all_latents)   # (N, latent_dim)

    result = {
        "reconstruction_errors": reconstruction_errors,
        "latent_vectors":        latent_vectors,
    }

    if threshold is not None:
        result["anomaly_flags"] = reconstruction_errors > threshold

    return result


# =============================================================================
# 7. QUICK SMOKE TEST  (replace with real data)
# =============================================================================

if __name__ == "__main__":
    # ── synthetic data ─────────────────────────────────────────────────────
    N = 2_000
    df = pd.DataFrame({
        "bytes_sent":       np.random.exponential(1e4, N),
        "bytes_recv":       np.random.exponential(5e4, N),
        "session_duration": np.random.gamma(2, 300, N),
        "severity_level":   np.random.choice(["low", "medium", "high"], N),
        "protocol":         np.random.choice(["TCP", "UDP", "ICMP"], N),
        "country_code":     np.random.choice(["IN", "US", "DE", "CN"], N),
        "user_agent_string": np.random.choice([
            "Mozilla/5.0 Chrome/120",
            "python-requests/2.31",
            "curl/7.88",
            "Googlebot/2.1",
        ], N),
    })

    column_config = {
        "numerical": ["bytes_sent", "bytes_recv", "session_duration"],
        "ordinal":   ["severity_level"],
        "nominal":   ["protocol", "country_code"],
        "text":      ["user_agent_string"],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── dataset + dataloaders ──────────────────────────────────────────────
    full_ds = TabularDFPDataset(df, column_config, fit=True, device=str(device))
    print(f"Input dimension: {full_ds.input_dim}")

    train_loader, val_loader = build_dataloaders(
        full_ds, val_fraction=0.15, batch_size=128, num_workers=0
    )

    # ── model ──────────────────────────────────────────────────────────────
    model = TabularAutoencoder(
        input_dim=full_ds.input_dim,
        latent_dim=32,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    )
    print(model)

    # ── training ───────────────────────────────────────────────────────────
    history = fit(
        model, train_loader, val_loader,
        epochs=20,
        lr=1e-3,
        patience=5,
        checkpoint_path="autoencoder_best.pth",
        device=device,
    )

    # ── reload best checkpoint ─────────────────────────────────────────────
    model_loaded, optimizer_loaded, meta = load_checkpoint(
        "autoencoder_best.pth", device=device
    )
    print(f"\nLoaded checkpoint from epoch {meta['epoch']}, "
          f"val_loss={meta['val_loss']:.6f}")

    # ── inference ──────────────────────────────────────────────────────────
    # Build an inference dataset reusing fitted transforms
    infer_ds = TabularDFPDataset(
        df.sample(100, random_state=0).reset_index(drop=True),
        column_config,
        fit=False,
        fitted_state=full_ds.fitted_state,
        device=str(device),
    )

    results = infer(model_loaded, infer_ds, device=device, threshold=0.05)
    print(f"\nReconstruction errors — mean: {results['reconstruction_errors'].mean():.4f}, "
          f"max: {results['reconstruction_errors'].max():.4f}")
    if "anomaly_flags" in results:
        n_anom = results["anomaly_flags"].sum().item()
        print(f"Anomalies flagged: {n_anom} / {len(infer_ds)}")
