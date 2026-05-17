"""
train.py  — main entrypoint
============================
Wires up every module in dfp_pipeline for a full run:

    python train.py [--config config.yaml]

Steps
-----
1. Load config (or use defaults below).
2. Build TabularDFPDataset with fit=True  (training data).
3. Build DataLoaders.
4. Instantiate TabularAutoencoder.
5. Train via Trainer (with TensorBoard logging).
6. Reload best checkpoint.
7. Run inference + compute anomaly threshold from the val set.
8. Flag anomalies on a fresh inference sample.

TensorBoard
-----------
While training (or after) open a second terminal and run:

    tensorboard --logdir runs/

Then navigate to http://localhost:6006 in your browser.

Tabs you'll see:
  Scalars      → Loss/train, Loss/val, LearningRate (per epoch)
  Graphs       → model computation graph
  Histograms   → weight & gradient distributions (every 5 epochs)
  Projector    → latent space embeddings (call trainer.log_embeddings())
  Custom Scalars → train + val on one chart
"""

import argparse
import torch
import numpy as np
import pandas as pd

from dfp_pipeline.data      import TabularDFPDataset, build_dataloaders
from dfp_pipeline.model     import TabularAutoencoder
from dfp_pipeline.training  import Trainer, load_checkpoint
from dfp_pipeline.inference import infer
from dfp_pipeline.utils     import set_seed


# ── default config (override via config.yaml) ─────────────────────────────────
DEFAULT_CONFIG = {
    "data": {
        "val_fraction": 0.15,
        "batch_size":   256,
        "num_workers":  0,          # set >0 on Linux; 0 is safest on Mac/Windows
    },
    "model": {
        "latent_dim":  32,
        "hidden_dims": [256, 128, 64],
        "dropout":     0.2,
    },
    "training": {
        "epochs":          50,
        "lr":              1e-3,
        "weight_decay":    1e-5,
        "patience":        7,
        "use_amp":         False,
        "checkpoint_path": "checkpoints/best.pth",
        "log_dir":         "runs/dfp",
        "histogram_every": 5,
    },
    "inference": {
        "batch_size":          512,
        "threshold_quantile":  0.99,
    },
}


def make_synthetic_df(n: int = 3_000) -> pd.DataFrame:
    """Minimal synthetic dataset for smoke-testing the pipeline."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "bytes_sent":        rng.exponential(1e4, n),
        "bytes_recv":        rng.exponential(5e4, n),
        "session_duration":  rng.gamma(2, 300, n),
        "severity_level":    rng.choice(["low", "medium", "high"], n),
        "protocol":          rng.choice(["TCP", "UDP", "ICMP"], n),
        "country_code":      rng.choice(["IN", "US", "DE", "CN"], n),
        "user_agent_string": rng.choice([
            "Mozilla/5.0 Chrome/120",
            "python-requests/2.31",
            "curl/7.88",
            "Googlebot/2.1",
        ], n),
    })


COLUMN_CONFIG = {
    "numerical": ["bytes_sent", "bytes_recv", "session_duration"],
    "ordinal":   ["severity_level"],
    "nominal":   ["protocol", "country_code"],
    "text":      ["user_agent_string"],
}


def main(cfg: dict):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. data ───────────────────────────────────────────────────────────
    df = make_synthetic_df()
    dcfg = cfg["data"]

    train_ds = TabularDFPDataset(
        df, COLUMN_CONFIG, fit=True, device=str(device)
    )
    print(f"Input dimension: {train_ds.input_dim}")

    train_loader, val_loader = build_dataloaders(
        train_ds,
        val_fraction=dcfg["val_fraction"],
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
    )

    # ── 2. model ──────────────────────────────────────────────────────────
    mcfg  = cfg["model"]
    model = TabularAutoencoder(
        input_dim=train_ds.input_dim, **mcfg
    )
    print(model, "\n")

    # ── 3. train ──────────────────────────────────────────────────────────
    tcfg    = cfg["training"]
    trainer = Trainer(
        model, train_loader, val_loader, device,
        lr               = tcfg["lr"],
        weight_decay     = tcfg["weight_decay"],
        patience         = tcfg["patience"],
        checkpoint_path  = tcfg["checkpoint_path"],
        log_dir          = tcfg["log_dir"],
        histogram_every  = tcfg["histogram_every"],
        use_amp          = tcfg["use_amp"],
    )
    history = trainer.fit(epochs=tcfg["epochs"])

    # Optional: log latent embeddings to TensorBoard Projector tab
    val_ds_for_embed = TabularDFPDataset(
        df, COLUMN_CONFIG, fit=False,
        fitted_state=train_ds.fitted_state, device=str(device)
    )
    trainer.log_embeddings(val_ds_for_embed, tag="latent_space", n=500)

    # ── 4. reload best checkpoint ─────────────────────────────────────────
    model_best, _, meta = load_checkpoint(tcfg["checkpoint_path"], device=device)
    print(f"\nReloaded checkpoint — epoch {meta['epoch']}, "
          f"val_loss={meta['val_loss']:.6f}")

    # ── 5. derive anomaly threshold from val reconstruction errors ─────────
    icfg   = cfg["inference"]
    val_ds = TabularDFPDataset(
        df, COLUMN_CONFIG, fit=False,
        fitted_state=train_ds.fitted_state, device=str(device)
    )
    val_results = infer(model_best, val_ds, device=device,
                        batch_size=icfg["batch_size"])
    threshold   = float(
        torch.quantile(
            val_results["reconstruction_errors"],
            icfg["threshold_quantile"]
        )
    )
    print(f"Anomaly threshold ({icfg['threshold_quantile']*100:.0f}th pct): "
          f"{threshold:.6f}")

    # ── 6. inference on new data ──────────────────────────────────────────
    new_df = make_synthetic_df(n=200)
    infer_ds = TabularDFPDataset(
        new_df, COLUMN_CONFIG, fit=False,
        fitted_state=train_ds.fitted_state, device=str(device)
    )
    results = infer(
        model_best, infer_ds,
        batch_size=icfg["batch_size"],
        device=device,
        threshold=threshold,
    )
    n_flagged = results["anomaly_flags"].sum().item()
    print(f"Inference on {len(infer_ds)} samples — flagged anomalies: {n_flagged}")
    print(f"  mean recon error: {results['reconstruction_errors'].mean():.6f}")
    print(f"  max  recon error: {results['reconstruction_errors'].max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config")
    args = parser.parse_args()

    if args.config:
        from dfp_pipeline.utils import load_config
        cfg = load_config(args.config)
    else:
        cfg = DEFAULT_CONFIG

    main(cfg)
