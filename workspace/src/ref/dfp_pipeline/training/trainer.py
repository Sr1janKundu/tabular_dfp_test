"""
dfp_pipeline/training/trainer.py
----------------------------------
Trainer — encapsulates the full training loop with:

  • AdamW optimiser + ReduceLROnPlateau scheduler
  • Per-epoch train / val MSE loss
  • Gradient clipping (safety net for large embedding inputs)
  • Mixed-precision (AMP) support
  • Early stopping on val loss
  • Best-checkpoint saving
  • TensorBoard logging

TensorBoard quickstart
-----------------------
Install:
    pip install tensorboard

Launch the UI (from your project root):
    tensorboard --logdir runs/

What gets logged (all visible in the TensorBoard UI):
  Scalars tab:
    Loss/train          — mean MSE per epoch, training set
    Loss/val            — mean MSE per epoch, validation set
    LearningRate        — current LR after scheduler step

  Histograms tab  (logged every `histogram_every` epochs):
    <layer_name>/weight — weight distributions for every Linear layer
    <layer_name>/grad   — gradient distributions (helps spot vanishing grads)

  Graph tab:
    The model computation graph (logged once at epoch 1 via add_graph).

  Custom Scalars layout:
    Loss/train and Loss/val plotted on the same chart for easy comparison.

  Embeddings tab  (optional, see log_embeddings):
    2-D UMAP/t-SNE projection of latent vectors from the val set.
    Useful to visually verify that the encoder is learning structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..model.autoencoder import TabularAutoencoder
from .checkpoint import save_checkpoint


class Trainer:
    """
    Parameters
    ----------
    model            : TabularAutoencoder instance.
    train_loader     : DataLoader for training split.
    val_loader       : DataLoader for validation split.
    device           : torch.device.
    lr               : initial learning rate for AdamW.
    weight_decay     : L2 regularisation coefficient.
    patience         : early-stopping patience (epochs without val improvement).
    checkpoint_path  : file path where the best checkpoint is saved.
    log_dir          : TensorBoard log directory (default: 'runs/dfp').
    histogram_every  : log weight/grad histograms every N epochs (expensive; default 5).
    use_amp          : enable automatic mixed precision on CUDA.
    """

    def __init__(
        self,
        model: TabularAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 7,
        checkpoint_path: str = "checkpoints/best.pth",
        log_dir: str = "runs/dfp",
        histogram_every: int = 5,
        use_amp: bool = False,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.patience       = patience
        self.checkpoint_path = checkpoint_path
        self.histogram_every = histogram_every

        self.criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.amp_scaler = (
            torch.cuda.amp.GradScaler()
            if (use_amp and device.type == "cuda")
            else None
        )

        # SummaryWriter writes event files consumed by tensorboard
        self.writer = SummaryWriter(log_dir=log_dir)
        self._log_dir = log_dir

    # ── public API ────────────────────────────────────────────────────────

    def fit(self, epochs: int = 50) -> dict:
        """
        Run the full training loop.

        Returns
        -------
        history : dict with 'train_loss' and 'val_loss' lists.
        """
        best_val_loss = float("inf")
        no_improve    = 0
        history       = {"train_loss": [], "val_loss": []}

        # Log the model graph once (needs a dummy input)
        self._log_graph()

        # Log a custom scalar layout so train + val appear on the same chart
        self._setup_custom_layout()

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # ── TensorBoard: scalars ──────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalars(
                "Loss",
                {"train": train_loss, "val": val_loss},
                global_step=epoch,
            )
            self.writer.add_scalar("LearningRate", current_lr, global_step=epoch)

            # ── TensorBoard: weight & gradient histograms (periodic) ──────
            if epoch % self.histogram_every == 0:
                self._log_histograms(epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"lr={current_lr:.2e}"
            )

            # ── early stopping + checkpoint ───────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    self.checkpoint_path
                )
                print(f"  ✓ checkpoint saved  (val={val_loss:.6f})")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        self.writer.flush()
        self.writer.close()
        print(f"\nTensorBoard logs → tensorboard --logdir {self._log_dir}")
        return history

    # ── private helpers ───────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total = 0.0

        for x, target in self.train_loader:
            x, target = (
                x.to(self.device, non_blocking=True),
                target.to(self.device, non_blocking=True),
            )
            self.optimizer.zero_grad(set_to_none=True)

            if self.amp_scaler is not None:
                with torch.autocast(device_type=self.device.type):
                    loss = self.criterion(self.model(x), target)
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                loss = self.criterion(self.model(x), target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total += loss.item() * x.size(0)

        return total / len(self.train_loader.dataset)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total = 0.0

        for x, target in self.val_loader:
            x, target = (
                x.to(self.device, non_blocking=True),
                target.to(self.device, non_blocking=True),
            )
            loss = self.criterion(self.model(x), target)
            total += loss.item() * x.size(0)

        return total / len(self.val_loader.dataset)

    def _log_graph(self):
        """Log model graph using a random dummy input (runs once)."""
        dummy = torch.zeros(1, self.model.input_dim, device=self.device)
        self.writer.add_graph(self.model, dummy)

    def _log_histograms(self, epoch: int):
        """
        Log weight and gradient histograms for every Linear layer.
        Histograms help you spot:
          - Dead neurons (weights collapsing to zero)
          - Exploding / vanishing gradients
          - Unhealthy weight distributions
        """
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if module.weight is not None:
                self.writer.add_histogram(
                    f"{name}/weight", module.weight.data, global_step=epoch
                )
            if module.weight.grad is not None:
                self.writer.add_histogram(
                    f"{name}/grad", module.weight.grad, global_step=epoch
                )

    def _setup_custom_layout(self):
        """
        Tell TensorBoard to display train and val loss on the same chart
        under the 'Custom Scalars' tab.

        This uses the multiline chart layout — requires
        tensorboard >= 1.14 and the google.protobuf package.
        Falls back silently if unavailable.
        """
        try:
            from torch.utils.tensorboard.summary import custom_scalars
            layout = {
                "Training Overview": {
                    "Loss (train vs val)": [
                        "Multiline",
                        ["Loss/train", "Loss/val"],
                    ],
                }
            }
            self.writer.add_custom_scalars(layout)
        except Exception:
            pass  # not critical

    @torch.no_grad()
    def log_embeddings(self, val_dataset, tag: str = "latent_space", n: int = 1000):
        """
        Log a sample of latent vectors to TensorBoard's Embeddings tab.

        The Embeddings projector performs interactive UMAP / PCA / t-SNE
        in the browser, letting you visually check that the encoder
        separates structure in the latent space.

        Usage (call after fit()):
            trainer.log_embeddings(val_dataset, n=500)

        Then in TensorBoard navigate to the Projector tab.

        Parameters
        ----------
        val_dataset : a TabularDFPDataset (val or inference split).
        tag         : label shown in TensorBoard.
        n           : number of samples to project (keep ≤ 5000 for speed).
        """
        from torch.utils.data import DataLoader as DL
        loader = DL(val_dataset, batch_size=256, shuffle=False)

        latents = []
        for x, _ in loader:
            x = x.to(self.device)
            latents.append(self.model.encode(x).cpu())
            if sum(t.shape[0] for t in latents) >= n:
                break

        mat = torch.cat(latents)[:n]
        self.writer.add_embedding(mat, tag=tag, global_step=0)
        self.writer.flush()
        print(f"Embeddings logged → open TensorBoard Projector tab ({tag})")
