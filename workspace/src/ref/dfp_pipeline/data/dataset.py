"""
dfp_pipeline/data/dataset.py
-----------------------------
TabularDFPDataset — handles mixed-type tabular columns:
  numerical  → StandardScaler   → float32
  ordinal    → OrdinalEncoder   → float32
  nominal    → OneHotEncoder    → float32
  text       → SentenceTransformer (pre-computed) → float32

All processed parts are concatenated into a single flat vector.
For an autoencoder, __getitem__ returns (x, x).

fit=True   : fit all transforms on THIS data (use for training split only).
fit=False  : reuse fitted_state from a prior fit=True call (val / test / infer).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sentence_transformers import SentenceTransformer


class TabularDFPDataset(Dataset):
    """
    Parameters
    ----------
    df              : raw pandas DataFrame
    column_config   : dict mapping role → list[str], e.g.
                        {
                          "numerical": ["bytes_sent", "duration"],
                          "ordinal":   ["severity"],
                          "nominal":   ["protocol", "country"],
                          "text":      ["user_agent"],
                        }
    fit             : if True, fit scalers/encoders on this data.
    fitted_state    : dict from a previous fit=True instance; required when fit=False.
    text_model_name : any sentence-transformers model (default: all-MiniLM-L6-v2).
    device          : device used for sentence-transformer inference ('cpu' or 'cuda').
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

        # ── numerical ──────────────────────────────────────────────────────
        num_cols = column_config.get("numerical", [])
        if num_cols:
            if fit:
                self.num_scaler = StandardScaler()
                num_arr = self.num_scaler.fit_transform(
                    df[num_cols].values.astype(np.float32)
                )
            else:
                self.num_scaler = fitted_state["num_scaler"]
                num_arr = self.num_scaler.transform(
                    df[num_cols].values.astype(np.float32)
                )
        else:
            num_arr = np.empty((len(df), 0), dtype=np.float32)
            self.num_scaler = None

        # ── ordinal ────────────────────────────────────────────────────────
        ord_cols = column_config.get("ordinal", [])
        if ord_cols:
            if fit:
                self.ord_encoder = OrdinalEncoder()
                ord_arr = self.ord_encoder.fit_transform(
                    df[ord_cols].values
                ).astype(np.float32)
            else:
                self.ord_encoder = fitted_state["ord_encoder"]
                ord_arr = self.ord_encoder.transform(
                    df[ord_cols].values
                ).astype(np.float32)
        else:
            ord_arr = np.empty((len(df), 0), dtype=np.float32)
            self.ord_encoder = None

        # ── nominal (one-hot) ──────────────────────────────────────────────
        nom_cols = column_config.get("nominal", [])
        if nom_cols:
            if fit:
                self.nom_encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
                nom_arr = self.nom_encoder.fit_transform(
                    df[nom_cols].values
                ).astype(np.float32)
            else:
                self.nom_encoder = fitted_state["nom_encoder"]
                nom_arr = self.nom_encoder.transform(
                    df[nom_cols].values
                ).astype(np.float32)
        else:
            nom_arr = np.empty((len(df), 0), dtype=np.float32)
            self.nom_encoder = None

        # ── text (sentence embeddings, pre-computed once at init) ──────────
        # Pre-computing here (not in __getitem__) is intentional:
        # - avoids re-running a transformer model per batch
        # - keeps the DataLoader worker logic simple and fast
        txt_cols = column_config.get("text", [])
        if txt_cols:
            sentences = (
                df[txt_cols]
                .fillna("")
                .apply(lambda row: " | ".join(row.values.astype(str)), axis=1)
                .tolist()
            )
            st_model = SentenceTransformer(text_model_name, device=device)
            with torch.no_grad():
                txt_arr = st_model.encode(
                    sentences,
                    batch_size=256,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
            self.text_embed_dim = txt_arr.shape[1]
            del st_model  # free GPU memory immediately after use
        else:
            txt_arr = np.empty((len(df), 0), dtype=np.float32)
            self.text_embed_dim = 0

        # ── concatenate all parts into one flat float32 vector ─────────────
        self.data = torch.from_numpy(
            np.concatenate([num_arr, ord_arr, nom_arr, txt_arr], axis=1)
        )
        self.input_dim = self.data.shape[1]

        # Pack fitted state so val/test/infer datasets can reuse transforms
        self.fitted_state = {
            "num_scaler":     self.num_scaler,
            "ord_encoder":    self.ord_encoder,
            "nom_encoder":    self.nom_encoder,
            "input_dim":      self.input_dim,
            "text_embed_dim": self.text_embed_dim,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # autoencoder: input == target
