"""
dfp_pipeline/utils/config.py
------------------------------
load_config() — load a YAML or JSON experiment config file.

Using a config file keeps hyperparameters out of code and makes
experiment tracking easier.  Example YAML:

    data:
      val_fraction: 0.15
      batch_size: 256
      num_workers: 4

    model:
      latent_dim: 32
      hidden_dims: [256, 128, 64]
      dropout: 0.2

    training:
      epochs: 50
      lr: 0.001
      weight_decay: 0.00001
      patience: 7
      use_amp: false
      checkpoint_path: checkpoints/best.pth
      log_dir: runs/dfp

    inference:
      batch_size: 512
      threshold_quantile: 0.99
"""

import json
import pathlib


def load_config(path: str) -> dict:
    """
    Load a .yaml / .yml or .json config file.

    PyYAML is used for YAML; falls back to json for .json files.
    Returns a plain dict.
    """
    p = pathlib.Path(path)
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("pip install pyyaml to load YAML configs") from e
        with open(p) as f:
            return yaml.safe_load(f)
    elif p.suffix == ".json":
        with open(p) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")
