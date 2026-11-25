"""Loading a trained model."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from humite.data.config import DataConfig
from humite.models.lightning_module import OmniXModule


def load_pretrained(folder: str, device: str = "cuda") -> Tuple[OmniXModule, DataConfig]:
    """The model and the dataset configuration stored in `folder`, which holds
    `config.yaml` (the `dataset` and `model` sections of the training config) and
    `model.ckpt`. The dataset paths are read from the environment when the data are used."""
    folder = Path(folder)
    cfg = OmegaConf.load(folder / "config.yaml")
    data = instantiate(cfg.dataset)
    module = instantiate(cfg.model, compile=False)(data=data)
    checkpoint = torch.load(folder / "model.ckpt", map_location="cpu", weights_only=True)
    module.load_state_dict(checkpoint["state_dict"])
    return module.to(device).eval(), data
