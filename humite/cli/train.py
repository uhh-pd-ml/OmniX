"""Train a model: `humite-train +experiment=omnix`."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from humite.data.datamodule import DataModule


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    # compile the forward as one graph, also under DDP
    torch._dynamo.config.optimize_ddp = False

    data = instantiate(cfg.dataset)
    model = instantiate(cfg.model)(data=data)
    datamodule = DataModule(data, **cfg.data)
    trainer = instantiate(
        cfg.trainer,
        callbacks=list(instantiate(cfg.callbacks).values()),
        logger=instantiate(cfg.logger),
    )
    if trainer.is_global_zero:
        # what humite.pretrained.load_pretrained needs to rebuild the model
        run_dir = Path(cfg.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        model_config = OmegaConf.masked_copy(cfg, ["dataset", "model"])
        (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(model_config, resolve=False))
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume_ckpt_path)


if __name__ == "__main__":
    main()
