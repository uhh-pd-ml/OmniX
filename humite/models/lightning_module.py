"""Training of OmniX: next-token loss and tagging loss on every batch."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

from humite.data.codec import build_codecs
from humite.data.config import DataConfig
from humite.data.tokenizer import shift_table
from humite.models.omnix import OmniX
from humite.models.transformer import TransformerConfig
from humite.utils.ranger import Ranger


def next_token_loss(
    logits: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy of `logits[:, p]` against `tokens[:, p + 1]`, averaged over the
    positions where `valid[:, p + 1]`."""
    vocab = logits.shape[-1]
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab), tokens[:, 1:].reshape(-1), reduction="none"
    )
    weight = valid[:, 1:]
    return (loss.view_as(weight).to(logits.dtype) * weight).sum() / weight.sum().clamp_min(1)


def mean_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1), reduction="none")
    return loss.sum() / loss.numel()


class OmniXModule(L.LightningModule):
    """Every step runs two passes over the batch: the generative pass (next-token loss on the
    particle count, the plan and the particle tokens) and the tagging pass (classification
    loss of the tasks that have labels for the batch's member)."""

    def __init__(
        self,
        data: DataConfig,
        transformer: TransformerConfig,
        n_condition_codes: int,
        tasks: Sequence[Mapping[str, Any]],
        pair_bias: Mapping[str, Any],
        optimizer: Mapping[str, Any],
        schedule: Mapping[str, Any],
        jet_decoder: Optional[Mapping[str, Any]] = None,
        compile: bool = False,
    ) -> None:
        """`tasks` lists the classification tasks with name, mode (`event` or `per_jet`),
        num_classes and label. An event task reads its label from the event value `label`, the
        per_jet task from the jet flavours. `optimizer` holds the arguments of
        humite.utils.ranger.Ranger, and `schedule` holds warmup_steps, decay_start, total_steps
        and min_lr."""
        super().__init__()
        self.model = OmniX(
            transformer=transformer,
            feature_codecs=build_codecs(data.particle_features),
            plan_codecs=build_codecs(data.plan),
            n_jet_slots=data.n_jet_slots,
            max_particles=data.max_particles,
            shift_table=shift_table(data),
            optional_features=data.optional_features,
            n_sources=data.n_sources,
            n_condition_codes=n_condition_codes,
            tasks=tasks,
            pair_bias=pair_bias,
            jet_decoder=jet_decoder,
        )
        self.tasks = [dict(task) for task in tasks]
        self.optimizer_options = dict(optimizer)
        self.schedule = dict(schedule)
        self.compile = bool(compile)
        self.val_auroc = nn.ModuleDict(
            {t["name"]: MulticlassAUROC(num_classes=int(t["num_classes"])) for t in self.tasks}
        )
        self.val_accuracy = nn.ModuleDict(
            {t["name"]: MulticlassAccuracy(num_classes=int(t["num_classes"])) for t in self.tasks}
        )

    def configure_model(self) -> None:
        if self.compile and not hasattr(self.model.forward, "_torchdynamo_orig_callable"):
            # one static graph per batch shape
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self.model.forward = torch.compile(self.model.forward, dynamic=False)

    def loss(
        self, batch: Dict[str, Any], stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the total loss of a batch and its parts. The keys carry no `stage` prefix."""
        generative = self.model(batch=batch, tagging=False)
        tagging = self.model(batch=batch, tagging=True)
        model = self.model
        metrics: Dict[str, torch.Tensor] = {}

        feature_losses = []
        for name in model.feature_names:
            logits = generative["features"][name]
            tokens = batch[f"{name}_token"].long()
            if name in ("d0", "dz"):
                # restrict the softmax to the tokens the particle's type allows
                allowed = model.impact_parameter_support(name, model.pid_of_predicted(name, batch))
                target = torch.cat([tokens[:, 1:], torch.zeros_like(tokens[:, :1])], dim=1)
                is_target = F.one_hot(target, logits.shape[-1]).bool()
                logits = logits.masked_fill(~(allowed | is_target), float("-inf"))
            feature_losses.append(next_token_loss(logits, tokens, batch["feature_valid"][name]))
            metrics[f"token_ce_{name}"] = feature_losses[-1]
        token_ce = torch.stack(feature_losses).sum()

        plan_losses = [mean_cross_entropy(generative["count"], batch["count_token"].long())]
        for name, logits in generative["plan"].items():
            plan_losses.append(mean_cross_entropy(logits, batch[f"plan_{name}_token"].long()))
        plan_ce = torch.stack(plan_losses).sum()
        total = token_ce + plan_ce

        for task in self.tasks:
            name = task["name"]
            logits = tagging["classes"][name]
            if task["mode"] == "event":
                labels = batch["event_values"].get(task["label"])
                labels = None if labels is None else labels.to(logits.device).reshape(-1).long()
                selected = None if labels is None else torch.ones_like(labels, dtype=torch.bool)
            else:
                labels = batch.get("jet_flavor")
                labels = None if labels is None else labels.to(logits.device).long()
                selected = None if labels is None else (labels >= 0) & tagging["jet_valid"]
            if labels is None or not bool(selected.any()):
                # no labels for this task in the batch; DDP needs every parameter in the graph
                total = total + logits.sum() * 0.0
                continue
            task_loss = F.cross_entropy(logits[selected], labels[selected])
            total = total + task_loss
            metrics[f"cls_{name}_loss"] = task_loss.detach()
            if stage == "val":
                probabilities = logits.softmax(dim=-1)[selected].detach()
                self.val_auroc[name].update(probabilities, labels[selected])
                self.val_accuracy[name].update(probabilities, labels[selected])

        metrics.update(loss=total, token_ce=token_ce, plan_ce=plan_ce)
        return total, metrics

    def _log(self, metrics: Dict[str, torch.Tensor], stage: str, batch: Dict[str, Any]) -> None:
        """Log every metric per member; the loss and the token loss also over all members."""
        member, batch_size = batch["member"], batch["attention_mask"].shape[0]
        for key, value in metrics.items():
            on_step = stage == "train" and key in ("loss", "token_ce")
            options = dict(on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log(f"{stage}/{member}/{key}", value, on_step=False, **options)
            if key in ("loss", "token_ce", "plan_ce"):
                self.log(f"{stage}/{key}", value, on_step=on_step, prog_bar=on_step, **options)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, metrics = self.loss(batch, "train")
        self._log(metrics, "train", batch)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        _, metrics = self.loss(batch, "val")
        self._log(metrics, "val", batch)

    def on_validation_epoch_end(self) -> None:
        for task in self.tasks:
            name = task["name"]
            if self.val_auroc[name].update_called:
                self.log(f"val/cls_{name}_auroc", self.val_auroc[name].compute(), sync_dist=False)
                self.log(f"val/cls_{name}_acc", self.val_accuracy[name].compute(), sync_dist=False)
            self.val_auroc[name].reset()
            self.val_accuracy[name].reset()

    def configure_optimizers(self):
        optimizer = Ranger(
            [p for p in self.parameters() if p.requires_grad], **self.optimizer_options
        )
        warmup = int(self.schedule["warmup_steps"])
        start = int(self.schedule["decay_start"])
        total = int(self.schedule["total_steps"])
        if not 0 <= warmup <= start < total:
            raise ValueError(
                f"need warmup_steps <= decay_start < total_steps, got {self.schedule}"
            )
        floor = float(self.schedule["min_lr"]) / float(self.optimizer_options["lr"])

        def factor(step: int) -> float:
            """Linear warm-up, constant, then a cosine decay to `min_lr` at `total_steps`."""
            if warmup > 0 and step < warmup:
                return step / warmup
            if step < start:
                return 1.0
            progress = min(1.0, (step - start) / float(total - start))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
