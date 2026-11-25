"""Training callbacks."""

from __future__ import annotations

from typing import Any, Dict

import lightning as L
from lightning.pytorch.callbacks import Callback


class MemberBalancedMonitor(Callback):
    """Logs `val/member_max`: the largest validation token loss of any dataset member,
    each relative to its own value at `reference_epoch`.

    The members' losses are not on one scale (an event has several times the tokens of a
    jet), so their mean would follow the largest. The relative losses are comparable, and
    their maximum only improves when no member gets worse.
    """

    def __init__(self, metric: str = "token_ce", reference_epoch: int = 1) -> None:
        super().__init__()
        self.metric = metric
        self.reference_epoch = int(reference_epoch)
        self.reference: Dict[str, float] = {}

    def state_dict(self) -> Dict[str, Any]:
        return {"reference": dict(self.reference)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.reference = dict(state_dict["reference"])

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        current = {}
        for key, value in trainer.callback_metrics.items():
            parts = key.split("/")
            if len(parts) == 3 and parts[0] == "val" and parts[2] == self.metric:
                current[parts[1]] = float(value)
        if not current:
            return
        if not self.reference and trainer.current_epoch >= self.reference_epoch:
            self.reference = dict(current)
        relative = [
            value / self.reference[member] if self.reference else 1.0
            for member, value in current.items()
        ]
        pl_module.log("val/member_max", max(relative), sync_dist=True)
