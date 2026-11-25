"""Lightning data module: member-pure, tokenized batches for training and validation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader

from humite.data.config import DataConfig
from humite.data.dataset import EventDataset, MixedBatchDataset
from humite.data.tokenizer import EventTokenizer

log = logging.getLogger(__name__)


class BatchBuilder:
    """Collates and tokenizes the events of one member into a model batch."""

    def __init__(self, data: DataConfig, exact_kinematics: bool = False) -> None:
        self.data = data
        self.tokenizers = {
            name: EventTokenizer(data, member.pad_length, exact_kinematics)
            for name, member in data.members.items()
        }

    def __call__(self, member: str, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tokenizer = self.tokenizers[member]
        batch = tokenizer(tokenizer.collate(events))
        batch["member"] = member
        batch["source"] = torch.full(
            (len(events),), self.data.members[member].source_id, dtype=torch.long
        )
        return batch


def member_batches(
    data: DataConfig,
    member: str,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    feature_mode: Optional[int] = None,
    exact_kinematics: bool = False,
) -> Iterable[Dict[str, Any]]:
    """One pass over a split of one member, as model batches. The condition is not given;
    `feature_mode` (1 = all features, 0 = kinematics only) overrides the fixed per-event
    choice of the validation stream; `exact_kinematics`: see EventTokenizer."""
    build = BatchBuilder(data, exact_kinematics)
    dataset = EventDataset(
        data.members[member],
        data,
        build.tokenizers[member].codecs,
        split,
        train=False,
        feature_mode=feature_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda events: build(member, events),
        pin_memory=True,
    )


class _EpochHookLoader(DataLoader):
    """A DataLoader that calls `on_epoch` whenever a pass over it starts."""

    def __init__(self, *args: Any, on_epoch=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_epoch = on_epoch

    def __iter__(self):
        self._on_epoch()
        return super().__iter__()


class DataModule(L.LightningDataModule):
    """Training batches follow `data.mix_weights`, validation batches `val_mix_weights`.

    The training stream continues across epochs (an epoch is `trainer.limit_train_batches`
    batches). Its position is saved in the checkpoint as the lengths of the epochs consumed
    so far, so a resumed training carries on with the events that come next.
    """

    def __init__(
        self, data: DataConfig, batch_size: int, num_workers: int = 0, prefetch_factor: int = 2
    ) -> None:
        super().__init__()
        self.data = data
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.build_batch = BatchBuilder(data)
        self._segments: List[int] = []
        self._segment_start: Optional[int] = None
        self._restored: Optional[Dict[str, Any]] = None

    def _dataset(self, split: str) -> MixedBatchDataset:
        train = split == "train"
        members = {
            name: EventDataset(
                member, self.data, self.build_batch.tokenizers[name].codecs, split, train=train
            )
            for name, member in self.data.members.items()
        }
        weights = self.data.mix_weights if train else self.data.val_mix_weights
        return MixedBatchDataset(members, self.batch_size, weights, continuous=train)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = self._dataset("train")
        self.val_dataset = self._dataset("val")
        log.info("batch pattern: %s", list(self.train_dataset.pattern))

    def _collate(self, item: Tuple[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        return self.build_batch(*item)

    def _loader_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "batch_size": None,
            "num_workers": self.num_workers,
            "collate_fn": self._collate,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            options["prefetch_factor"] = self.prefetch_factor
        return options

    def _consumed_batches(self) -> int:
        return int(self.trainer.global_step) * int(self.trainer.accumulate_grad_batches)

    def _stream_settings(self) -> Dict[str, Any]:
        limit = self.trainer.limit_train_batches
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pattern": list(self.train_dataset.pattern),
            "epoch_batches": limit if isinstance(limit, int) and limit > 0 else None,
        }

    def train_dataloader(self) -> DataLoader:
        consumed = self._consumed_batches()
        settings = self._stream_settings()
        epoch_batches = settings["epoch_batches"]
        restored, self._restored = self._restored, None
        segments = None
        if restored is not None:
            saved = [int(n) for n in restored["segments"]]
            if all(restored.get(k) == v for k, v in settings.items()) and sum(saved) == consumed:
                segments = saved
        if segments is None:
            if consumed:
                log.warning(
                    "no exact position of the training stream for %d consumed batches; "
                    "assuming uninterrupted epochs",
                    consumed,
                )
            if consumed and epoch_batches:
                full, rest = divmod(consumed, epoch_batches)
                segments = [epoch_batches] * full + ([rest] if rest else [])
            else:
                segments = [consumed] if consumed else []
        self._segments = segments
        self._segment_start = None
        self.train_dataset.set_progress(epoch_batches, segments)
        return _EpochHookLoader(
            self.train_dataset, on_epoch=self._start_segment, **self._loader_options()
        )

    def _start_segment(self) -> None:
        consumed = self._consumed_batches()
        if self._segment_start is not None and consumed > self._segment_start:
            self._segments.append(consumed - self._segment_start)
        self._segment_start = consumed

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self._loader_options())

    def state_dict(self) -> Dict[str, Any]:
        segments = list(self._segments)
        if self._segment_start is not None:
            running = self._consumed_batches() - self._segment_start
            if running > 0:
                segments.append(running)
        return {"train_stream": {**self._stream_settings(), "segments": segments}}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._restored = state_dict.get("train_stream")
