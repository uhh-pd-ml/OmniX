"""Event streams: one member's events, and member-pure batches of several members."""

from __future__ import annotations

import glob
import itertools
import logging
import os
import random
import zlib
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import awkward as ak
import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from humite.data.config import DataConfig, MemberConfig
from humite.data.readers import PARTICLE_COLUMNS, build_reader, row_count

log = logging.getLogger(__name__)

# Seed offsets of the random choices made per event: arbitrary distinct constants, so that the
# choices are independent of each other. A validation stream uses the same seeds on every pass.
SEED_OFFSETS = {
    "events": 2,
    "particles": 1,
    "order": 7919,
    "condition": 104729,
    "mode": 49979687,
}


def validation_feature_mode(event_uid: int, seed: int) -> int:
    """The feature mode of a validation event: a fixed half of the events has all features."""
    return int(zlib.crc32(f"{int(event_uid)}:feature_mode:{int(seed)}".encode()) % 2)


class EventDataset(IterableDataset):
    """One pass over the events of one member.

    Files are split between GPUs (rows are, if there are fewer files than GPUs) and the rows
    of a file between the workers. An event is a dict with its particles in random order
    (`[n, len(PARTICLE_COLUMNS)]`), its event values, and the three per-event choices of the
    model's input: the feature order, the condition code and the feature mode. With `train`
    those are drawn at random; otherwise they are fixed functions of the event, the condition
    is never given, and every pass yields the same stream.
    """

    def __init__(
        self,
        member: MemberConfig,
        data: DataConfig,
        codecs: Mapping[str, Any],
        split: str,
        train: bool,
        feature_mode: Optional[int] = None,
    ) -> None:
        self.member = member
        self.split = split
        self.train = bool(train)
        self.seed = int(data.seed)
        self.n_feature_orders = int(data.n_feature_orders)
        self.p_all_features = float(data.p_all_features)
        self.p_condition = float(data.p_condition)
        self.forced_feature_mode = feature_mode
        self.reader = build_reader(
            member.reader, member.reader_options, codecs, data.n_jet_slots, split
        )
        self.files = [f for pattern in member.files[split] for f in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"no files match {member.files[split]}")
        # number of passes started; a new pass draws a new file order and new random choices
        self.passes = 0
        # events to skip before the next one is yielded (set when a training is resumed)
        self.skip_events = 0
        # (index, count) of this worker among the workers that serve this member
        self.worker_shard: Optional[Tuple[int, int]] = None

    @staticmethod
    def _rank_and_world() -> Tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return int(os.environ.get("RANK", "0")), max(1, int(os.environ.get("WORLD_SIZE", "1")))

    def _worker(self) -> Tuple[int, int]:
        if self.worker_shard is not None:
            return self.worker_shard
        info = get_worker_info()
        return (0, 1) if info is None else (int(info.id), int(info.num_workers))

    def _files_of_pass(self, pass_seed: int) -> Tuple[List[str], Optional[Tuple[int, int]]]:
        """The files this GPU reads in a pass and, if GPUs share files, its (rank, world)."""
        files = list(self.files)
        random.Random(pass_seed + self.seed).shuffle(files)
        rank, world = self._rank_and_world()
        if world == 1:
            return files, None
        if len(files) < world:
            return files, (rank, world)
        files = files[: (len(files) // world) * world]
        return files[rank::world], None

    def _row_range(self, path: str, rank_shard: Optional[Tuple[int, int]]) -> range:
        """The rows of a file this GPU and worker read."""
        start, count = 0, row_count(path)
        for index, parts in filter(None, (rank_shard, self._worker())):
            share, remainder = divmod(count, parts)
            start += index * share + min(index, remainder)
            count = share + (1 if index < remainder else 0)
        return range(start, start + count)

    def _load(self, path: str, rows: range) -> List[Dict[str, Any]]:
        block = self.reader.read(path, rows.start, rows.stop)
        particles = block.particles
        keep = (particles["pt"] > 0.0) & np.isfinite(particles["pt"])
        counts = np.asarray(ak.to_numpy(ak.sum(keep, axis=1)), dtype=np.int64)
        flat_keep = ak.flatten(keep)
        columns = [
            np.asarray(ak.to_numpy(ak.flatten(particles[name])[flat_keep], allow_missing=False))
            for name in PARTICLE_COLUMNS
        ]
        table = np.stack(columns, axis=1).astype(np.float32)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        values = {name: array.tolist() for name, array in block.values.items()}
        name = os.path.basename(path)
        events = []
        for i in range(len(counts)):
            event = {
                "particles": table[offsets[i] : offsets[i + 1]],
                "uid": zlib.crc32(f"{name}:{rows.start + i}".encode()),
                "values": {key: float(column[i]) for key, column in values.items()},
            }
            if block.jet_flavor is not None:
                event["jet_flavor"] = block.jet_flavor[i]
            events.append(event)
        return events

    def _choose(self, event: Dict[str, Any], rngs: Dict[str, Any]) -> Dict[str, Any]:
        """The event with shuffled particles and its feature order, condition and mode."""
        out = dict(event)
        n = len(event["particles"])
        if n > 1:
            out["particles"] = event["particles"][rngs["particles"].permutation(n)]
        uid = event["uid"]
        if self.train:
            out["feature_order"] = int(rngs["order"].integers(self.n_feature_orders))
        else:
            out["feature_order"] = (
                zlib.crc32(f"{uid}:{self.seed}".encode()) % self.n_feature_orders
            )
        label = self.member.condition_label
        out["condition"] = 0
        if label is not None and self.train and rngs["condition"].random() < self.p_condition:
            out["condition"] = int(event["values"][label]) + 1 + self.member.condition_offset
        if self.train:
            out["feature_mode"] = int(rngs["mode"].random() < self.p_all_features)
        elif self.forced_feature_mode is not None:
            out["feature_mode"] = int(self.forced_feature_mode)
        else:
            out["feature_mode"] = validation_feature_mode(uid, self.seed)
        return out

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        skip, self.skip_events = int(self.skip_events), 0
        while True:
            pass_seed = self.seed + self.passes
            if self.train:
                self.passes += 1
            files, rank_shard = self._files_of_pass(pass_seed)
            yielded = False
            for path in files:
                rows = self._row_range(path, rank_shard)
                if len(rows) == 0:
                    continue
                if skip >= len(rows):
                    skip -= len(rows)
                    continue
                events = self._load(path, rows)
                random.Random(SEED_OFFSETS["events"] + pass_seed).shuffle(events)
                rngs = {
                    name: np.random.default_rng(self.seed + pass_seed + SEED_OFFSETS[name])
                    for name in ("particles", "order", "condition", "mode")
                }
                for event in events:
                    chosen = self._choose(event, rngs)
                    if skip > 0:
                        skip -= 1
                        continue
                    yielded = True
                    yield chosen
            # a pass that was skipped entirely is followed by the next one
            if yielded or skip == 0:
                return


def mix_pattern(weights: Mapping[str, float], max_period: int = 64) -> Tuple[str, ...]:
    """A repeating sequence of member names with each member's share of the entries given by
    its weight, the entries of a member spread as evenly as possible. Deterministic."""
    names = sorted(weights)
    values = [float(weights[n]) for n in names]
    if any(v <= 0 for v in values):
        raise ValueError(f"mix weights must be positive, got {dict(weights)}")
    total = sum(values)
    period = min(int(max(len(names), round(total / min(values)))), int(max_period))
    exact = [v / total * period for v in values]
    counts = [int(x) for x in exact]
    by_remainder = sorted(range(len(names)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in by_remainder[: period - sum(counts)]:
        counts[i] += 1
    counts = [max(c, 1) for c in counts]
    period = sum(counts)
    pattern: List[str] = []
    credit = [0.0] * len(names)
    emitted = [0] * len(names)
    for _ in range(period):
        for i in range(len(names)):
            credit[i] += counts[i] / period
        pick = max(range(len(names)), key=lambda i: (credit[i] - emitted[i], -i))
        pattern.append(names[pick])
        emitted[pick] += 1
    return tuple(pattern)


def batches_served(
    pattern: Sequence[str], segments: Sequence[int], worker_id: int, num_workers: int
) -> Dict[str, int]:
    """Batches of each member that one worker produced over `segments`, the lengths of the
    epochs (complete or interrupted) consumed so far."""
    served = {name: 0 for name in pattern}
    for n in segments:
        for j in range(worker_id, int(n), num_workers):
            served[pattern[j % len(pattern)]] += 1
    return served


class MixedBatchDataset(IterableDataset):
    """Batches of events, each from a single member, the members following `mix_pattern`.

    Yields `(member, events)`; use with `DataLoader(batch_size=None)`. Batch `k` of an
    epoch comes from member `pattern[k % len(pattern)]` on every GPU, and worker `w` of a
    loader produces the batches `k = w, w + num_workers, ...`.

    With `continuous` (training) the member streams carry on from one epoch to the next,
    an epoch is `epoch_batches` long, and `set_progress` moves the streams to where an
    interrupted training stopped. Otherwise every epoch restarts the streams.
    """

    def __init__(
        self,
        members: Dict[str, EventDataset],
        batch_size: int,
        weights: Mapping[str, float],
        continuous: bool,
    ) -> None:
        self.members = members
        self.batch_size = int(batch_size)
        self.pattern = mix_pattern(weights)
        self.continuous = bool(continuous)
        self.epoch_batches: Optional[int] = None
        self._segments: List[int] = []
        self._first_epoch_batches: Optional[int] = None
        self._streams: Optional[Dict[str, Iterator[Any]]] = None

    def __getstate__(self) -> Dict[str, Any]:
        return {**self.__dict__, "_streams": None}

    def set_progress(self, epoch_batches: Optional[int], segments: Sequence[int]) -> None:
        """Set the epoch length and the epochs already consumed, before the first epoch."""
        self.epoch_batches = epoch_batches
        self._segments = [int(n) for n in segments]
        self._first_epoch_batches = epoch_batches
        if epoch_batches:
            self._first_epoch_batches = epoch_batches - sum(self._segments) % epoch_batches
        self._streams = None

    @staticmethod
    def _cycle(dataset: EventDataset) -> Iterator[Any]:
        while True:
            yield from dataset

    def _open_streams(self, worker_id: int, num_workers: int) -> Dict[str, Iterator[Any]]:
        """Split each member's rows between the workers that serve it, then open it."""
        period = len(self.pattern)
        serves = [
            {self.pattern[(w + num_workers * i) % period] for i in range(period)}
            for w in range(num_workers)
        ]
        for name, dataset in self.members.items():
            serving = [w for w in range(num_workers) if name in serves[w]]
            shared = worker_id in serving and 0 < len(serving) < num_workers
            dataset.worker_shard = (serving.index(worker_id), len(serving)) if shared else None
        return {name: self._cycle(dataset) for name, dataset in self.members.items()}

    def __iter__(self) -> Iterator[Tuple[str, List[Any]]]:
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (int(info.id), int(info.num_workers))
        if not self.continuous:
            streams = self._open_streams(worker_id, num_workers)
            indices: Any = itertools.count(worker_id, num_workers)
        else:
            n = self.epoch_batches
            if self._streams is None:
                served = batches_served(self.pattern, self._segments, worker_id, num_workers)
                for name, dataset in self.members.items():
                    dataset.skip_events = served.get(name, 0) * self.batch_size
                self._streams = self._open_streams(worker_id, num_workers)
                n = self._first_epoch_batches
            streams = self._streams
            indices = (
                itertools.count(worker_id, num_workers)
                if n is None
                else range(worker_id, n, num_workers)
            )
        for k in indices:
            member = self.pattern[k % len(self.pattern)]
            yield member, [next(streams[member]) for _ in range(self.batch_size)]
