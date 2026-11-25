#!/usr/bin/env python3
"""Tagging metrics of a released model on the test split of a dataset.

    python scripts/classify.py checkpoints/omnix jetclass --n 100000
    python scripts/classify.py checkpoints/omnix collide2v --n 20000

jetclass: the ten jet types of JetClass. collide2v: b, c or light for every jet of a
COLLIDE-2V event, `--n` events of each process. Reported are the accuracy, the mean
one-vs-rest AUC and, for the flavours, the AUC of each pair of classes.

As for the paper's numbers, the model sees all particle features, the exact kinematics as
continuous inputs, and no class or source token. The precision is that of the paper's
evaluations too: float32 for JetClass, bfloat16 autocast for the flavours.
"""

import argparse
from itertools import combinations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from humite.data.datamodule import member_batches
from humite.data.readers import JetClassReader
from humite.pretrained import load_pretrained

TASK = {"jetclass": "jetclass_class", "collide2v": "c2v_flavor"}
CLASSES = {"jetclass": JetClassReader.CLASSES, "collide2v": ("light", "c", "b")}
MIXED_PRECISION = {"jetclass": False, "collide2v": True}


@torch.no_grad()
def score(
    module,
    data,
    member: str,
    task: str,
    n: int,
    batch_size: int,
    num_workers: int,
    mixed_precision: bool,
):
    """Class probabilities and labels of the first `n` test events of a member."""
    probabilities, labels, seen = [], [], 0
    per_jet = task == TASK["collide2v"]
    batches = member_batches(
        data,
        member,
        "test",
        batch_size,
        num_workers=num_workers,
        feature_mode=1,
        exact_kinematics=True,
    )
    for batch in batches:
        batch = {
            key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()
        }
        batch["event_values"] = {k: v.cuda() for k, v in batch["event_values"].items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            out = module.model(batch=batch, tagging=True)
        logits = out["classes"][task].float()
        if per_jet:
            target = batch["jet_flavor"]
            selected = (target >= 0) & out["jet_valid"]
            logits, target = logits[selected], target[selected]
        else:
            target = batch["event_values"]["jc_label"].long()
        probabilities.append(logits.softmax(dim=-1).double().cpu().numpy())
        labels.append(target.cpu().numpy())
        seen += batch["attention_mask"].shape[0]
        if seen >= n:
            break
    return np.concatenate(probabilities), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="folder with config.yaml and model.ckpt")
    parser.add_argument("dataset", choices=list(TASK))
    parser.add_argument("--n", type=int, default=100000, help="test events per member")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    module, data = load_pretrained(args.model)
    task = TASK[args.dataset]
    if task not in module.model.class_heads:
        parser.error(f"this model has no {task} task")
    members = [name for name, member in data.members.items() if member.reader == args.dataset]
    scored = [
        score(
            module,
            data,
            member,
            task,
            args.n,
            args.batch_size,
            args.num_workers,
            MIXED_PRECISION[args.dataset],
        )
        for member in members
    ]
    probabilities = np.concatenate([p for p, _ in scored])
    labels = np.concatenate([y for _, y in scored])
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    names = CLASSES[args.dataset]
    print(f"{len(labels)} {'jets' if args.dataset == 'jetclass' else 'jets in events'}")
    print(f"accuracy  {np.mean(probabilities.argmax(axis=1) == labels):.4f}")
    auc = roc_auc_score(labels, probabilities, average="macro", multi_class="ovr")
    print(f"AUC       {auc:.4f}  (mean one-vs-rest)")
    if args.dataset == "collide2v":
        for negative, positive in combinations(range(len(names)), 2):
            pair = (labels == positive) | (labels == negative)
            ratio = probabilities[pair, positive] / (
                probabilities[pair, positive] + probabilities[pair, negative]
            )
            pair_auc = roc_auc_score(labels[pair] == positive, ratio)
            print(f"AUC       {pair_auc:.4f}  ({names[positive]} vs {names[negative]})")


if __name__ == "__main__":
    main()
