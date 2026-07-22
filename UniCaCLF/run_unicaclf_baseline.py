"""Train/evaluate the released UniCaCLF localisation model on LAV-DF.

Run from the parent Deepfake directory:
    python -m UniCaCLF.run_unicaclf_baseline --mode train ...

This runner intentionally preserves UniCaCLF's Contextformer/CaPFormer,
classification head, boundary head and contrastive loss.  It only replaces
the unreleased dataset and NMS glue in the public repository.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import types
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm

from .lavdf_features import LAVDFFeatureDataset, trivial_batch_collator
from .runtime import Contextformer


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation exactly once per sample, without DDP padding."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.length = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return (self.length - self.rank + self.world_size - 1) // self.world_size


def distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not distributed_enabled() or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


class SafeIntraSampleInfoNCE(torch.nn.Module):
    """Keep authentic full-truth samples valid in a mixed LAV-DF minibatch.

    The released loss has no valid positive/negative pair for an all-authentic
    minibatch and attempts to stack an empty list.  Its intended contribution
    in that case is zero; classification loss still supervises background.
    """
    def __init__(self, delegate):
        super().__init__()
        self.delegate = delegate

    def forward(self, fpn_feats, fpn_gfeats, gt_fpn_frame):
        try:
            return self.delegate(fpn_feats, fpn_gfeats, gt_fpn_frame)
        except RuntimeError as error:
            if "non-empty TensorList" in str(error):
                return fpn_feats[0].sum() * 0.0
            raise


def segment_iou(segment: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(targets) == 0:
        return np.empty(0, dtype=np.float32)
    inter = np.maximum(0.0, np.minimum(segment[1], targets[:, 1]) - np.maximum(segment[0], targets[:, 0]))
    union = (segment[1] - segment[0]) + (targets[:, 1] - targets[:, 0]) - inter
    return inter / np.maximum(union, 1e-8)


def average_precision(predictions, ground_truth, threshold: float) -> float:
    n_gt = sum(len(x) for x in ground_truth.values())
    if n_gt == 0:
        return float("nan")
    claimed = {key: np.zeros(len(value), dtype=bool) for key, value in ground_truth.items()}
    ordered = sorted(predictions, key=lambda x: x[1], reverse=True)
    tp, fp = [], []
    for vid, _, segment in ordered:
        target = ground_truth.get(vid, np.empty((0, 2), dtype=np.float32))
        ious = segment_iou(segment, target)
        if len(ious):
            best = int(ious.argmax())
            matched = ious[best] >= threshold and not claimed[vid][best]
        else:
            matched = False
        if matched:
            claimed[vid][best] = True
            tp.append(1.0); fp.append(0.0)
        else:
            tp.append(0.0); fp.append(1.0)
    if not tp:
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall, precision = tp / n_gt, tp / np.maximum(tp + fp, 1e-8)
    # Standard interpolated AP.
    recall = np.r_[0.0, recall, 1.0]
    precision = np.r_[0.0, precision, 0.0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]) * 100.0)


def average_recall(predictions_by_video, ground_truth, topk: int, thresholds) -> float:
    recalls = []
    for threshold in thresholds:
        total, found = 0, 0
        for vid, gt in ground_truth.items():
            total += len(gt)
            preds = predictions_by_video.get(vid, [])[:topk]
            for target in gt:
                if any(segment_iou(np.asarray(pred), target[None])[0] >= threshold for _, pred in preds):
                    found += 1
        recalls.append(found / max(total, 1))
    return float(np.mean(recalls) * 100.0)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    # Evaluate the underlying module.  This avoids DDP gradient/buffer
    # synchronisation while every rank independently handles its data shard.
    model = unwrap_model(model)
    model.eval()
    predictions, per_video, ground_truth = [], {}, {}
    for batch in tqdm(loader, desc="Evaluate", leave=False, disable=not is_main_process()):
        outputs = model(batch)
        for item, output in zip(batch, outputs):
            vid = item["video_id"]
            gt = item["ori_segments"].cpu().numpy()
            ground_truth[vid] = gt
            segs = output["segments"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            ranked = sorted(zip(scores.tolist(), segs.tolist()), key=lambda x: x[0], reverse=True)
            per_video[vid] = ranked
            predictions.extend((vid, score, np.asarray(seg, dtype=np.float32)) for score, seg in ranked)

    if distributed_enabled():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (predictions, per_video, ground_truth))
        if not is_main_process():
            return None
        predictions, per_video, ground_truth = [], {}, {}
        for local_predictions, local_per_video, local_ground_truth in gathered:
            predictions.extend(local_predictions)
            per_video.update(local_per_video)
            ground_truth.update(local_ground_truth)

    thresholds = np.arange(0.50, 1.00, 0.05)
    all_ap = {f"AP@{t:.2f}": average_precision(predictions, ground_truth, float(t)) for t in thresholds}
    # Keep exactly the metrics required for this LAV-DF experiment.  AP at
    # the intermediate thresholds is used only to form mAP and is not saved.
    results = {
        "AP@0.50": all_ap["AP@0.50"],
        "AP@0.75": all_ap["AP@0.75"],
        "AP@0.90": all_ap["AP@0.90"],
        "AP@0.95": all_ap["AP@0.95"],
        "mAP@0.50:0.95": float(np.nanmean(list(all_ap.values()))),
    }
    for k in (5, 10, 20, 30, 50):
        results[f"AR@{k}"] = average_recall(per_video, ground_truth, k, thresholds)
    return results


def make_loader(args, split: str, training: bool):
    dataset = LAVDFFeatureDataset(
        args.metadata, args.feature_root, split,
        max_seq_len=args.max_seq_len, training=training,
        video_dim=args.video_dim, audio_dim=args.audio_dim,
    )
    if distributed_enabled():
        sampler = (
            DistributedSampler(dataset, shuffle=True, drop_last=training)
            if training else DistributedEvalSampler(dataset, dist.get_rank(), dist.get_world_size())
        )
    else:
        sampler = None
    return DataLoader(
        dataset, batch_size=args.batch_size if training else 1,
        shuffle=training and sampler is None, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
        collate_fn=trivial_batch_collator, drop_last=training,
    )


def build_model(args):
    model = Contextformer(
        input_dim=args.video_dim,
        audio_input_dim=args.audio_dim,
        max_seq_len=args.max_seq_len,
        backbone_arch=(2, 1, 5),
        # UniCaCLF's released source uses these default CaPFormer settings.
        backbone_type="CapFormer", fpn_type="capidentity",
    )
    model.info_loss = SafeIntraSampleInfoNCE(model.info_loss)

    # The public source returns two targets for a full-truth video although
    # its caller always unpacks three.  Keep the original fake-video path and
    # supply the all-zero pyramid frame labels only for this missing case.
    original_label_points = model.label_points_single_video

    def full_truth_compatible_label_points(self, concat_points, gt_segment, gt_label, gt_frame_label):
        if gt_segment.shape[0] != 0:
            return original_label_points(concat_points, gt_segment, gt_label, gt_frame_label)
        count = concat_points.shape[0]
        cls = gt_segment.new_zeros((count, self.num_classes))
        offsets = gt_segment.new_zeros((count, 2))
        frames = (gt_frame_label,)
        for index in range(self.backbone_arch[2] - 2):
            pooled = torch.nn.AvgPool1d(kernel_size=2 ** (index + 1), stride=2 ** (index + 1))(gt_frame_label[None, None])
            frames += (torch.zeros_like(pooled.squeeze()),)
        return cls, offsets, frames

    model.label_points_single_video = types.MethodType(full_truth_compatible_label_points, model)
    return model


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": unwrap_model(model).state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "metrics": metrics, "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--metadata", required=True, help="LAV-DF metadata.json")
    parser.add_argument("--feature-root", required=True, help="Directory containing tsn/ and byola/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", help="Required for --mode eval; optional resume for train")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--video-dim", type=int, default=4096)
    parser.add_argument("--audio-dim", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-backend", default="nccl", choices=("nccl", "gloo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "eval" and not args.checkpoint:
        parser.error("--checkpoint is required in eval mode")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed execution requires CUDA/NCCL in this runner")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    rank = dist.get_rank() if distributed_enabled() else 0
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed + rank)
    output = Path(args.output_dir)
    if is_main_process():
        output.mkdir(parents=True, exist_ok=True)
    if distributed_enabled():
        dist.barrier()

    model = build_model(args).to(device)
    payload = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"] if "model" in payload else payload, strict=True)
    if distributed_enabled():
        # Some full-truth minibatches bypass contrastive-loss branches.
        # Detecting unused parameters makes DDP robust to that valid case.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # UniCaCLF paper, Implementation Details: Adam, lr=1e-3, batch=8,
    # betas=(0.9,0.999), eps=1e-8.  Weight decay is disabled by default.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    if payload is not None and args.mode == "train" and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    if args.mode == "eval":
        metrics = evaluate(model, make_loader(args, args.test_split, False), device)
        if is_main_process():
            (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(json.dumps(metrics, indent=2))
        if distributed_enabled():
            dist.barrier(); dist.destroy_process_group()
        return

    train_loader = make_loader(args, args.train_split, True)
    dev_loader = make_loader(args, args.dev_split, False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-7)
    best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train(); totals = {}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main_process()):
            optimizer.zero_grad(set_to_none=True)
            losses = model(batch)
            losses["final_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        train_metrics = {f"train_{name}": value / max(len(train_loader), 1) for name, value in totals.items()}
        if distributed_enabled():
            keys = sorted(train_metrics)
            reduced = torch.tensor([train_metrics[key] for key in keys], device=device)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            train_metrics = {key: float(value) / world_size for key, value in zip(keys, reduced.tolist())}

        dev_metrics = evaluate(model, dev_loader, device)
        # Every rank must step its own scheduler identically.  Only rank 0
        # holds the complete distributed-evaluation metrics.
        score = torch.zeros(1, device=device)
        if is_main_process():
            score.fill_(dev_metrics["mAP@0.50:0.95"])
        if distributed_enabled():
            dist.broadcast(score, src=0)
        scheduler.step(float(score.item()))

        if is_main_process():
            metrics = {**train_metrics, **dev_metrics, "epoch": epoch}
            print(json.dumps(metrics, indent=2))
            (output / "last_metrics.json").write_text(json.dumps(metrics, indent=2))
            save_checkpoint(output / "last.pt", model, optimizer, epoch, metrics, args)
            if dev_metrics["mAP@0.50:0.95"] > best:
                best = dev_metrics["mAP@0.50:0.95"]
                save_checkpoint(output / "best.pt", model, optimizer, epoch, metrics, args)
        if distributed_enabled():
            dist.barrier()

    if distributed_enabled():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
