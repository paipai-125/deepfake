"""Dependency-free 1-D NMS used by the released UniCaCLF inference code."""
from __future__ import annotations

import torch


def _iou_1d(segment: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    left = torch.maximum(segment[0], segments[:, 0])
    right = torch.minimum(segment[1], segments[:, 1])
    inter = (right - left).clamp(min=0)
    union = (segment[1] - segment[0]).clamp(min=1e-6) + (segments[:, 1] - segments[:, 0]).clamp(min=1e-6) - inter
    return inter / union.clamp(min=1e-6)


def _nms_1d(segs: torch.Tensor, scores: torch.Tensor, iou_threshold: float, soft: bool, sigma: float):
    keep = []
    scores = scores.clone()
    candidates = torch.arange(scores.numel(), device=scores.device)
    while candidates.numel():
        best_pos = scores[candidates].argmax()
        best = candidates[best_pos]
        keep.append(best)
        rest = torch.cat((candidates[:best_pos], candidates[best_pos + 1:]))
        if not rest.numel():
            break
        iou = _iou_1d(segs[best], segs[rest])
        if soft:
            scores[rest] *= torch.exp(-(iou * iou) / max(float(sigma), 1e-6))
            rest = rest[scores[rest] > 0]
        else:
            rest = rest[iou <= iou_threshold]
        candidates = rest
    keep = torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=segs.device)
    return keep, scores


def batched_nms(
    segs: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    min_score: float,
    max_seg_num: int,
    use_soft_nms: bool = False,
    multiclass: bool = False,
    sigma: float = 0.75,
    voting_thresh: float = 0.9,
):
    """ActionFormer-compatible subset of batched_nms for 1-D temporal segments.

    `voting_thresh` is accepted for API compatibility; UniCaCLF does not require
    score voting for the LAV-DF one-class setting.
    """
    del voting_thresh
    valid = scores >= min_score
    segs, scores, labels = segs[valid], scores[valid], labels[valid]
    if not scores.numel():
        return segs, scores, labels
    groups = labels.unique() if multiclass else [None]
    selected, updated_scores = [], scores.clone()
    for group in groups:
        idx = torch.arange(scores.numel()) if group is None else torch.where(labels == group)[0]
        keep, local_scores = _nms_1d(segs[idx], scores[idx], iou_threshold, use_soft_nms, sigma)
        updated_scores[idx] = local_scores
        selected.append(idx[keep])
    selected = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)
    selected = selected[updated_scores[selected].argsort(descending=True)[:max_seg_num]]
    return segs[selected], updated_scores[selected], labels[selected]
