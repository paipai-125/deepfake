"""LAV-DF loader for the original UniCaCLF offline-feature interface.

Expected reusable UMMAFormer layout::

    <feature_root>/tsn/rgb/<split>/<id>.npy      # T_v x 2048
    <feature_root>/tsn/flow/<split>/<id>.npy     # T_v x 2048
    <feature_root>/byola/<split>/<id>.npy        # T_a x 2048

It produces the 4096-D TSN and 2048-D BYOL-A sequence used by the completed
baseline.  Each modality is linearly resampled to `max_seq_len`; annotations
remain in seconds in metadata.json and are converted to feature-grid units.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _resample(sequence: np.ndarray, length: int) -> torch.Tensor:
    if sequence.ndim != 2:
        raise ValueError(f"Expected a T x C .npy array, got shape {sequence.shape}")
    x = torch.from_numpy(np.ascontiguousarray(sequence.astype(np.float32).T))
    if x.shape[1] != length:
        x = F.interpolate(x.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)
    return x


class LAVDFFeatureDataset(Dataset):
    def __init__(
        self,
        metadata: str | Path,
        feature_root: str | Path,
        split: str,
        max_seq_len: int = 768,
        training: bool = False,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        require_audio: bool = True,
    ):
        self.metadata = Path(metadata)
        self.root = Path(feature_root)
        self.split = split.lower()
        self.max_seq_len = max_seq_len
        self.training = training
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.require_audio = require_audio
        records = json.loads(self.metadata.read_text(encoding="utf-8"))
        self.records = []
        missing = 0
        for item in records:
            if item.get("split", "").lower() != self.split:
                continue
            key = Path(item["file"]).stem
            paths = self._paths(key)
            if not all(p.exists() for p in paths):
                missing += 1
                continue
            self.records.append((item, key))
        if not self.records:
            raise RuntimeError(
                f"No usable {self.split} items. Expected TSN/BYOL-A features under {self.root}."
            )
        if missing:
            print(f"[LAVDF] skipped {missing} {self.split} items with incomplete features")
        print(f"[LAVDF] loaded {len(self.records)} {self.split} items (training={training})")

    def _paths(self, key: str) -> tuple[Path, ...]:
        rgb = self.root / "tsn" / "rgb" / self.split / f"{key}.npy"
        flow = self.root / "tsn" / "flow" / self.split / f"{key}.npy"
        if not self.require_audio:
            return rgb, flow
        return rgb, flow, self.root / "byola" / self.split / f"{key}.npy"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        item, key = self.records[index]
        paths = self._paths(key)
        rgb, flow = np.load(paths[0]), np.load(paths[1])
        if rgb.shape[0] != flow.shape[0]:
            # The two TSN streams encode the same video but can differ by a
            # frame at decoding boundaries; interpolate, never use np.resize.
            flow_t = _resample(flow, rgb.shape[0]).T.numpy()
            flow = flow_t
        video = np.concatenate((rgb, flow), axis=1)
        if video.shape[1] != self.video_dim:
            raise ValueError(f"{key}: TSN dimension is {video.shape[1]}, expected {self.video_dim}")
        feats = _resample(video, self.max_seq_len)
        if self.require_audio:
            audio = np.load(paths[2])
            if audio.shape[1] != self.audio_dim:
                raise ValueError(f"{key}: BYOL-A dimension is {audio.shape[1]}, expected {self.audio_dim}")
            feats = torch.cat((feats, _resample(audio, self.max_seq_len)), dim=0)

        duration = float(item["duration"])
        fps = float(item.get("video_frames", 0) or 0) / max(duration, 1e-6)
        if fps <= 0:
            fps = 25.0
        # All offline streams are linearly resampled to ``max_seq_len``.
        # One resulting feature position therefore spans this many *original
        # video frames*, not ``T_cached / max_seq_len`` cached positions.
        # Using cached T here incorrectly moves every forged interval by the
        # extraction stride (e.g. 4 or 16) and leaves all regression targets
        # outside the valid 768-point grid.
        feat_stride = fps * duration / self.max_seq_len
        feat_num_frames = feat_stride
        feat_offset = 0.5
        periods = item.get("fake_periods") or []
        segments_seconds = torch.tensor(periods, dtype=torch.float32).reshape(-1, 2)
        if periods:
            segments = segments_seconds * fps / feat_stride - feat_offset
            labels = torch.zeros(len(periods), dtype=torch.long)
        else:
            segments = torch.empty((0, 2), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
        # UniCaCLF downsamples these labels with AvgPool1d in its CaCL loss,
        # which requires floating-point input.  They are still compared to
        # 0/1 afterwards, so this does not change their semantics.
        frame_labels = torch.zeros(self.max_seq_len, dtype=torch.float32)
        for start, end in segments:
            lo = max(0, int(torch.floor(start).item()))
            hi = min(self.max_seq_len, int(torch.ceil(end).item()))
            frame_labels[lo:hi] = 1
        return {
            "video_id": key,
            "feats": feats,
            "segments": segments,
            "labels": labels,
            "frame_labels": frame_labels,
            "fps": fps,
            "duration": duration,
            "feat_stride": feat_stride,
            "feat_num_frames": feat_num_frames,
            "ori_segments": segments_seconds,
        }


def trivial_batch_collator(batch: Iterable[dict]) -> list[dict]:
    return list(batch)
