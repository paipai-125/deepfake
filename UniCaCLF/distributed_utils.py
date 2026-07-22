"""Small single-node distributed helpers for offline probing jobs."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

import torch
import torch.distributed as dist


def init_distributed(device_arg: str = "cuda") -> tuple[torch.device, int, int]:
    """Initialise NCCL when launched by torchrun; otherwise return one device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return torch.device(device_arg), 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun probing requires CUDA GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank), dist.get_rank(), world_size


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def gather_objects(local: Any) -> list[Any]:
    """Collect one small Python object per rank; single-GPU stays local."""
    if not is_distributed():
        return [local]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, local)
    return values


def audit_pair_coverage(
    expected_ids: list[str], local_attempted: list[str], local_success: list[str], local_failed: list[str],
) -> dict[str, Any] | None:
    """Verify that distributed shards process every manifest pair exactly once.

    This checks IDs, not aggregate counts, so a duplicated / missing shard
    cannot silently change channel statistics.  Only rank 0 returns the audit.
    """
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise RuntimeError("Pair manifest contains duplicate IDs; refusing distributed probing")
    reports = gather_objects({
        "attempted": list(local_attempted), "success": list(local_success), "failed": list(local_failed),
    })
    if not is_main_process():
        return None
    attempted = [item for report in reports for item in report["attempted"]]
    success = [item for report in reports for item in report["success"]]
    failed = [item for report in reports for item in report["failed"]]
    expected_set, attempted_set = set(expected), set(attempted)
    repeated = sorted(key for key, count in Counter(attempted).items() if count != 1)
    if attempted_set != expected_set or repeated:
        missing = sorted(expected_set - attempted_set)
        unexpected = sorted(attempted_set - expected_set)
        raise RuntimeError(
            "Distributed pair partition is invalid: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, repeated={repeated[:8]}"
        )
    if set(success) | set(failed) != expected_set or set(success) & set(failed):
        raise RuntimeError("A pair was neither uniquely successful nor uniquely recorded as failed")
    if len(success) != len(set(success)) or len(failed) != len(set(failed)):
        raise RuntimeError("A pair was reported more than once")
    return {
        "world_size": len(reports),
        "expected_pairs": len(expected),
        "attempted_pairs": len(attempted),
        "successful_pairs": len(success),
        "failed_pairs": len(failed),
        "per_rank": [
            {"attempted": len(report["attempted"]), "successful": len(report["success"]), "failed": len(report["failed"])}
            for report in reports
        ],
    }
