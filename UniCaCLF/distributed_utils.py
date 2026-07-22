"""Small single-node distributed helpers for offline probing jobs."""
from __future__ import annotations

import os

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
