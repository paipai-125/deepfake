"""Import shim for the released UniCaCLF model.

The upstream source imports ``utils.nms`` from a private parent project.  The
released repository is otherwise self-contained, so register the local 1-D NMS
implementation under that historical module name before importing the model.
"""
from __future__ import annotations

import sys
import types

from .nms import batched_nms

utils = types.ModuleType("utils")
nms = types.ModuleType("utils.nms")
nms.batched_nms = batched_nms
utils.nms = nms
sys.modules.setdefault("utils", utils)
sys.modules.setdefault("utils.nms", nms)

from .context_archs import Contextformer  # noqa: E402

__all__ = ["Contextformer"]
