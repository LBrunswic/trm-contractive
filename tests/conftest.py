"""Shared pytest setup: CPU-only, eager mode, repo root importable."""
import os
import sys

# Tests are CPU-only by design; never compile.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DISABLE_COMPILE", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import torch  # noqa: E402

# The test models are tiny: a few threads are faster than one per core.
torch.set_num_threads(min(4, torch.get_num_threads()))
