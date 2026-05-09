"""Five-place seed for Zindi reproducibility.

Zindi audits the top-10 finishers' code end-to-end. `set_all_seeds` is called once
at process entry (training, inference, and test code). `transformers.set_seed` is
also called from the training entry point to cover the HF-internal RNG.
"""

from __future__ import annotations

import os
import random


def set_all_seeds(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """Set seeds in all five places Zindi reproducibility requires.

    Args:
        seed: the seed value. Default 42 (project convention).
        deterministic_cudnn: if True, force cuDNN deterministic algorithms.
            Slower but bit-identical across runs on the same hardware.
            Default False — turn on only for the repro-verification pass.
    """
    # 1. Python built-in random
    random.seed(seed)

    # 2. NumPy
    import numpy as np

    np.random.seed(seed)

    # 3. Torch CPU
    import torch

    torch.manual_seed(seed)

    # 4. Torch CUDA (all devices, if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 5. Python hash seed — must be set before any dict/set-order-sensitive code runs.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch also exposes a stronger flag — kept behind the same switch.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch builds lack this; silently skip.
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn that derives a deterministic per-worker seed."""
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
