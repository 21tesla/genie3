"""
Model I/O utilities.

Functions for loading and saving model checkpoints and weights.
"""

import os
import glob
import numpy as np


def get_ckpt_path(ckptdir: str, step: int = None) -> str:
    """
    Get checkpoint file path from directory.

    Args:
        ckptdir: Directory containing checkpoint files
        step: Specific step number (if None, returns latest checkpoint)

    Returns:
        str: Path to checkpoint file, or None if no checkpoints found
    """
    if step is None:
        steps = [
            int(filepath.split("/")[-1].split("=")[-1].split(".")[0])
            for filepath in glob.glob(os.path.join(ckptdir, "step=*.ckpt"))
        ]
        step = np.max(steps) if len(steps) > 0 else None
    ckpt_path = os.path.join(ckptdir, f"step={step}.ckpt") if step else None
    return ckpt_path
