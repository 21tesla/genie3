"""
PyTorch Lightning callbacks.

Custom callbacks for training monitoring and checkpointing.
"""

import os
import signal
from lightning.pytorch.callbacks import Callback


class SaveOnTermination(Callback):
    """
    PyTorch Lightning callback for graceful checkpoint saving on termination signals.

    This callback registers signal handlers for SIGTERM and SIGUSR1. When either
    signal is received, it saves the current training checkpoint and then exits
    the process cleanly.
    """

    def __init__(self, dirpath: str):
        """
        Initializes the SaveOnTermination callback.

        Args:
            dirpath: The directory path where checkpoints should be saved.
        """
        super().__init__()
        self.dirpath = dirpath
        self.trainer_ref = None

    def on_train_start(self, trainer, pl_module):
        """
        Called at the start of training.

        Registers signal handlers for SIGTERM and SIGUSR1 to allow graceful
        checkpointing before termination.

        Args:
            trainer: The PyTorch Lightning Trainer instance.
            pl_module: The LightningModule being trained.
        """
        self.trainer_ref = trainer

        # Register signal handlers for SIGTERM and SIGUSR1
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGUSR1, self.handle_signal)
        print(
            f"Training started. Send SIGTERM or SIGUSR1 to process {os.getpid()} to save checkpoint and stop gracefully."
        )

    def handle_signal(self, signum: int, frame):
        """
        Handles received signals (SIGTERM, SIGUSR1) by saving a checkpoint
        and then exiting the process.

        Args:
            signum: The signal number received.
            frame: The current stack frame.
        """
        # Save checkpoint
        if self.trainer_ref:
            global_step = self.trainer_ref.global_step
            filename = os.path.join(self.dirpath, f"step={global_step}.ckpt")
            print(
                f"Received signal {signum}. Saving checkpoint at step {global_step} to '{filename}' before shutdown..."
            )
            self.trainer_ref.save_checkpoint(filename)

        # Exit immediately to prevent hanging
        os._exit(0)
