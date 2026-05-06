"""
PyTorch Lightning DataModule for Genie training and sampling.

Provides data loading for training (from data generators) and sampling/testing
(from sample datasets). Handles distributed training, checkpointing, and batching.
"""

import math
import logging
import numpy as np
from ml_collections import ConfigDict
from typing import Dict, List, Optional, Any, Tuple
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset, Sampler

from genie3.generation.data.data_generator.base import DataGenerator
from genie3.generation.data.data_generator.registry import get_data_generator
from genie3.generation.data.feature_pipeline.base import FeaturePipeline
from genie3.generation.data.sample_dataset.registry import get_sample_dataset
from genie3.generation.data.feature_pipeline.registry import get_feature_pipeline
from genie3.generation.utils.config_utils import create_batch_config_from_task
from genie3.generation.utils.feat_utils import batchify_np_features
from genie3.generation.utils.pdb_utils import parse_pdb


class GenieDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for Genie training and sampling.

    This DataModule supports two stages:

    1) Training ("fit"):
        - Builds one or more DataGenerator instances (e.g., AFDB / Pinder / DDI)
        - Samples generators according to their weights
        - Maps each sampled "task" to a FeaturePipeline
        - Produces batched NumPy features suitable for the model

    2) Testing / Sampling ("test"):
        - Builds a SampleDataset (unconditional / sidechain / motif / target)
        - Uses a custom sampler that yields batches as lists/arrays of indices
        - Produces batched NumPy features suitable for the model

    Key design notes:
        - Training uses an IterableDataset (GenieDataset) that yields endlessly.
        - Distributed training is handled by a global_step-based sharding scheme:
            only tasks where (global_step-1) % world_size == rank are yielded.
        - Oversized batches (token count > max_n_token) are skipped.
        - Optional strictness: if drop_if_missing_atom=True, any missing in atom
          mask triggers a skip via exception.
    """

    def __init__(self, config: ConfigDict, reset_state_dict: bool = False):
        """
        Initialize Genie data module.

        Args:
            config:
                Configuration object providing:
                    - config.dataset: mapping of generator name -> generator config
                    - config.pipeline: mapping of task -> pipeline config
                    - config.max_n_token: maximum number of tokens allowed per batch
                    - config.seq_pred_mode: sequence prediction mode used in batch config
                    - config.drop_if_missing_atom: skip samples with missing atoms
                    - config.batch_size: used during test stage sampler
            reset_state_dict:
                If True, ignore checkpoint state when calling load_state_dict().
                Useful for "fresh start" training runs.
        """
        super().__init__()
        self.config = config
        self.reset_state_dict = reset_state_dict

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets/samplers for the requested stage.

        Args:
            stage:
                Lightning stage string, typically 'fit' or 'test'.
        """
        if stage == "fit":

            # Define
            self.generators = []
            self.weights = []
            self.tasks = []

            # Create data generators with weights
            for name in self.config.dataset:
                generator = get_data_generator(self.config.dataset[name])
                generator_weight = generator.get_weight()
                generator_tasks = generator.get_tasks()
                self.generators.append(generator)
                self.weights.append(generator_weight)
                self.tasks.extend(generator_tasks)

            # Create data pipelines
            self.pipelines = dict(
                [
                    (task, get_feature_pipeline(self.config.pipeline[task]))
                    for task in set(self.tasks)
                ]
            )
            del self.tasks

            # Fetch distribution information
            rank = self.trainer.global_rank if self.trainer else 0
            world_size = self.trainer.world_size if self.trainer else 1

            # Create dataset
            self.dataset = GenieDataset(
                generators=self.generators,
                weights=self.weights,
                pipelines=self.pipelines,
                max_n_token=self.config.max_n_token,
                seq_pred_mode=self.config.seq_pred_mode,
                drop_if_missing_atom=self.config.drop_if_missing_atom,
                rank=rank,
                world_size=world_size,
            )

        elif stage == "test":

            self.dataset = get_sample_dataset(self.config)
            self.sampler = GenieTestSampler(
                n_sample=len(self.dataset), batch_size=self.config.batch_size
            )

    def train_dataloader(self) -> DataLoader:
        """
        Training dataloader.

        Notes:
            - Uses batch_size=1 because GenieDataset yields pre-batched items:
                (batch_config, batch_features)
            - IterableDataset streams indefinitely.
        """
        return DataLoader(self.dataset, batch_size=1, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        """
        Test/Sampling dataloader.

        Notes:
            - Uses custom sampler that yields batches of indices.
            - batch_size=1 because __getitem__ expects a batch of indices.
            - shuffle is disabled; sampler defines ordering.
        """
        return DataLoader(
            self.dataset, batch_size=1, sampler=self.sampler, shuffle=False
        )

    def state_dict(self) -> Dict[str, Any]:
        """
        Lightning checkpoint hook.

        Returns:
            dict containing dataset checkpoint state (global_step + generator states).
        """
        return {"dataset": self.dataset.get_checkpoint()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Lightning checkpoint restore hook.

        If reset_state_dict=True, restoration is skipped.

        Args:
            state_dict:
                Output from state_dict().
        """
        if not self.reset_state_dict:
            self.dataset.load_checkpoint(state_dict["dataset"])


class GenieDataset(IterableDataset):
    """
    Infinite streaming dataset for training.

    Each iteration:
        1) Randomly selects a DataGenerator according to provided weights
        2) Draws a task from that generator
        3) Converts raw structure filepaths into model-ready features using
           the task's FeaturePipeline
        4) Applies size constraints and missing-atom filtering
        5) Yields (batch_config, batch_np_features)

    Distributed sharding:
        - All ranks still advance all generators, but only one rank yields a
          given global_step:
              if (global_step-1) % world_size == rank: yield
              else: consume (advance generator) without yielding
    """

    def __init__(
        self,
        generators: List[DataGenerator],
        weights: List[float],
        pipelines: Dict[str, FeaturePipeline],
        max_n_token: int,
        seq_pred_mode: str,
        drop_if_missing_atom: bool,
        world_size: int,
        rank: int,
    ):
        """
        Initialize training dataset.

        Args:
            generators:
                List of generator objects that yield task dicts.
            weights:
                Sampling probabilities across generators (should sum to 1).
            pipelines:
                Mapping from task name -> pipeline instance.
            max_n_token:
                Maximum allowed token length. Batches above this are skipped.
            seq_pred_mode:
                Passed through to create_batch_config_from_task().
            drop_if_missing_atom:
                If True, any missing atom in gt_atom_mask triggers skipping the task.
            world_size:
                Number of distributed workers.
            rank:
                Rank index of this worker.
        """
        super().__init__()
        self.generators = generators
        self.weights = weights
        self.pipelines = pipelines
        self.max_n_token = max_n_token
        self.seq_pred_mode = seq_pred_mode
        self.drop_if_missing_atom = drop_if_missing_atom
        self.world_size = world_size
        self.rank = rank

        # Used for deterministic sharding across ranks
        self.global_step = 0

    def __iter__(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Infinite iterator.

        Yields:
            tuple:
                (batch_config, batch_np_features)
        """
        while True:

            # Select generator
            generator_index = np.random.choice(
                range(len(self.generators)), p=self.weights, size=1
            )[0]
            generator = self.generators[generator_index]

            # Increment gloval step
            self.global_step += 1

            # Yield data
            if (self.global_step - 1) % self.world_size == self.rank:
                try:
                    task = next(generator)
                    batch = self._process(task, self.drop_if_missing_atom)
                    if batch[1]["token_mask"].shape[1] <= self.max_n_token:
                        yield batch
                    else:
                        logging.info(f"Skipping due to oversize: {task}")
                except:
                    logging.error(f"Failed to handle task: {task}")
            else:
                _ = next(generator)

    def get_checkpoint(self) -> Dict[str, Any]:
        """
        Capture checkpoint state for resuming training.

        Returns:
            dict:
                {
                    'global_step': int,
                    'generators': [generator_state_dict, ...]
                }
        """
        return {
            "global_step": self.global_step,
            "generators": [g.get_state() for g in self.generators],
        }

    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Restore dataset/generator states from checkpoint.

        Args:
            checkpoint:
                Output from get_checkpoint().
        """
        self.global_step = checkpoint["global_step"]
        for g, state in zip(self.generators, checkpoint["generators"]):
            g.load_state(state)

    def _process(
        self, task: Dict[str, Any], drop_if_missing_atom: bool
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Convert a generator task into a model-ready batch.

        Steps:
            1. Select the FeaturePipeline based on task['task']
            2. Build a batch_config (model metadata / conditioning config)
            3. Parse each structure filepath into a Structure
            4. Create per-structure np_features via the pipeline
            5. Batchify into a single dict
            6. Optionally reject samples with missing atoms

        Args:
            task:
                Dict produced by DataGenerator.generate(), typically containing:
                    - 'task': task name used to lookup pipeline
                    - 'synthetic': whether structure should be treated as synthetic
                    - 'filepaths': list of PDB / PDB.GZ paths for this batch
            drop_if_missing_atom:
                If True, raise ValueError when any missing atom mask exists.

        Returns:
            tuple:
                (batch_config, batch_np_features)

        Raises:
            ValueError:
                If missing atoms are found and drop_if_missing_atom is True.
        """
        pipeline = self.pipelines[task["task"]]

        # Build model batch metadata / conditioning config
        batch_config = create_batch_config_from_task(
            task=task, seq_pred_mode=self.seq_pred_mode
        )

        # Convert each filepath to features
        list_np_features = []
        for filepath in task["filepaths"]:

            # Parse supported file types
            if filepath.endswith(".pdb") or filepath.endswith(".pdb.gz"):
                structure = parse_pdb(filepath=filepath, synthetic=task["synthetic"])
            else:
                logging.error(
                    f'Invalid data file postfix: {filepath.strip().split(".")[-1]}'
                )
                exit(0)

            # Create features and add to list
            list_np_features.append(pipeline.create_np_features(structure))

        # Batch features across all structures in this task
        batch = batchify_np_features(list_np_features)

        # Reject missing atoms if requested (caught in __iter__ and task skipped)
        if (batch["gt_atom_mask"] == 0).sum() > 0 and drop_if_missing_atom:
            raise ValueError

        return (batch_config, batch)


class GenieTestSampler(Sampler):
    """
    Sampler that yields batches of indices for sampling datasets.

    This sampler is designed for datasets where __getitem__ expects a batch
    of indices (list/array) rather than a single index.

    Distributed sharding is handled automatically by Lightning's
    DistributedSampler wrapper — this class does not need to shard manually.
    """

    def __init__(self, n_sample: int, batch_size: int):
        """
        Initialize inference sampler.

        Args:
            n_sample:
                Total number of samples in the dataset.
            batch_size:
                Number of samples per batch.
        """
        self.batch_size = batch_size
        self.sample_idx = np.arange(n_sample)
        self.num_batches = math.ceil(n_sample / batch_size)

    def __len__(self) -> int:
        """
        Number of batches yielded by this sampler.
        """
        return self.num_batches

    def __iter__(self) -> np.ndarray:
        """
        Yield batches of sample indices.

        Yields:
            batch_idx:
                Array of indices for the batch.
        """
        batch_idx = [
            self.sample_idx[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(self.num_batches)
        ]
        yield from batch_idx
