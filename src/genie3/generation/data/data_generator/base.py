"""
Base class for task-aware data generators.

Provides infrastructure for weighted task sampling, batch generation,
and checkpointing for training data sources.
"""

import random
import logging
import numpy as np
from abc import ABC
from typing import Dict, List, Iterator


class DataGenerator(ABC):
    """
    Base class for a task-aware data generator.

    This generator:
    - Samples tasks according to user-defined probabilities.
    - Iterates over filepaths sampled from clusters.
    - Produces batches indefinitely via an internal generator.
    - Supports checkpointing and restoring state.

    Expected attributes set externally:
        self.weight: weight of the data generator
        self.clusters: list of lists (each cluster contains filepaths)
        self.tasks: list of task names
        self.task_weights: list of task weights (order corresponding to self.tasks)
        self.batch_size: number of samples per batch
        self.synthetic: boolean flag for whether the dataset is synthetic / distillated
    """

    ##########################
    ###   Getter methods   ###
    ##########################

    def get_weight(self) -> float:
        """
        Get the weight of this data generator.

        Returns:
            float: Generator weight for sampling
        """
        return self.weight

    def get_tasks(self) -> List[str]:
        """
        Get list of tasks supported by this generator.

        Returns:
            list: Task names
        """
        return self.tasks

    ##########################
    ###   Setter methods   ###
    ##########################

    def set_synthetic(self, synthetic: bool):
        """
        Set whether this dataset is synthetic/distilled.

        Args:
            synthetic: True if synthetic, False otherwise
        """
        self.synthetic = synthetic

    def set_weight(self, weight: float):
        """
        Set the weight of this data generator.

        Args:
            weight: Generator weight for sampling
        """
        self.weight = weight

    def set_batch_size(self, batch_size: int):
        """
        Set batch size for this generator.

        Args:
            batch_size: Number of samples per batch
        """
        self.batch_size = batch_size

    def set_assignment(self, assignment: Dict[str, float]):
        """
        Set task assignment probabilities.

        Args:
            assignment (dict):
                Mapping {task_name: probability}

        Requirements:
            - Probabilities must sum to 1.
        """
        # Validate sum of task weight
        if np.sum([assignment[task] for task in assignment]) != 1:
            logging.error("Task weights should sum up to 1.")
            exit(0)

        # Set up tasks with weights
        self.tasks = [task for task in assignment]
        self.task_weights = [assignment[task] for task in self.tasks]

    #####################
    ###   Generator   ###
    #####################

    def reset(self):
        """
        Reset generator state.

        - Resets sample index.
        - Resamples one filepath from each cluster.
        - Shuffles resulting filepaths.
        - Recreates generation iterator.
        """

        # Reset sample index
        self.sample_index = 0

        # Resample filepaths
        self.filepaths = []
        for cluster in self.clusters:
            self.filepaths.append(random.choice(cluster))
        random.shuffle(self.filepaths)

        # Create generation iterator
        self.iterator = self.generate()

    def generate(self) -> Iterator[Dict]:
        """
        Infinite generator.

        Each iteration:
            1. Samples a task according to task_weights.
            2. Samples a batch of filepaths sequentially.
            3. Resets when filepaths are exhausted.
            4. Yields a dictionary with task + metadata.
        """
        while True:

            # Sample task
            task_index = np.random.choice(
                range(len(self.tasks)), p=self.task_weights, size=1
            )[0]
            task = self.tasks[task_index]

            # Sample filepaths
            filepaths = []
            for _ in range(self.batch_size):
                filepaths.append(self.filepaths[self.sample_index])
                self.sample_index += 1
                if self.sample_index == len(self.filepaths):
                    self.reset()

            yield {"task": task, "synthetic": self.synthetic, "filepaths": filepaths}

    ####################################
    ###   Python iterator protocol   ###
    ####################################

    def __next__(self) -> Dict:
        """Return next generated batch."""
        return next(self.iterator)

    #########################
    ###   Checkpointing   ###
    #########################

    def get_state(self) -> Dict:
        """
        Return generator state for checkpointing.

        Useful for:
            - Resuming training
            - Reproducibility
        """
        return {
            "clusters": self.clusters,
            "filepaths": self.filepaths,
            "sample_index": self.sample_index,
        }

    def load_state(self, state: Dict):
        """
        Restore generator state from checkpoint.

        Args:
            state: Output of get_state()
        """
        self.clusters = state["clusters"]
        self.filepaths = state["filepaths"]
        self.sample_index = state["sample_index"]
        self.iterator = self.generate()
