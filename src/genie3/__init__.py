"""Genie 3: fast, all-atom SE(3)-equivariant protein design.

Entry points
------------
- CLI:  ``genie3 run|generate|evaluate|status|train`` (see ``genie3.cli``)
- API:  ``genie3.generation.workflow.run_generation``
        ``genie3.evaluation.workflow.run_evaluation``

Sub-packages
------------
config      Experiment configuration models and YAML loader.
runtime     Run context, structured logging, progress reporting, profiling.
generation  Diffusion model, samplers, data loading, training infrastructure.
evaluation  Inverse folding, structure prediction, metric reduction.
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
