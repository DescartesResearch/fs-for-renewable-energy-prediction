"""
Experiment generation and management for paper experiments.

Provides utilities for defining experiment matrices in YAML and generating
individual experiment configurations with parameter validation.
"""

from .matrix_generator import MatrixGenerator
from .launcher import ExperimentLauncher
from .utils import ExperimentUtils

__all__ = [
    "MatrixGenerator",
    "ExperimentLauncher",
    "ExperimentUtils",
]
