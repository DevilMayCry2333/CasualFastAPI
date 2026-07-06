"""
exploration — Exploration Layer: generates hypotheses, designs experiments,
and simulates multi-world scenarios.

Rules:
    - Cannot execute actions directly
    - All outputs are "candidates" for the Cognitive Layer
    - High stochasticity by design
    - Proposes experiments, does not run them
"""

from runtime_kernel.runtime.exploration.hypothesis_generator import StochasticHypothesisGenerator
from runtime_kernel.runtime.exploration.experiment_scheduler import ExperimentScheduler
from runtime_kernel.runtime.exploration.multi_world import MultiWorldSimulator

__all__ = [
    "StochasticHypothesisGenerator",
    "ExperimentScheduler",
    "MultiWorldSimulator",
]
