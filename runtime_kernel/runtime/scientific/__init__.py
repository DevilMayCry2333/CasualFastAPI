"""
scientific — Autonomous Scientific Agent (ASA) subsystem.

Full scientific method loop:
    Question → Hypothesis → Experiment → Observation → Causal Analysis → Theory Update

All modules use structured data. No self-modifying code.
All updates are causally traceable.

Integration:
    RuntimeEngine creates a ScientificLoop in __init__.
    After each autonomous step, if should_run() is true, run_cycle() is called.
    Results feed into the existing World Model and evidence pipeline.
"""

from runtime_kernel.runtime.scientific.models import (
    CausalEdge,
    CycleSummary,
    ExperimentResult,
    ExperimentStep,
    Hypothesis,
    ScientificQuestion,
)
from runtime_kernel.runtime.scientific.loop import ScientificLoop

__all__ = [
    "CausalEdge",
    "CycleSummary",
    "ExperimentResult",
    "ExperimentStep",
    "Hypothesis",
    "ScientificLoop",
    "ScientificQuestion",
]
