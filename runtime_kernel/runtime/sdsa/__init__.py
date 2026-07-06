"""
sdsa — Self-Driven Scientific Agent Runtime.

The autonomous daemon loop that runs continuously:
    Generate Goals → Enqueue Experiments → Execute through Core → Causal Update → World Model Update

This is the system's "self-drive" engine. It does not depend on user requests.
"""

from runtime_kernel.runtime.sdsa.models import ExperimentEntry, ResearchGoal, SDSACycleResult
from runtime_kernel.runtime.sdsa.goal_generator import generate_goals
from runtime_kernel.runtime.sdsa.experiment_queue import ExperimentQueue
from runtime_kernel.runtime.sdsa.daemon_loop import AutonomousDaemonLoop

__all__ = [
    "AutonomousDaemonLoop",
    "ExperimentEntry",
    "ExperimentQueue",
    "ResearchGoal",
    "SDSACycleResult",
    "generate_goals",
]
