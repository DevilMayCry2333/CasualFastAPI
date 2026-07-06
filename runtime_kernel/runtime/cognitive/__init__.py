"""Cognitive Architecture — the agent's internal cognitive models.

This package replaces flat state-with-everything with specialized
cognitive models, each with a clear boundary:

  SelfModel      — "who I am" (identity, beliefs, goals, drives)
  WorldModel     — "what the world is" (places, objects, events, rules)
  SocialModel    — "who others are" (trust, cooperation, interaction history)
  KnowledgeModel — "what I know" (facts, hypotheses, evidence, contradictions)
  TheoryOfMind   — "what others believe" (perceived beliefs, goals, confidence)
  WorkingMemory  — "what I'm thinking about now" (current focus, active question)
  Attention      — "what I notice" (salience-based event filtering)
  Perception     — "how I perceive" (event-to-cognition pipeline)
"""

from runtime_kernel.runtime.cognitive.attention import filter_events
from runtime_kernel.runtime.cognitive.knowledge_model import KnowledgeModel
from runtime_kernel.runtime.cognitive.perception import (
    format_perception_for_prompt,
    perceive,
)
from runtime_kernel.runtime.cognitive.self_model import SelfModel
from runtime_kernel.runtime.cognitive.social_model import AgentSocialProfile, SocialModel
from runtime_kernel.runtime.cognitive.theory_of_mind import MentalState, TheoryOfMind
from runtime_kernel.runtime.cognitive.working_memory import WorkingMemory
from runtime_kernel.runtime.cognitive.world_model import WorldModel
from runtime_kernel.runtime.cognitive.causal_graph import CausalEdge as CausalGraphEdge, CausalGraph
from runtime_kernel.runtime.cognitive.probabilistic_wm import ProbabilisticWorldModel

__all__ = [
    "AgentSocialProfile",
    "filter_events",
    "format_perception_for_prompt",
    "KnowledgeModel",
    "MentalState",
    "perceive",
    "SelfModel",
    "SocialModel",
    "TheoryOfMind",
    "WorkingMemory",
    "CausalGraph",
    "CausalGraphEdge",
    "ProbabilisticWorldModel",
    "WorldModel",
]
