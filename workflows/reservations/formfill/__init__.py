"""
Generic web-form filling — read any business's booking or contact form and fill
it from the user's durable profile, with no per-site selectors.

  harvest.py  the DOM walk: form fields + their human labels, stable refs
  mapper.py   which fact fills which field (LLM sees fact keys, never values)
  agent.py    plan / execute / verify, and the dry-run CLI
"""

from .agent import FormAgent, FormOutcome, FormPlan, PlanEntry
from .harvest import FormField, FormOption, FormSnapshot, harvest, locator_for
from .mapper import Assignment, FieldMap, MissingFact, map_fields

__all__ = [
    "FormAgent", "FormOutcome", "FormPlan", "PlanEntry",
    "FormField", "FormOption", "FormSnapshot", "harvest", "locator_for",
    "Assignment", "FieldMap", "MissingFact", "map_fields",
]
