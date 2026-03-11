"""Legal analysis agents."""

from .parser_agent import ParserAgent
from .structure_agent import StructureAgent
from .timeline_agent import TimelineAgent
from .contradiction_agent import ContradictionAgent
from .research_agent import ResearchAgent

__all__ = [
    "ParserAgent",
    "StructureAgent",
    "TimelineAgent",
    "ContradictionAgent",
    "ResearchAgent",
]
