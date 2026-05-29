"""Provenance Scorer - Credibility and AI-generated content likelihood scoring.

Exports:
- ProvenanceScorer: Main scorer with scoring gate and rescore support
- ProvenanceScore: Score result dataclass
- ScoredDocument: Document with attached scores
"""

from backend.provenance_scorer.scorer import ProvenanceScore, ProvenanceScorer, ScoredDocument

__all__ = [
    "ProvenanceScorer",
    "ProvenanceScore",
    "ScoredDocument",
]
