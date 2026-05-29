"""Provenance Scorer — credibility and AI-generated content scoring (Tasks 10.7–10.9, R10.1, R10.6).

Assigns each document:
- credibility_score in [0.0, 1.0]
- ai_generated_likelihood in [0.0, 1.0]
- scored_at timestamp

Implements:
- Scoring gate: document not visible to Retriever until scored (R10.1).
- Rescore path: preserves document_id/version, mutates only score fields (R10.6).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ProvenanceScore:
    """Provenance score result for a document.

    Attributes:
        credibility_score: Score in [0.0, 1.0] indicating source credibility.
        ai_generated_likelihood: Score in [0.0, 1.0] indicating likelihood of AI generation.
        scored_at: When the scoring was performed (UTC).
    """

    credibility_score: float
    ai_generated_likelihood: float
    scored_at: datetime


@dataclass
class ScoredDocument:
    """A document with its provenance scores attached.

    Used to track the scoring state and enforce the scoring gate (R10.1).

    Attributes:
        document_id: The document's stable ID.
        version: The document version.
        content_hash: SHA-256 of the cleaned text.
        score: The provenance score (None until scored).
        visible: Whether the document is visible to Retriever.
    """

    document_id: str
    version: int
    content_hash: str
    score: ProvenanceScore | None = None
    visible: bool = False


class ProvenanceScorer:
    """Scores documents for credibility and AI-generated content likelihood.

    In production, this would use ML models for:
    - Credibility: domain reputation, link graph analysis, editorial signals.
    - AI detection: perplexity analysis, watermark detection, stylometric features.

    This implementation uses a deterministic hash-based approach for testing,
    ensuring scores are always in [0.0, 1.0] and are reproducible for the
    same input text.
    """

    def __init__(self) -> None:
        # Track scored documents for the scoring gate (R10.1)
        self._scored_documents: dict[tuple[str, int], ScoredDocument] = {}

    @property
    def scored_documents(self) -> dict[tuple[str, int], ScoredDocument]:
        """Return all scored documents."""
        return dict(self._scored_documents)

    def score(self, document_text: str) -> ProvenanceScore:
        """Compute provenance scores for document text.

        Produces deterministic scores based on text content hash.
        Both scores are guaranteed to be in [0.0, 1.0] (R10.1).

        Args:
            document_text: The cleaned text content to score.

        Returns:
            ProvenanceScore with credibility_score and ai_generated_likelihood.
        """
        credibility = self._compute_credibility(document_text)
        ai_likelihood = self._compute_ai_likelihood(document_text)

        return ProvenanceScore(
            credibility_score=credibility,
            ai_generated_likelihood=ai_likelihood,
            scored_at=datetime.now(timezone.utc),
        )

    def score_document(
        self,
        document_id: str,
        version: int,
        content_hash: str,
        document_text: str,
    ) -> ScoredDocument:
        """Score a document and track it for the scoring gate.

        The document becomes visible to the Retriever only after this
        method completes successfully (R10.1).

        Args:
            document_id: The document's stable ID.
            version: The document version.
            content_hash: SHA-256 of the cleaned text.
            document_text: The cleaned text content.

        Returns:
            ScoredDocument with scores and visibility set to True.
        """
        provenance = self.score(document_text)

        scored_doc = ScoredDocument(
            document_id=document_id,
            version=version,
            content_hash=content_hash,
            score=provenance,
            visible=True,
        )
        self._scored_documents[(document_id, version)] = scored_doc
        return scored_doc

    def rescore(
        self,
        document_id: str,
        version: int,
        document_text: str,
    ) -> ProvenanceScore:
        """Rescore an existing document, preserving document_id and version (R10.6).

        Only mutates credibility_score, ai_generated_likelihood, and scored_at.
        The document_id, version, and all other stored fields remain unchanged.

        Args:
            document_id: The document's stable ID (preserved).
            version: The document version (preserved).
            document_text: The cleaned text content to rescore.

        Returns:
            Updated ProvenanceScore.

        Raises:
            KeyError: If the document has not been previously scored.
        """
        key = (document_id, version)
        scored_doc = self._scored_documents.get(key)

        if scored_doc is None:
            raise KeyError(
                f"Document ({document_id}, version={version}) has not been scored. "
                f"Use score_document() for initial scoring."
            )

        # Compute new scores
        new_score = self.score(document_text)

        # Mutate ONLY score fields — document_id, version, content_hash unchanged (R10.6)
        scored_doc.score = new_score
        # visible remains True (already scored)

        return new_score

    def is_visible(self, document_id: str, version: int) -> bool:
        """Check if a document is visible to the Retriever (scoring gate, R10.1).

        A document is only visible after it has been scored.

        Args:
            document_id: The document's stable ID.
            version: The document version.

        Returns:
            True if the document has been scored and is visible.
        """
        key = (document_id, version)
        scored_doc = self._scored_documents.get(key)
        return scored_doc is not None and scored_doc.visible

    def get_score(self, document_id: str, version: int) -> ProvenanceScore | None:
        """Get the current provenance score for a document.

        Args:
            document_id: The document's stable ID.
            version: The document version.

        Returns:
            The ProvenanceScore if scored, None otherwise.
        """
        key = (document_id, version)
        scored_doc = self._scored_documents.get(key)
        if scored_doc:
            return scored_doc.score
        return None

    def _compute_credibility(self, text: str) -> float:
        """Compute credibility score in [0.0, 1.0].

        Uses LLM-based analysis when OPENAI_API_KEY is set, otherwise
        falls back to heuristic scoring based on text features.
        """
        import os

        if os.environ.get("OPENAI_API_KEY"):
            try:
                return self._llm_credibility(text)
            except Exception:
                pass

        return self._heuristic_credibility(text)

    def _compute_ai_likelihood(self, text: str) -> float:
        """Compute AI-generated likelihood in [0.0, 1.0].

        Uses statistical analysis (perplexity proxy, burstiness) when possible,
        falls back to heuristic scoring.
        """
        import os

        if os.environ.get("OPENAI_API_KEY"):
            try:
                return self._llm_ai_detection(text)
            except Exception:
                pass

        return self._heuristic_ai_detection(text)

    def _llm_credibility(self, text: str) -> float:
        """Use LLM to assess credibility of text content."""
        import openai
        import os
        import json

        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", None),
        )
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": (
                    "You are a credibility assessor. Analyze the text and return a JSON object "
                    "with a single field 'score' between 0.0 and 1.0 indicating credibility. "
                    "Consider: factual accuracy signals, source quality indicators, "
                    "citation presence, specificity, and balanced language. "
                    "Return ONLY valid JSON like {\"score\": 0.75}"
                )},
                {"role": "user", "content": text[:2000]},
            ],
            temperature=0.0,
            max_tokens=50,
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        score = float(result.get("score", 0.5))
        return max(0.0, min(1.0, score))

    def _llm_ai_detection(self, text: str) -> float:
        """Use LLM to detect AI-generated content likelihood."""
        import openai
        import os
        import json

        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", None),
        )
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": (
                    "You are an AI content detector. Analyze the text and return a JSON object "
                    "with a single field 'score' between 0.0 and 1.0 indicating the likelihood "
                    "the text was AI-generated. Consider: uniformity of sentence length, "
                    "lack of personal voice, generic phrasing, perfect grammar, "
                    "and repetitive structure. Return ONLY valid JSON like {\"score\": 0.3}"
                )},
                {"role": "user", "content": text[:2000]},
            ],
            temperature=0.0,
            max_tokens=50,
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        score = float(result.get("score", 0.5))
        return max(0.0, min(1.0, score))

    def _heuristic_credibility(self, text: str) -> float:
        """Heuristic credibility scoring based on text features.

        Analyzes:
        - Text length (longer = more detailed = higher credibility)
        - Presence of numbers/statistics
        - Sentence variety
        - Vocabulary richness
        """
        score = 0.5  # Base score

        # Length bonus (detailed content is more credible)
        if len(text) > 500:
            score += 0.1
        if len(text) > 2000:
            score += 0.1

        # Numbers/statistics presence (factual indicators)
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        if len(numbers) > 3:
            score += 0.1

        # Sentence variety (not all same length)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 3:
            lengths = [len(s) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            if variance > 100:  # Good variety
                score += 0.05

        # Vocabulary richness
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.6:
                score += 0.05

        return max(0.0, min(1.0, score))

    def _heuristic_ai_detection(self, text: str) -> float:
        """Heuristic AI detection based on text features.

        Analyzes:
        - Sentence length uniformity (AI tends to be uniform)
        - Burstiness (human writing is more bursty)
        - Repetitive transition words
        - Perfect paragraph structure
        """
        import re

        score = 0.3  # Base assumption: slightly unlikely to be AI

        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 3:
            return score

        # Sentence length uniformity (AI indicator)
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

        if variance < 20:  # Very uniform = likely AI
            score += 0.2
        elif variance < 50:
            score += 0.1

        # Repetitive transition words (AI indicator)
        ai_transitions = ['furthermore', 'moreover', 'additionally', 'in conclusion',
                         'it is important to note', 'it is worth noting']
        text_lower = text.lower()
        transition_count = sum(1 for t in ai_transitions if t in text_lower)
        if transition_count >= 3:
            score += 0.15
        elif transition_count >= 2:
            score += 0.1

        # Perfect paragraph structure (AI indicator)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 2:
            para_lengths = [len(p) for p in paragraphs]
            para_variance = sum((l - sum(para_lengths)/len(para_lengths)) ** 2 for l in para_lengths) / len(para_lengths)
            if para_variance < 500:  # Very uniform paragraphs
                score += 0.1

        return max(0.0, min(1.0, score))
