"""LLM provider abstraction for streaming token generation.

Implements a protocol/interface pattern with:
- LLMProvider: Protocol defining the streaming interface.
- MockLLMProvider: Test implementation that yields predetermined tokens.
- OpenAIProvider / AnthropicProvider: Placeholder implementations for real providers.

The provider abstraction supports streaming token output and handles
model failures gracefully (R6.5).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an LLM provider."""

    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class GenerationRequest:
    """Request to generate an answer from source documents."""

    query: str
    context_documents: list[dict[str, str]]  # [{document_id, title, text}]
    system_prompt: str = ""
    max_tokens: int = 4096


class LLMProviderError(Exception):
    """Raised when the LLM provider encounters an error."""

    pass


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers with streaming token output."""

    async def stream_tokens(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM provider.

        Yields individual tokens as strings.
        Raises LLMProviderError on model failure.
        """
        ...  # pragma: no cover


class MockLLMProvider:
    """Mock LLM provider for testing.

    Yields predetermined tokens with configurable delays and failure modes.
    """

    def __init__(
        self,
        tokens: list[str] | None = None,
        delay_seconds: float = 0.0,
        fail_after: int | None = None,
        hang_after: int | None = None,
    ):
        """Initialize mock provider.

        Args:
            tokens: List of tokens to yield. Defaults to a simple response.
            delay_seconds: Delay between tokens (simulates streaming).
            fail_after: Raise LLMProviderError after this many tokens.
            hang_after: Stop yielding (simulate silence) after this many tokens.
        """
        self.tokens = tokens if tokens is not None else ["The ", "answer ", "is ", "42."]
        self.delay_seconds = delay_seconds
        self.fail_after = fail_after
        self.hang_after = hang_after
        self._tokens_yielded = 0

    async def stream_tokens(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Stream predetermined tokens with optional failure simulation."""
        self._tokens_yielded = 0
        for token in self.tokens:
            if self.fail_after is not None and self._tokens_yielded >= self.fail_after:
                raise LLMProviderError("Simulated model failure")

            if self.hang_after is not None and self._tokens_yielded >= self.hang_after:
                # Simulate indefinite silence — caller's timeout will trigger
                await asyncio.sleep(3600)
                return

            if self.delay_seconds > 0:
                await asyncio.sleep(self.delay_seconds)

            yield token
            self._tokens_yielded += 1


class OpenAIProvider:
    """Real OpenAI provider implementation with streaming.

    Uses the OpenAI API for streaming token generation.
    Falls back to MockLLMProvider if openai package is not installed
    or OPENAI_API_KEY is not set.
    """

    def __init__(self, config: ProviderConfig | None = None):
        self.config = config or ProviderConfig(model="gpt-4o-mini")
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                import openai
                import os

                api_key = os.environ.get("OPENAI_API_KEY", "")
                if not api_key:
                    raise LLMProviderError("OPENAI_API_KEY environment variable not set")

                # Support custom base URL (for Groq, Together, etc.)
                base_url = os.environ.get("OPENAI_BASE_URL", None)
                self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                raise LLMProviderError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    async def stream_tokens(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from OpenAI API."""
        client = self._get_client()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Build user message with context
        user_content = request.query
        if request.context_documents:
            context = "\n\n".join(
                f"[Source: {doc.get('document_id', 'unknown')}] {doc.get('title', '')}\n{doc.get('text', '')}"
                for doc in request.context_documents
            )
            user_content = f"Context:\n{context}\n\nQuestion: {request.query}"

        messages.append({"role": "user", "content": user_content})

        try:
            stream = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=request.max_tokens or self.config.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise LLMProviderError(f"OpenAI API error: {str(e)}") from e


class AnthropicProvider:
    """Real Anthropic provider implementation with streaming.

    Uses the Anthropic API for streaming token generation.
    Falls back gracefully if anthropic package is not installed
    or ANTHROPIC_API_KEY is not set.
    """

    def __init__(self, config: ProviderConfig | None = None):
        self.config = config or ProviderConfig(model="claude-sonnet-4-20250514")
        self._client = None

    def _get_client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                import os

                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    raise LLMProviderError("ANTHROPIC_API_KEY environment variable not set")
                self._client = anthropic.AsyncAnthropic(api_key=api_key)
            except ImportError:
                raise LLMProviderError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    async def stream_tokens(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Anthropic API."""
        client = self._get_client()

        # Build user message with context
        user_content = request.query
        if request.context_documents:
            context = "\n\n".join(
                f"[Source: {doc.get('document_id', 'unknown')}] {doc.get('title', '')}\n{doc.get('text', '')}"
                for doc in request.context_documents
            )
            user_content = f"Context:\n{context}\n\nQuestion: {request.query}"

        try:
            async with client.messages.stream(
                model=self.config.model,
                max_tokens=request.max_tokens or self.config.max_tokens,
                system=request.system_prompt or "You are a research assistant.",
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise LLMProviderError(f"Anthropic API error: {str(e)}") from e
