"""Performance benchmarks for the LangGraph LiveKit adapter.

These benchmarks target the message conversion hot path — the functions called
for every chunk during real-time audio streaming.
"""

import asyncio
import pytest

from langchain_core.messages import AIMessageChunk, HumanMessage
from livekit.agents import llm

from langgraph_livekit_agents import LangGraphStream
from langgraph_livekit_agents.types import TypedLivekit


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_text_chat_message(text: str = "Hello, how are you?") -> llm.ChatMessage:
    """Create a simple text ChatMessage."""
    return llm.ChatMessage(role="user", content=text)


def _make_image_chat_message() -> llm.ChatMessage:
    """Create a ChatMessage with mixed text and image content."""
    return llm.ChatMessage(
        role="user",
        content=[
            "Describe this image",
            llm.ChatImage(image="https://example.com/image.png"),
        ],
    )


# _to_message uses `cls` as first param name but never references it.
# We pass None as the instance to call it without constructing LangGraphStream.
_to_message = LangGraphStream._to_message


# ---------------------------------------------------------------------------
# Benchmarks: _to_message  (sync, hot path for every user utterance)
# ---------------------------------------------------------------------------


class TestToMessageBenchmarks:
    """Benchmark LangGraphStream._to_message which converts LiveKit messages
    to LangChain HumanMessage objects."""

    def test_to_message_text(self, benchmark):
        msg = _make_text_chat_message()
        result = benchmark(_to_message, None, msg)
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello, how are you?"

    def test_to_message_multimodal(self, benchmark):
        msg = _make_image_chat_message()
        result = benchmark(_to_message, None, msg)
        assert isinstance(result, HumanMessage)
        assert isinstance(result.content, list)
        assert len(result.content) == 2

    def test_to_message_empty(self, benchmark):
        msg = llm.ChatMessage(role="user", content="")
        result = benchmark(_to_message, None, msg)
        assert isinstance(result, HumanMessage)
        assert result.content == ""


# ---------------------------------------------------------------------------
# Benchmarks: _create_livekit_chunk  (sync, called per streamed token)
# ---------------------------------------------------------------------------


class TestCreateChunkBenchmarks:
    """Benchmark LangGraphStream._create_livekit_chunk which builds LiveKit
    ChatChunk objects — called for every streamed token."""

    def test_create_chunk_short(self, benchmark):
        result = benchmark(LangGraphStream._create_livekit_chunk, "Hi")
        assert result is not None
        assert result.choices[0].delta.content == "Hi"

    def test_create_chunk_long(self, benchmark):
        long_text = "This is a longer response that simulates a full sentence from the LLM. " * 10
        result = benchmark(LangGraphStream._create_livekit_chunk, long_text)
        assert result is not None

    def test_create_chunk_with_id(self, benchmark):
        result = benchmark(
            LangGraphStream._create_livekit_chunk, "Hello", id="msg-123"
        )
        assert result is not None
        assert result.request_id == "msg-123"


# ---------------------------------------------------------------------------
# Benchmarks: _to_livekit_chunk  (async, called per streamed token)
# ---------------------------------------------------------------------------


def _run_async(coro_fn, *args):
    """Helper to run an async function synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro_fn(*args))
    finally:
        loop.close()


class TestToLivekitChunkBenchmarks:
    """Benchmark LangGraphStream._to_livekit_chunk which converts various
    message types to LiveKit ChatChunk objects."""

    def test_to_livekit_chunk_string(self, benchmark):
        result = benchmark(_run_async, LangGraphStream._to_livekit_chunk, "Hello world")
        assert result is not None

    def test_to_livekit_chunk_ai_message(self, benchmark):
        chunk = AIMessageChunk(content="streaming token", id="chunk-1")
        result = benchmark(_run_async, LangGraphStream._to_livekit_chunk, chunk)
        assert result is not None

    def test_to_livekit_chunk_dict(self, benchmark):
        data = {"id": "msg-42", "content": "dict-based content"}
        result = benchmark(_run_async, LangGraphStream._to_livekit_chunk, data)
        assert result is not None

    def test_to_livekit_chunk_none(self, benchmark):
        result = benchmark(_run_async, LangGraphStream._to_livekit_chunk, None)
        assert result is None


# ---------------------------------------------------------------------------
# Benchmarks: TypedLivekit  (sync helper used in graph nodes)
# ---------------------------------------------------------------------------


class TestTypedLivekitBenchmarks:
    """Benchmark TypedLivekit which wraps the LangGraph StreamWriter for
    voice events."""

    def test_say(self, benchmark):
        events = []
        writer = TypedLivekit(writer=events.append)
        benchmark(writer.say, "Hello from the agent")
        assert len(events) > 0
        assert events[-1]["type"] == "say"

    def test_flush(self, benchmark):
        events = []
        writer = TypedLivekit(writer=events.append)
        benchmark(writer.flush)
        assert len(events) > 0
        assert events[-1]["type"] == "flush"
