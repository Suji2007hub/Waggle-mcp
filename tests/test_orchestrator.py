from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from waggle.orchestrator import (
    AsyncMemoryOrchestrator,
    ConversationTurn,
    MemoryScope,
    RetrieveRequest,
)


@dataclass
class FakeGraph:
    observed: list[dict[str, str]] = field(default_factory=list)
    queries: list[dict[str, object]] = field(default_factory=list)

    def observe_conversation(
        self,
        *,
        user_message: str,
        assistant_response: str,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> dict[str, object]:
        self.observed.append(
            {
                "user_message": user_message,
                "assistant_response": assistant_response,
                "agent_id": agent_id,
                "project": project,
                "session_id": session_id,
            }
        )
        return {"ok": True}

    def query(
        self,
        *,
        query: str,
        max_nodes: int = 20,
        max_depth: int = 2,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
        retrieval_mode: str = "graph",
    ) -> dict[str, object]:
        payload = {
            "query": query,
            "max_nodes": max_nodes,
            "max_depth": max_depth,
            "agent_id": agent_id,
            "project": project,
            "session_id": session_id,
            "retrieval_mode": retrieval_mode,
        }
        self.queries.append(payload)
        return payload

    def prime_context(
        self,
        *,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        max_nodes: int = 25,
    ) -> dict[str, object]:
        return {
            "project": project,
            "agent_id": agent_id,
            "session_id": session_id,
            "max_nodes": max_nodes,
        }


@pytest.mark.asyncio
async def test_on_assistant_turn_enqueues_and_ingests() -> None:
    graph = FakeGraph()
    orchestrator = AsyncMemoryOrchestrator(graph)
    scope = MemoryScope(project="MCP", session_id="thread-1", agent_id="codex")
    turn = ConversationTurn(
        user_message="Please remember that we decided to publish to MCP registry.",
        assistant_response="Decision captured. We'll publish to registry and keep pyproject in sync.",
    )

    await orchestrator.start()
    try:
        plan = await orchestrator.on_assistant_turn(scope=scope, turn=turn)
        assert plan.should_ingest is True
        await asyncio.wait_for(orchestrator.flush(), timeout=2)
    finally:
        await orchestrator.stop()

    assert len(graph.observed) == 1
    observed = graph.observed[0]
    assert observed["project"] == "MCP"
    assert observed["session_id"] == "thread-1"


@pytest.mark.asyncio
async def test_on_assistant_turn_deduplicates_same_turn() -> None:
    graph = FakeGraph()
    orchestrator = AsyncMemoryOrchestrator(graph)
    scope = MemoryScope(project="MCP", session_id="thread-1", agent_id="codex")
    turn = ConversationTurn(
        user_message="Remember that MCP registry publishing needs server.json.",
        assistant_response="I'll keep that registry requirement in memory.",
    )

    await orchestrator.start()
    try:
        first = await orchestrator.on_assistant_turn(scope=scope, turn=turn)
        second = await orchestrator.on_assistant_turn(scope=scope, turn=turn)
        await asyncio.wait_for(orchestrator.flush(), timeout=2)
    finally:
        await orchestrator.stop()

    assert first.should_ingest is True
    assert second.should_ingest is False
    assert second.reason == "duplicate turn"
    assert len(graph.observed) == 1


@pytest.mark.asyncio
async def test_build_context_applies_token_budget() -> None:
    graph = FakeGraph()
    orchestrator = AsyncMemoryOrchestrator(graph)
    scope = MemoryScope(project="MCP", session_id="thread-2", agent_id="codex")

    result = await orchestrator.build_context(
        scope=scope,
        request=RetrieveRequest(
            query="what did we decide about pypi and registry publishing",
            max_context_tokens=340,
            max_nodes=20,
            max_depth=2,
        ),
    )

    assert result is not None
    assert len(graph.queries) == 1
    query_call = graph.queries[0]
    assert query_call["max_nodes"] == 4
    assert query_call["max_depth"] == 1
    assert query_call["project"] == "MCP"
