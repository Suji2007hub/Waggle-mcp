from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from waggle.chat_runtime import OrchestratedChatRuntime
from waggle.orchestrator import AsyncMemoryOrchestrator, MemoryScope


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


@dataclass
class FakeModel:
    calls: list[dict[str, object]] = field(default_factory=list)

    async def generate(
        self,
        *,
        user_message: str,
        context: dict[str, object] | None,
        scope: MemoryScope,
    ) -> str:
        self.calls.append(
            {
                "user_message": user_message,
                "context": context,
                "scope": scope,
            }
        )
        return "Stored and answered."


@pytest.mark.asyncio
async def test_runtime_automates_retrieve_then_ingest() -> None:
    graph = FakeGraph()
    model = FakeModel()
    orchestrator = AsyncMemoryOrchestrator(graph)
    runtime = OrchestratedChatRuntime(model=model, orchestrator=orchestrator)
    scope = MemoryScope(project="MCP", session_id="thread-42", agent_id="codex")

    await runtime.start()
    try:
        result = await runtime.handle_turn(
            user_message="What did we decide about MCP registry publishing?",
            scope=scope,
            turn_id="thread-42:msg-1",
        )
        await runtime.flush()
    finally:
        await runtime.stop()

    assert result.context is not None
    assert result.ingest_plan.should_ingest is True
    assert len(graph.queries) == 1
    assert len(graph.observed) == 1
    assert graph.queries[0]["project"] == "MCP"
    assert graph.observed[0]["session_id"] == "thread-42"
    assert model.calls[0]["context"] is not None


@pytest.mark.asyncio
async def test_runtime_can_skip_retrieval_but_still_ingest() -> None:
    graph = FakeGraph()
    model = FakeModel()
    orchestrator = AsyncMemoryOrchestrator(graph)
    runtime = OrchestratedChatRuntime(model=model, orchestrator=orchestrator)
    scope = MemoryScope(project="MCP", session_id="thread-43", agent_id="codex")

    await runtime.start()
    try:
        result = await runtime.handle_turn(
            user_message="Remember this decision.",
            scope=scope,
            turn_id="thread-43:msg-1",
            retrieve=False,
        )
        await runtime.flush()
    finally:
        await runtime.stop()

    assert result.context is None
    assert len(graph.queries) == 0
    assert len(graph.observed) == 1
