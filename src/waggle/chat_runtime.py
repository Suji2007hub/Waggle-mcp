from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from waggle.models import SubgraphResult
from waggle.orchestrator import (
    AsyncMemoryOrchestrator,
    ConversationTurn,
    IngestPlan,
    MemoryScope,
    RetrieveRequest,
)


class ModelAdapter(Protocol):
    async def generate(
        self,
        *,
        user_message: str,
        context: SubgraphResult | None,
        scope: MemoryScope,
    ) -> str: ...


@dataclass(slots=True, frozen=True)
class RuntimeTurnResult:
    user_message: str
    assistant_response: str
    context: SubgraphResult | None
    ingest_plan: IngestPlan


class OrchestratedChatRuntime:
    """
    Runtime integration that automates memory retrieval and ingestion per turn.

    - Before answer: query scoped memory context.
    - After answer: enqueue conversation observation for durable memory storage.
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        orchestrator: AsyncMemoryOrchestrator,
        retrieval_mode: str = "graph",
        max_context_tokens: int = 1000,
        max_nodes: int = 12,
        max_depth: int = 2,
    ) -> None:
        self.model = model
        self.orchestrator = orchestrator
        self.retrieval_mode = retrieval_mode
        self.max_context_tokens = max_context_tokens
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    async def start(self) -> None:
        await self.orchestrator.start()

    async def stop(self) -> None:
        await self.orchestrator.stop()

    async def flush(self) -> None:
        await self.orchestrator.flush()

    async def handle_turn(
        self,
        *,
        user_message: str,
        scope: MemoryScope,
        turn_id: str = "",
        retrieve: bool = True,
    ) -> RuntimeTurnResult:
        message = user_message.strip()
        if not message:
            raise ValueError("user_message cannot be empty.")

        context: SubgraphResult | None = None
        if retrieve:
            context = await self.orchestrator.build_context(
                scope=scope,
                request=RetrieveRequest(
                    query=message,
                    retrieval_mode=self.retrieval_mode,
                    max_context_tokens=self.max_context_tokens,
                    max_nodes=self.max_nodes,
                    max_depth=self.max_depth,
                ),
            )

        assistant_response = (
            await self.model.generate(
                user_message=message,
                context=context,
                scope=scope,
            )
        ).strip()
        if not assistant_response:
            raise ValueError("Model returned an empty assistant response.")

        ingest_plan = await self.orchestrator.on_assistant_turn(
            scope=scope,
            turn=ConversationTurn(
                user_message=message,
                assistant_response=assistant_response,
                turn_id=turn_id,
            ),
        )
        return RuntimeTurnResult(
            user_message=message,
            assistant_response=assistant_response,
            context=context,
            ingest_plan=ingest_plan,
        )
