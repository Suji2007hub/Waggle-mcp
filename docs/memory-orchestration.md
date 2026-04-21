# Event-Driven Memory Orchestration

This architecture removes manual memory tool calls from end users. The chat runtime owns memory ingestion/retrieval and calls Waggle in the background.

## Why this pattern

- Deterministic behavior: memory writes are governed by policy, not prompt luck.
- Better UX: users chat normally; no `store_node` micromanagement.
- Lower token burn: ingestion is async and retrieval is bounded by budget.
- Auditable operation: each turn produces a clear ingest/retrieve decision.

## Runtime flow

1. User sends message.
2. Runtime asks orchestrator for retrieval context before model call.
3. Model generates response with retrieved memory context.
4. Runtime hands the completed `(user, assistant)` turn to orchestrator.
5. Orchestrator enqueues async ingestion and continues serving next turns.

## MCP-native guidance

Waggle also exposes memory policy through MCP so clients and models can discover the intended automatic behavior:

- Prompt: `waggle_memory_policy`
- Resource: `graph://memory-policy`
- Retrieval tools: `prime_context`, `query_graph`
- Ingestion tool: `observe_conversation`

Clients that support MCP prompts should load `waggle_memory_policy` into the assistant/system instructions. Clients that do not support prompts can read `graph://memory-policy` or hard-code the same policy in their runtime.

## Reference implementation

Use [orchestrator.py](../src/waggle/orchestrator.py) and [chat_runtime.py](../src/waggle/chat_runtime.py):

- `AsyncMemoryOrchestrator`: async queue + worker for ingestion.
- `MemoryPolicy`: controls ingest/retrieve gating.
- `MemoryScope`: namespace key (`tenant/project/agent/session/model`).
- `RetrieveRequest`: token-budgeted query request.
- `ConversationTurn.turn_id`: optional idempotency key for deduplicating retries.
- `OrchestratedChatRuntime`: concrete chat loop integration (retrieve before answer, ingest after answer).

## Integration skeleton

```python
import asyncio

from waggle import AsyncMemoryOrchestrator, MemoryScope, OrchestratedChatRuntime
from waggle.graph import MemoryGraph
from waggle.embeddings import EmbeddingModel


graph = MemoryGraph("~/.waggle/memory.db", EmbeddingModel("all-MiniLM-L6-v2"))
orchestrator = AsyncMemoryOrchestrator(graph)
runtime = OrchestratedChatRuntime(model=my_model_adapter, orchestrator=orchestrator)


async def main() -> None:
    await runtime.start()
    try:
        scope = MemoryScope(project="MCP", session_id="thread-123", agent_id="codex", model_id="gpt-5.4")
        turn = await runtime.handle_turn(
            user_message="What did we decide about the database?",
            scope=scope,
            turn_id="thread-123:msg-9",
        )
        print(turn.assistant_response)
    finally:
        await runtime.stop()

asyncio.run(main())
```

## Operational guidelines

- Ingest policy:
  - ingest decisions/preferences/constraints
  - skip acknowledgements/chatter
  - guard ingest frequency with per-scope interval
  - pass a stable `turn_id` when the chat platform has message IDs
- Retrieval policy:
  - query only when user prompt is meaningful
  - cap context by token budget, then fit `max_nodes` and `max_depth`
  - default to `retrieval_mode=graph`, escalate to `fusion` only when needed
- Scope isolation:
  - always pass stable `project`, `agent_id`, and `session_id`
  - avoid global unscoped writes in shared environments

## Next hardening steps

1. Add a persistent idempotency store if retries can cross process restarts.
2. Add dead-letter queue for repeated ingest failures.
3. Export metrics: queue depth, ingest success rate, retrieval hit rate, token overhead.
4. Apply PII redaction before ingestion in regulated workloads.
