from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import mcp.server.stdio
import mcp.types as types
import uvicorn
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.lowlevel.server import request_ctx
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route

from waggle import __version__
from waggle.config import AppConfig
from waggle.embeddings import EmbeddingModel
from waggle.errors import (
    AuthenticationError,
    WaggleError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    ValidationFailure,
)
from waggle.graph import MemoryGraph
from waggle.logging_utils import configure_logging
from waggle.metrics import MetricsRegistry
from waggle.models import (
    ConflictEntry,
    ConflictListResult,
    ContextBundleExportResult,
    ContextScopeResult,
    GraphDiffResult,
    GraphStats,
    MarkdownVaultExportResult,
    MarkdownVaultImportResult,
    Node,
    NodeHistoryResult,
    NodeType,
    ObservationResult,
    PrimeContextResult,
    RelationType,
    SubgraphResult,
    TimelineResult,
    TopicResult,
)
from waggle.rate_limit import RateLimiter
from waggle.runtime_context import runtime_context
from waggle.serializer import (
    serialize_conflict_entry,
    serialize_conflicts,
    serialize_context_bundle_export,
    serialize_graph_diff,
    serialize_node_history,
    serialize_observation_result,
    serialize_prime_context,
    serialize_recent_nodes,
    serialize_stats,
    serialize_subgraph,
    serialize_timeline,
    serialize_topics,
)

LOGGER = logging.getLogger(__name__)
WRITE_HEAVY_TOOLS = {
    "store_node",
    "store_edge",
    "decompose_and_store",
    "observe_conversation",
    "import_graph_backup",
    "import_markdown_vault",
}
REQUIRED_RUNTIME_METHODS = (
    "export_context_bundle",
    "export_markdown_vault",
    "list_context_scopes",
    "get_node_history",
    "import_markdown_vault",
    "timeline",
    "list_conflicts",
    "resolve_conflict",
)

MEMORY_AUTOMATION_POLICY = """Waggle automatic memory policy

The user should not manually manage memory. The assistant/runtime is responsible for using Waggle tools.

Before answering:
- Use prime_context at the start of a new session when project, agent, or session scope is known.
- Use query_graph before answering questions that may depend on prior decisions, preferences, constraints, project state, or earlier conversation context.
- Keep retrieval narrow: start with max_nodes 8-12, max_depth 1-2, retrieval_mode graph. Use fusion only when transcript replay is needed.

After answering:
- Use observe_conversation after completed turns that contain durable information: decisions, preferences, constraints, requirements, user corrections, project facts, or meaningful task outcomes.
- Do not call store_node for normal conversation memory unless the user explicitly gives one atomic fact and no inference is needed.
- Skip memory writes for acknowledgements, greetings, short chatter, or failed/aborted work.

Scoping:
- Always pass stable project, agent_id, and session_id when available.
- Prefer scoped memory over global memory in shared workspaces.
"""


def _object_input_schema(
    properties: dict[str, Any] | None = None,
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _scope_properties() -> dict[str, dict[str, Any]]:
    return {
        "agent_id": {
            "type": "string",
            "default": "",
            "description": "Optional agent or client identifier used to partition memory.",
        },
        "project": {
            "type": "string",
            "default": "",
            "description": "Optional project or workspace name used to partition memory.",
        },
        "session_id": {
            "type": "string",
            "default": "",
            "description": "Optional conversation or run identifier used to partition memory.",
        },
    }


def _assert_runtime_feature_parity() -> None:
    missing = [name for name in REQUIRED_RUNTIME_METHODS if not hasattr(MemoryGraph, name)]
    if not missing:
        return
    joined = ", ".join(missing)
    raise RuntimeError(
        "Detected a stale waggle runtime on the import path. Missing methods: "
        f"{joined}. This usually means an older copied package in site-packages is "
        "shadowing the current source tree or editable install. Recreate the virtualenv "
        "or uninstall old waggle/graph-memory-mcp builds before running waggle-mcp."
    )


def _build_backend(config: AppConfig) -> Any:
    embedding_model = EmbeddingModel(config.model_name)
    if config.backend == "sqlite":
        return MemoryGraph(
            config.db_path,
            embedding_model,
            tenant_id=config.default_tenant_id,
            export_dir=config.export_dir,
        )
    from waggle.neo4j_graph import Neo4jMemoryGraph

    return Neo4jMemoryGraph(
        uri=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database or None,
        embedding_model=embedding_model,
        tenant_id=config.default_tenant_id,
        export_dir=config.export_dir,
    )


class WaggleServer:
    """MCP server wrapper with tenant-aware graph resolution."""

    def __init__(
        self,
        graph: Any | None = None,
        *,
        config: AppConfig | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.metrics = metrics or MetricsRegistry()
        self._static_graph = graph
        self._root_graph = graph or _build_backend(self.config)
        self.server = Server("waggle")
        self._register_handlers()

    @property
    def graph(self) -> Any:
        return self.current_graph()

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return self.build_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            return await anyio.to_thread.run_sync(self.handle_tool_call, name, arguments or {})

        @self.server.list_resources()
        async def list_resources(
            request: types.ListResourcesRequest | None = None,
        ) -> types.ListResourcesResult:
            del request
            return self.build_resources()

        @self.server.read_resource()
        async def read_resource(uri: Any) -> str:
            return self.read_resource_text(str(uri))

        @self.server.list_prompts()
        async def list_prompts() -> list[types.Prompt]:
            return self.build_prompts()

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
            return self.get_prompt_result(name, arguments or {})

    def build_tools(self) -> list[types.Tool]:
        return [
            types.Tool(
                name="store_node",
                description=(
                    "Store a piece of knowledge as a node in the persistent memory graph. "
                    "Call this whenever you learn something important from the user: facts, "
                    "preferences, decisions, entities, concepts, or questions. Prefer atomic facts."
                ),
                inputSchema=_object_input_schema(
                    {
                        "label": {"type": "string", "description": "Short label for the knowledge being stored."},
                        "content": {"type": "string", "description": "Full natural-language description for this node."},
                        "node_type": {
                            "type": "string",
                            "enum": [node_type.value for node_type in NodeType],
                            "description": "Category of knowledge represented by the node.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization.",
                            "default": [],
                        },
                        "source_prompt": {
                            "type": "string",
                            "description": "Optional original prompt that produced this knowledge.",
                            "default": "",
                        },
                        **_scope_properties(),
                    },
                    required=["label", "content", "node_type"],
                ),
            ),
            types.Tool(
                name="store_edge",
                description=(
                    "Create a relationship between two stored nodes. Use this immediately after "
                    "storing related nodes so the memory graph preserves structure, updates, and conflicts."
                ),
                inputSchema=_object_input_schema(
                    {
                        "source_id": {"type": "string", "description": "Source node ID."},
                        "target_id": {"type": "string", "description": "Target node ID."},
                        "relationship": {
                            "type": "string",
                            "enum": [relation.value for relation in RelationType],
                            "description": "Relationship between the two nodes.",
                        },
                        "weight": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 1.0,
                            "description": "Optional strength of the relationship.",
                        },
                    },
                    required=["source_id", "target_id", "relationship"],
                ),
            ),
            types.Tool(
                name="query_graph",
                description=(
                    "Automatically search the memory graph before answering questions that may depend on prior context, "
                    "user preferences, project decisions, constraints, or earlier conversation state. "
                    "Returns a serialized subgraph with matching nodes and their connected neighborhood. "
                    "Understands temporal references such as 'recently', 'latest', 'originally', and 'last week'."
                ),
                inputSchema=_object_input_schema(
                    {
                        "query": {"type": "string", "description": "Natural-language search query."},
                        "max_nodes": {
                            "type": "integer",
                            "default": 20,
                            "minimum": 1,
                            "description": "Maximum number of matching nodes to return.",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 2,
                            "minimum": 0,
                            "description": "Relationship traversal depth around matching nodes.",
                        },
                        **_scope_properties(),
                        "retrieval_mode": {
                            "type": "string",
                            "enum": ["graph", "replay", "fusion"],
                            "default": "graph",
                            "description": "Retrieval strategy: graph-only, transcript replay, or fused graph plus replay results.",
                        },
                    },
                    required=["query"],
                ),
            ),
            types.Tool(
                name="get_related",
                description=(
                    "Fetch the neighborhood around a specific memory node. Use when you already have a node ID "
                    "and need its connected context. Returns matching nodes and edges as a serialized subgraph."
                ),
                inputSchema=_object_input_schema(
                    {
                        "node_id": {"type": "string", "description": "ID of the node whose neighborhood should be returned."},
                        "max_depth": {
                            "type": "integer",
                            "default": 2,
                            "minimum": 0,
                            "description": "Relationship traversal depth from the starting node.",
                        },
                    },
                    required=["node_id"],
                ),
            ),
            types.Tool(
                name="get_node_history",
                description=(
                    "Inspect one memory node's evidence, validity window, and connected context. Use when auditing "
                    "why a memory exists or how it changed. Returns the node, evidence records, related nodes, and edges."
                ),
                inputSchema=_object_input_schema(
                    {
                        "node_id": {"type": "string", "description": "ID of the node to inspect."},
                        "max_depth": {
                            "type": "integer",
                            "default": 2,
                            "minimum": 0,
                            "description": "Relationship traversal depth for related context.",
                        },
                    },
                    required=["node_id"],
                ),
            ),
            types.Tool(
                name="list_context_scopes",
                description=(
                    "List known agent, project, and session scope values stored in the current tenant graph. "
                    "Use before filtering memory by scope. Returns arrays of scope identifiers."
                ),
                inputSchema=_object_input_schema(),
            ),
            types.Tool(
                name="timeline",
                description=(
                    "Build a chronological view of memory changes for a node, a query result, or the whole tenant. "
                    "Use when order and evidence matter. Returns timestamped timeline items."
                ),
                inputSchema=_object_input_schema(
                    {
                        "node_id": {"type": "string", "description": "Optional node ID to anchor the timeline."},
                        "query": {"type": "string", "description": "Optional natural-language query to select relevant memories."},
                        "limit": {
                            "type": "integer",
                            "default": 25,
                            "minimum": 1,
                            "description": "Maximum number of timeline items to return.",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 2,
                            "minimum": 0,
                            "description": "Relationship traversal depth when a node ID or query is supplied.",
                        },
                        "include_evidence": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to include evidence records alongside node and edge events.",
                        },
                    },
                ),
            ),
            types.Tool(
                name="list_conflicts",
                description=(
                    "List contradiction and update edges, with unresolved conflicts shown by default. "
                    "Use to review memory disagreements before resolving them. Returns conflict entries with source and target nodes."
                ),
                inputSchema=_object_input_schema(
                    {
                        "include_resolved": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to include conflicts that were already marked resolved.",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 25,
                            "minimum": 1,
                            "description": "Maximum number of conflicts to return.",
                        },
                    },
                ),
            ),
            types.Tool(
                name="resolve_conflict",
                description=(
                    "Mark a contradiction or update edge as resolved without deleting the underlying history. "
                    "Use after deciding how competing memories should be interpreted. Returns the resolved conflict entry."
                ),
                inputSchema=_object_input_schema(
                    {
                        "edge_id": {"type": "string", "description": "ID of the conflict edge to mark resolved."},
                        "resolution_note": {
                            "type": "string",
                            "default": "",
                            "description": "Optional human-readable note explaining the resolution.",
                        },
                    },
                    required=["edge_id"],
                ),
            ),
            types.Tool(
                name="update_node",
                description=(
                    "Update an existing memory node's content, label, or tags. Use when a stored memory needs correction "
                    "without deleting its identity. Returns the updated node."
                ),
                inputSchema=_object_input_schema(
                    {
                        "node_id": {"type": "string", "description": "ID of the node to update."},
                        "content": {"type": "string", "description": "Replacement natural-language content for the node."},
                        "label": {"type": "string", "description": "Replacement short label for the node."},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Replacement tag list for the node.",
                        },
                    },
                    required=["node_id"],
                ),
            ),
            types.Tool(
                name="delete_node",
                description="Delete a node and all connected edges from persistent memory.",
                inputSchema=_object_input_schema(
                    {"node_id": {"type": "string", "description": "ID of the node to delete."}},
                    required=["node_id"],
                ),
            ),
            types.Tool(
                name="decompose_and_store",
                description=(
                    "Break long or complex content into atomic memory nodes, store them automatically, and create inferred edges. "
                    "Use for notes, summaries, or multi-fact passages. Returns the stored subgraph."
                ),
                inputSchema=_object_input_schema(
                    {
                        "content": {"type": "string", "description": "Long-form content to decompose into memory nodes."},
                        "context": {
                            "type": "string",
                            "default": "",
                            "description": "Optional background that helps classify and connect extracted memories.",
                        },
                    },
                    required=["content"],
                ),
            ),
            types.Tool(
                name="observe_conversation",
                description=(
                    "Automatically observe a completed user-assistant turn, extract durable information, and store it in the graph. "
                    "Call this after turns containing preferences, decisions, constraints, requirements, corrections, project facts, "
                    "or meaningful task outcomes. Do not ask the user to trigger this."
                ),
                inputSchema=_object_input_schema(
                    {
                        "user_message": {"type": "string", "description": "The user's message from the completed turn."},
                        "assistant_response": {"type": "string", "description": "The assistant's response from the completed turn."},
                        **_scope_properties(),
                    },
                    required=["user_message", "assistant_response"],
                ),
            ),
            types.Tool(
                name="graph_diff",
                description=(
                    "Show what changed in the memory graph recently, including added nodes, updated nodes, created edges, "
                    "and contradiction edges. Use for review or handoff. Returns a serialized graph diff."
                ),
                inputSchema=_object_input_schema(
                    {
                        "since": {
                            "type": "string",
                            "default": "24h",
                            "description": "Lookback window such as '24h', '7d', or an ISO-like timestamp.",
                        }
                    }
                ),
            ),
            types.Tool(
                name="prime_context",
                description=(
                    "Automatically build a compact context brief at the start of a scoped conversation or before work that needs continuity. "
                    "Use to hydrate an assistant with the most relevant scoped memories. Returns summary text plus nodes and edges."
                ),
                inputSchema=_object_input_schema(_scope_properties()),
            ),
            types.Tool(
                name="get_topics",
                description=(
                    "Detect topic clusters in the graph using community detection. Use to understand the main themes "
                    "in memory. Returns labeled clusters with representative nodes and tags."
                ),
                inputSchema=_object_input_schema(),
            ),
            types.Tool(
                name="get_stats",
                description=(
                    "Return high-level statistics about the current memory graph. Use for health checks or quick summaries. "
                    "Returns node and edge counts, node type breakdowns, and recent or highly connected nodes."
                ),
                inputSchema=_object_input_schema(),
            ),
            types.Tool(
                name="export_graph_html",
                description=(
                    "Export the current memory graph as an interactive HTML visualization. "
                    "Use when a human needs to inspect the graph visually. Returns the output path and graph counts."
                ),
                inputSchema=_object_input_schema(
                    {
                        "output_path": {
                            "type": "string",
                            "description": "Optional destination HTML file path. If omitted, Waggle chooses an export path.",
                        },
                        "include_physics": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether the visualization should use physics-based node layout.",
                        },
                    },
                ),
            ),
            types.Tool(
                name="export_graph_backup",
                description=(
                    "Export the current graph as a portable JSON backup. Use for migration, restore drills, or offline archive. "
                    "Returns backup path, schema version, and object counts."
                ),
                inputSchema=_object_input_schema(
                    {
                        "output_path": {
                            "type": "string",
                            "description": "Optional destination JSON file path. If omitted, Waggle chooses an export path.",
                        }
                    }
                ),
            ),
            types.Tool(
                name="export_context_bundle",
                description=(
                    "Export a portable Markdown and/or JSON context bundle for handing memory to another AI or a human. "
                    "Use for cross-tool context transfer, audits, and resumable work. Returns file paths, counts, and render hints."
                ),
                inputSchema=_object_input_schema(
                    {
                        "mode": {
                            "type": "string",
                            "enum": ["prime", "query", "graph"],
                            "default": "prime",
                            "description": "Bundle selection mode: scoped prime context, query result, or broad graph export.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Natural-language query used when mode is 'query'.",
                        },
                        **_scope_properties(),
                        "max_nodes": {
                            "type": "integer",
                            "default": 25,
                            "minimum": 1,
                            "description": "Maximum number of nodes to include.",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 2,
                            "minimum": 0,
                            "description": "Relationship traversal depth for query or prime context modes.",
                        },
                        "retrieval_mode": {
                            "type": "string",
                            "enum": ["graph", "replay", "fusion"],
                            "default": "graph",
                            "description": "Retrieval strategy for query mode: graph-only, transcript replay, or fused results.",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "json", "both"],
                            "default": "both",
                            "description": "Output format to write.",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional destination file path or directory prefix for the bundle.",
                        },
                        "include_edges": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether relationship edges should be included in the export.",
                        },
                        "include_timestamps": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether created and updated timestamps should be included.",
                        },
                        "include_source_prompt": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether original source prompts should be included when available.",
                        },
                        "audience": {
                            "type": "string",
                            "enum": ["llm", "human"],
                            "default": "llm",
                            "description": "Target audience used to tune bundle rendering.",
                        },
                    },
                ),
            ),
            types.Tool(
                name="import_graph_backup",
                description=(
                    "Import a portable JSON graph backup into the current backend. Use for restores or migrations. "
                    "Returns counts for created and updated nodes and edges."
                ),
                inputSchema=_object_input_schema(
                    {"input_path": {"type": "string", "description": "Path to the JSON backup file to import."}},
                    required=["input_path"],
                ),
            ),
            types.Tool(
                name="export_markdown_vault",
                description=(
                    "Export the current graph as an Obsidian-compatible Markdown vault. "
                    "Use when a human wants browsable note files with graph links. Returns written files and graph counts."
                ),
                inputSchema=_object_input_schema(
                    {
                        "root_path": {"type": "string", "description": "Destination directory for the Markdown vault."},
                        **_scope_properties(),
                    },
                    required=["root_path"],
                ),
            ),
            types.Tool(
                name="import_markdown_vault",
                description=(
                    "Import an Obsidian-compatible Markdown vault into the current graph non-destructively. "
                    "Use to sync edited vault notes back into memory. Returns created, updated, deleted-edge, and conflict counts."
                ),
                inputSchema=_object_input_schema(
                    {"root_path": {"type": "string", "description": "Source directory of the Markdown vault to import."}},
                    required=["root_path"],
                ),
            ),
        ]

    def build_prompts(self) -> list[types.Prompt]:
        return [
            types.Prompt(
                name="waggle_memory_policy",
                title="Waggle Memory Policy",
                description=(
                    "Instructions for automatic memory retrieval and ingestion. "
                    "Use this prompt to make the assistant handle memory without user-triggered tool calls."
                ),
                arguments=[
                    types.PromptArgument(
                        name="project",
                        description="Optional project/workspace scope to pass to Waggle tools.",
                        required=False,
                    ),
                    types.PromptArgument(
                        name="agent_id",
                        description="Optional agent/client identifier to pass to Waggle tools.",
                        required=False,
                    ),
                    types.PromptArgument(
                        name="session_id",
                        description="Optional conversation/session identifier to pass to Waggle tools.",
                        required=False,
                    ),
                ],
            )
        ]

    def get_prompt_result(self, name: str, arguments: dict[str, str]) -> types.GetPromptResult:
        if name != "waggle_memory_policy":
            raise ValidationFailure(f"Unknown prompt: {name}")
        project = str(arguments.get("project", "")).strip()
        agent_id = str(arguments.get("agent_id", "")).strip()
        session_id = str(arguments.get("session_id", "")).strip()
        scope_lines = []
        if project:
            scope_lines.append(f"- project: {project}")
        if agent_id:
            scope_lines.append(f"- agent_id: {agent_id}")
        if session_id:
            scope_lines.append(f"- session_id: {session_id}")
        scope_text = "\n".join(scope_lines) if scope_lines else "- no explicit scope supplied"
        text = f"{MEMORY_AUTOMATION_POLICY}\nSuggested scope for this conversation:\n{scope_text}\n"
        return types.GetPromptResult(
            description="Automatic Waggle memory policy.",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=text),
                )
            ],
        )

    def _get_request(self) -> Request | None:
        try:
            current = request_ctx.get()
        except LookupError:
            return None
        return current.request if isinstance(current.request, Request) else None

    def current_graph(self) -> Any:
        request = self._get_request()
        if request is not None and getattr(request.state, "tenant_id", ""):
            return self._root_graph.for_tenant(request.state.tenant_id)
        return self._root_graph.for_tenant(self.config.default_tenant_id)

    def validate_startup(self) -> None:
        graph = self.current_graph()
        started = time.perf_counter()
        graph.ensure_tenant(graph.tenant_id)
        graph.embedding_model.embed("startup validation")
        self.metrics.observe("waggle_startup_validation_seconds", time.perf_counter() - started, backend=self.config.backend)

    def build_resources(self) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(uri="graph://stats", name="Graph Stats", description="Current graph statistics.", mimeType="text/plain"),
                types.Resource(uri="graph://recent", name="Recent Graph Nodes", description="The 10 most recently updated nodes.", mimeType="text/plain"),
                types.Resource(
                    uri="graph://memory-policy",
                    name="Automatic Memory Policy",
                    description="Policy for when assistants should retrieve and write Waggle memory automatically.",
                    mimeType="text/plain",
                ),
            ]
        )

    def read_resource_text(self, uri: str) -> str:
        graph = self.current_graph()
        if uri == "graph://stats":
            return serialize_stats(graph.get_stats())
        if uri == "graph://recent":
            return serialize_recent_nodes(graph.list_recent_nodes(limit=10))
        if uri == "graph://memory-policy":
            return MEMORY_AUTOMATION_POLICY
        raise ValidationFailure(f"Unknown resource: {uri}")

    def initialization_options(self) -> InitializationOptions:
        return InitializationOptions(
            server_name="waggle",
            server_version="0.2.0",
            capabilities=self.server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
        )

    def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        graph = self.current_graph()
        request = self._get_request()
        request_id = ""
        if request is not None:
            request_id = getattr(request.state, "request_id", "")
        else:
            try:
                request_id = str(request_ctx.get().request_id)
            except LookupError:
                request_id = ""

        started = time.perf_counter()
        with runtime_context(
            request_id=request_id,
            tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
            transport="http" if request is not None else "stdio",
            backend=self.config.backend,
            api_key_id=getattr(getattr(request, "state", object()), "api_key_id", "") if request is not None else "",
            tool_name=name,
        ):
            try:
                self._validate_tool_payload(name, arguments)
                LOGGER.info("tool_call_started")
                if name == "store_node":
                    store_result = graph.add_node(
                        label=arguments["label"],
                        content=arguments["content"],
                        node_type=NodeType(arguments["node_type"]),
                        tags=arguments.get("tags", []),
                        source_prompt=arguments.get("source_prompt", ""),
                        agent_id=arguments.get("agent_id", ""),
                        project=arguments.get("project", ""),
                        session_id=arguments.get("session_id", ""),
                    )
                    node = store_result.node
                    text = (
                        f"Stored node '{node.label}' with id {node.id}."
                        if store_result.created
                        else f"Reused existing node '{node.label}' with id {node.id}."
                    )
                    if store_result.conflicts:
                        text += f" Detected {len(store_result.conflicts)} potential conflict(s)."
                    if not store_result.created:
                        self.metrics.increment(
                            "waggle_dedup_hits_total",
                            tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
                            dedup_reason=store_result.dedup_reason or "unknown",
                        )
                    if store_result.conflicts:
                        self.metrics.increment(
                            "waggle_conflicts_total",
                            value=len(store_result.conflicts),
                            tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
                        )
                    result = self._tool_result(
                        text,
                        {
                            **self._node_payload(node),
                            "created": store_result.created,
                            "dedup_reason": store_result.dedup_reason,
                            "similarity": store_result.similarity,
                            "conflicts": [
                                {
                                    "other_node_id": conflict.other_node_id,
                                    "other_node_label": conflict.other_node_label,
                                    "relationship": conflict.relationship,
                                    "reason": conflict.reason,
                                }
                                for conflict in store_result.conflicts
                            ],
                        },
                    )
                elif name == "store_edge":
                    edge = graph.add_edge(
                        source_id=arguments["source_id"],
                        target_id=arguments["target_id"],
                        relationship=arguments["relationship"],
                        weight=float(arguments.get("weight", 1.0)),
                    )
                    result = self._tool_result(
                        f"Created edge {edge.id} linking {edge.source_id} to {edge.target_id} as {edge.relationship}.",
                        self._edge_payload(edge),
                    )
                elif name == "query_graph":
                    subgraph = graph.query(
                        query=arguments["query"],
                        max_nodes=int(arguments.get("max_nodes", 20)),
                        max_depth=int(arguments.get("max_depth", 2)),
                        agent_id=arguments.get("agent_id", ""),
                        project=arguments.get("project", ""),
                        session_id=arguments.get("session_id", ""),
                        retrieval_mode=arguments.get("retrieval_mode", "graph"),
                    )
                    result = self._tool_result(
                        serialize_subgraph(subgraph),
                        self._subgraph_payload(subgraph),
                    )
                elif name == "list_context_scopes":
                    scopes = graph.list_context_scopes()
                    result = self._tool_result(
                        f"Known scopes: {len(scopes.agent_ids)} agents, {len(scopes.projects)} projects, {len(scopes.session_ids)} sessions.",
                        self._context_scope_payload(scopes),
                    )
                elif name == "get_related":
                    subgraph = graph.get_related(node_id=arguments["node_id"], max_depth=int(arguments.get("max_depth", 2)))
                    result = self._tool_result(serialize_subgraph(subgraph), self._subgraph_payload(subgraph))
                elif name == "get_node_history":
                    history = graph.get_node_history(node_id=arguments["node_id"], max_depth=int(arguments.get("max_depth", 2)))
                    result = self._tool_result(serialize_node_history(history), self._node_history_payload(history))
                elif name == "timeline":
                    timeline = graph.timeline(
                        node_id=arguments.get("node_id", ""),
                        query=arguments.get("query", ""),
                        limit=int(arguments.get("limit", 25)),
                        max_depth=int(arguments.get("max_depth", 2)),
                        include_evidence=bool(arguments.get("include_evidence", True)),
                    )
                    result = self._tool_result(serialize_timeline(timeline), self._timeline_payload(timeline))
                elif name == "list_conflicts":
                    conflicts = graph.list_conflicts(
                        include_resolved=bool(arguments.get("include_resolved", False)),
                        limit=int(arguments.get("limit", 25)),
                    )
                    result = self._tool_result(serialize_conflicts(conflicts), self._conflict_list_payload(conflicts))
                elif name == "resolve_conflict":
                    resolved = graph.resolve_conflict(
                        edge_id=arguments["edge_id"],
                        resolution_note=arguments.get("resolution_note", ""),
                    )
                    result = self._tool_result(serialize_conflict_entry(resolved), self._conflict_entry_payload(resolved))
                elif name == "update_node":
                    node = graph.update_node(
                        node_id=arguments["node_id"],
                        content=arguments.get("content"),
                        label=arguments.get("label"),
                        tags=arguments.get("tags"),
                    )
                    result = self._tool_result(f"Updated node '{node.label}' ({node.id}).", self._node_payload(node))
                elif name == "delete_node":
                    node = graph.delete_node(node_id=arguments["node_id"])
                    result = self._tool_result(
                        f"Deleted node '{node.label}' ({node.id}) and its connected edges.",
                        {"id": node.id, "label": node.label, "tenant_id": node.tenant_id},
                    )
                elif name == "decompose_and_store":
                    subgraph = graph.decompose_and_store(content=arguments["content"], context=arguments.get("context", ""))
                    result = self._tool_result(serialize_subgraph(subgraph), self._subgraph_payload(subgraph))
                elif name == "observe_conversation":
                    observation = graph.observe_conversation(
                        user_message=arguments["user_message"],
                        assistant_response=arguments["assistant_response"],
                        agent_id=arguments.get("agent_id", ""),
                        project=arguments.get("project", ""),
                        session_id=arguments.get("session_id", ""),
                    )
                    result = self._tool_result(
                        serialize_observation_result(observation),
                        self._observation_payload(observation),
                    )
                elif name == "graph_diff":
                    diff = graph.graph_diff(since=arguments.get("since", "24h"))
                    result = self._tool_result(serialize_graph_diff(diff), self._graph_diff_payload(diff))
                elif name == "prime_context":
                    context_result = graph.prime_context(
                        project=arguments.get("project", ""),
                        agent_id=arguments.get("agent_id", ""),
                        session_id=arguments.get("session_id", ""),
                    )
                    result = self._tool_result(serialize_prime_context(context_result), self._prime_context_payload(context_result))
                elif name == "get_topics":
                    topics = graph.get_topics()
                    result = self._tool_result(serialize_topics(topics), self._topic_payload(topics))
                elif name == "get_stats":
                    stats = graph.get_stats()
                    result = self._tool_result(serialize_stats(stats), self._stats_payload(stats))
                elif name == "export_graph_html":
                    output_path = graph.export_graph_html(
                        output_path=arguments.get("output_path"),
                        include_physics=bool(arguments.get("include_physics", True)),
                    )
                    stats = graph.get_stats()
                    result = self._tool_result(
                        f"Exported graph visualization to {output_path}.",
                        {
                            "output_path": str(output_path),
                            "tenant_id": graph.tenant_id,
                            "total_nodes": stats.total_nodes,
                            "total_edges": stats.total_edges,
                        },
                    )
                elif name == "export_graph_backup":
                    backup = graph.export_graph_backup(output_path=arguments.get("output_path"))
                    result = self._tool_result(
                        f"Exported graph backup to {backup.output_path}.",
                        {
                            "output_path": backup.output_path,
                            "tenant_id": backup.tenant_id,
                            "schema_version": backup.schema_version,
                            "node_count": backup.node_count,
                            "edge_count": backup.edge_count,
                        },
                    )
                elif name == "export_context_bundle":
                    exported = graph.export_context_bundle(
                        mode=arguments.get("mode", "prime"),
                        query=arguments.get("query", ""),
                        project=arguments.get("project", ""),
                        agent_id=arguments.get("agent_id", ""),
                        session_id=arguments.get("session_id", ""),
                        max_nodes=int(arguments.get("max_nodes", 25)),
                        max_depth=int(arguments.get("max_depth", 2)),
                        retrieval_mode=arguments.get("retrieval_mode", "graph"),
                        format=arguments.get("format", "both"),
                        output_path=arguments.get("output_path"),
                        include_edges=bool(arguments.get("include_edges", True)),
                        include_timestamps=bool(arguments.get("include_timestamps", True)),
                        include_source_prompt=bool(arguments.get("include_source_prompt", False)),
                        audience=arguments.get("audience", "llm"),
                    )
                    result = self._tool_result(
                        serialize_context_bundle_export(exported),
                        self._context_bundle_payload(exported),
                    )
                elif name == "import_graph_backup":
                    imported = graph.import_graph_backup(input_path=arguments["input_path"])
                    result = self._tool_result(
                        f"Imported graph backup from {imported.input_path}.",
                        {
                            "input_path": imported.input_path,
                            "tenant_id": imported.tenant_id,
                            "schema_version": imported.schema_version,
                            "nodes_created": imported.nodes_created,
                            "nodes_updated": imported.nodes_updated,
                            "edges_created": imported.edges_created,
                            "edges_updated": imported.edges_updated,
                        },
                    )
                elif name == "export_markdown_vault":
                    exported = graph.export_markdown_vault(
                        root_path=arguments["root_path"],
                        project=arguments.get("project", ""),
                        agent_id=arguments.get("agent_id", ""),
                        session_id=arguments.get("session_id", ""),
                    )
                    result = self._tool_result(
                        f"Exported Markdown vault to {exported.root_path}.",
                        self._markdown_vault_export_payload(exported),
                    )
                elif name == "import_markdown_vault":
                    imported = graph.import_markdown_vault(root_path=arguments["root_path"])
                    result = self._tool_result(
                        f"Imported Markdown vault from {imported.root_path}.",
                        self._markdown_vault_import_payload(imported),
                    )
                else:
                    raise ValidationFailure(f"Unknown tool: {name}")

                elapsed = time.perf_counter() - started
                self.metrics.increment(
                    "waggle_tool_requests_total",
                    tool=name,
                    status="success",
                    tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
                )
                self.metrics.observe("waggle_tool_latency_seconds", elapsed, tool=name)
                self._record_graph_size(graph)
                LOGGER.info("tool_call_completed")
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - started
                self.metrics.increment(
                    "waggle_tool_requests_total",
                    tool=name,
                    status="error",
                    tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
                )
                self.metrics.observe("waggle_tool_latency_seconds", elapsed, tool=name)
                if isinstance(exc, AuthenticationError):
                    self.metrics.increment("waggle_auth_failures_total", tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id))
                LOGGER.exception("tool_call_failed")
                return self._error_result(exc)

    def _tool_result(self, text: str, structured: dict[str, Any]) -> types.CallToolResult:
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], structuredContent=structured)

    def _error_result(self, exc: Exception) -> types.CallToolResult:
        if isinstance(exc, WaggleError):
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error [{exc.code}]: {exc}")],
                structuredContent={"error": str(exc), "error_type": type(exc).__name__, "error_code": exc.code, "status_code": exc.status_code},
                isError=True,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {exc}")],
            structuredContent={"error": str(exc), "error_type": type(exc).__name__},
            isError=True,
        )

    def _node_payload(self, node: Node) -> dict[str, Any]:
        return {
            "id": node.id,
            "tenant_id": node.tenant_id,
            "agent_id": node.agent_id,
            "project": node.project,
            "session_id": node.session_id,
            "label": node.label,
            "content": node.content,
            "node_type": node.node_type.value,
            "tags": node.tags,
            "source_prompt": node.source_prompt,
            "evidence_records": [
                {
                    "evidence_id": record.evidence_id,
                    "session_id": record.session_id,
                    "turn_index": record.turn_index,
                    "source_role": record.source_role,
                    "source_text": record.source_text,
                    "source_span_start": record.source_span_start,
                    "source_span_end": record.source_span_end,
                    "observed_at": record.observed_at.isoformat(),
                }
                for record in node.evidence_records
            ],
            "valid_from": node.valid_from.isoformat() if node.valid_from is not None else None,
            "valid_to": node.valid_to.isoformat() if node.valid_to is not None else None,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "access_count": node.access_count,
        }

    def _edge_payload(self, edge: Any) -> dict[str, Any]:
        return {
            "id": edge.id,
            "tenant_id": getattr(edge, "tenant_id", ""),
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relationship": edge.relationship,
            "weight": edge.weight,
            "metadata": edge.metadata,
            "created_at": edge.created_at.isoformat(),
        }

    def _subgraph_payload(self, result: SubgraphResult) -> dict[str, Any]:
        return {
            "query": result.query,
            "retrieval_mode": result.retrieval_mode,
            "total_nodes_in_graph": result.total_nodes_in_graph,
            "nodes": [self._node_payload(node) for node in result.nodes],
            "edges": [self._edge_payload(edge) for edge in result.edges],
            "replay_hits": [
                {
                    "score": hit.score,
                    "session_id": hit.session_id,
                    "turn_index": hit.turn_index,
                    "role": hit.role,
                    "transcript_text": hit.transcript_text,
                    "transcript_snippet": hit.transcript_snippet,
                    "observed_at": hit.observed_at.isoformat(),
                }
                for hit in result.replay_hits
            ],
            "fusion_hits": [
                {
                    "content": hit.content,
                    "score": hit.score,
                    "source_lane": hit.source_lane,
                    "graph_rank": hit.graph_rank,
                    "replay_rank": hit.replay_rank,
                    "fused_rank": hit.fused_rank,
                    "node_id": hit.node_id,
                    "node_type": hit.node_type,
                    "edges": hit.edges,
                    "session_id": hit.session_id,
                    "transcript_snippet": hit.transcript_snippet,
                    "turn_index": hit.turn_index,
                }
                for hit in result.fusion_hits
            ],
        }

    def _observation_payload(self, result: ObservationResult) -> dict[str, Any]:
        return {
            "stored_nodes": [self._node_payload(node) for node in result.stored_nodes],
            "created_count": result.created_count,
            "reused_count": result.reused_count,
            "conflicts": [
                {
                    "other_node_id": conflict.other_node_id,
                    "other_node_label": conflict.other_node_label,
                    "relationship": conflict.relationship,
                    "reason": conflict.reason,
                }
                for conflict in result.conflicts
            ],
        }

    def _node_history_payload(self, result: NodeHistoryResult) -> dict[str, Any]:
        return {
            "node": self._node_payload(result.node),
            "related_nodes": [self._node_payload(node) for node in result.related_nodes],
            "edges": [self._edge_payload(edge) for edge in result.edges],
        }

    def _timeline_payload(self, result: TimelineResult) -> dict[str, Any]:
        return {
            "scope": result.scope,
            "items": [
                {
                    "kind": item.kind,
                    "timestamp": item.timestamp.isoformat(),
                    "label": item.label,
                    "summary": item.summary,
                    "node_id": item.node_id,
                    "edge_id": item.edge_id,
                }
                for item in result.items
            ],
        }

    def _conflict_entry_payload(self, entry: ConflictEntry) -> dict[str, Any]:
        return {
            "edge": self._edge_payload(entry.edge),
            "source_node": self._node_payload(entry.source_node),
            "target_node": self._node_payload(entry.target_node),
            "resolved": entry.resolved,
            "resolution_note": entry.resolution_note,
            "resolved_at": entry.resolved_at.isoformat() if entry.resolved_at is not None else None,
        }

    def _conflict_list_payload(self, result: ConflictListResult) -> dict[str, Any]:
        return {
            "include_resolved": result.include_resolved,
            "conflicts": [self._conflict_entry_payload(entry) for entry in result.conflicts],
        }

    def _context_scope_payload(self, result: ContextScopeResult) -> dict[str, Any]:
        return {
            "agent_ids": result.agent_ids,
            "projects": result.projects,
            "session_ids": result.session_ids,
        }

    def _graph_diff_payload(self, result: GraphDiffResult) -> dict[str, Any]:
        return {
            "since": result.since,
            "generated_at": result.generated_at.isoformat(),
            "added_nodes": [self._node_payload(node) for node in result.added_nodes],
            "updated_nodes": [self._node_payload(node) for node in result.updated_nodes],
            "created_edges": [self._edge_payload(edge) for edge in result.created_edges],
            "contradiction_edges": [self._edge_payload(edge) for edge in result.contradiction_edges],
        }

    def _prime_context_payload(self, result: PrimeContextResult) -> dict[str, Any]:
        return {
            "project": result.project,
            "summary": result.summary,
            "total_nodes_in_graph": result.total_nodes_in_graph,
            "nodes": [self._node_payload(node) for node in result.nodes],
            "edges": [self._edge_payload(edge) for edge in result.edges],
        }

    def _context_bundle_payload(self, result: ContextBundleExportResult) -> dict[str, Any]:
        return {
            "tenant_id": result.tenant_id,
            "project": result.project,
            "mode": result.mode,
            "retrieval_mode": result.retrieval_mode,
            "query": result.query,
            "summary": result.summary,
            "markdown_path": result.markdown_path,
            "json_path": result.json_path,
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "render_hints": {
                "token_estimate": result.bundle.render_hints.token_estimate,
                "recommended_paste_order": result.bundle.render_hints.recommended_paste_order,
                "truncation_flags": result.bundle.render_hints.truncation_flags,
                "chunk_count": result.bundle.render_hints.chunk_count,
            },
        }

    def _topic_payload(self, result: TopicResult) -> dict[str, Any]:
        return {
            "total_clusters": result.total_clusters,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "label": cluster.label,
                    "node_count": cluster.node_count,
                    "top_tags": cluster.top_tags,
                    "nodes": [self._node_payload(node) for node in cluster.nodes],
                }
                for cluster in result.clusters
            ],
        }

    def _stats_payload(self, stats: GraphStats) -> dict[str, Any]:
        return {
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "node_type_breakdown": stats.node_type_breakdown,
            "most_connected_nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "node_type": node.node_type.value,
                    "connection_count": node.connection_count,
                }
                for node in stats.most_connected_nodes
            ],
            "most_recent_nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "node_type": node.node_type.value,
                    "updated_at": node.updated_at.isoformat(),
                }
                for node in stats.most_recent_nodes
            ],
        }

    def _markdown_vault_export_payload(self, result: MarkdownVaultExportResult) -> dict[str, Any]:
        return {
            "root_path": result.root_path,
            "tenant_id": result.tenant_id,
            "project": result.project,
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "files_written": result.files_written,
        }

    def _markdown_vault_import_payload(self, result: MarkdownVaultImportResult) -> dict[str, Any]:
        return {
            "root_path": result.root_path,
            "tenant_id": result.tenant_id,
            "nodes_created": result.nodes_created,
            "nodes_updated": result.nodes_updated,
            "edges_created": result.edges_created,
            "edges_deleted": result.edges_deleted,
            "stub_nodes_created": result.stub_nodes_created,
            "conflicts": result.conflicts,
        }

    def _validate_tool_payload(self, name: str, arguments: dict[str, Any]) -> None:
        limit = self.config.max_payload_bytes
        if name == "store_node":
            self._assert_payload_size(arguments.get("label", ""), limit, "store_node.label")
            self._assert_payload_size(arguments.get("content", ""), limit, "store_node.content")
            self._assert_payload_size(arguments.get("source_prompt", ""), limit, "store_node.source_prompt")
            self._assert_payload_size(arguments.get("agent_id", ""), limit, "store_node.agent_id")
            self._assert_payload_size(arguments.get("project", ""), limit, "store_node.project")
            self._assert_payload_size(arguments.get("session_id", ""), limit, "store_node.session_id")
            return
        if name == "decompose_and_store":
            self._assert_payload_size(arguments.get("content", ""), limit, "decompose_and_store.content")
            self._assert_payload_size(arguments.get("context", ""), limit, "decompose_and_store.context")
            return
        if name == "observe_conversation":
            self._assert_payload_size(arguments.get("user_message", ""), limit, "observe_conversation.user_message")
            self._assert_payload_size(arguments.get("assistant_response", ""), limit, "observe_conversation.assistant_response")
            self._assert_payload_size(arguments.get("agent_id", ""), limit, "observe_conversation.agent_id")
            self._assert_payload_size(arguments.get("project", ""), limit, "observe_conversation.project")
            self._assert_payload_size(arguments.get("session_id", ""), limit, "observe_conversation.session_id")
            return
        if name == "query_graph":
            self._assert_payload_size(arguments.get("query", ""), limit, "query_graph.query")
            self._assert_payload_size(arguments.get("agent_id", ""), limit, "query_graph.agent_id")
            self._assert_payload_size(arguments.get("project", ""), limit, "query_graph.project")
            self._assert_payload_size(arguments.get("session_id", ""), limit, "query_graph.session_id")
            return
        if name == "export_context_bundle":
            self._assert_payload_size(arguments.get("query", ""), limit, "export_context_bundle.query")
            self._assert_payload_size(arguments.get("project", ""), limit, "export_context_bundle.project")
            self._assert_payload_size(arguments.get("agent_id", ""), limit, "export_context_bundle.agent_id")
            self._assert_payload_size(arguments.get("session_id", ""), limit, "export_context_bundle.session_id")
            self._assert_payload_size(arguments.get("output_path", ""), limit, "export_context_bundle.output_path")
            return
        if name == "timeline":
            self._assert_payload_size(arguments.get("query", ""), limit, "timeline.query")
            self._assert_payload_size(arguments.get("node_id", ""), limit, "timeline.node_id")
            return
        if name == "resolve_conflict":
            self._assert_payload_size(arguments.get("edge_id", ""), limit, "resolve_conflict.edge_id")
            self._assert_payload_size(arguments.get("resolution_note", ""), limit, "resolve_conflict.resolution_note")

    @staticmethod
    def _assert_payload_size(value: Any, limit: int, field_name: str) -> None:
        if value is None:
            return
        size = len(str(value).encode("utf-8"))
        if size > limit:
            raise PayloadTooLargeError(f"{field_name} exceeds the configured payload limit.")

    def _record_graph_size(self, graph: Any) -> None:
        stats = graph.get_stats()
        tenant_id = getattr(graph, "tenant_id", self.config.default_tenant_id)
        self.metrics.set_gauge("waggle_graph_nodes", stats.total_nodes, tenant_id=tenant_id)
        self.metrics.set_gauge("waggle_graph_edges", stats.total_edges, tenant_id=tenant_id)


class MCPHttpApp:
    def __init__(self, app_server: WaggleServer, config: AppConfig) -> None:
        self.app_server = app_server
        self.config = config
        self.metrics = app_server.metrics
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_rpm,
            max_concurrent_requests=config.max_concurrent_requests,
            write_requests_per_minute=config.write_rate_limit_rpm,
        )
        self.transport: StreamableHTTPServerTransport | None = None
        self.ready = False
        self.draining = False

    @asynccontextmanager
    async def lifespan(self, app: Starlette):
        self.transport = StreamableHTTPServerTransport(mcp_session_id=None, is_json_response_enabled=False)
        async with self.transport.connect() as (read_stream, write_stream):
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    self.app_server.server.run,
                    read_stream,
                    write_stream,
                    self.app_server.initialization_options(),
                    False,
                    True,
                )
                self.app_server.validate_startup()
                self.ready = True
                self.metrics.set_gauge("waggle_ready", 1)
                app.state.http_service = self
                try:
                    yield
                finally:
                    self.draining = True
                    self.ready = False
                    self.metrics.set_gauge("waggle_ready", 0)
                    tg.cancel_scope.cancel()

    async def mcp_asgi(self, scope: Any, receive: Any, send: Any) -> None:
        started = time.perf_counter()
        method = scope["method"]
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        request_id = headers.get("x-request-id", str(uuid.uuid4()))
        status_holder = {"status": 500}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
            await send(message)

        try:
            if self.draining:
                raise ServiceUnavailableError("Server is draining.")
            body = b""
            receive_callable = receive
            if method == "POST":
                request = Request(scope, receive)
                body = await request.body()
                if len(body) > self.config.max_payload_bytes:
                    raise PayloadTooLargeError()
                receive_callable = self._replay_receive(body)
                headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}

            raw_api_key = headers.get("x-api-key", "")
            if not raw_api_key:
                raise AuthenticationError("Missing X-API-Key header.")
            principal = self.app_server._root_graph.authenticate_api_key(raw_api_key)
            scope.setdefault("state", {})
            scope["state"]["tenant_id"] = principal.tenant_id
            scope["state"]["api_key_id"] = principal.api_key_id
            scope["state"]["request_id"] = request_id

            tool_name = self._extract_tool_name(body)
            await self.rate_limiter.check_rate(principal.api_key_id, is_write=tool_name in WRITE_HEAVY_TOOLS)
            async with self.rate_limiter.concurrency_slot(principal.api_key_id):
                with runtime_context(
                    request_id=request_id,
                    tenant_id=principal.tenant_id,
                    transport="http",
                    backend=self.config.backend,
                    api_key_id=principal.api_key_id,
                    tool_name=tool_name,
                ):
                    with anyio.fail_after(self.config.request_timeout_seconds):
                        assert self.transport is not None
                        await self.transport.handle_request(scope, receive_callable, send_wrapper)
        except WaggleError as exc:
            LOGGER.warning("http_request_failed", extra={"error_code": exc.code, "status_code": exc.status_code})
            if isinstance(exc, AuthenticationError):
                self.metrics.increment("waggle_auth_failures_total")
            if exc.code == "rate_limited":
                self.metrics.increment("waggle_rate_limit_rejections_total")
            await JSONResponse({"error": exc.code, "message": str(exc)}, status_code=exc.status_code)(scope, receive, send)
            status_holder["status"] = exc.status_code
        finally:
            elapsed = time.perf_counter() - started
            self.metrics.increment(
                "waggle_http_requests_total",
                path="/mcp",
                method=method,
                status=str(status_holder["status"]),
            )
            self.metrics.observe("waggle_http_request_latency_seconds", elapsed, path="/mcp", method=method)

    @staticmethod
    def _extract_tool_name(body: bytes) -> str:
        if not body:
            return ""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return ""
        params = payload.get("params", {})
        if isinstance(params, dict):
            return str(params.get("name", ""))
        return ""

    @staticmethod
    def _replay_receive(body: bytes):
        delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        return receive


def create_http_application(app_server: WaggleServer, config: AppConfig) -> Starlette:
    service = MCPHttpApp(app_server, config)

    async def live(_: Request) -> Response:
        return JSONResponse({"status": "live"})

    async def ready(request: Request) -> Response:
        http_service: MCPHttpApp = request.app.state.http_service
        if not http_service.ready or http_service.draining:
            return JSONResponse({"status": "not-ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    async def metrics_endpoint(request: Request) -> Response:
        return PlainTextResponse(request.app.state.http_service.metrics.render_prometheus())

    app = Starlette(
        routes=[
            Route("/health/live", live),
            Route("/health/ready", ready),
            Route("/metrics", metrics_endpoint),
            Mount("/mcp", app=service.mcp_asgi),
        ],
        lifespan=service.lifespan,
    )
    return app


_APP: WaggleServer | None = None


def _default_graph(config: AppConfig | None = None) -> Any:
    try:
        return _build_backend(config or AppConfig.from_env())
    except ValidationFailure as exc:
        raise RuntimeError(str(exc)) from exc


def get_app(config: AppConfig | None = None) -> WaggleServer:
    global _APP
    if _APP is None:
        _APP = WaggleServer(config=config or AppConfig.from_env())
    return _APP


async def run_stdio(config: AppConfig) -> None:
    app = get_app(config)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.server.run(read_stream, write_stream, app.initialization_options())


def run_http(config: AppConfig) -> None:
    app_server = get_app(config)
    http_app = create_http_application(app_server, config)
    uvicorn.run(http_app, host=config.http_host, port=config.http_port, log_level=config.log_level.lower())


_FEATURES_GUIDE = """\
waggle-mcp feature guide
========================

Core graph workflow
-------------------
1. Ingest memory
   - observe_conversation : extract structured facts/decisions from a finished turn
   - decompose_and_store  : split long content into linked atomic nodes
   - store_node           : save one standalone node directly
   - store_edge           : connect nodes explicitly when relationship is known

2. Retrieve context
   - query_graph          : semantic retrieval over nodes plus graph/replay/fusion context
   - get_related          : walk outward from one node to inspect connected context
   - get_node_history     : inspect one node's evidence, validity, and linked context
   - timeline             : see recent graph events chronologically
   - prime_context        : build a compact briefing for a new model/session

3. Resolve conflicts
   - list_conflicts       : list contradiction/update edges
   - resolve_conflict     : mark a conflict as resolved without deleting history

4. Export / handoff
   - export_context_bundle : create markdown/json context packages for another model
   - export_markdown_vault : export one-file-per-node markdown for manual editing
   - import_markdown_vault : re-import edited markdown vault files
   - export_graph_backup   : portable json backup
   - import_graph_backup   : restore backup into the active backend
   - export_graph_html     : interactive graph visualization

5. Inspect the graph
   - get_stats            : node/edge counts and high-level graph stats
   - list_context_scopes  : available project / agent / session scopes
   - get_topics           : topic clusters via community detection
   - graph_diff           : recently changed nodes and edges

Important behavior
------------------
- store_node alone does not create edges.
- Edges come from:
  - explicit store_edge calls
  - observe_conversation extraction
  - decompose_and_store inferred structure
  - automatic contradiction/update detection in some cases
- The graph-aware tools are what bring connected context back to the model:
  - query_graph
  - get_related
  - get_node_history
  - prime_context
  - export_context_bundle

Common workflows
----------------
- Quick setup:
  waggle-mcp init

- Start the MCP server:
  waggle-mcp serve

- Export a handoff bundle:
  waggle-mcp export-context-bundle --mode query --query "why did we choose PostgreSQL?"

- Edit memory as markdown:
  waggle-mcp export-markdown-vault --root-path ./vault
  waggle-mcp import-markdown-vault --root-path ./vault
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="waggle-mcp",
        description=(
            "Persistent graph memory for MCP-compatible AI clients. "
            "Use 'waggle-mcp features' for a detailed guide to ingestion, graph retrieval, "
            "conflict handling, and export workflows."
        ),
        epilog="Examples: 'waggle-mcp init', 'waggle-mcp serve', 'waggle-mcp features'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the MCP server using the configured stdio or HTTP transport.")

    create_tenant = subparsers.add_parser("create-tenant", help="Create or update a tenant record in the active backend.")
    create_tenant.add_argument("--tenant-id", required=True)
    create_tenant.add_argument("--name", default="")

    create_api_key = subparsers.add_parser("create-api-key", help="Issue an API key for a tenant.")
    create_api_key.add_argument("--tenant-id", required=True)
    create_api_key.add_argument("--name", default="")

    list_api_keys = subparsers.add_parser("list-api-keys", help="List API keys for a tenant.")
    list_api_keys.add_argument("--tenant-id", required=True)

    revoke_api_key = subparsers.add_parser("revoke-api-key", help="Revoke an API key.")
    revoke_api_key.add_argument("--api-key-id", required=True)

    migrate_sqlite = subparsers.add_parser("migrate-sqlite", help="Export a SQLite graph and import it into the configured Neo4j backend.")
    migrate_sqlite.add_argument("--db-path", required=True)
    migrate_sqlite.add_argument("--tenant-id", required=True)

    export_context_bundle = subparsers.add_parser(
        "export-context-bundle",
        help="Export a markdown/json context package for another model or conversation.",
        description=(
            "Build a portable context bundle from the graph. "
            "Use mode=query for question-focused handoff, mode=prime for a fresh-session brief, "
            "and mode=graph for a broader graph export."
        ),
    )
    export_context_bundle.add_argument("--mode", choices=["prime", "query", "graph"], default="prime")
    export_context_bundle.add_argument("--query", default="")
    export_context_bundle.add_argument("--project", default="")
    export_context_bundle.add_argument("--agent-id", default="")
    export_context_bundle.add_argument("--session-id", default="")
    export_context_bundle.add_argument("--max-nodes", type=int, default=25)
    export_context_bundle.add_argument("--max-depth", type=int, default=2)
    export_context_bundle.add_argument("--retrieval-mode", choices=["graph", "replay", "fusion"], default="graph")
    export_context_bundle.add_argument("--format", choices=["markdown", "json", "both"], default="both")
    export_context_bundle.add_argument("--output-path", default=None)
    export_context_bundle.add_argument("--include-edges", action=argparse.BooleanOptionalAction, default=True)
    export_context_bundle.add_argument("--include-timestamps", action=argparse.BooleanOptionalAction, default=True)
    export_context_bundle.add_argument("--include-source-prompt", action=argparse.BooleanOptionalAction, default=False)
    export_context_bundle.add_argument("--audience", choices=["llm", "human"], default="llm")

    export_markdown_vault = subparsers.add_parser(
        "export-markdown-vault",
        help="Export graph memory as an Obsidian-style markdown vault.",
    )
    export_markdown_vault.add_argument("--root-path", required=True)
    export_markdown_vault.add_argument("--project", default="")
    export_markdown_vault.add_argument("--agent-id", default="")
    export_markdown_vault.add_argument("--session-id", default="")

    import_markdown_vault = subparsers.add_parser(
        "import-markdown-vault",
        help="Import an edited markdown vault back into the graph.",
    )
    import_markdown_vault.add_argument("--root-path", required=True)

    subparsers.add_parser("init", help="Interactive setup wizard — configure an MCP client to use waggle-mcp.")
    subparsers.add_parser(
        "features",
        help="Explain the main tools, graph workflows, and how connected context reaches the model.",
        description="Print a detailed guide to the waggle-mcp feature surface.",
    )
    return parser


def _run_admin_command(config: AppConfig, args: argparse.Namespace) -> int:
    backend = _build_backend(config)
    if args.command == "create-tenant":
        tenant = backend.ensure_tenant(args.tenant_id, args.name)
        print(json.dumps(tenant.model_dump(), indent=2))
        return 0
    if args.command == "create-api-key":
        created = backend.create_api_key(args.tenant_id, args.name)
        print(
            json.dumps(
                {
                    "api_key_id": created.record.api_key_id,
                    "tenant_id": created.record.tenant_id,
                    "name": created.record.name,
                    "status": created.record.status,
                    "raw_api_key": created.raw_api_key,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "list-api-keys":
        print(json.dumps([record.model_dump(mode="json") for record in backend.list_api_keys(args.tenant_id)], indent=2))
        return 0
    if args.command == "revoke-api-key":
        backend.revoke_api_key(args.api_key_id)
        print(json.dumps({"revoked": args.api_key_id}))
        return 0
    if args.command == "migrate-sqlite":
        if config.backend != "neo4j":
            raise ValidationFailure("migrate-sqlite requires WAGGLE_BACKEND=neo4j for the target environment.")
        source = MemoryGraph(args.db_path, EmbeddingModel(config.model_name), tenant_id=args.tenant_id, export_dir=config.export_dir)
        target = backend.for_tenant(args.tenant_id)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            temp_path = Path(handle.name)
        backup = source.export_graph_backup(output_path=temp_path)
        imported = target.import_graph_backup(input_path=temp_path)
        print(
            json.dumps(
                {
                    "backup": backup.model_dump(),
                    "import": imported.model_dump(),
                },
                indent=2,
            )
        )
        temp_path.unlink(missing_ok=True)
        return 0
    if args.command == "export-context-bundle":
        exported = backend.export_context_bundle(
            mode=args.mode,
            query=args.query,
            project=args.project,
            agent_id=getattr(args, "agent_id", ""),
            session_id=getattr(args, "session_id", ""),
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            retrieval_mode=getattr(args, "retrieval_mode", "graph"),
            format=args.format,
            output_path=args.output_path,
            include_edges=args.include_edges,
            include_timestamps=args.include_timestamps,
            include_source_prompt=args.include_source_prompt,
            audience=args.audience,
        )
        print(json.dumps(exported.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "export-markdown-vault":
        exported = backend.export_markdown_vault(
            root_path=args.root_path,
            project=getattr(args, "project", ""),
            agent_id=getattr(args, "agent_id", ""),
            session_id=getattr(args, "session_id", ""),
        )
        print(json.dumps(exported.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "import-markdown-vault":
        imported = backend.import_markdown_vault(root_path=args.root_path)
        print(json.dumps(imported.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "features":
        print(_FEATURES_GUIDE)
        return 0
    raise ValidationFailure(f"Unknown command: {args.command}")


# ---------------------------------------------------------------------------
# init wizard
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_NO_COLOR = not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Apply ANSI colour code, or return plain text when not a tty."""
    return text if _NO_COLOR else f"{code}{text}{_RESET}"


def _prompt_choice(question: str, choices: list[str]) -> str:
    """Render an arrow-key style menu (keyboard fallback: number entry)."""
    print(f"\n{_c(_BOLD, question)}")
    for i, choice in enumerate(choices, 1):
        print(f"  {_c(_CYAN, str(i) + '.')} {choice}")
    while True:
        raw = input(f"  Enter number [1-{len(choices)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            selected = choices[int(raw) - 1]
            print(f"  {_c(_GREEN, '>')} {selected}")
            return selected
        print(f"  {_c(_RED, 'Invalid choice, try again.')}")


def _prompt_path(question: str, default: str) -> str:
    """Prompt for a file path, showing the default."""
    print(f"\n{_c(_BOLD, question)}")
    raw = input(f"  [{default}]: ").strip()
    result = raw or default
    print(f"  {_c(_GREEN, '>')} {result}")
    return result


def _ok(msg: str) -> None:
    print(f"  {_c(_GREEN, chr(0x2705))} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_c(_RED, chr(0x274C))} {msg}")


def _python_exe() -> str:
    """Return the Python executable used to run this process."""
    return sys.executable


# ── client config writers ────────────────────────────────────────────────────

def _write_claude_desktop(db_path: str, python_exe: str) -> Path:
    """Write ~/.config/claude/claude_desktop_config.json (macOS/Linux)."""
    if sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "Claude"
    else:
        config_dir = Path.home() / ".config" / "claude"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "claude_desktop_config.json"

    existing: dict = {}
    if config_file.exists():
        try:
            existing = json.loads(config_file.read_text())
        except json.JSONDecodeError:
            pass

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["waggle"] = {
        "command": python_exe,
        "args": ["-m", "waggle.server"],
        "env": {
            "WAGGLE_TRANSPORT": "stdio",
            "WAGGLE_BACKEND": "sqlite",
            "WAGGLE_DB_PATH": db_path,
            "WAGGLE_DEFAULT_TENANT_ID": "local-default",
            "WAGGLE_MODEL": "all-MiniLM-L6-v2",
        },
    }
    config_file.write_text(json.dumps(existing, indent=2))
    return config_file


def _write_cursor(db_path: str, python_exe: str) -> Path:
    """Write ~/.cursor/mcp.json."""
    config_dir = Path.home() / ".cursor"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "mcp.json"

    existing: dict = {}
    if config_file.exists():
        try:
            existing = json.loads(config_file.read_text())
        except json.JSONDecodeError:
            pass

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["waggle"] = {
        "command": python_exe,
        "args": ["-m", "waggle.server"],
        "env": {
            "WAGGLE_TRANSPORT": "stdio",
            "WAGGLE_BACKEND": "sqlite",
            "WAGGLE_DB_PATH": db_path,
            "WAGGLE_DEFAULT_TENANT_ID": "local-default",
            "WAGGLE_MODEL": "all-MiniLM-L6-v2",
        },
    }
    config_file.write_text(json.dumps(existing, indent=2))
    return config_file


def _write_codex(db_path: str, python_exe: str) -> Path:
    """Write ~/codex_mcp.toml (printable TOML snippet — Codex uses a TOML file)."""
    config_file = Path.home() / "codex_mcp.toml"
    toml_block = (
        '[mcp_servers.waggle]\n'
        f'command = "{python_exe}"\n'
        'args = ["-m", "waggle.server"]\n'
        'env = {\n'
        '  WAGGLE_TRANSPORT = "stdio",\n'
        '  WAGGLE_BACKEND = "sqlite",\n'
        f'  WAGGLE_DB_PATH = "{db_path}",\n'
        '  WAGGLE_DEFAULT_TENANT_ID = "local-default",\n'
        '  WAGGLE_MODEL = "all-MiniLM-L6-v2"\n'
        '}\n'
    )
    config_file.write_text(toml_block)
    return config_file


def _write_other(db_path: str, python_exe: str) -> Path:
    """Write a generic JSON snippet to ~/waggle-mcp-config.json."""
    config_file = Path.home() / "waggle-mcp-config.json"
    snippet = {
        "command": python_exe,
        "args": ["-m", "waggle.server"],
        "env": {
            "WAGGLE_TRANSPORT": "stdio",
            "WAGGLE_BACKEND": "sqlite",
            "WAGGLE_DB_PATH": db_path,
            "WAGGLE_DEFAULT_TENANT_ID": "local-default",
            "WAGGLE_MODEL": "all-MiniLM-L6-v2",
        },
    }
    config_file.write_text(json.dumps(snippet, indent=2))
    return config_file


_CLIENT_WRITERS = {
    "Claude Desktop": _write_claude_desktop,
    "Cursor": _write_cursor,
    "Codex": _write_codex,
    "Other": _write_other,
}

_RESTART_HINTS = {
    "Claude Desktop": "Restart Claude Desktop to activate.",
    "Cursor": "Reload the Cursor window (Cmd/Ctrl+Shift+P → 'Reload Window') to activate.",
    "Codex": "Copy the TOML block into your Codex config file, then restart Codex.",
    "Other": "Add the JSON config to your MCP client's server list, then restart it.",
}


def _run_init() -> int:
    """Interactive setup wizard for waggle-mcp."""
    print()
    print(_c(_BOLD, "waggle-mcp setup wizard"))
    print(_c(_CYAN, "─" * 40))

    clients = list(_CLIENT_WRITERS.keys())
    client = _prompt_choice("Which MCP client are you using?", clients)

    default_db = str(Path.home() / ".waggle" / "memory.db")
    db_path_raw = _prompt_path("Where should the database be stored?", default_db)
    db_path = str(Path(db_path_raw).expanduser().resolve())

    python_exe = _python_exe()

    print()

    # Write client config
    writer = _CLIENT_WRITERS[client]
    try:
        config_file = writer(db_path, python_exe)
        _ok(f"Config written to {config_file}")
    except OSError as exc:
        _fail(f"Could not write config: {exc}")
        return 1

    # Create database directory
    db_dir = Path(db_path).parent
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        _ok(f"Database directory ready at {db_dir}")
    except OSError as exc:
        _fail(f"Could not create database directory: {exc}")
        return 1

    # Restart hint
    print(f"  {_c(_CYAN, chr(0x27A1))}  {_RESTART_HINTS[client]}")
    print()
    return 0


def main() -> None:
    _assert_runtime_feature_parity()
    parser = _build_parser()
    args = parser.parse_args()
    command = args.command or "serve"

    # Commands that should work before full backend/app initialization.
    if command == "init":
        sys.exit(_run_init())
    if command == "features":
        print(_FEATURES_GUIDE)
        return

    config = AppConfig.from_env()
    log_stream = sys.stderr if config.transport == "stdio" else sys.stdout
    configure_logging(config.log_level, stream=log_stream)
    LOGGER.info("waggle_startup")
    if command != "serve":
        _run_admin_command(config, args)
        return
    if config.transport == "http":
        run_http(config)
        return
    if config.backend == "sqlite":
        Path(config.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_stdio(config))


if __name__ == "__main__":
    main()
