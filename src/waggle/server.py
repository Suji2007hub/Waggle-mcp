from __future__ import annotations

import argparse
import asyncio
import base64
import getpass
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from copy import deepcopy
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
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from waggle import __version__
from waggle.abhi import abhi_to_snapshot, build_abhi_document, execute_abhi_query, load_abhi_document, validate_abhi_document
from waggle.config import AppConfig, STARTUP_MODE_FAST, STARTUP_MODE_STRICT
from waggle.embeddings import EmbeddingModel, EMBEDDING_FREE_TOOLS, STATUS_READY, STATUS_DISABLED
from waggle.errors import (
    AuthenticationError,
    WaggleError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    ValidationFailure,
)
from waggle.graph import MemoryGraph
from waggle.graph_ui import render_graph_editor_html
from waggle.logging_utils import configure_logging
from waggle.metrics import MetricsRegistry
from waggle.oolong_benchmark import evaluate_oolong
from waggle.rlm import (
    DEFAULT_WAGGLE_RLM_SYSTEM_PROMPT,
    build_gemini_backend_kwargs,
    build_groq_openai_backend_kwargs,
    build_subprocess_response_fn,
    run_gemini_one_shot,
    run_groq_one_shot,
)
from waggle.models import (
    ConflictEntry,
    ConflictListResult,
    ContextBundleExportResult,
    ContextScopeResult,
    ContextWindow,
    ContextWindowEdge,
    DrivePullResult,
    DrivePushResult,
    DriveShareResult,
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
    TranscriptIngestionInput,
    TranscriptMessage,
)
from waggle.rate_limit import RateLimiter
from waggle.runtime_context import runtime_context
from waggle.serializer import (
    serialize_abhi_chunk_load,
    serialize_abhi_inspect,
    serialize_abhi_diff,
    serialize_abhi_merge,
    serialize_abhi_query,
    serialize_abhi_validation,
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
_DRIVE_SYNC_IMPORT_ERROR: Exception | None = None

try:
    from waggle.drive_sync import (
        download_drive_file,
        ensure_drive_credentials,
        merge_downloaded_abhi,
        push_file_to_drive,
        resolve_drive_file_id,
        share_drive_file,
    )
except Exception as exc:  # pragma: no cover - depends on optional Google libs
    download_drive_file = None
    ensure_drive_credentials = None
    merge_downloaded_abhi = None
    push_file_to_drive = None
    resolve_drive_file_id = None
    share_drive_file = None
    _DRIVE_SYNC_IMPORT_ERROR = exc

WRITE_HEAVY_TOOLS = {
    "store_node",
    "store_edge",
    "decompose_and_store",
    "observe_conversation",
    "import_graph_backup",
    "import_abhi",
    "merge_abhi",
    "load_abhi_chunks",
    "query_abhi",
    "import_markdown_vault",
}
REQUIRED_RUNTIME_METHODS = (
    "export_context_bundle",
    "export_markdown_vault",
    "export_abhi",
    "diff_abhi",
    "import_abhi",
    "merge_abhi",
    "load_abhi_chunks",
    "query_abhi",
    "validate_abhi",
    "inspect_abhi",
    "list_context_scopes",
    "get_node_history",
    "import_markdown_vault",
    "timeline",
    "list_conflicts",
    "resolve_conflict",
)

_EXPORT_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("JWT token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9._-]{8,}\.[A-Za-z0-9._-]{8,}\b")),
    ("Password assignment", re.compile(r"(?i)\b(password|passwd|pwd)\b\s*[:=]\s*['\"]?\S+")),
    ("Secret/token assignment", re.compile(r"(?i)\b(api[_ -]?key|secret[_ -]?key|access[_ -]?token)\b\s*[:=]\s*['\"]?\S+")),
)


def _resolve_passphrase(args: argparse.Namespace) -> str:
    env_name = str(getattr(args, "passphrase_env", "") or "").strip()
    if env_name:
        return os.environ.get(env_name, "").strip()
    if bool(getattr(args, "encrypt", False)):
        return getpass.getpass("ABHI passphrase: ").strip()
    return ""


def _resolve_drive_token_path(args: argparse.Namespace, config: AppConfig) -> Path:
    raw = str(getattr(args, "token_path", "") or "").strip()
    if raw:
        return Path(raw).expanduser()
    export_root = Path(config.export_dir).expanduser() if config.export_dir else Path.home() / ".waggle"
    return export_root / "google-drive-token.json"


def _require_drive_sync() -> None:
    if _DRIVE_SYNC_IMPORT_ERROR is None:
        return
    raise ValidationFailure(
        "Google Drive sync requires optional Google API dependencies in the active environment. "
        f"Original import error: {_DRIVE_SYNC_IMPORT_ERROR}"
    )


def _scan_export_transcripts_for_secrets(
    backend: MemoryGraph,
    *,
    project: str = "",
    agent_id: str = "",
    session_id: str = "",
    scope: str = "all",
    since_date: str = "",
    max_findings: int = 10,
) -> list[dict[str, Any]]:
    snapshot = backend.get_graph_snapshot()
    document = build_abhi_document(
        snapshot,
        scope=scope,
        project=project,
        agent_id=agent_id,
        session_id=session_id,
        since_date=since_date,
        include_embeddings=False,
        encrypted=False,
    )
    findings: list[dict[str, Any]] = []
    for row in document.get("transcripts", []):
        text = str(row.get("transcript_text", ""))
        if not text.strip():
            continue
        for label, pattern in _EXPORT_SECRET_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            secret = match.group(0)
            preview = text.replace(secret, "[REDACTED]")
            findings.append(
                {
                    "pattern": label,
                    "transcript_id": str(row.get("id", "")),
                    "session_id": str(row.get("session_id", "")),
                    "turn_index": int(row.get("turn_index", 0) or 0),
                    "role": str(row.get("role", "")),
                    "preview": preview[:180],
                }
            )
            break
        if len(findings) >= max_findings:
            break
    return findings


def _assert_export_safe(
    backend: MemoryGraph,
    *,
    force: bool,
    project: str = "",
    agent_id: str = "",
    session_id: str = "",
    scope: str = "all",
    since_date: str = "",
) -> None:
    findings = _scan_export_transcripts_for_secrets(
        backend,
        project=project,
        agent_id=agent_id,
        session_id=session_id,
        scope=scope,
        since_date=since_date,
    )
    if findings and not force:
        summary = "; ".join(
            f"{item['pattern']} in {item['role']} turn {item['turn_index']} of session {item['session_id'] or 'default'}"
            for item in findings[:3]
        )
        raise ValidationFailure(
            "Export refused because transcript_records appear to contain secrets. "
            f"Run again with --force only after redacting or confirming the export scope is safe. Findings: {summary}."
        )

MEMORY_AUTOMATION_POLICY = """Waggle automatic memory policy

The user should not manually manage memory. The assistant/runtime is responsible for using Waggle tools.
Waggle should remember relevant conversational context automatically. If memory looks empty, the likely issue is that this session is not loading the automatic memory policy or is bypassing the orchestrated runtime hooks.

Before answering:
- Use prime_context at the start of a new session when project, agent, or session scope is known.
- Use query_graph before answering questions that may depend on prior decisions, preferences, constraints, project state, or earlier conversation context.
- Keep retrieval narrow: start with max_nodes 8-12, max_depth 1-2, retrieval_mode hybrid. Use graph only when transcript evidence is not needed.

After answering:
- Use observe_conversation after completed turns that contain durable information: decisions, preferences, constraints, requirements, user corrections, project facts, or meaningful task outcomes.
- Do not call store_node for normal conversation memory unless the user explicitly gives one atomic fact and no inference is needed.
- Skip memory writes for acknowledgements, greetings, short chatter, or failed/aborted work.

Scoping:
- Always pass stable project, agent_id, and session_id when available.
- Prefer scoped memory over global memory in shared workspaces.
"""

AUTOMATIC_MEMORY_RULE_TEXT = """Use Waggle automatically for conversational memory.

At the start of a new session, if project, agent, or session scope is known, call prime_context.

Before answering questions that may depend on prior decisions, preferences, constraints, project state, or earlier conversation context, call query_graph with the narrowest relevant scope.

After completed turns that contain durable information such as decisions, preferences, constraints, requirements, user corrections, project facts, or meaningful task outcomes, call observe_conversation automatically.

Waggle should remember relevant context automatically. If memory appears empty, the session is likely missing the automatic memory policy or the runtime hooks that call build_context before answers and on_assistant_turn after answers.

Do not ask the user to trigger Waggle manually. Use it in the background when relevant.
"""

_AGENTS_MEMORY_BLOCK_HEADER = "<!-- waggle:auto-memory:start -->"
_AGENTS_MEMORY_BLOCK_FOOTER = "<!-- waggle:auto-memory:end -->"
_AGENTS_MEMORY_BLOCK = (
    f"{_AGENTS_MEMORY_BLOCK_HEADER}\n"
    "## Waggle Automatic Memory\n\n"
    f"{AUTOMATIC_MEMORY_RULE_TEXT.rstrip()}\n"
    f"{_AGENTS_MEMORY_BLOCK_FOOTER}\n"
)


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
    # Disable ML entirely in fast/inspection mode.
    if config.is_fast_mode:
        embedding_model.disable_warmup()
    if config.backend == "sqlite":
        return MemoryGraph(
            config.db_path,
            embedding_model,
            tenant_id=config.default_tenant_id,
            recency_half_life_days=config.recency_half_life_days,
            tiered_retrieval=config.tiered_retrieval,
            tiered_retrieval_top_k_windows=config.tiered_retrieval_top_k_windows,
            hybrid_retrieval_config=config.hybrid_retrieval_config(),
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
                name="aggregate_graph",
                description=(
                    "Retrieve a broad set of nodes bypassing standard semantic limits, optimized for "
                    "global aggregation and map-reduce tasks. Supports filtering by node_type and tags."
                ),
                inputSchema=_object_input_schema(
                    {
                        "query": {"type": "string", "description": "Optional natural-language search query to rank the broad retrieval."},
                        "node_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of node types to filter by (e.g., 'fact', 'entity')."
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of tags to require."
                        },
                        "max_nodes": {"type": "integer", "description": "Maximum number of nodes to return (default 100, up to 1000)."},
                        "max_depth": {"type": "integer", "description": "Relationship traversal depth around matching nodes."},
                        **_scope_properties(),
                    },
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
                        "expand_depth": {
                            "type": "integer",
                            "default": 0,
                            "minimum": 0,
                            "description": "Optional support expansion depth. At 1, graph mode may return up to twice max_nodes.",
                        },
                        **_scope_properties(),
                        "retrieval_mode": {
                            "type": "string",
                            "enum": ["graph", "verbatim", "hybrid"],
                            "default": "hybrid",
                            "description": "Retrieval strategy: graph-only, verbatim transcript retrieval, or hybrid fusion with reranking.",
                        },
                    },
                    required=["query"],
                ),
            ),
            types.Tool(
                name="debug_retrieval",
                description=(
                    "Diagnose memory retrieval ranking for a query. Returns query embedding preview, context-window "
                    "routing scores, selected windows, flat top nodes, and tiered top nodes for comparison."
                ),
                inputSchema=_object_input_schema(
                    {
                        "query": {"type": "string", "description": "Natural-language search query to diagnose."},
                        "max_nodes": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "description": "Maximum number of flat and tiered node matches to include.",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 2,
                            "minimum": 0,
                            "description": "Relationship traversal depth for the flat retrieval comparison.",
                        },
                        "retrieval_mode": {
                            "type": "string",
                            "enum": ["graph", "verbatim", "hybrid"],
                            "default": "hybrid",
                            "description": "Which retrieval stack to diagnose.",
                        },
                        **_scope_properties(),
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
                name="list_context_windows",
                description=(
                    "List context windows for a project. Use to inspect chat/session-level memory containers, "
                    "their status, node counts, and update times."
                ),
                inputSchema=_object_input_schema(
                    {
                        "project": {
                            "type": "string",
                            "description": "Optional project/repository scope to filter windows.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "closed", "archived"],
                            "description": "Optional status filter for returned windows.",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "minimum": 1,
                            "description": "Maximum number of context windows to return.",
                        },
                    }
                ),
            ),
            types.Tool(
                name="get_context_window",
                description=(
                    "Inspect one context window, including its nodes and links to other context windows. "
                    "Use when auditing what a conversation/session contributed to memory."
                ),
                inputSchema=_object_input_schema(
                    {
                        "window_id": {"type": "string", "description": "ID of the context window to inspect."},
                        "include_nodes": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to include memory nodes stored in this context window.",
                        },
                    },
                    required=["window_id"],
                ),
            ),
            types.Tool(
                name="close_context_window",
                description=(
                    "Close a context window, recompute its final graph embedding, refresh node counts, "
                    "and derive cross-window edges. Use when a chat/session is complete."
                ),
                inputSchema=_object_input_schema(
                    {"window_id": {"type": "string", "description": "ID of the context window to close."}},
                    required=["window_id"],
                ),
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
                    "or meaningful task outcomes. Do not ask the user to trigger this. "
                    "Required fields: 'user_message' (the user's text) and 'assistant_response' (the assistant's reply). "
                    "Do NOT use 'user_text' or 'assistant_text' — those field names are not accepted."
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
                    "in memory. Returns labeled clusters with representative nodes and tags. "
                    "Note: scope filtering (project, agent_id, session_id) is optional and silently ignored — "
                    "topic detection always runs across the full tenant graph."
                ),
                inputSchema=_object_input_schema(_scope_properties()),
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
                name="window_graph_viz",
                description=(
                    "Export the context-window graph as an interactive HTML visualization. "
                    "Each node is a chat/session window and edges show overlap, supersession, temporal order, or shared scope."
                ),
                inputSchema=_object_input_schema(
                    {
                        "project": {
                            "type": "string",
                            "description": "Optional project/repository scope whose context-window graph should be exported.",
                        },
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
                name="export_abhi",
                description=(
                    "Export the current memory graph as a validated .abhi file. "
                    "The file includes graph data, schema, constraints, AI rules, versions, queries, integrity metadata, and event definitions."
                ),
                inputSchema=_object_input_schema(
                    {
                        "output_path": {
                            "type": "string",
                            "description": "Optional destination .abhi file path. If omitted, Waggle chooses an export path.",
                        },
                        "force": {
                            "type": "boolean",
                            "default": False,
                            "description": "Override the secret-scan refusal if transcript records contain likely secrets. Use only after deliberate review.",
                        },
                        **_scope_properties(),
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
                            "enum": ["graph", "verbatim", "hybrid"],
                            "default": "hybrid",
                            "description": "Retrieval strategy for query mode: graph-only, verbatim transcript retrieval, or hybrid fusion.",
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
                name="import_abhi",
                description=(
                    "Import an .abhi file with integrity verification, schema validation, and constraint checks before merging it into memory."
                ),
                inputSchema=_object_input_schema(
                    {"input_path": {"type": "string", "description": "Path to the .abhi file to import."}},
                    required=["input_path"],
                ),
            ),
            types.Tool(
                name="diff_abhi",
                description=(
                    "Compare two .abhi files and report structural graph changes plus lightweight semantic changes."
                ),
                inputSchema=_object_input_schema(
                    {
                        "input_path_a": {"type": "string", "description": "Path to the first .abhi file."},
                        "input_path_b": {"type": "string", "description": "Path to the second .abhi file."},
                    },
                    required=["input_path_a", "input_path_b"],
                ),
            ),
            types.Tool(
                name="merge_abhi",
                description=(
                    "Three-way merge branching .abhi files into one output file with explicit conflict reporting."
                ),
                inputSchema=_object_input_schema(
                    {
                        "base_input_path": {"type": "string", "description": "Path to the common base .abhi file."},
                        "left_input_path": {"type": "string", "description": "Path to the left branch .abhi file."},
                        "right_input_path": {"type": "string", "description": "Path to the right branch .abhi file."},
                        "output_path": {"type": "string", "description": "Destination path for the merged .abhi file."},
                        "merge_strategy": {
                            "type": "string",
                            "enum": ["prefer_right", "prefer_left", "last_write_wins"],
                            "default": "prefer_right",
                            "description": "Winner strategy when both sides changed the same object differently.",
                        },
                    },
                    required=["base_input_path", "left_input_path", "right_input_path", "output_path"],
                ),
            ),
            types.Tool(
                name="query_abhi",
                description=(
                    "Execute a saved or ad hoc query against an .abhi file and trigger its on_query event actions."
                ),
                inputSchema=_object_input_schema(
                    {
                        "input_path": {"type": "string", "description": "Path to the .abhi file to query."},
                        "query_id": {"type": "string", "description": "Optional saved query id from the file."},
                        "query_text": {"type": "string", "description": "Optional ad hoc query text to execute."},
                    },
                    required=["input_path"],
                ),
            ),
            types.Tool(
                name="load_abhi_chunks",
                description=(
                    "Load only selected or query-relevant chunks from an .abhi file for partial graph inspection."
                ),
                inputSchema=_object_input_schema(
                    {
                        "input_path": {"type": "string", "description": "Path to the .abhi file to inspect."},
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional explicit chunk ids to load.",
                        },
                        "query_id": {"type": "string", "description": "Optional saved query id used to select chunks."},
                        "query_text": {"type": "string", "description": "Optional ad hoc query text used to select chunks."},
                    },
                    required=["input_path"],
                ),
            ),
            types.Tool(
                name="validate_abhi",
                description=(
                    "Validate an .abhi file without importing it. Verifies integrity hash, schema compliance, and constraint satisfaction."
                ),
                inputSchema=_object_input_schema(
                    {"input_path": {"type": "string", "description": "Path to the .abhi file to validate."}},
                    required=["input_path"],
                ),
            ),
            types.Tool(
                name="inspect_abhi",
                description=(
                    "Inspect an .abhi file without loading it into the memory graph. Returns summary stats, type breakdowns, and metadata counts."
                ),
                inputSchema=_object_input_schema(
                    {"input_path": {"type": "string", "description": "Path to the .abhi file to inspect."}},
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
        em = graph.embedding_model
        if self.config.is_fast_mode:
            # Fast mode: zero ML overhead. Schema inspection is the goal.
            LOGGER.info(
                "startup_fast_mode",
                extra={"startup_mode": self.config.startup_mode},
            )
        elif self.config.is_strict_mode:
            # Strict mode: block here until the model is fully loaded.
            LOGGER.info(
                "startup_strict_mode_waiting_for_embedding",
                extra={"model": em.model_name},
            )
            try:
                em.embed("startup validation", wait_timeout=120.0)
                if em.warmup_status != STATUS_READY:
                    LOGGER.warning(
                        "startup_strict_mode_embedding_not_ready",
                        extra={"status": em.warmup_status, "error": em.warmup_error},
                    )
            except Exception:
                LOGGER.exception("startup_strict_mode_embedding_failed")
        else:
            # Normal mode: embedding loads in background; startup is instant.
            # Fire a quick embed so the model is warm for the first real call;
            # failures are non-fatal and captured in warmup_status.
            try:
                em.embed("startup validation", wait_timeout=0.5)
            except Exception:
                LOGGER.debug("startup_embedding_probe_skipped")
        self.metrics.observe(
            "waggle_startup_validation_seconds",
            time.perf_counter() - started,
            backend=self.config.backend,
        )

    def build_resources(self) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(uri="graph://stats", name="Graph Stats", description="Current graph statistics.", mimeType="text/plain"),
                types.Resource(uri="graph://recent", name="Recent Graph Nodes", description="The 10 most recently updated nodes.", mimeType="text/plain"),
                types.Resource(uri="graph://windows", name="Context Windows", description="Recent context windows grouped by project/session.", mimeType="text/plain"),
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
        if uri == "graph://windows":
            windows = graph.list_context_windows(limit=50)
            if not windows:
                return "=== Context Windows: No context windows stored ==="
            lines = ["=== Context Windows ==="]
            for window in windows:
                lines.append(
                    f"• {window.id} [{window.status}] session={window.session_id} "
                    f"nodes={window.node_count} updated={window.updated_at.isoformat()}"
                )
            lines.append("=== End Context Windows ===")
            return "\n".join(lines)
        if uri == "graph://memory-policy":
            return MEMORY_AUTOMATION_POLICY
        raise ValidationFailure(f"Unknown resource: {uri}")

    def initialization_options(self) -> InitializationOptions:
        return InitializationOptions(
            server_name="waggle",
            server_version="0.2.0",
            capabilities=self.server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
        )

    def _check_embedding_available(
        self, name: str, graph: Any, arguments: dict[str, Any]
    ) -> types.CallToolResult | None:
        """Return a degraded response if fast mode blocks semantic tools."""
        if not self.config.is_fast_mode:
            return None
        if name in EMBEDDING_FREE_TOOLS:
            return None
        em = graph.embedding_model
        if em.warmup_status in (STATUS_READY,):  # shouldn't happen in fast mode, but guard it
            return None
        # Best-effort: if the tool supports retrieval_mode, try replay fallback.
        retrieval_mode = arguments.get("retrieval_mode", "") if name in ("query_graph", "export_context_bundle") else ""
        if retrieval_mode in ("verbatim", "lexical"):
            return None  # let it through — no embeddings needed
        return self._tool_result(
            f"Tool '{name}' requires semantic embeddings which are unavailable in fast/inspection mode "
            f"(WAGGLE_STARTUP_MODE={self.config.startup_mode}). "
            "Use retrieval_mode='verbatim' for transcript-only retrieval, or restart with "
            "WAGGLE_STARTUP_MODE=normal.",
            {
                "status": "unavailable",
                "reason": "fast_mode",
                "startup_mode": self.config.startup_mode,
                "embedding_status": STATUS_DISABLED,
                "tool": name,
            },
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

        # Fast-mode guard: return structured degraded response for semantic tools.
        fast_mode_result = self._check_embedding_available(name, graph, arguments)
        if fast_mode_result is not None:
            return fast_mode_result

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
                elif name == "aggregate_graph":
                    subgraph = graph.aggregate(
                        query=arguments.get("query", ""),
                        node_types=arguments.get("node_types"),
                        tags=arguments.get("tags"),
                        max_nodes=int(arguments.get("max_nodes", 100)),
                        max_depth=int(arguments.get("max_depth", 1)),
                        agent_id=arguments.get("agent_id", ""),
                        project=arguments.get("project", ""),
                        session_id=arguments.get("session_id", ""),
                    )
                    result = self._tool_result(
                        serialize_subgraph(subgraph),
                        self._subgraph_payload(subgraph),
                    )
                elif name == "query_graph":
                    subgraph = graph.query(
                        query=arguments["query"],
                        max_nodes=int(arguments.get("max_nodes", 20)),
                        max_depth=int(arguments.get("max_depth", 2)),
                        expand_depth=int(arguments.get("expand_depth", 0)),
                        agent_id=arguments.get("agent_id", ""),
                        project=arguments.get("project", ""),
                        session_id=arguments.get("session_id", ""),
                        retrieval_mode=arguments.get("retrieval_mode", "hybrid"),
                    )
                    result = self._tool_result(
                        serialize_subgraph(subgraph),
                        self._subgraph_payload(subgraph),
                    )
                elif name == "debug_retrieval":
                    debug = graph.debug_retrieval(
                        query=arguments["query"],
                        max_nodes=int(arguments.get("max_nodes", 10)),
                        max_depth=int(arguments.get("max_depth", 2)),
                        agent_id=arguments.get("agent_id", ""),
                        project=arguments.get("project", ""),
                        session_id=arguments.get("session_id", ""),
                        retrieval_mode=arguments.get("retrieval_mode", "hybrid"),
                    )
                    result = self._tool_result(
                        json.dumps(debug, indent=2),
                        debug,
                    )
                elif name == "list_context_scopes":
                    scopes = graph.list_context_scopes()
                    result = self._tool_result(
                        f"Known scopes: {len(scopes.agent_ids)} agents, {len(scopes.projects)} projects, {len(scopes.session_ids)} sessions.",
                        self._context_scope_payload(scopes),
                    )
                elif name == "list_context_windows":
                    windows = graph.list_context_windows(
                        project=arguments.get("project", ""),
                        status=arguments.get("status", ""),
                        limit=int(arguments.get("limit", 20)),
                    )
                    result = self._tool_result(
                        f"Context windows: {len(windows)}",
                        {"windows": [self._context_window_payload(window) for window in windows]},
                    )
                elif name == "get_context_window":
                    window = graph.get_context_window(arguments["window_id"])
                    edges = graph.get_context_window_edges(window.id)
                    nodes = graph.get_window_nodes(window.id) if bool(arguments.get("include_nodes", True)) else []
                    result = self._tool_result(
                        f"Context window {window.id}: {window.node_count} nodes, {len(edges)} connected window edge(s).",
                        {
                            "window": self._context_window_payload(window),
                            "nodes": [self._node_payload(node) for node in nodes],
                            "window_edges": [self._context_window_edge_payload(edge) for edge in edges],
                        },
                    )
                elif name == "close_context_window":
                    window = graph.close_context_window(arguments["window_id"])
                    edges = graph.get_context_window_edges(window.id)
                    result = self._tool_result(
                        f"Closed context window {window.id} with {window.node_count} nodes and {len(edges)} connected window edge(s).",
                        {
                            "window": self._context_window_payload(window),
                            "window_edges": [self._context_window_edge_payload(edge) for edge in edges],
                        },
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
                    # Scope parameters are accepted by the schema but ignored —
                    # topic detection runs across the full tenant graph.
                    topics = graph.get_topics()
                    result = self._tool_result(serialize_topics(topics), self._topic_payload(topics))
                elif name == "get_stats":
                    stats = graph.get_stats()
                    em = graph.embedding_model
                    stats_payload = self._stats_payload(stats)
                    # Augment with live embedding status — never triggers ML load.
                    embedding_status = getattr(em, "warmup_status", "unknown")
                    embedding_error = getattr(em, "warmup_error", "")
                    stats_payload["embedding_status"] = embedding_status
                    if embedding_error:
                        stats_payload["embedding_error"] = embedding_error
                    stats_payload["startup_mode"] = self.config.startup_mode
                    result = self._tool_result(
                        serialize_stats(stats)
                        + f"\nEmbedding status: {embedding_status}"
                        + (f" (error: {embedding_error}" + ")" if embedding_error else ""),
                        stats_payload,
                    )
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
                elif name == "window_graph_viz":
                    output_path = graph.export_window_graph_html(
                        project=arguments.get("project", ""),
                        output_path=arguments.get("output_path"),
                        include_physics=bool(arguments.get("include_physics", True)),
                    )
                    windows = graph.list_context_windows(project=arguments.get("project", ""), limit=10_000)
                    edge_count = sum(len(graph.get_context_window_edges(window.id)) for window in windows)
                    result = self._tool_result(
                        f"Exported context-window graph visualization to {output_path}.",
                        {
                            "output_path": str(output_path),
                            "tenant_id": graph.tenant_id,
                            "project": arguments.get("project", ""),
                            "total_context_windows": len(windows),
                            "total_context_window_edges": edge_count,
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
                elif name == "export_abhi":
                    _assert_export_safe(
                        graph,
                        force=bool(arguments.get("force", False)),
                        project=arguments.get("project", ""),
                        agent_id=arguments.get("agent_id", ""),
                        session_id=arguments.get("session_id", ""),
                    )
                    exported = graph.export_abhi(
                        output_path=arguments.get("output_path"),
                        project=arguments.get("project", ""),
                        agent_id=arguments.get("agent_id", ""),
                        session_id=arguments.get("session_id", ""),
                    )
                    result = self._tool_result(
                        f"Exported ABHI memory file to {exported.output_path}.",
                        {
                            "output_path": exported.output_path,
                            "tenant_id": exported.tenant_id,
                            "schema_version": exported.schema_version,
                            "abhi_spec_version": exported.abhi_spec_version,
                            "node_count": exported.node_count,
                            "edge_count": exported.edge_count,
                            "content_hash": exported.content_hash,
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
                        retrieval_mode=arguments.get("retrieval_mode", "hybrid"),
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
                elif name == "import_abhi":
                    imported = graph.import_abhi(input_path=arguments["input_path"])
                    result = self._tool_result(
                        f"Imported ABHI memory file from {imported.input_path}.",
                        {
                            "input_path": imported.input_path,
                            "tenant_id": imported.tenant_id,
                            "schema_version": imported.schema_version,
                            "abhi_spec_version": imported.abhi_spec_version,
                            "nodes_created": imported.nodes_created,
                            "nodes_updated": imported.nodes_updated,
                            "edges_created": imported.edges_created,
                            "edges_updated": imported.edges_updated,
                            "hash_verified": imported.hash_verified,
                        },
                    )
                elif name == "diff_abhi":
                    diff = graph.diff_abhi(
                        input_path_a=arguments["input_path_a"],
                        input_path_b=arguments["input_path_b"],
                    )
                    result = self._tool_result(
                        serialize_abhi_diff(diff),
                        diff.model_dump(mode="json"),
                    )
                elif name == "merge_abhi":
                    merged = graph.merge_abhi(
                        base_input_path=arguments["base_input_path"],
                        left_input_path=arguments["left_input_path"],
                        right_input_path=arguments["right_input_path"],
                        output_path=arguments["output_path"],
                        merge_strategy=arguments.get("merge_strategy", "prefer_right"),
                    )
                    result = self._tool_result(
                        serialize_abhi_merge(merged),
                        merged.model_dump(mode="json"),
                    )
                elif name == "query_abhi":
                    queried = graph.query_abhi(
                        input_path=arguments["input_path"],
                        query_id=arguments.get("query_id", ""),
                        query_text=arguments.get("query_text", ""),
                    )
                    result = self._tool_result(
                        serialize_abhi_query(queried),
                        queried.model_dump(mode="json"),
                    )
                elif name == "load_abhi_chunks":
                    loaded = graph.load_abhi_chunks(
                        input_path=arguments["input_path"],
                        chunk_ids=list(arguments.get("chunk_ids", [])),
                        query_id=arguments.get("query_id", ""),
                        query_text=arguments.get("query_text", ""),
                    )
                    result = self._tool_result(
                        serialize_abhi_chunk_load(loaded),
                        loaded.model_dump(mode="json"),
                    )
                elif name == "validate_abhi":
                    validation = graph.validate_abhi(input_path=arguments["input_path"])
                    result = self._tool_result(
                        serialize_abhi_validation(validation),
                        validation.model_dump(mode="json"),
                    )
                elif name == "inspect_abhi":
                    inspection = graph.inspect_abhi(input_path=arguments["input_path"])
                    result = self._tool_result(
                        serialize_abhi_inspect(inspection),
                        inspection.model_dump(mode="json"),
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
            "context_window_id": node.context_window_id,
            "label": node.label,
            "content": node.content,
            "node_type": node.node_type.value,
            "tags": node.tags,
            "source_prompt": node.source_prompt,
            "metadata": node.metadata,
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
            "similarity_score": node.similarity_score,
            "recency_score": node.recency_score,
            "edge_score": node.edge_score,
            "final_score": node.final_score,
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
            "hybrid_hits": [
                {
                    "content": hit.content,
                    "score": hit.score,
                    "source": hit.source,
                    "turn_pair_id": hit.turn_pair_id,
                    "node_ids": hit.node_ids,
                    "reasoning_from_reranker": hit.reasoning_from_reranker,
                    "observed_at": hit.observed_at.isoformat() if hit.observed_at is not None else None,
                    "layer_scores": hit.layer_scores,
                }
                for hit in result.hybrid_hits
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
                    "recency_score": item.recency_score,
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

    def _context_window_payload(self, window: ContextWindow) -> dict[str, Any]:
        return {
            "id": window.id,
            "tenant_id": window.tenant_id,
            "repo_id": window.repo_id,
            "session_id": window.session_id,
            "title": window.title,
            "status": window.status,
            "node_count": window.node_count,
            "embedding_stale": window.embedding_stale,
            "created_at": window.created_at.isoformat(),
            "updated_at": window.updated_at.isoformat(),
            "closed_at": window.closed_at.isoformat() if window.closed_at is not None else None,
        }

    def _context_window_edge_payload(self, edge: ContextWindowEdge) -> dict[str, Any]:
        return {
            "id": edge.id,
            "tenant_id": edge.tenant_id,
            "source_window_id": edge.source_window_id,
            "target_window_id": edge.target_window_id,
            "edge_type": edge.edge_type,
            "shared_entities": edge.shared_entities,
            "weight": edge.weight,
            "metadata": edge.metadata,
            "created_at": edge.created_at.isoformat(),
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
            "total_repos": stats.total_repos,
            "total_context_windows": stats.total_context_windows,
            "context_window_status_breakdown": stats.context_window_status_breakdown,
            "total_context_window_edges": stats.total_context_window_edges,
            "context_window_edge_type_breakdown": stats.context_window_edge_type_breakdown,
            "windows_with_embeddings": stats.windows_with_embeddings,
            "windows_with_stale_embeddings": stats.windows_with_stale_embeddings,
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
        if name == "aggregate_graph":
            self._assert_payload_size(arguments.get("query", ""), limit, "aggregate_graph.query")
            self._assert_payload_size(arguments.get("agent_id", ""), limit, "aggregate_graph.agent_id")
            self._assert_payload_size(arguments.get("project", ""), limit, "aggregate_graph.project")
            self._assert_payload_size(arguments.get("session_id", ""), limit, "aggregate_graph.session_id")
            return
        if name == "query_graph":
            self._assert_payload_size(arguments.get("query", ""), limit, "query_graph.query")
            self._assert_payload_size(arguments.get("agent_id", ""), limit, "query_graph.agent_id")
            self._assert_payload_size(arguments.get("project", ""), limit, "query_graph.project")
            self._assert_payload_size(arguments.get("session_id", ""), limit, "query_graph.session_id")
            return
        if name == "debug_retrieval":
            self._assert_payload_size(arguments.get("query", ""), limit, "debug_retrieval.query")
            self._assert_payload_size(arguments.get("agent_id", ""), limit, "debug_retrieval.agent_id")
            self._assert_payload_size(arguments.get("project", ""), limit, "debug_retrieval.project")
            self._assert_payload_size(arguments.get("session_id", ""), limit, "debug_retrieval.session_id")
            return
        if name == "export_context_bundle":
            self._assert_payload_size(arguments.get("query", ""), limit, "export_context_bundle.query")
            self._assert_payload_size(arguments.get("project", ""), limit, "export_context_bundle.project")
            self._assert_payload_size(arguments.get("agent_id", ""), limit, "export_context_bundle.agent_id")
            self._assert_payload_size(arguments.get("session_id", ""), limit, "export_context_bundle.session_id")
            self._assert_payload_size(arguments.get("output_path", ""), limit, "export_context_bundle.output_path")
            return
        if name == "window_graph_viz":
            self._assert_payload_size(arguments.get("project", ""), limit, "window_graph_viz.project")
            self._assert_payload_size(arguments.get("output_path", ""), limit, "window_graph_viz.output_path")
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
        # Kick off background embedding warmup for HTTP transport (non-blocking).
        em = self.app_server._root_graph.embedding_model
        if (
            not self.config.is_fast_mode
            and hasattr(em, "start_background_warmup")
            and not getattr(em, "_warmup_started", False)
        ):
            em.start_background_warmup()
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

    def _scope_from_request(request: Request) -> dict[str, str]:
        return {
            "project": request.query_params.get("project", "").strip(),
            "agent_id": request.query_params.get("agent_id", "").strip(),
            "session_id": request.query_params.get("session_id", "").strip(),
        }

    def _build_scoped_abhi(scope: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        snapshot = app_server.graph.get_graph_snapshot(**scope)
        return snapshot, build_abhi_document(snapshot)

    def _validate_live_snapshot(snapshot: dict[str, Any]) -> None:
        validation = validate_abhi_document(build_abhi_document(snapshot), input_path="live://graph")
        blocking_errors = [
            error
            for error in validation.errors
            if "cannot originate from node type" not in error and "cannot target node type" not in error
        ]
        if blocking_errors:
            raise ValidationFailure("; ".join(blocking_errors))

    def _node_snapshot_payload(
        *,
        snapshot: dict[str, Any],
        payload: dict[str, Any],
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = existing or {}
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return {
            "id": str(current.get("id") or payload.get("id") or f"preview-{uuid.uuid4()}").strip(),
            "tenant_id": str(snapshot.get("tenant_id", "")),
            "agent_id": str(payload.get("agent_id", current.get("agent_id", "")) or current.get("agent_id", "")),
            "project": str(payload.get("project", current.get("project", "")) or current.get("project", "")),
            "session_id": str(payload.get("session_id", current.get("session_id", "")) or current.get("session_id", "")),
            "context_window_id": current.get("context_window_id"),
            "label": str(payload.get("label", current.get("label", "")) or current.get("label", "")).strip(),
            "content": str(payload.get("content", current.get("content", "")) or current.get("content", "")).strip(),
            "node_type": str(payload.get("node_type", current.get("node_type", "note")) or current.get("node_type", "note")).strip(),
            "tags": payload.get("tags", current.get("tags", [])) or [],
            "source_prompt": current.get("source_prompt", ""),
            "metadata": current.get("metadata", {}),
            "evidence_records": current.get("evidence_records", []),
            "valid_from": current.get("valid_from"),
            "valid_to": current.get("valid_to"),
            "created_at": current.get("created_at", now),
            "updated_at": now,
            "access_count": current.get("access_count", 0),
        }

    def _edge_snapshot_payload(
        *,
        snapshot: dict[str, Any],
        payload: dict[str, Any],
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = existing or {}
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return {
            "id": str(current.get("id") or payload.get("id") or f"preview-{uuid.uuid4()}").strip(),
            "tenant_id": str(snapshot.get("tenant_id", "")),
            "source_id": str(payload.get("source_id", current.get("source_id", "")) or current.get("source_id", "")).strip(),
            "target_id": str(payload.get("target_id", current.get("target_id", "")) or current.get("target_id", "")).strip(),
            "relationship": str(payload.get("relationship", current.get("relationship", "")) or current.get("relationship", "")).strip(),
            "weight": float(payload.get("weight", current.get("weight", 1.0)) or current.get("weight", 1.0)),
            "metadata": current.get("metadata", {}),
            "created_at": current.get("created_at", now),
        }

    def _json_safe_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        safe = deepcopy(snapshot)
        for node in safe.get("nodes", []):
            node.pop("embedding", None)
        for transcript in safe.get("transcripts", []):
            transcript.pop("embedding", None)
        return safe

    async def graph_editor(request: Request) -> Response:
        mode = request.query_params.get("mode", "edit").strip().lower()
        if mode not in {"edit", "view"}:
            mode = "edit"
        scope = _scope_from_request(request)
        return HTMLResponse(
            render_graph_editor_html(
                mode=mode,
                project=scope["project"],
                agent_id=scope["agent_id"],
                session_id=scope["session_id"],
            )
        )

    async def graph_snapshot(request: Request) -> Response:
        scope = _scope_from_request(request)
        graph = app_server.graph
        include_source_prompt = request.query_params.get("include_source_prompt", "").strip().lower() in {"1", "true", "yes"}
        try:
            snapshot = graph.get_graph_snapshot(include_source_prompt=include_source_prompt, **scope)
        except TypeError:
            snapshot = graph.get_graph_snapshot(**scope)
        return JSONResponse(
            {
                "tenant_id": snapshot.get("tenant_id", ""),
                "schema_version": snapshot.get("schema_version", 1),
                "nodes": snapshot.get("nodes", []),
                "edges": snapshot.get("edges", []),
                "ui": snapshot.get("ui", {}),
                "node_types": [node_type.value for node_type in NodeType],
                "relation_types": [relation.value for relation in RelationType],
            }
        )

    async def graph_transcripts(request: Request) -> Response:
        scope = _scope_from_request(request)
        limit = int(request.query_params.get("limit", "200") or "200")
        query_text = request.query_params.get("query", "").strip()
        graph = app_server.graph
        if query_text and hasattr(graph, "search_transcript_records"):
            hits = graph.search_transcript_records(query=query_text, limit=limit, **scope)
            return JSONResponse(
                {
                    "mode": "hybrid",
                    "query": query_text,
                    "hits": [hit.model_dump(mode="json") for hit in hits],
                }
            )
        if not hasattr(graph, "list_transcript_records"):
            raise ValidationFailure("Transcript listing is not available in this backend.")
        records = graph.list_transcript_records(limit=limit, **scope)
        return JSONResponse(
            {
                "mode": "chronological",
                "records": [record.model_dump(mode="json") for record in records],
            }
        )

    async def graph_retrieval_debug(request: Request) -> Response:
        payload = await request.json()
        scope = {
            "project": str(payload.get("project", "")).strip(),
            "agent_id": str(payload.get("agent_id", "")).strip(),
            "session_id": str(payload.get("session_id", "")).strip(),
        }
        query_text = str(payload.get("query", "")).strip()
        if not query_text:
            raise ValidationFailure("query is required.")
        max_nodes = int(payload.get("max_nodes", 8) or 8)
        max_depth = int(payload.get("max_depth", 1) or 1)
        graph = app_server.graph
        debug = graph.debug_retrieval(query=query_text, max_nodes=max_nodes, max_depth=max_depth, retrieval_mode="hybrid", **scope)
        fusion = graph.query(
            query=query_text,
            retrieval_mode="hybrid",
            max_nodes=max_nodes,
            max_depth=max_depth,
            **scope,
        )
        token_estimate = sum(max(1, len((hit.content if hasattr(hit, "content") else "") or "").split()) for hit in fusion.fusion_hits) * 1.33
        fused_ranking: list[dict[str, Any]] = []
        for hit in fusion.fusion_hits:
            reasoning: list[str] = []
            if hit.source_lane == "graph":
                reasoning.append("semantic graph node")
            if hit.graph_rank is not None:
                reasoning.append(f"graph rank {hit.graph_rank}")
            if hit.replay_rank is not None:
                reasoning.append(f"replay rank {hit.replay_rank}")
            if hit.session_id:
                reasoning.append(f"session {hit.session_id}")
            fused_ranking.append(
                {
                    "content": hit.content,
                    "score": hit.score,
                    "source_lane": hit.source_lane,
                    "graph_rank": hit.graph_rank,
                    "replay_rank": hit.replay_rank,
                    "fused_rank": hit.fused_rank,
                    "node_id": hit.node_id,
                    "session_id": hit.session_id,
                    "turn_index": hit.turn_index,
                    "transcript_snippet": hit.transcript_snippet,
                    "reasoning": ", ".join(reasoning) or "ranked by reciprocal-rank fusion",
                }
            )
        return JSONResponse(
            {
                "debug": debug,
                "replay_hits": [hit.model_dump(mode="json") for hit in fusion.replay_hits],
                "fusion_hits": fused_ranking,
                "token_estimate": int(token_estimate),
            }
        )

    async def graph_abhi_preview(request: Request) -> Response:
        scope = _scope_from_request(request)
        snapshot, document = _build_scoped_abhi(scope)
        validation = validate_abhi_document(document, input_path="live://graph")
        return JSONResponse(
            {
                "tenant_id": snapshot.get("tenant_id", ""),
                "scope": scope,
                "schema": document.get("schema", {}),
                "constraints": document.get("constraints", []),
                "queries": document.get("queries", {}),
                "events": document.get("events", {}),
                "versions": document.get("versions", []),
                "integrity": document.get("integrity", {}),
                "validation": validation.model_dump(mode="json"),
            }
        )

    async def graph_query(request: Request) -> Response:
        payload = await request.json()
        scope = {
            "project": str(payload.get("project", "")).strip(),
            "agent_id": str(payload.get("agent_id", "")).strip(),
            "session_id": str(payload.get("session_id", "")).strip(),
        }
        _, document = _build_scoped_abhi(scope)
        result = execute_abhi_query(
            document,
            query_id=str(payload.get("query_id", "")).strip(),
            query_text=str(payload.get("query", "")).strip(),
        )
        return JSONResponse(result)

    async def graph_diff_feed(request: Request) -> Response:
        since = request.query_params.get("since", "24h").strip() or "24h"
        diff = app_server.graph.graph_diff(since=since)
        return JSONResponse(
            {
                "since": diff.since,
                "added_nodes": [node.model_dump(mode="json") for node in diff.added_nodes],
                "updated_nodes": [node.model_dump(mode="json") for node in diff.updated_nodes],
                "created_edges": [edge.model_dump(mode="json") for edge in diff.created_edges],
                "contradiction_edges": [edge.model_dump(mode="json") for edge in diff.contradiction_edges],
            }
        )

    async def graph_save_ui(request: Request) -> Response:
        payload = await request.json()
        graph = app_server.graph
        saved = graph.save_ui_state(
            project=str(payload.get("project", "")).strip(),
            agent_id=str(payload.get("agent_id", "")).strip(),
            session_id=str(payload.get("session_id", "")).strip(),
            positions=payload.get("positions"),
            zoom=float(payload["zoom"]) if "zoom" in payload and payload.get("zoom") is not None else None,
            viewport=payload.get("viewport"),
            groups=payload.get("groups"),
            collapsed_groups=payload.get("collapsed_groups"),
            selected_nodes=payload.get("selected_nodes"),
        )
        return JSONResponse(saved)

    async def graph_restore(request: Request) -> Response:
        payload = await request.json()
        scope = {
            "project": str(payload.get("project", "")).strip(),
            "agent_id": str(payload.get("agent_id", "")).strip(),
            "session_id": str(payload.get("session_id", "")).strip(),
        }
        graph = app_server.graph
        current = graph.get_graph_snapshot(**scope)
        desired_nodes = {
            str(node.get("id", "")).strip(): node
            for node in payload.get("nodes", [])
            if str(node.get("id", "")).strip()
        }
        desired_edges = {
            str(edge.get("id", "")).strip(): edge
            for edge in payload.get("edges", [])
            if str(edge.get("id", "")).strip()
        }
        current_nodes = {
            str(node.get("id", "")).strip(): node
            for node in current.get("nodes", [])
            if str(node.get("id", "")).strip()
        }
        current_edges = {
            str(edge.get("id", "")).strip(): edge
            for edge in current.get("edges", [])
            if str(edge.get("id", "")).strip()
        }

        for edge_id in list(current_edges):
            if edge_id not in desired_edges:
                graph.delete_edge(edge_id=edge_id)

        for node_id in list(current_nodes):
            if node_id not in desired_nodes:
                graph.delete_node(node_id=node_id)

        for node_id, node_payload in desired_nodes.items():
            tags = [str(tag).strip() for tag in node_payload.get("tags", []) if str(tag).strip()]
            if node_id in current_nodes:
                graph.update_node(
                    node_id=node_id,
                    label=node_payload.get("label"),
                    content=node_payload.get("content"),
                    tags=tags,
                )
                continue
            graph.add_node(
                node_id=node_id,
                label=str(node_payload.get("label", "")).strip(),
                content=str(node_payload.get("content", "")).strip(),
                node_type=NodeType(str(node_payload.get("node_type", "note")).strip() or "note"),
                tags=tags,
                agent_id=str(node_payload.get("agent_id", scope["agent_id"])).strip(),
                project=str(node_payload.get("project", scope["project"])).strip(),
                session_id=str(node_payload.get("session_id", scope["session_id"])).strip(),
            )

        for edge_id, edge_payload in desired_edges.items():
            if edge_id in current_edges:
                graph.update_edge(
                    edge_id=edge_id,
                    source_id=str(edge_payload.get("source_id", "")).strip() or None,
                    target_id=str(edge_payload.get("target_id", "")).strip() or None,
                    relationship=str(edge_payload.get("relationship", "")).strip() or None,
                    weight=float(edge_payload["weight"]) if "weight" in edge_payload and edge_payload.get("weight") is not None else None,
                )
                continue
            graph.add_edge(
                edge_id=edge_id,
                source_id=str(edge_payload.get("source_id", "")).strip(),
                target_id=str(edge_payload.get("target_id", "")).strip(),
                relationship=str(edge_payload.get("relationship", "")).strip(),
                weight=float(edge_payload.get("weight", 1.0)),
            )

        ui = payload.get("ui", {}) or {}
        saved = graph.save_ui_state(
            project=scope["project"],
            agent_id=scope["agent_id"],
            session_id=scope["session_id"],
            positions=ui.get("positions"),
            zoom=float(ui["zoom"]) if "zoom" in ui and ui.get("zoom") is not None else None,
            viewport=ui.get("viewport"),
            groups=ui.get("groups"),
            collapsed_groups=ui.get("collapsed_groups"),
            selected_nodes=ui.get("selected_nodes"),
        )
        restored = graph.get_graph_snapshot(**scope)
        return JSONResponse(
            {
                "nodes": restored.get("nodes", []),
                "edges": restored.get("edges", []),
                "ui": saved,
            }
        )

    async def graph_create_node(request: Request) -> Response:
        payload = await request.json()
        graph = app_server.graph
        scope = {
            "project": str(payload.get("project", "")).strip(),
            "agent_id": str(payload.get("agent_id", "")).strip(),
            "session_id": str(payload.get("session_id", "")).strip(),
        }
        snapshot = graph.get_graph_snapshot(**scope)
        snapshot["nodes"] = [*snapshot.get("nodes", []), _node_snapshot_payload(snapshot=snapshot, payload=payload)]
        _validate_live_snapshot(snapshot)
        created = graph.add_node(
            node_id=str(payload.get("id", "")).strip() or None,
            label=str(payload.get("label", "")).strip(),
            content=str(payload.get("content", "")).strip(),
            node_type=NodeType(str(payload.get("node_type", "note")).strip() or "note"),
            tags=[str(tag).strip() for tag in payload.get("tags", []) if str(tag).strip()],
            agent_id=str(payload.get("agent_id", "")).strip(),
            project=str(payload.get("project", "")).strip(),
            session_id=str(payload.get("session_id", "")).strip(),
        )
        return JSONResponse(created.node.model_dump(mode="json"))

    async def graph_update_node(request: Request) -> Response:
        node_id = request.path_params["node_id"]
        payload = await request.json()
        graph = app_server.graph
        snapshot = graph.get_graph_snapshot()
        existing = next((node for node in snapshot.get("nodes", []) if str(node.get("id", "")).strip() == node_id), None)
        if existing is None:
            raise ValidationFailure(f"Node not found: {node_id}")
        snapshot["nodes"] = [
            _node_snapshot_payload(snapshot=snapshot, payload=payload, existing=existing)
            if str(node.get("id", "")).strip() == node_id
            else node
            for node in snapshot.get("nodes", [])
        ]
        _validate_live_snapshot(snapshot)
        updated = graph.update_node(
            node_id=node_id,
            label=payload.get("label"),
            content=payload.get("content"),
            tags=payload.get("tags"),
        )
        return JSONResponse(updated.model_dump(mode="json"))

    async def graph_delete_node(request: Request) -> Response:
        node_id = request.path_params["node_id"]
        graph = app_server.graph
        deleted = graph.delete_node(node_id=node_id)
        return JSONResponse(deleted.model_dump(mode="json"))

    async def graph_create_edge(request: Request) -> Response:
        payload = await request.json()
        graph = app_server.graph
        snapshot = graph.get_graph_snapshot()
        snapshot["edges"] = [*snapshot.get("edges", []), _edge_snapshot_payload(snapshot=snapshot, payload=payload)]
        _validate_live_snapshot(snapshot)
        edge = graph.add_edge(
            edge_id=str(payload.get("id", "")).strip() or None,
            source_id=str(payload.get("source_id", "")).strip(),
            target_id=str(payload.get("target_id", "")).strip(),
            relationship=str(payload.get("relationship", "")).strip(),
            weight=float(payload.get("weight", 1.0)),
        )
        return JSONResponse(edge.model_dump(mode="json"))

    async def graph_update_edge(request: Request) -> Response:
        edge_id = request.path_params["edge_id"]
        payload = await request.json()
        graph = app_server.graph
        snapshot = graph.get_graph_snapshot()
        existing = next((item for item in snapshot.get("edges", []) if str(item.get("id", "")).strip() == edge_id), None)
        if existing is None:
            raise ValidationFailure(f"Edge not found: {edge_id}")
        snapshot["edges"] = [
            _edge_snapshot_payload(snapshot=snapshot, payload=payload, existing=existing)
            if str(item.get("id", "")).strip() == edge_id
            else item
            for item in snapshot.get("edges", [])
        ]
        _validate_live_snapshot(snapshot)
        edge = graph.update_edge(
            edge_id=edge_id,
            source_id=str(payload.get("source_id", "")).strip() or None,
            target_id=str(payload.get("target_id", "")).strip() or None,
            relationship=str(payload.get("relationship", "")).strip() or None,
            weight=float(payload["weight"]) if "weight" in payload and payload.get("weight") is not None else None,
        )
        return JSONResponse(edge.model_dump(mode="json"))

    async def graph_delete_edge(request: Request) -> Response:
        edge_id = request.path_params["edge_id"]
        graph = app_server.graph
        edge = graph.delete_edge(edge_id=edge_id)
        return JSONResponse(edge.model_dump(mode="json"))

    async def graph_export(request: Request) -> Response:
        scope = _scope_from_request(request)
        export_format = request.query_params.get("format", "abhi").strip().lower()
        graph = app_server.graph
        if export_format == "abhi":
            exported = graph.export_abhi(**scope)
            content = Path(exported.output_path).read_bytes()
            return Response(
                content,
                media_type="application/octet-stream",
                headers={"Content-Disposition": 'attachment; filename="waggle-memory.abhi"'},
            )
        if export_format == "json":
            snapshot = graph.get_graph_snapshot(**scope)
            return Response(
                json.dumps(snapshot, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="waggle-backup.json"'},
            )
        raise ValidationFailure("format must be one of: abhi, json.")

    async def graph_import(request: Request) -> Response:
        payload = await request.json()
        import_format = str(payload.get("format", "abhi")).strip().lower()
        content = str(payload.get("content", ""))
        content_base64 = str(payload.get("content_base64", ""))
        suffix = ".abhi" if import_format == "abhi" else ".json"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            if import_format == "abhi" and content_base64:
                temp_path.write_bytes(base64.b64decode(content_base64))
            else:
                temp_path.write_text(content, encoding="utf-8")
            graph = app_server.graph
            imported_node_ids: list[str] = []
            if import_format == "abhi":
                document = load_abhi_document(temp_path)
                imported_node_ids = [str(node.get("id", "")).strip() for node in abhi_to_snapshot(document, fallback_tenant_id=graph.tenant_id).get("nodes", []) if str(node.get("id", "")).strip()]
                imported = graph.import_abhi(input_path=temp_path)
            elif import_format == "json":
                snapshot = json.loads(content)
                imported_node_ids = [str(node.get("id", "")).strip() for node in snapshot.get("nodes", []) if str(node.get("id", "")).strip()]
                imported = graph.import_graph_backup(input_path=temp_path)
            else:
                raise ValidationFailure("format must be one of: abhi, json.")
            return JSONResponse({**imported.model_dump(mode="json"), "imported_node_ids": imported_node_ids})
        finally:
            temp_path.unlink(missing_ok=True)

    async def graph_import_preview(request: Request) -> Response:
        payload = await request.json()
        import_format = str(payload.get("format", "abhi")).strip().lower()
        content = str(payload.get("content", ""))
        content_base64 = str(payload.get("content_base64", ""))
        suffix = ".abhi" if import_format == "abhi" else ".json"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            if import_format == "abhi" and content_base64:
                temp_path.write_bytes(base64.b64decode(content_base64))
            else:
                temp_path.write_text(content, encoding="utf-8")
            graph = app_server.graph
            if import_format == "abhi":
                validation = graph.validate_abhi(input_path=temp_path)
                inspect_result = graph.inspect_abhi(input_path=temp_path)
                document = load_abhi_document(temp_path)
                snapshot = abhi_to_snapshot(document, fallback_tenant_id=graph.tenant_id)
                return JSONResponse(
                    {
                        "validation": validation.model_dump(mode="json"),
                        "inspect": inspect_result.model_dump(mode="json"),
                        "snapshot": {
                            "tenant_id": snapshot.get("tenant_id", ""),
                            "nodes": _json_safe_snapshot(snapshot).get("nodes", []),
                            "edges": snapshot.get("edges", []),
                            "ui": snapshot.get("ui", {}),
                        },
                        "imported_node_ids": [str(node.get("id", "")).strip() for node in snapshot.get("nodes", []) if str(node.get("id", "")).strip()],
                    }
                )
            if import_format == "json":
                snapshot = json.loads(content)
                return JSONResponse(
                    {
                        "validation": {"valid": True, "errors": []},
                        "inspect": {"node_count": len(snapshot.get("nodes", [])), "edge_count": len(snapshot.get("edges", []))},
                        "snapshot": snapshot,
                        "imported_node_ids": [str(node.get("id", "")).strip() for node in snapshot.get("nodes", []) if str(node.get("id", "")).strip()],
                    }
                )
            raise ValidationFailure("format must be one of: abhi, json.")
        finally:
            temp_path.unlink(missing_ok=True)

    async def graph_abhi_diff(request: Request) -> Response:
        payload = await request.json()
        content_a = str(payload.get("content_a", ""))
        content_b = str(payload.get("content_b", ""))
        content_a_base64 = str(payload.get("content_a_base64", ""))
        content_b_base64 = str(payload.get("content_b_base64", ""))
        with tempfile.NamedTemporaryFile(suffix=".abhi", delete=False) as handle_a:
            path_a = Path(handle_a.name)
        with tempfile.NamedTemporaryFile(suffix=".abhi", delete=False) as handle_b:
            path_b = Path(handle_b.name)
        try:
            if content_a_base64:
                path_a.write_bytes(base64.b64decode(content_a_base64))
            else:
                path_a.write_text(content_a, encoding="utf-8")
            if content_b_base64:
                path_b.write_bytes(base64.b64decode(content_b_base64))
            else:
                path_b.write_text(content_b, encoding="utf-8")
            graph = app_server.graph
            diff = graph.diff_abhi(input_path_a=path_a, input_path_b=path_b)
            snapshot_a = abhi_to_snapshot(load_abhi_document(path_a), fallback_tenant_id=graph.tenant_id)
            snapshot_b = abhi_to_snapshot(load_abhi_document(path_b), fallback_tenant_id=graph.tenant_id)
            return JSONResponse(
                {
                    "diff": diff.model_dump(mode="json"),
                    "left": {"nodes": _json_safe_snapshot(snapshot_a).get("nodes", []), "edges": snapshot_a.get("edges", [])},
                    "right": {"nodes": _json_safe_snapshot(snapshot_b).get("nodes", []), "edges": snapshot_b.get("edges", [])},
                }
            )
        finally:
            path_a.unlink(missing_ok=True)
            path_b.unlink(missing_ok=True)

    app = Starlette(
        routes=[
            Route("/health/live", live),
            Route("/health/ready", ready),
            Route("/metrics", metrics_endpoint),
            Route("/graph", graph_editor),
            Route("/api/graph", graph_snapshot, methods=["GET"]),
            Route("/api/graph/transcripts", graph_transcripts, methods=["GET"]),
            Route("/api/graph/retrieval-debug", graph_retrieval_debug, methods=["POST"]),
            Route("/api/graph/abhi", graph_abhi_preview, methods=["GET"]),
            Route("/api/graph/abhi/preview-import", graph_import_preview, methods=["POST"]),
            Route("/api/graph/abhi/diff", graph_abhi_diff, methods=["POST"]),
            Route("/api/graph/query", graph_query, methods=["POST"]),
            Route("/api/graph/diff", graph_diff_feed, methods=["GET"]),
            Route("/api/graph/ui", graph_save_ui, methods=["PATCH"]),
            Route("/api/graph/restore", graph_restore, methods=["POST"]),
            Route("/api/graph/nodes", graph_create_node, methods=["POST"]),
            Route("/api/graph/nodes/{node_id:str}", graph_update_node, methods=["PATCH"]),
            Route("/api/graph/nodes/{node_id:str}", graph_delete_node, methods=["DELETE"]),
            Route("/api/graph/edges", graph_create_edge, methods=["POST"]),
            Route("/api/graph/edges/{edge_id:str}", graph_update_edge, methods=["PATCH"]),
            Route("/api/graph/edges/{edge_id:str}", graph_delete_edge, methods=["DELETE"]),
            Route("/api/graph/export", graph_export, methods=["GET"]),
            Route("/api/graph/import", graph_import, methods=["POST"]),
            Mount("/graph-assets", app=StaticFiles(packages=[("waggle", "static/graph")], html=False, check_dir=False)),
            Mount("/mcp", app=service.mcp_asgi),
        ],
        lifespan=service.lifespan,
    )
    return app


def _run_graph_editor_command(config: AppConfig, args: argparse.Namespace) -> int:
    host = str(getattr(args, "host", "") or config.http_host or "127.0.0.1")
    port = int(getattr(args, "port", 8686) or 8686)
    should_open = bool(getattr(args, "open", True))
    mode = str(getattr(args, "command", "edit-graph")).strip().lower()
    page_mode = "view" if mode == "view-graph" else "edit"

    config.transport = "http"
    config.http_host = host
    config.http_port = port
    app_server = WaggleServer(config=config)
    http_app = create_http_application(app_server, config)
    url = f"http://{host}:{port}/graph?mode={page_mode}"

    print(f"Launching Waggle Graph Studio at {url}")
    print("Use Ctrl+C in this terminal to stop the editor server.")

    if should_open:
        def _open_browser() -> None:
            try:
                webbrowser.open(url)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("graph_editor_open_failed", extra={"error": str(exc), "url": url})

        timer = threading.Timer(0.4, _open_browser)
        timer.daemon = True
        timer.start()

    uvicorn.run(http_app, host=host, port=port, log_level=config.log_level.lower())
    return 0


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
    graph = app._root_graph
    em = graph.embedding_model
    if (
        not config.is_fast_mode
        and hasattr(em, "start_background_warmup")
        and not getattr(em, "_warmup_started", False)
    ):
        # Kick off background warmup so the first semantic call is fast.
        em.start_background_warmup()
    if config.is_strict_mode:
        # Block until the model is ready before accepting any requests.
        LOGGER.info("stdio_strict_mode_waiting_for_embedding", extra={"model": em.model_name})
        if hasattr(em, "_ready_event"):
            em._ready_event.wait(timeout=120.0)
        LOGGER.info(
            "stdio_strict_mode_embedding_status",
            extra={"status": getattr(em, "warmup_status", "unknown"), "error": getattr(em, "warmup_error", "")},
        )
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
  waggle-mcp setup --yes

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
        epilog="Examples: 'waggle-mcp setup --yes', 'waggle-mcp serve', 'waggle-mcp features'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the MCP server using the configured stdio or HTTP transport.")
    graph_editor = subparsers.add_parser(
        "edit-graph",
        help="Launch the visual graph editor in a browser window.",
        description=(
            "Start the Waggle HTTP app for local graph editing and open /graph in the browser by default. "
            "Use this for mouse-and-keyboard graph editing: add, remove, connect, reposition, import, and export."
        ),
    )
    graph_editor.add_argument("--host", default="127.0.0.1")
    graph_editor.add_argument("--port", type=int, default=8686)
    graph_editor.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)

    graph_viewer = subparsers.add_parser(
        "view-graph",
        help="Launch the visual graph viewer/editor in a browser window.",
        description="Alias for edit-graph. Starts the local graph UI and opens it in the browser by default.",
    )
    graph_viewer.add_argument("--host", default="127.0.0.1")
    graph_viewer.add_argument("--port", type=int, default=8686)
    graph_viewer.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)

    graph_ui = subparsers.add_parser(
        "ui",
        help="Launch the local graph UI in the browser.",
        description="Alias for edit-graph. Starts the localhost Graph Studio and opens it by default.",
    )
    graph_ui.add_argument("--host", default="127.0.0.1")
    graph_ui.add_argument("--port", type=int, default=8686)
    graph_ui.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)

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

    export_abhi = subparsers.add_parser(
        "export",
        help="Export the current memory graph in a portable format.",
    )
    export_abhi.add_argument("--output", dest="output_path", default=None)
    export_abhi.add_argument("--project", default="")
    export_abhi.add_argument("--agent-id", default="")
    export_abhi.add_argument("--session-id", default="")
    export_abhi.add_argument("--scope", choices=["all", "project", "session", "since-date"], default="all")
    export_abhi.add_argument("--since-date", default="")
    export_abhi.add_argument("--include-embeddings", action=argparse.BooleanOptionalAction, default=True)
    export_abhi.add_argument("--encrypt", action="store_true")
    export_abhi.add_argument("--sign", action="store_true")
    export_abhi.add_argument("--signing-key-dir", default="~/.waggle/keys")
    export_abhi.add_argument("--redact", dest="redact_patterns", action="append", default=[])
    export_abhi.add_argument("--passphrase-env", default="")
    export_abhi.add_argument("--force", action="store_true", help="Export even if transcript secret scan finds likely credentials or tokens.")

    import_abhi = subparsers.add_parser(
        "import",
        help="Import a portable memory file into the active backend.",
    )
    import_abhi.add_argument("input_path", nargs="?")
    import_abhi.add_argument("--input", dest="input_path_flag", default="")
    import_abhi.add_argument("--namespace", default="")
    import_abhi.add_argument("--merge-strategy", choices=["skip-existing", "overwrite", "branch"], default="skip-existing")
    import_abhi.add_argument("--verify-signature", action="store_true")
    import_abhi.add_argument("--read-only", action="store_true")
    import_abhi.add_argument("--reembed-on-mismatch", action="store_true")
    import_abhi.add_argument("--passphrase-env", default="")

    validate_abhi = subparsers.add_parser(
        "validate",
        help="Validate a portable .abhi memory file.",
    )
    validate_abhi.add_argument("--input", dest="input_path", required=True)
    validate_abhi.add_argument("--passphrase-env", default="")

    inspect_abhi = subparsers.add_parser(
        "inspect",
        help="Inspect an .abhi memory file without importing it.",
    )
    inspect_abhi.add_argument("--input", dest="input_path", required=True)
    inspect_abhi.add_argument("--passphrase-env", default="")

    diff_abhi = subparsers.add_parser(
        "diff",
        help="Compare two .abhi memory files.",
    )
    diff_abhi.add_argument("input_path_a", nargs="?")
    diff_abhi.add_argument("input_path_b", nargs="?")
    diff_abhi.add_argument("--file-a", dest="input_path_a_flag", default="")
    diff_abhi.add_argument("--file-b", dest="input_path_b_flag", default="")

    merge_abhi = subparsers.add_parser(
        "merge",
        help="Three-way merge .abhi memory files.",
    )
    merge_abhi.add_argument("left_input_path", nargs="?")
    merge_abhi.add_argument("right_input_path", nargs="?")
    merge_abhi.add_argument("--base", dest="base_input_path", default="")
    merge_abhi.add_argument("--left", dest="left_input_path_flag", default="")
    merge_abhi.add_argument("--right", dest="right_input_path_flag", default="")
    merge_abhi.add_argument("--output", dest="output_path", required=True)
    merge_abhi.add_argument("--merge-strategy", choices=["prefer_right", "prefer_left", "last_write_wins"], default="prefer_right")

    query_abhi = subparsers.add_parser(
        "query",
        help="Execute a query against an .abhi memory file.",
    )
    query_abhi.add_argument("--input", dest="input_path", required=True)
    query_abhi.add_argument("--query-id", default="")
    query_abhi.add_argument("--query-text", default="")
    query_abhi.add_argument("--passphrase-env", default="")

    load_abhi_chunks = subparsers.add_parser(
        "load-chunks",
        help="Load selected or query-relevant chunks from an .abhi memory file.",
    )
    load_abhi_chunks.add_argument("--input", dest="input_path", required=True)
    load_abhi_chunks.add_argument("--chunk-id", dest="chunk_ids", action="append", default=[])
    load_abhi_chunks.add_argument("--query-id", default="")
    load_abhi_chunks.add_argument("--query-text", default="")
    load_abhi_chunks.add_argument("--passphrase-env", default="")

    push_abhi = subparsers.add_parser(
        "push",
        help="Export the current graph to .abhi and upload it to Google Drive.",
    )
    push_abhi.add_argument("--drive", action="store_true")
    push_abhi.add_argument("--output", dest="output_path", default=None)
    push_abhi.add_argument("--project", default="")
    push_abhi.add_argument("--agent-id", default="")
    push_abhi.add_argument("--session-id", default="")
    push_abhi.add_argument("--scope", choices=["all", "project", "session", "since-date"], default="all")
    push_abhi.add_argument("--since-date", default="")
    push_abhi.add_argument("--include-embeddings", action=argparse.BooleanOptionalAction, default=True)
    push_abhi.add_argument("--encrypt", action=argparse.BooleanOptionalAction, default=True)
    push_abhi.add_argument("--passphrase-env", default="")
    push_abhi.add_argument("--force", action="store_true", help="Export/upload even if transcript secret scan finds likely credentials or tokens.")
    push_abhi.add_argument("--folder-id", default="")
    push_abhi.add_argument("--remote-name", default="")
    push_abhi.add_argument("--client-secret-path", default="~/.waggle/google-client-secret.json")
    push_abhi.add_argument("--token-path", default="")
    push_abhi.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)

    pull_abhi = subparsers.add_parser(
        "pull",
        help="Download a Google Drive .abhi file and merge it into the current graph.",
    )
    pull_abhi.add_argument("file_ref")
    pull_abhi.add_argument("--folder-id", default="")
    pull_abhi.add_argument("--download-path", default="")
    pull_abhi.add_argument("--merged-output", default="")
    pull_abhi.add_argument("--passphrase-env", default="")
    pull_abhi.add_argument("--client-secret-path", default="~/.waggle/google-client-secret.json")
    pull_abhi.add_argument("--token-path", default="")
    pull_abhi.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    pull_abhi.add_argument("--namespace", default="")
    pull_abhi.add_argument("--merge-strategy", choices=["skip-existing", "overwrite", "branch"], default="skip-existing")
    pull_abhi.add_argument("--verify-signature", action="store_true")
    pull_abhi.add_argument("--read-only", action="store_true")
    pull_abhi.add_argument("--reembed-on-mismatch", action="store_true")

    share_abhi = subparsers.add_parser(
        "share",
        help="Create an anyone-with-link share URL for a Google Drive file.",
    )
    share_abhi.add_argument("file_ref")
    share_abhi.add_argument("--folder-id", default="")
    share_abhi.add_argument("--client-secret-path", default="~/.waggle/google-client-secret.json")
    share_abhi.add_argument("--token-path", default="")
    share_abhi.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)

    benchmark_oolong = subparsers.add_parser(
        "benchmark-oolong",
        help="Run OOLONG long-context evaluation over Waggle retrieval.",
        description=(
            "Load a local OOLONG JSON/JSONL export, index context windows into Waggle, "
            "retrieve scoped context per question, and optionally score an external LLM over that retrieved bundle."
        ),
    )
    benchmark_oolong.add_argument("dataset_path")
    benchmark_oolong.add_argument("--dataset-kind", choices=["auto", "real", "synth", "custom"], default="auto")
    benchmark_oolong.add_argument("--context-field", default="auto")
    benchmark_oolong.add_argument("--eval-mode", choices=["retrieval_only", "waggle_llm", "waggle_rlm"], default="retrieval_only")
    benchmark_oolong.add_argument("--llm-command", default="")
    benchmark_oolong.add_argument("--llm-backend", choices=["command", "groq", "gemini"], default="command")
    benchmark_oolong.add_argument("--llm-model", default="llama-3.3-70b-versatile")
    benchmark_oolong.add_argument("--llm-api-key-env", default="GROQ_API_KEY")
    benchmark_oolong.add_argument("--llm-max-tokens", type=int, default=512)
    benchmark_oolong.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    benchmark_oolong.add_argument("--retrieval-mode", choices=["graph", "hybrid", "verbatim", "aggregate"], default="graph")
    benchmark_oolong.add_argument("--max-nodes", type=int, default=8)
    benchmark_oolong.add_argument("--max-depth", type=int, default=1)
    benchmark_oolong.add_argument("--chunk-lines", type=int, default=12)
    benchmark_oolong.add_argument("--chunk-overlap-lines", type=int, default=3)
    benchmark_oolong.add_argument("--rlm-system-prompt-file", default="")
    benchmark_oolong.add_argument("--rlm-max-iterations", type=int, default=6)
    benchmark_oolong.add_argument("--limit", type=int, default=None)
    benchmark_oolong.add_argument("--output", default="")

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
    export_context_bundle.add_argument("--retrieval-mode", choices=["graph", "verbatim", "hybrid"], default="graph")
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

    backfill_windows = subparsers.add_parser(
        "backfill-windows",
        help="Retroactively assign legacy nodes to context windows grouped by project/session.",
    )
    backfill_windows.add_argument(
        "--dry-run",
        action="store_true",
        help="Show backfill stats without assigning nodes or creating windows.",
    )

    ingest_transcript_handoff = subparsers.add_parser(
        "ingest-transcript-handoff",
        help="Ingest a full session transcript as a rollover handoff, extract memory, and export a context bundle.",
        description=(
            "Client-triggered rollover handoff: pass the full ordered transcript as JSON, "
            "Waggle stores all messages as transcript provenance, extracts durable memory from "
            "logical user->assistant turns, and exports a session-scoped context bundle. "
            "Supported backend: SQLite only in v1. Neo4j support is deferred."
        ),
    )
    ingest_transcript_handoff.add_argument(
        "--input",
        default="-",
        metavar="PATH_OR_DASH",
        help="Path to the JSON transcript file, or '-' to read from stdin (default: stdin).",
    )
    ingest_transcript_handoff.add_argument("--project", default="", help="Scope override: project name.")
    ingest_transcript_handoff.add_argument("--agent-id", default="", help="Scope override: agent identifier.")
    ingest_transcript_handoff.add_argument("--session-id", default="", help="Scope override: session identifier.")
    ingest_transcript_handoff.add_argument("--output-path", default=None, help="Optional export output path prefix.")
    ingest_transcript_handoff.add_argument(
        "--export-format",
        choices=["markdown", "json", "both"],
        default="both",
        help="Export format for the post-ingestion context bundle (default: both).",
    )
    ingest_transcript_handoff.add_argument(
        "--max-nodes",
        type=int,
        default=25,
        help="Maximum nodes included in the exported context bundle (default: 25).",
    )
    ingest_transcript_handoff.add_argument(
        "--max-input-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="Hard input-size cap in bytes (default: 16 MiB). Oversized payloads fail with exit code 1.",
    )

    setup = subparsers.add_parser(
        "setup",
        help="Non-interactive one-line setup — auto-patch supported MCP clients.",
        description=(
            "Patch supported MCP client config files without prompts. "
            "By default, --clients auto updates detected clients; use --yes to run from install scripts."
        ),
    )
    setup.add_argument(
        "--yes",
        action="store_true",
        help="Confirm non-interactive setup. Required unless --dry-run is used.",
    )
    setup.add_argument(
        "--clients",
        default="auto",
        help=(
            "Comma-separated clients to configure, or 'auto'. "
            "Supported: codex, claude-desktop, cursor, gemini, antigravity, other."
        ),
    )
    setup.add_argument("--db", default="", help="Database path. Default: ~/.waggle/memory.db")
    setup.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model for client env. Use 'deterministic' for offline-safe startup.",
    )
    setup.add_argument(
        "--project-instructions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write supported project instruction files, currently Codex AGENTS.md.",
    )
    setup.add_argument("--dry-run", action="store_true", help="Show what would change without writing files.")
    setup.add_argument("--run-doctor", action=argparse.BooleanOptionalAction, default=True)

    subparsers.add_parser("init", help="Interactive setup wizard — configure an MCP client to use waggle-mcp.")
    subparsers.add_parser(
        "features",
        help="Explain the main tools, graph workflows, and how connected context reaches the model.",
        description="Print a detailed guide to the waggle-mcp feature surface.",
    )
    doctor = subparsers.add_parser(
        "doctor",
        help="Check your Waggle installation: config files, embedding model status, DB path, and common mistakes.",
        description=(
            "Inspect the waggle-mcp environment and surface any configuration issues before they become runtime errors. "
            "Checks: config file locations per MCP client, embedding model cache, DB path writability, "
            "WAGGLE_STARTUP_MODE, stdout encoding (Windows), and known API gotchas."
        ),
    )
    doctor.add_argument("--fix", action="store_true", help="Re-embed stale transcript/node rows to the current model ID.")
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
    if args.command == "export":
        _assert_export_safe(
            backend,
            force=bool(getattr(args, "force", False)),
            project=getattr(args, "project", ""),
            agent_id=getattr(args, "agent_id", ""),
            session_id=getattr(args, "session_id", ""),
            scope=getattr(args, "scope", "all"),
            since_date=getattr(args, "since_date", ""),
        )
        exported = backend.export_abhi(
            output_path=args.output_path,
            project=getattr(args, "project", ""),
            agent_id=getattr(args, "agent_id", ""),
            session_id=getattr(args, "session_id", ""),
            scope=getattr(args, "scope", "all"),
            since_date=getattr(args, "since_date", ""),
            include_embeddings=bool(getattr(args, "include_embeddings", True)),
            passphrase=_resolve_passphrase(args),
            redact_patterns=list(getattr(args, "redact_patterns", []) or []),
            sign=bool(getattr(args, "sign", False)),
            signing_key_dir=getattr(args, "signing_key_dir", "~/.waggle/keys"),
        )
        print(json.dumps(exported.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "import":
        input_path = getattr(args, "input_path", None) or getattr(args, "input_path_flag", "")
        imported = backend.import_abhi(
            input_path=input_path,
            passphrase=_resolve_passphrase(args),
            namespace=getattr(args, "namespace", ""),
            merge_strategy=getattr(args, "merge_strategy", "skip-existing"),
            verify_signature=bool(getattr(args, "verify_signature", False)),
            read_only=bool(getattr(args, "read_only", False)),
            reembed_on_mismatch=bool(getattr(args, "reembed_on_mismatch", False)),
        )
        print(json.dumps(imported.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "validate":
        validation = backend.validate_abhi(input_path=args.input_path, passphrase=_resolve_passphrase(args))
        print(json.dumps(validation.model_dump(mode="json"), indent=2))
        return 0 if validation.valid else 1
    if args.command == "inspect":
        inspected = backend.inspect_abhi(input_path=args.input_path, passphrase=_resolve_passphrase(args))
        print(json.dumps(inspected.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "diff":
        input_path_a = getattr(args, "input_path_a", None) or getattr(args, "input_path_a_flag", "")
        input_path_b = getattr(args, "input_path_b", None) or getattr(args, "input_path_b_flag", "")
        diff = backend.diff_abhi(input_path_a=input_path_a, input_path_b=input_path_b)
        print(json.dumps(diff.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "merge":
        left_input_path = getattr(args, "left_input_path_flag", "") or args.left_input_path
        right_input_path = getattr(args, "right_input_path_flag", "") or args.right_input_path
        merged = backend.merge_abhi(
            base_input_path=args.base_input_path or left_input_path,
            left_input_path=left_input_path,
            right_input_path=right_input_path,
            output_path=args.output_path,
            merge_strategy=args.merge_strategy,
        )
        print(json.dumps(merged.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "query":
        queried = backend.query_abhi(
            input_path=args.input_path,
            query_id=getattr(args, "query_id", ""),
            query_text=getattr(args, "query_text", ""),
            passphrase=_resolve_passphrase(args),
        )
        print(json.dumps(queried.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "load-chunks":
        loaded = backend.load_abhi_chunks(
            input_path=args.input_path,
            chunk_ids=getattr(args, "chunk_ids", []),
            query_id=getattr(args, "query_id", ""),
            query_text=getattr(args, "query_text", ""),
            passphrase=_resolve_passphrase(args),
        )
        print(json.dumps(loaded.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "push":
        _require_drive_sync()
        passphrase = _resolve_passphrase(args)
        _assert_export_safe(
            backend,
            force=bool(getattr(args, "force", False)),
            project=getattr(args, "project", ""),
            agent_id=getattr(args, "agent_id", ""),
            session_id=getattr(args, "session_id", ""),
            scope=getattr(args, "scope", "all"),
            since_date=getattr(args, "since_date", ""),
        )
        exported = backend.export_abhi(
            output_path=args.output_path,
            project=getattr(args, "project", ""),
            agent_id=getattr(args, "agent_id", ""),
            session_id=getattr(args, "session_id", ""),
            scope=getattr(args, "scope", "all"),
            since_date=getattr(args, "since_date", ""),
            include_embeddings=bool(getattr(args, "include_embeddings", True)),
            passphrase=passphrase,
        )
        credentials = ensure_drive_credentials(
            client_secret_path=args.client_secret_path,
            token_path=_resolve_drive_token_path(args, config),
            open_browser=bool(getattr(args, "open_browser", True)),
        )
        pushed = push_file_to_drive(
            local_path=exported.output_path,
            folder_id=str(getattr(args, "folder_id", "") or ""),
            credentials=credentials,
            remote_name=str(getattr(args, "remote_name", "") or ""),
            encrypted=bool(passphrase),
        )
        print(json.dumps(pushed.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "pull":
        _require_drive_sync()
        passphrase = _resolve_passphrase(args)
        credentials = ensure_drive_credentials(
            client_secret_path=args.client_secret_path,
            token_path=_resolve_drive_token_path(args, config),
            open_browser=bool(getattr(args, "open_browser", True)),
        )
        remote_file_id, remote_name = resolve_drive_file_id(
            file_ref=args.file_ref,
            credentials=credentials,
            folder_id=str(getattr(args, "folder_id", "") or ""),
        )
        download_path = Path(getattr(args, "download_path", "") or backend.export_dir / f"{remote_file_id}.abhi")
        _, resolved_name = download_drive_file(
            file_id=remote_file_id,
            destination_path=download_path,
            credentials=credentials,
        )
        local_document = build_abhi_document(backend.get_graph_snapshot())
        remote_document = load_abhi_document(download_path, passphrase=passphrase)
        merged_output = Path(getattr(args, "merged_output", "") or backend.export_dir / f"merged-{remote_file_id}.abhi")
        merged_path = merge_downloaded_abhi(
            local_document=local_document,
            remote_document=remote_document,
            output_path=merged_output,
        )
        imported = backend.import_abhi(
            input_path=merged_path,
            passphrase=passphrase,
            namespace=getattr(args, "namespace", ""),
            merge_strategy=getattr(args, "merge_strategy", "skip-existing"),
            verify_signature=bool(getattr(args, "verify_signature", False)),
            read_only=bool(getattr(args, "read_only", False)),
            reembed_on_mismatch=bool(getattr(args, "reembed_on_mismatch", False)),
        )
        result = DrivePullResult(
            remote_file_id=remote_file_id,
            remote_name=resolved_name or remote_name,
            downloaded_path=str(download_path),
            merged_output_path=merged_path,
            merge_strategy="last_write_wins",
            nodes_created=imported.nodes_created,
            nodes_updated=imported.nodes_updated,
            edges_created=imported.edges_created,
            edges_updated=imported.edges_updated,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "share":
        _require_drive_sync()
        credentials = ensure_drive_credentials(
            client_secret_path=args.client_secret_path,
            token_path=_resolve_drive_token_path(args, config),
            open_browser=bool(getattr(args, "open_browser", True)),
        )
        remote_file_id, _ = resolve_drive_file_id(
            file_ref=args.file_ref,
            credentials=credentials,
            folder_id=str(getattr(args, "folder_id", "") or ""),
        )
        shared = share_drive_file(file_id=remote_file_id, credentials=credentials)
        print(json.dumps(shared.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "benchmark-oolong":
        llm_answerer = None
        rlm_backend = "openai"
        rlm_backend_kwargs: dict[str, Any] | None = None
        rlm_mock_response_fn = None
        if args.llm_backend == "command":
            if args.llm_command:
                from waggle.oolong_benchmark import run_subprocess_llm

                llm_answerer = lambda prompt: run_subprocess_llm(prompt, command_template=args.llm_command)
                rlm_mock_response_fn = build_subprocess_response_fn(args.llm_command)
        elif args.llm_backend == "groq":
            api_key = os.environ.get(args.llm_api_key_env, "").strip()
            if not api_key:
                raise ValidationFailure(f"{args.llm_api_key_env} is required for --llm-backend groq.")
            rlm_backend_kwargs = build_groq_openai_backend_kwargs(api_key=api_key, model_name=args.llm_model)
            llm_answerer = lambda prompt: run_groq_one_shot(
                prompt=prompt,
                api_key=api_key,
                model_name=args.llm_model,
                max_tokens=args.llm_max_tokens,
                timeout_seconds=args.llm_timeout_seconds,
            )
        elif args.llm_backend == "gemini":
            api_key = os.environ.get(args.llm_api_key_env, "").strip()
            if not api_key:
                raise ValidationFailure(f"{args.llm_api_key_env} is required for --llm-backend gemini.")
            rlm_backend = "gemini"
            rlm_backend_kwargs = build_gemini_backend_kwargs(api_key=api_key, model_name=args.llm_model)
            llm_answerer = lambda prompt: run_gemini_one_shot(
                prompt=prompt,
                api_key=api_key,
                model_name=args.llm_model,
                timeout_seconds=args.llm_timeout_seconds,
            )

        rlm_system_prompt = DEFAULT_WAGGLE_RLM_SYSTEM_PROMPT
        if args.rlm_system_prompt_file:
            rlm_system_prompt = Path(args.rlm_system_prompt_file).read_text(encoding="utf-8")

        report = evaluate_oolong(
            args.dataset_path,
            dataset_kind=args.dataset_kind,
            context_field=args.context_field,
            embedding_model=EmbeddingModel(config.model_name),
            eval_mode=args.eval_mode,
            retrieval_mode=args.retrieval_mode,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            chunk_lines=args.chunk_lines,
            chunk_overlap_lines=args.chunk_overlap_lines,
            limit=args.limit,
            llm_answerer=llm_answerer,
            rlm_system_prompt=rlm_system_prompt,
            rlm_max_iterations=args.rlm_max_iterations,
            rlm_backend=rlm_backend,
            rlm_backend_kwargs=rlm_backend_kwargs,
            rlm_mock_response_fn=rlm_mock_response_fn,
        )
        payload = report.to_dict()
        if args.output:
            Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
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
    if args.command == "backfill-windows":
        from waggle.backfill import backfill_context_windows

        if not isinstance(backend, MemoryGraph):
            raise ValidationFailure("backfill-windows is currently supported only for the SQLite backend.")
        stats = backfill_context_windows(backend, dry_run=bool(args.dry_run))
        print(json.dumps(stats.model_dump(mode="json"), indent=2))
        return 1 if stats.errors else 0
    if args.command == "ingest-transcript-handoff":
        return _run_ingest_transcript_handoff(config, args)
    if args.command == "features":
        print(_FEATURES_GUIDE)
        return 0
    if args.command == "doctor":
        return _run_doctor(config, fix=bool(getattr(args, "fix", False)))
    raise ValidationFailure(f"Unknown command: {args.command}")


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------

_KNOWN_CONFIG_PATHS: list[tuple[str, str]] = [
    # (label, path_template)
    # %APPDATA% and ~ are expanded at runtime
    ("Claude Desktop (macOS/Linux)", "~/.config/claude/claude_desktop_config.json"),
    ("Claude Desktop (macOS alt)", "~/Library/Application Support/Claude/claude_desktop_config.json"),
    ("Claude Desktop (Windows)", "%APPDATA%\\Claude\\claude_desktop_config.json"),
    ("Cursor (macOS/Linux)", "~/.cursor/mcp.json"),
    ("Cursor (Windows)", "%APPDATA%\\Cursor\\User\\mcp.json"),
    ("Antigravity AI agent (macOS/Linux)", "~/.gemini/antigravity/mcp_config.json"),
    ("Antigravity AI agent (Windows)", "%USERPROFILE%\\.gemini\\antigravity\\mcp_config.json"),
    ("VS Code extension (Windows)", "%APPDATA%\\Antigravity\\User\\mcp.json"),
    ("Codex", "~/.codex/config.toml"),
]

_DOCTOR_KNOWN_GOTCHAS = """\
Known API gotchas (from the Waggle field error log):
  • observe_conversation requires fields 'user_message' and 'assistant_response'.
    Do NOT use 'user_text' / 'assistant_text' — those will be rejected.
  • get_topics does not support scope filtering. Passing 'project' etc. used to
    cause "additional properties" errors. It now accepts (and ignores) scope fields.
  • Use the official 'mcp' Python package for stdio clients, not hand-rolled JSON-RPC.
    Install: pip install mcp
    Use:     from mcp import ClientSession, StdioServerParameters
             from mcp.client.stdio import stdio_client
  • On Windows, stdio framing requires the official mcp client; raw subprocess I/O
    with manual \\n line endings will produce garbled or dropped messages.
  • Set WAGGLE_MODEL=deterministic for offline/offline-first environments.
    The default (all-MiniLM-L6-v2) downloads ~420 MB on first run and will
    block store_node indefinitely if no network is available.
"""


def _run_doctor(config: AppConfig, *, fix: bool = False) -> int:
    """waggle-mcp doctor — surface configuration and environment issues."""
    issues: list[str] = []
    ok_items: list[str] = []

    print()
    print(_c(_BOLD, "waggle-mcp doctor"))
    print(_c(_CYAN, "─" * 50))

    # ── 1. Config file locations ─────────────────────────────────────────────
    print(_c(_BOLD, "\n[1] MCP client config files"))
    waggle_found_in: list[str] = []
    for label, template in _KNOWN_CONFIG_PATHS:
        raw = template.replace("%APPDATA%", os.environ.get("APPDATA", "")).replace(
            "%USERPROFILE%", str(Path.home())
        )
        path = Path(raw).expanduser()
        if path.exists():
            try:
                raw_text = path.read_text(encoding="utf-8", errors="replace")
                if path.suffix == ".toml":
                    has_waggle = "[mcp_servers.waggle]" in raw_text
                else:
                    data = json.loads(raw_text)
                    servers = data.get("mcpServers", data.get("tools", {}) if isinstance(data, dict) else {})
                    has_waggle = isinstance(servers, dict) and "waggle" in servers
                if has_waggle:
                    waggle_found_in.append(label)
                    _ok(f"{label}\n     {path}  [waggle entry found]")
                else:
                    print(f"  {_c(_CYAN, chr(0x2022))} {label}\n     {path}  [exists, no waggle entry]")
            except Exception:
                print(f"  {_c(_CYAN, chr(0x2022))} {label}\n     {path}  [exists, could not parse]")
        # only show missing paths that are plausible for this platform
        elif (sys.platform == "darwin" and ("macOS" in label or "Cursor" in label or "Antigravity" in label or "Codex" in label)) or \
             (sys.platform == "win32" and "Windows" in label) or \
             (sys.platform.startswith("linux") and ("Linux" in label or "Cursor" in label)):
            print(f"  {_c(_CYAN, chr(0x2022))} {label}\n     {path}  [not found]")

    if not waggle_found_in:
        issues.append(
            "No MCP client config file contains a 'waggle' server entry. "
            "Run 'waggle-mcp setup --yes' to create one, or add it manually."
        )
    else:
        ok_items.append(f"Waggle found in: {', '.join(waggle_found_in)}")

    # ── 2. DB path ───────────────────────────────────────────────────────────
    print(_c(_BOLD, "\n[2] Database path"))
    db_path = Path(config.db_path)
    db_dir = db_path.parent
    if db_path.exists():
        _ok(f"DB file exists: {db_path}")
        ok_items.append("DB file found")
    elif db_dir.exists():
        _ok(f"DB directory exists (file will be created on first run): {db_path}")
        ok_items.append("DB directory writable")
    else:
        issues.append(
            f"DB directory does not exist: {db_dir}. "
            "Create it with: mkdir -p <dir>"
        )
        _fail(f"DB directory missing: {db_dir}")

    # ── 3. Embedding model ───────────────────────────────────────────────────
    print(_c(_BOLD, "\n[3] Embedding model"))
    model_name = config.model_name
    hf_home = os.environ.get("HF_HOME") or os.environ.get("SENTENCE_TRANSFORMERS_HOME") or str(Path.home() / ".cache" / "huggingface")
    st_cache = Path(os.environ.get("SENTENCE_TRANSFORMERS_HOME", Path(hf_home) / "hub"))

    if model_name.strip().lower() in {"fake", "fake-model", "deterministic", "offline-demo"}:
        _ok(f"Model: {model_name!r}  (deterministic — no download, always offline-safe)")
        ok_items.append("Deterministic model — no download needed")
    else:
        # Heuristic: look for a cached sentence-transformers directory
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        possible_dirs = [
            st_cache / f"models--{safe_name.replace('_', '--', 1)}",
            st_cache / safe_name,
            Path(hf_home) / "hub" / f"models--{safe_name.replace('_', '--', 1)}",
        ]
        cached = any(p.exists() for p in possible_dirs)
        if cached:
            _ok(f"Model: {model_name!r}  (cached locally — fast startup)")
            ok_items.append("Embedding model cached")
        else:
            print(
                f"  {_c(_CYAN, chr(0x2139))} Model: {model_name!r}  — NOT found in local cache.\n"
                f"    First store_node/query_graph call will download ~420 MB from HuggingFace.\n"
                f"    To avoid: set WAGGLE_MODEL=deterministic, or pre-download with:\n"
                f"      python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\""
            )
            issues.append(
                f"Embedding model '{model_name}' not found in cache. "
                "First semantic call will block for a network download. "
                "Set WAGGLE_MODEL=deterministic for offline-safe mode."
            )

    # ── 4. WAGGLE_STARTUP_MODE ───────────────────────────────────────────────
    print(_c(_BOLD, "\n[4] Embedding store"))
    graph = _default_graph(config)
    if isinstance(graph, MemoryGraph):
        store_health = graph.get_embedding_store_health()
        if fix and (store_health["transcript_stale_rows"] or store_health["node_stale_rows"]):
            repair = graph.reembed_stale_embeddings(batch_size=100)
            _ok(
                "Re-embedded stale rows: "
                f"{repair['transcript_rows_updated']} transcript rows, {repair['node_rows_updated']} node rows."
            )
            store_health = graph.get_embedding_store_health()
        transcript_models = store_health["transcript_model_counts"]
        node_models = store_health["node_model_counts"]
        print(f"  Current model id: {store_health['current_model_id']}")
        print(f"  Transcript model ids: {transcript_models or '{}'}")
        print(f"  Node model ids: {node_models or '{}'}")
        print(f"  Stale transcript rows: {store_health['transcript_stale_rows']}")
        print(f"  Stale node rows: {store_health['node_stale_rows']}")
        if store_health["mixed_models"]:
            issues.append("Mixed embedding_model_id values detected in the store.")
            _fail("Mixed embedding model IDs detected across transcript_records/nodes.")
        else:
            _ok("Store model IDs are consistent.")

    # ── 5. WAGGLE_STARTUP_MODE ───────────────────────────────────────────────
    print(_c(_BOLD, "\n[5] Startup mode"))
    print(f"  WAGGLE_STARTUP_MODE = {config.startup_mode!r}")
    if config.is_fast_mode:
        _ok("fast mode: zero ML overhead. Schema/tool listing only. Semantic tools return 'unavailable'.")
    elif config.is_strict_mode:
        _ok("strict mode: server blocks on startup until embedding model is ready.")
    else:
        _ok("normal mode: embedding loads in background. First semantic call may wait up to ~30 s.")

    # ── 6. Windows stdout encoding ───────────────────────────────────────────
    if sys.platform == "win32":
        print(_c(_BOLD, "\n[6] Windows stdout encoding"))
        enc = getattr(sys.stdout, "encoding", "unknown")
        if enc.lower().replace("-", "") in ("utf8", "utf8"):
            _ok(f"stdout encoding: {enc}")
        else:
            _fail(
                f"stdout encoding is {enc!r} (not UTF-8). "
                "Unicode characters (emoji, accented text) will cause UnicodeEncodeError.\n"
                "    Fix: run with 'python -X utf8' or add at script top:\n"
                "      import sys; sys.stdout.reconfigure(encoding='utf-8')"
            )
            issues.append(f"Windows stdout encoding is {enc!r} — set PYTHONUTF8=1 or use python -X utf8.")

    # ── 7. Known gotchas ─────────────────────────────────────────────────────
    print(_c(_BOLD, "\n[7] Known API gotchas"))
    print(_DOCTOR_KNOWN_GOTCHAS)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(_c(_BOLD, "─" * 50))
    if issues:
        print(_c(_RED, f"Found {len(issues)} issue(s):"))
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        return 1
    else:
        print(_c(_GREEN, f"All checks passed ({len(ok_items)} OK). Waggle looks healthy."))
        print()
        return 0


# ---------------------------------------------------------------------------
# ingest-transcript-handoff CLI helpers
# ---------------------------------------------------------------------------


def _emit_cli_error(code: str, message: str, details: dict[str, Any]) -> None:
    """Write a structured failure JSON object to stderr."""
    payload = {"code": code, "message": message, "details": details}
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.stderr.flush()


def _run_ingest_transcript_handoff(config: AppConfig, args: argparse.Namespace) -> int:
    """Execute the ingest-transcript-handoff CLI command.

    Exit codes:
      0  — success
      1  — input or validation failure
      2  — backend / graph / export failure
      3  — unexpected internal error
    """
    max_bytes: int = args.max_input_bytes

    # ── Load raw input ──────────────────────────────────────────────────────
    try:
        input_arg: str = args.input
        if input_arg == "-":
            raw = sys.stdin.buffer.read(max_bytes + 1)
        else:
            path = Path(input_arg)
            if not path.exists():
                _emit_cli_error("input_not_found", f"Input file not found: {input_arg}", {})
                return 1
            raw = path.read_bytes()[: max_bytes + 1]
    except OSError as exc:
        _emit_cli_error("input_read_error", str(exc), {})
        return 1

    if len(raw) > max_bytes:
        _emit_cli_error(
            "payload_too_large",
            f"Input exceeds --max-input-bytes ({max_bytes} bytes).",
            {"max_input_bytes": max_bytes},
        )
        return 1

    # ── Parse JSON ──────────────────────────────────────────────────────────
    try:
        payload_dict = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        _emit_cli_error("malformed_json", f"JSON parse error: {exc}", {})
        return 1

    if not isinstance(payload_dict, dict):
        _emit_cli_error("invalid_format", "Input must be a JSON object.", {})
        return 1

    if "messages" not in payload_dict:
        _emit_cli_error("missing_field", "Input JSON must contain a 'messages' key.", {})
        return 1

    # ── Validate via Pydantic ───────────────────────────────────────────────
    try:
        payload = TranscriptIngestionInput.model_validate(payload_dict)
    except Exception as exc:
        _emit_cli_error("validation_error", str(exc), {})
        return 1

    # ── CLI scope flags override JSON fields ────────────────────────────────
    if getattr(args, "project", ""):
        payload.project = args.project.strip()
    if getattr(args, "agent_id", ""):
        payload.agent_id = args.agent_id.strip()
    if getattr(args, "session_id", ""):
        payload.session_id = args.session_id.strip()

    # ── Run ingestion ───────────────────────────────────────────────────────
    try:
        backend = _build_backend(config)
        result = backend.ingest_transcript_handoff(
            payload,
            export_format=args.export_format,
            output_path=args.output_path,
            max_nodes=args.max_nodes,
        )
    except ValidationFailure as exc:
        _emit_cli_error("validation_error", str(exc), {})
        return 1
    except WaggleError as exc:
        _emit_cli_error(exc.code, str(exc), {"status_code": exc.status_code})
        return 2
    except Exception as exc:
        _emit_cli_error("backend_error", str(exc), {"type": type(exc).__name__})
        return 2

    # ── Emit success JSON to stdout ─────────────────────────────────────────
    output: dict[str, Any] = {
        "scope": {
            "project": result.project,
            "agent_id": result.agent_id,
            "session_id": result.session_id,
        },
        "input_message_count": result.input_message_count,
        "transcript_records_written": result.transcript_records_written,
        "transcript_records_skipped": result.transcript_records_skipped,
        "logical_turns_processed": result.logical_turns_processed,
        "unpaired_trailing_blocks": result.unpaired_trailing_blocks,
        "nodes_created": result.nodes_created,
        "nodes_reused": result.nodes_reused,
        "conflicts": result.conflicts,
    }
    if result.export_skipped:
        output["export_skipped"] = True
        output["export_skipped_reason"] = result.export_skipped_reason
    else:
        output["export_skipped"] = False
        output["markdown_path"] = result.markdown_path
        output["json_path"] = result.json_path
        output["export_node_count"] = result.export_node_count
        output["export_edge_count"] = result.export_edge_count

    print(json.dumps(output, indent=2))
    return 0


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


def _write_gemini(db_path: str, python_exe: str) -> Path:
    """Write or update ~/.gemini/settings.json."""
    config_dir = Path.home() / ".gemini"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "settings.json"

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
        "trust": False,
    }
    config_file.write_text(json.dumps(existing, indent=2))
    return config_file


def _write_antigravity(db_path: str, python_exe: str) -> Path:
    """Write or update the Antigravity AI agent MCP config."""
    if sys.platform == "win32":
        config_dir = Path.home() / ".gemini" / "antigravity"
    else:
        config_dir = Path.home() / ".gemini" / "antigravity"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "mcp_config.json"

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
    """Write or update the Waggle MCP server block in ~/.codex/config.toml."""
    config_dir = Path.home() / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    toml_block = (
        '[mcp_servers.waggle]\n'
        f'command = "{python_exe}"\n'
        'args = ["-m", "waggle.server"]\n'
        '\n'
        '[mcp_servers.waggle.env]\n'
        'WAGGLE_TRANSPORT = "stdio"\n'
        'WAGGLE_BACKEND = "sqlite"\n'
        f'WAGGLE_DB_PATH = "{db_path}"\n'
        'WAGGLE_DEFAULT_TENANT_ID = "local-default"\n'
        'WAGGLE_MODEL = "all-MiniLM-L6-v2"\n'
    )
    existing = config_file.read_text() if config_file.exists() else ""
    pattern = re.compile(
        r"(?ms)^\[mcp_servers\.waggle\]\n.*?(?=^\[(?!mcp_servers\.waggle(?:\.env)?\])[^\n]+\]\n|\Z)"
    )
    replacement = toml_block.rstrip() + "\n"
    if pattern.search(existing):
        updated = pattern.sub(replacement, existing, count=1)
    else:
        separator = "\n\n" if existing.strip() else ""
        updated = existing.rstrip() + separator + replacement
    config_file.write_text(updated)
    return config_file


def _write_codex_agents(root_dir: Path | None = None) -> Path:
    """Write or update a Waggle-managed automatic-memory block in AGENTS.md."""
    project_root = (root_dir or Path.cwd()).resolve()
    agents_file = project_root / "AGENTS.md"
    existing = agents_file.read_text() if agents_file.exists() else ""
    pattern = re.compile(
        rf"(?ms)^\s*{re.escape(_AGENTS_MEMORY_BLOCK_HEADER)}\n.*?^\s*{re.escape(_AGENTS_MEMORY_BLOCK_FOOTER)}\n?"
    )

    block = _AGENTS_MEMORY_BLOCK
    if pattern.search(existing):
        updated = pattern.sub(block, existing, count=1)
    else:
        separator = "\n\n" if existing.strip() else ""
        updated = existing.rstrip() + separator + block
    agents_file.write_text(updated.rstrip() + "\n")
    return agents_file


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
    "Gemini CLI": _write_gemini,
    "Antigravity": _write_antigravity,
    "Codex": _write_codex,
    "Other": _write_other,
}

_RESTART_HINTS = {
    "Claude Desktop": "Restart Claude Desktop to activate.",
    "Cursor": "Reload the Cursor window (Cmd/Ctrl+Shift+P → 'Reload Window') to activate.",
    "Gemini CLI": "Restart Gemini CLI, then run /mcp to confirm Waggle is connected.",
    "Antigravity": "Restart Antigravity to activate the AI agent MCP config.",
    "Codex": "Restart Codex to activate.",
    "Other": "Add the JSON config to your MCP client's server list, then restart it.",
}

_CLIENT_ALIASES = {
    "claude": "Claude Desktop",
    "claude-desktop": "Claude Desktop",
    "claude_desktop": "Claude Desktop",
    "cursor": "Cursor",
    "gemini": "Gemini CLI",
    "gemini-cli": "Gemini CLI",
    "gemini_cli": "Gemini CLI",
    "antigravity": "Antigravity",
    "codex": "Codex",
    "other": "Other",
}


def _client_config_probe_paths() -> dict[str, list[Path]]:
    return {
        "Claude Desktop": [
            Path.home() / ".config" / "claude" / "claude_desktop_config.json",
            Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        ],
        "Cursor": [
            Path.home() / ".cursor" / "mcp.json",
            Path(os.environ.get("APPDATA", "")) / "Cursor" / "User" / "mcp.json",
        ],
        "Gemini CLI": [Path.home() / ".gemini" / "settings.json"],
        "Antigravity": [Path.home() / ".gemini" / "antigravity" / "mcp_config.json"],
        "Codex": [Path.home() / ".codex" / "config.toml"],
    }


def _normalize_setup_clients(raw_clients: str) -> list[str]:
    clients: list[str] = []
    for raw_client in raw_clients.split(","):
        key = raw_client.strip().lower()
        if not key:
            continue
        client = _CLIENT_ALIASES.get(key)
        if client is None:
            supported = ", ".join(sorted(_CLIENT_ALIASES))
            raise ValidationFailure(f"Unsupported setup client: {raw_client!r}. Supported values: auto, {supported}.")
        if client not in clients:
            clients.append(client)
    if not clients:
        raise ValidationFailure("No setup clients were provided.")
    return clients


def _detect_setup_clients() -> list[str]:
    detected: list[str] = []
    for client, paths in _client_config_probe_paths().items():
        if any(path.exists() for path in paths):
            detected.append(client)
    if not detected and (Path.cwd() / "AGENTS.md").exists():
        detected.append("Codex")
    return detected


def _setup_clients_from_args(raw_clients: str) -> list[str]:
    if raw_clients.strip().lower() == "auto":
        detected = _detect_setup_clients()
        return detected or ["Codex"]
    return _normalize_setup_clients(raw_clients)


def _run_setup(args: argparse.Namespace) -> int:
    """Non-interactive setup command for one-line installs."""
    if not args.yes and not args.dry_run:
        _fail("Refusing to patch config without --yes. Use --dry-run to preview changes.")
        return 1

    db_path_raw = args.db or str(Path.home() / ".waggle" / "memory.db")
    db_path = str(Path(db_path_raw).expanduser().resolve())
    python_exe = _python_exe()
    clients = _setup_clients_from_args(args.clients)

    print()
    print(_c(_BOLD, "waggle-mcp setup"))
    print(_c(_CYAN, "─" * 40))
    print(f"  clients: {', '.join(clients)}")
    print(f"  database: {db_path}")
    print(f"  model: {args.model}")
    if args.dry_run:
        print(f"  mode: dry-run")
        for client in clients:
            _ok(f"Would configure {client}")
        if args.project_instructions and "Codex" in clients:
            agents_path = (Path.cwd() / "AGENTS.md").resolve()
            _ok(f"Would write Codex automatic-memory instructions to {agents_path}")
        print()
        return 0

    for client in clients:
        writer = _CLIENT_WRITERS[client]
        try:
            config_file = writer(db_path, python_exe)
            if args.model != "all-MiniLM-L6-v2":
                _replace_model_in_client_config(config_file, args.model)
            _ok(f"{client} config written to {config_file}")
        except OSError as exc:
            _fail(f"Could not write {client} config: {exc}")
            return 1

    if args.project_instructions and "Codex" in clients:
        try:
            agents_file = _write_codex_agents()
            _ok(f"Codex automatic-memory instructions written to {agents_file}")
        except OSError as exc:
            _fail(f"Could not write AGENTS.md instructions: {exc}")
            return 1

    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _ok(f"Database directory ready at {Path(db_path).parent}")
    except OSError as exc:
        _fail(f"Could not create database directory: {exc}")
        return 1

    for client in clients:
        print(f"  {_c(_CYAN, chr(0x27A1))}  {_RESTART_HINTS[client]}")
    print()

    if args.run_doctor:
        doctor_config = AppConfig.from_env()
        doctor_config.db_path = db_path
        doctor_config.model_name = args.model
        doctor_exit = _run_doctor(doctor_config)
        if doctor_exit:
            print(_c(_CYAN, "Setup completed; doctor reported follow-up warnings above."))
        return 0
    return 0


def _replace_model_in_client_config(config_file: Path, model_name: str) -> None:
    """Patch WAGGLE_MODEL after using the existing config writers."""
    if config_file.suffix == ".toml":
        text = config_file.read_text()
        text = re.sub(r'WAGGLE_MODEL = "[^"]*"', f'WAGGLE_MODEL = "{model_name}"', text)
        config_file.write_text(text)
        return

    try:
        payload = json.loads(config_file.read_text())
    except json.JSONDecodeError:
        return

    if config_file.name == "waggle-mcp-config.json":
        env = payload.setdefault("env", {})
        if isinstance(env, dict):
            env["WAGGLE_MODEL"] = model_name
    else:
        servers = payload.get("mcpServers")
        if isinstance(servers, dict) and isinstance(servers.get("waggle"), dict):
            env = servers["waggle"].setdefault("env", {})
            if isinstance(env, dict):
                env["WAGGLE_MODEL"] = model_name
    config_file.write_text(json.dumps(payload, indent=2))


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

    if client == "Codex":
        try:
            agents_file = _write_codex_agents()
            _ok(f"Automatic memory instructions written to {agents_file}")
        except OSError as exc:
            _fail(f"Could not write AGENTS.md instructions: {exc}")
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
    # ── Windows UTF-8 guard (Error 3 from field bug log) ────────────────────
    # Windows consoles default to cp1252. Unicode log lines / emoji cause
    # UnicodeEncodeError that crashes test scripts or corrupts stdio framing.
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # best-effort; don't break startup over encoding
    _assert_runtime_feature_parity()
    parser = _build_parser()
    args = parser.parse_args()
    command = args.command or "serve"

    # Commands that should work before full backend/app initialization.
    if command == "setup":
        try:
            sys.exit(_run_setup(args))
        except ValidationFailure as exc:
            _emit_cli_error("validation_error", str(exc), {})
            sys.exit(1)
    if command == "init":
        sys.exit(_run_init())
    if command == "features":
        print(_FEATURES_GUIDE)
        return
    if command == "doctor":
        # Doctor only needs the config — not a live backend connection.
        config = AppConfig.from_env()
        sys.exit(_run_doctor(config, fix=bool(getattr(args, "fix", False))))

    config = AppConfig.from_env()
    log_stream = sys.stderr if config.transport == "stdio" else sys.stdout
    configure_logging(config.log_level, stream=log_stream)
    LOGGER.info("waggle_startup")
    if command in {"edit-graph", "view-graph", "ui"}:
        try:
            exit_code = _run_graph_editor_command(config, args)
        except ValidationFailure as exc:
            _emit_cli_error("validation_error", str(exc), {})
            sys.exit(1)
        except WaggleError as exc:
            _emit_cli_error(exc.code, str(exc), {"status_code": exc.status_code})
            sys.exit(2)
        except Exception as exc:
            _emit_cli_error("internal_error", str(exc), {"type": type(exc).__name__})
            sys.exit(3)
        sys.exit(exit_code or 0)
    if command != "serve":
        try:
            exit_code = _run_admin_command(config, args)
        except ValidationFailure as exc:
            _emit_cli_error("validation_error", str(exc), {})
            sys.exit(1)
        except WaggleError as exc:
            _emit_cli_error(exc.code, str(exc), {"status_code": exc.status_code})
            sys.exit(2)
        except Exception as exc:
            _emit_cli_error("internal_error", str(exc), {"type": type(exc).__name__})
            sys.exit(3)
        sys.exit(exit_code or 0)
        return
    if config.transport == "http":
        run_http(config)
        return
    if config.backend == "sqlite":
        Path(config.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_stdio(config))


if __name__ == "__main__":
    main()
