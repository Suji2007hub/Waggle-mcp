"""Persistent graph memory MCP server."""

from importlib.metadata import PackageNotFoundError, version

from waggle.graph import MemoryGraph
from waggle.models import (
    ApiKeyCreateResult,
    ApiKeyRecord,
    BackupResult,
    ConflictEntry,
    ConflictListResult,
    ConflictRecord,
    ContextBundle,
    ContextBundleExportResult,
    ContextRenderHints,
    ContextScopeResult,
    ContextTimelineItem,
    Edge,
    EvidenceRecord,
    GraphDiffResult,
    GraphStats,
    ImportResult,
    Node,
    NodeHistoryResult,
    NodeStoreResult,
    NodeType,
    ObservationResult,
    PrimeContextResult,
    RelationType,
    SubgraphResult,
    TenantRecord,
    TimelineResult,
    TopicCluster,
    TopicResult,
)
from waggle.orchestrator import (
    AsyncMemoryOrchestrator,
    ConversationTurn,
    IngestPlan,
    MemoryPolicy,
    MemoryScope,
    RetrievePlan,
    RetrieveRequest,
)
from waggle.chat_runtime import (
    ModelAdapter,
    OrchestratedChatRuntime,
    RuntimeTurnResult,
)

try:  # pragma: no cover
    from waggle.neo4j_graph import Neo4jMemoryGraph
except Exception:  # pragma: no cover
    Neo4jMemoryGraph = None

__all__ = [
    "Edge",
    "ApiKeyCreateResult",
    "ApiKeyRecord",
    "BackupResult",
    "ConflictEntry",
    "ConflictListResult",
    "ConflictRecord",
    "ContextBundle",
    "ContextBundleExportResult",
    "ContextRenderHints",
    "ContextScopeResult",
    "ContextTimelineItem",
    "GraphStats",
    "GraphDiffResult",
    "EvidenceRecord",
    "ImportResult",
    "MemoryGraph",
    "Neo4jMemoryGraph",
    "Node",
    "NodeHistoryResult",
    "NodeStoreResult",
    "NodeType",
    "ObservationResult",
    "PrimeContextResult",
    "RelationType",
    "SubgraphResult",
    "TenantRecord",
    "TimelineResult",
    "TopicCluster",
    "TopicResult",
    "AsyncMemoryOrchestrator",
    "ConversationTurn",
    "IngestPlan",
    "MemoryPolicy",
    "MemoryScope",
    "RetrievePlan",
    "RetrieveRequest",
    "ModelAdapter",
    "OrchestratedChatRuntime",
    "RuntimeTurnResult",
]

try:  # pragma: no cover
    __version__ = version("waggle-mcp")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.4"
