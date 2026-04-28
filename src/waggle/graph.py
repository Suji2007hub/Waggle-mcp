from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import networkx as nx
import numpy as np

from waggle.abhi import (
    ABHI_SPEC_VERSION,
    abhi_to_snapshot,
    dispatch_abhi_event,
    diff_abhi_files,
    filter_snapshot_by_scope,
    inspect_abhi_document,
    load_abhi_chunk_file,
    load_abhi_document,
    merge_abhi_files,
    query_abhi_file,
    validate_abhi_document,
    write_abhi_document,
)
from waggle.auth import generate_api_key, hash_api_key, verify_api_key
from waggle.context_bundle import build_context_bundle, build_query_summary, export_context_bundle_files
from waggle.embeddings import EmbeddingModel
from waggle.evidence import build_observation_evidence, merge_evidence_records, merge_validity_windows
from waggle.errors import AuthenticationError, ValidationFailure
from waggle.intelligence import (
    compatible_node_types,
    canonical_concept_overlap,
    contains_conflicting_months,
    contains_conflicting_numbers,
    describes_rejected_or_limited_option,
    content_token_jaccard,
    detect_conflict_reason,
    extract_choice_entity,
    extract_conversation_candidates,
    infer_label,
    infer_node_type,
    infer_relationship,
    infer_temporal_hints,
    is_acronym_match,
    label_similarity,
    lexical_overlap,
    normalize_text,
    paraphrase_dedup_score,
    parse_since_value,
    split_atomic_items,
    summarize_topic,
    temporal_score_adjustment,
    tokenize_text,
    type_aware_dedup_threshold,
    within_time_window,
)
from waggle.markdown_vault import (
    evidence_from_lines,
    iter_vault_documents,
    render_node_document,
    slugify,
    vault_filename,
)
from waggle.models import (
    AbhiChunkLoadResult,
    AbhiDiffResult,
    AbhiExportResult,
    AbhiImportResult,
    AbhiInspectResult,
    AbhiMergeResult,
    AbhiQueryResult,
    AbhiValidationResult,
    ApiKeyCreateResult,
    ApiKeyRecord,
    BackupResult,
    ConflictEntry,
    ConflictListResult,
    ContextWindow,
    ContextWindowEdge,
    ConflictRecord,
    ConnectedNodeStat,
    ContextBundleExportResult,
    ContextScopeResult,
    ContextTimelineItem,
    Edge,
    EvidenceRecord,
    GraphDiffResult,
    GraphStats,
    ImportResult,
    MarkdownVaultExportResult,
    MarkdownVaultImportResult,
    Node,
    NodeHistoryResult,
    NodeStoreResult,
    NodeType,
    ObservationResult,
    PrimeContextResult,
    FusionHit,
    ReplayHit,
    RecentNodeStat,
    RelationType,
    SubgraphResult,
    TranscriptIngestionInput,
    TranscriptIngestionResult,
    TranscriptMessage,
    TranscriptRecord,
    normalize_relationship,
    TenantRecord,
    TimelineResult,
    TopicCluster,
    TopicResult,
    utc_now,
)

SCHEMA_VERSION = 6

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpansionMeta:
    via_relation: str
    from_node: str
    effective_priority: float


class _NeutralTemporalHints:
    """Neutral temporal hints for operations without query-driven time intent."""
    recency_mode: str = "none"
    time_window_start = None
    time_window_end = None


RELATION_SCORE_BOOST: dict[str, float] = {
    "contradicts": 0.15,
    "updates": 0.12,
    "depends_on": 0.08,
    "derived_from": 0.05,
    "part_of": 0.03,
    "relates_to": 0.00,
    "similar_to": -0.05,
    "seed": 0.00,
}
TOPIC_RELEVANCE_THRESHOLD = 0.35
TOPIC_SEMANTIC_ONLY_THRESHOLD = 0.70
TEMPORAL_TOPIC_MARGIN = 0.03
NEGATION_QUERY_TERMS = (
    "not",
    "never",
    "reject",
    "rejected",
    "blocked",
    "forbid",
    "forbidden",
    "ruled out",
    "avoid",
    "must not",
    "should not",
    "off limits",
    "disallowed",
    "prohibit",
    "prohibited",
)
NEGATION_NODE_TERMS = (
    "must not",
    "do not",
    "cannot",
    "can not",
    "rejected",
    "blocked",
    "forbidden",
    "ruled out",
    "not allowed",
    "off limits",
    "disallowed",
    "prohibited",
    "mustn't",
)
NEGATION_SCORE_BOOST = 0.28

QUERY_ALIAS_TERMS: tuple[tuple[str, str], ...] = (
    ("ingestion and export", "ingestion import ndjson export csv parquet warehouse sync"),
    ("ingestion", "ingestion import ndjson streaming imports"),
    ("export", "export csv parquet warehouse sync signed download links"),
    ("enterprise data export policy", "enterprise data export policy admin approval signed download links"),
    ("privacy export stance", "privacy export policy admin approval signed download links"),
    ("privacy export", "privacy export policy admin approval signed download links"),
    ("export policy", "export policy admin approval signed download links"),
    ("database", "postgresql mysql sqlite database production"),
    ("production database choice", "postgresql production database choice current parity safer migrations"),
    ("auto rollback", "auto rollback deployments incident 5xx"),
    ("acid compliance", "acid compliance transactions consistency postgres decision reason"),
    ("justified by", "reason rationale because requirement constraint"),
    ("deployment platform", "cloud run ecs deployment deploy autoscaling"),
    ("deployment", "cloud run ecs deployment deploy autoscaling"),
    ("deploy", "cloud run ecs deployment rollback"),
    ("api deploy", "api deploy cloud run ecs autoscaling"),
    ("deploy now", "current deployment cloud run autoscaling"),
    ("auth", "jwt token expiry refresh authentication"),
    ("jwt expiry", "jwt token expiry 15m 1h authentication"),
    ("session cache backend", "session cache backend redis keydb ttl failover"),
    ("cache backend", "session cache backend redis keydb ttl failover"),
    ("workflow backend", "workflow backend temporal celery redis retries visibility"),
    ("workflow backend do we use now", "current workflow backend temporal retries visibility"),
    ("mobile offline", "offline queue sync mobile edits"),
    ("production incidents", "incident rollback auto-rollback 5xx error rate"),
    ("incidents", "incident rollback auto-rollback 5xx error rate"),
    ("observability", "traces slos logs metrics service-level objectives"),
    ("workflow engine", "temporal workflows celery redis queue backend retries visibility"),
    ("workflow", "temporal workflows celery redis queue backend"),
    ("schema migration", "alembic migrations autogenerate manual review"),
    ("migration tool", "alembic migrations autogenerate manual review"),
    ("feature flags", "flags control plane env vars"),
    ("access permissions", "access control rbac abac role attribute rules"),
    ("permissions", "access control rbac abac role attribute rules"),
    ("upstream changes", "webhooks polling sync missed events"),
    ("notified", "webhooks polling sync notifications"),
    ("notified", "notifications email slack alerts webhooks"),
    ("alert on", "notifications email slack ops alerts"),
    ("alert", "notifications email slack ops alerts"),
    ("workflow engine", "temporal workflows celery redis queue backend retries visibility"),
    ("scaling issue", "concurrent writes concurrency blocker scaling"),
    ("schema migration tool", "alembic migrations manual review schema"),
    ("enterprise-sensitive actions", "enterprise export approval signed links admin approval"),
    ("enterprise-sensitive", "enterprise export approval signed links admin approval"),
    ("privileged", "break-glass shared admin named ownership privileged actions"),
    ("model deployment", "model rollout canary approval auto-promote"),
    ("model rollout", "model rollout canary approval auto-promote product-manager approval"),
    ("canary approval", "canary approval product-manager approval no auto-promote"),
    ("pm gate", "product-manager approval no auto-promote canary"),
    ("refund flow", "refunds one-click refunds manual review rules engine"),
    ("refunds", "refund rules engine one-click refunds manual review"),
    ("risky automation", "rules engine manual review blocked no auto-promote one-click refunds"),
    ("monitoring was missing", "abuse monitoring one-click refunds blocked"),
    ("missing monitoring", "abuse monitoring one-click refunds blocked"),
    ("storage costs", "storage cold uploads s3 intelligent tiering cost"),
    ("data retention", "audit logs retention compliance 90 days"),
    ("retention compliance", "audit logs retention compliance 90 days"),
    ("emergency access", "break-glass access per-user accounts audit trails"),
    ("security review", "security review break-glass raw api keys shared admins"),
    ("logs", "logs raw api keys audit retention"),
    ("named accountability", "named ownership per-user accounts admin approval signed links"),
    ("deeper requirement", "requirement supported choice concurrency realtime"),
    ("supported that choice", "requirement supported choice concurrency realtime"),
    ("fastapi", "fastapi async concurrency realtime websockets"),
)

MUST_PAIR_RELATIONS: frozenset[str] = frozenset({
    "contradicts",
    "updates",
    "depends_on",
})

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    name TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    api_key_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    name TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    last_used_at TEXT DEFAULT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    agent_id TEXT DEFAULT '',
    project TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    context_window_id TEXT DEFAULT NULL,
    label TEXT NOT NULL,
    content TEXT NOT NULL,
    node_type TEXT NOT NULL CHECK(
        node_type IN ('fact', 'entity', 'concept', 'preference', 'decision', 'question', 'note')
    ),
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    embedding BLOB,
    source_prompt TEXT DEFAULT '',
    evidence_records TEXT DEFAULT '[]',
    valid_from TEXT DEFAULT NULL,
    valid_to TEXT DEFAULT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS repos (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(tenant_id, name)
);

CREATE TABLE IF NOT EXISTS context_windows (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    repo_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    title TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'closed', 'archived')),
    node_count INTEGER DEFAULT 0,
    embedding BLOB,
    embedding_stale INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    closed_at TEXT DEFAULT NULL,
    FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE,
    UNIQUE(tenant_id, repo_id, session_id)
);

CREATE TABLE IF NOT EXISTS context_window_edges (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    source_window_id TEXT NOT NULL,
    target_window_id TEXT NOT NULL,
    edge_type TEXT NOT NULL CHECK(edge_type IN (
        'entity_overlap',
        'supersedes',
        'temporal_sequence',
        'continuation',
        'shared_scope'
    )),
    shared_entities TEXT DEFAULT '[]',
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_window_id) REFERENCES context_windows(id) ON DELETE CASCADE,
    FOREIGN KEY (target_window_id) REFERENCES context_windows(id) ON DELETE CASCADE,
    UNIQUE(tenant_id, source_window_id, target_window_id, edge_type, shared_entities)
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS transcript_records (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    agent_id TEXT DEFAULT '',
    project TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    observed_at TEXT NOT NULL,
    turn_index INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT '',
    transcript_text TEXT NOT NULL,
    embedding BLOB,
    metadata TEXT DEFAULT '{}',
    message_identity TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS graph_ui_state (
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    agent_id TEXT DEFAULT '',
    project TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    positions TEXT DEFAULT '{}',
    zoom REAL DEFAULT 1.0,
    viewport TEXT DEFAULT '{}',
    groups_json TEXT DEFAULT '[]',
    collapsed_groups TEXT DEFAULT '[]',
    selected_nodes TEXT DEFAULT '[]',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, agent_id, project, session_id)
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_tenant_type ON nodes(tenant_id, node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_tenant_updated ON nodes(tenant_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_nodes_context_window ON nodes(context_window_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_relationship ON edges(relationship);
CREATE INDEX IF NOT EXISTS idx_edges_tenant_relationship ON edges(tenant_id, relationship);
CREATE INDEX IF NOT EXISTS idx_transcripts_tenant_observed ON transcript_records(tenant_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_transcripts_tenant_session_turn ON transcript_records(tenant_id, session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_repos_tenant_name ON repos(tenant_id, name);
CREATE INDEX IF NOT EXISTS idx_context_windows_repo ON context_windows(repo_id);
CREATE INDEX IF NOT EXISTS idx_context_windows_session ON context_windows(session_id);
CREATE INDEX IF NOT EXISTS idx_context_windows_status ON context_windows(status);
CREATE INDEX IF NOT EXISTS idx_cw_edges_source ON context_window_edges(source_window_id);
CREATE INDEX IF NOT EXISTS idx_cw_edges_target ON context_window_edges(target_window_id);
CREATE INDEX IF NOT EXISTS idx_cw_edges_type ON context_window_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_graph_ui_scope ON graph_ui_state(tenant_id, project, agent_id, session_id);
"""

RELATION_WEIGHTS: dict[str, float] = {
    "contradicts": 1.00,
    "updates": 0.95,
    "depends_on": 0.85,
    "derived_from": 0.75,
    "part_of": 0.70,
    "relates_to": 0.50,
    "similar_to": 0.30,
}


def _parse_datetime(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _encode_evidence_records(records: list[EvidenceRecord]) -> str:
    return json.dumps([record.model_dump(mode="json") for record in records], sort_keys=True)


def _decode_evidence_records(raw: Any) -> list[EvidenceRecord]:
    if raw in (None, ""):
        return []
    if isinstance(raw, list):
        return [EvidenceRecord.model_validate(item) for item in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(decoded, list):
            return []
        return [EvidenceRecord.model_validate(item) for item in decoded]
    return []


def _encode_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True)


def _decode_metadata(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def recency_weight(
    updated_at: float,
    now: float | None = None,
    half_life_days: float = 30.0,
) -> float:
    if now is None:
        now = time.time()
    age_days = (now - updated_at) / 86400.0
    if age_days < 0:
        age_days = 0.0
    return math.exp(-0.693 * age_days / half_life_days)


def score_node(
    similarity: float,
    updated_at: float,
    edge_weight: float = 1.0,
    *,
    now: float | None = None,
    half_life_days: float = 30.0,
    similarity_weight: float = 0.6,
    recency_weight_factor: float = 0.3,
    edge_weight_factor: float = 0.1,
    superseded: bool = False,
    superseded_penalty: float = 0.2,
) -> float:
    r = recency_weight(updated_at, now, half_life_days)
    e = max(0.0, min(1.0, edge_weight))
    score = (similarity * similarity_weight) + (r * recency_weight_factor) + (e * edge_weight_factor)
    if superseded:
        score *= superseded_penalty
    return score


def _scope_matches(node: Node, *, agent_id: str = "", project: str = "", session_id: str = "") -> bool:
    normalized_agent = agent_id.strip().lower()
    normalized_project = project.strip().lower()
    normalized_session = session_id.strip().lower()
    if normalized_agent and node.agent_id.strip().lower() != normalized_agent:
        return False
    if normalized_session and node.session_id.strip().lower() != normalized_session:
        return False
    if normalized_project:
        project_tags = {str(tag).strip().lower() for tag in node.tags}
        if node.project.strip().lower() != normalized_project and normalized_project not in project_tags and f"project:{normalized_project}" not in project_tags:
            return False
    return True


def _retrieval_session_scope(*, agent_id: str = "", project: str = "", session_id: str = "") -> str:
    if session_id.strip() and (agent_id.strip() or project.strip()):
        return ""
    return session_id


def _merge_scope_value(existing: str, incoming: str) -> str:
    return existing.strip() or incoming.strip()


class MemoryGraph:
    """SQLite-backed graph memory with embedding-assisted retrieval."""

    def __init__(
        self,
        db_path: str | Path,
        embedding_model: EmbeddingModel,
        *,
        tenant_id: str = "local-default",
        dedup_similarity_threshold: float = 0.97,
        dedup_same_label_threshold: float = 0.9,
        recency_half_life_days: float = 30.0,
        tiered_retrieval: bool = False,
        tiered_retrieval_top_k_windows: int = 3,
        export_dir: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.embedding_model = embedding_model
        self.tenant_id = tenant_id.strip() or "local-default"
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.dedup_same_label_threshold = dedup_same_label_threshold
        self.recency_half_life_days = recency_half_life_days
        self.tiered_retrieval = tiered_retrieval
        self.tiered_retrieval_top_k_windows = max(1, tiered_retrieval_top_k_windows)
        self.export_dir = Path(export_dir).expanduser() if export_dir is not None else self.db_path.parent / "exports"
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize_database(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(SCHEMA_SQL)
            self._migrate_legacy_schema(connection)
            connection.executescript(INDEX_SQL)
            created_at = utc_now().isoformat()
            connection.execute(
                """
                INSERT INTO tenants (tenant_id, name, status, created_at)
                VALUES (?, '', 'active', ?)
                ON CONFLICT(tenant_id) DO NOTHING
                """,
                (self.tenant_id, created_at),
            )

    def for_tenant(self, tenant_id: str) -> "MemoryGraph":
        clone = object.__new__(MemoryGraph)
        clone.db_path = self.db_path
        clone.embedding_model = self.embedding_model
        clone.tenant_id = tenant_id.strip() or "local-default"
        clone.dedup_similarity_threshold = self.dedup_similarity_threshold
        clone.dedup_same_label_threshold = self.dedup_same_label_threshold
        clone.recency_half_life_days = self.recency_half_life_days
        clone.tiered_retrieval = self.tiered_retrieval
        clone.tiered_retrieval_top_k_windows = self.tiered_retrieval_top_k_windows
        clone.export_dir = self.export_dir
        clone._lock = self._lock
        clone.ensure_tenant(clone.tenant_id)
        return clone

    def ensure_tenant(self, tenant_id: str, name: str = "") -> TenantRecord:
        normalized_tenant_id = tenant_id.strip()
        if not normalized_tenant_id:
            raise ValidationFailure("Tenant ID cannot be empty.")
        created_at = utc_now().isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tenants (tenant_id, name, status, created_at)
                VALUES (?, ?, 'active', ?)
                ON CONFLICT(tenant_id) DO UPDATE SET name = CASE WHEN excluded.name != '' THEN excluded.name ELSE tenants.name END
                """,
                (normalized_tenant_id, name.strip(), created_at),
            )
            row = connection.execute(
                "SELECT tenant_id, name, status, created_at FROM tenants WHERE tenant_id = ?",
                (normalized_tenant_id,),
            ).fetchone()
        return TenantRecord(
            tenant_id=row["tenant_id"],
            name=row["name"] or "",
            status=row["status"],
            created_at=_parse_datetime(row["created_at"]),
        )

    @staticmethod
    def _normalize_ui_scope(*, project: str = "", agent_id: str = "", session_id: str = "") -> tuple[str, str, str]:
        return (project.strip(), agent_id.strip(), session_id.strip())

    def get_ui_state(
        self,
        *,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        normalized_project, normalized_agent, normalized_session = self._normalize_ui_scope(
            project=project, agent_id=agent_id, session_id=session_id
        )
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT positions, zoom, viewport, groups_json, collapsed_groups, selected_nodes
                FROM graph_ui_state
                WHERE tenant_id = ? AND project = ? AND agent_id = ? AND session_id = ?
                """,
                (self.tenant_id, normalized_project, normalized_agent, normalized_session),
            ).fetchone()
        if row is None:
            return {
                "positions": {},
                "zoom": 1.0,
                "viewport": {"center_x": 0, "center_y": 0},
                "groups": [],
                "collapsed_groups": [],
                "selected_nodes": [],
            }
        return {
            "positions": json.loads(row["positions"] or "{}"),
            "zoom": float(row["zoom"] if row["zoom"] is not None else 1.0),
            "viewport": json.loads(row["viewport"] or "{}") or {"center_x": 0, "center_y": 0},
            "groups": json.loads(row["groups_json"] or "[]"),
            "collapsed_groups": json.loads(row["collapsed_groups"] or "[]"),
            "selected_nodes": json.loads(row["selected_nodes"] or "[]"),
        }

    def save_ui_state(
        self,
        *,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        positions: dict[str, Any] | None = None,
        zoom: float | None = None,
        viewport: dict[str, Any] | None = None,
        groups: list[dict[str, Any]] | None = None,
        collapsed_groups: list[str] | None = None,
        selected_nodes: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_project, normalized_agent, normalized_session = self._normalize_ui_scope(
            project=project, agent_id=agent_id, session_id=session_id
        )
        current = self.get_ui_state(
            project=normalized_project,
            agent_id=normalized_agent,
            session_id=normalized_session,
        )
        merged = {
            "positions": positions if positions is not None else current["positions"],
            "zoom": float(zoom if zoom is not None else current["zoom"]),
            "viewport": viewport if viewport is not None else current["viewport"],
            "groups": groups if groups is not None else current["groups"],
            "collapsed_groups": collapsed_groups if collapsed_groups is not None else current["collapsed_groups"],
            "selected_nodes": selected_nodes if selected_nodes is not None else current["selected_nodes"],
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO graph_ui_state (
                    tenant_id, agent_id, project, session_id,
                    positions, zoom, viewport, groups_json, collapsed_groups, selected_nodes, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, agent_id, project, session_id)
                DO UPDATE SET
                    positions = excluded.positions,
                    zoom = excluded.zoom,
                    viewport = excluded.viewport,
                    groups_json = excluded.groups_json,
                    collapsed_groups = excluded.collapsed_groups,
                    selected_nodes = excluded.selected_nodes,
                    updated_at = excluded.updated_at
                """,
                (
                    self.tenant_id,
                    normalized_agent,
                    normalized_project,
                    normalized_session,
                    json.dumps(merged["positions"], sort_keys=True),
                    merged["zoom"],
                    json.dumps(merged["viewport"], sort_keys=True),
                    json.dumps(merged["groups"], sort_keys=True),
                    json.dumps(merged["collapsed_groups"], sort_keys=True),
                    json.dumps(merged["selected_nodes"], sort_keys=True),
                    utc_now().isoformat(),
                ),
            )
        return merged

    def create_api_key(self, tenant_id: str, name: str = "") -> ApiKeyCreateResult:
        tenant = self.ensure_tenant(tenant_id)
        raw_api_key = generate_api_key()
        record = ApiKeyRecord(
            api_key_id=str(uuid4()),
            tenant_id=tenant.tenant_id,
            key_hash=hash_api_key(raw_api_key),
            name=name.strip(),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO api_keys (api_key_id, tenant_id, key_hash, name, status, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.api_key_id,
                    record.tenant_id,
                    record.key_hash,
                    record.name,
                    record.status,
                    record.created_at.isoformat(),
                    None,
                ),
            )
        return ApiKeyCreateResult(record=record, raw_api_key=raw_api_key)

    def list_api_keys(self, tenant_id: str) -> list[ApiKeyRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT api_key_id, tenant_id, key_hash, name, status, created_at, last_used_at
                FROM api_keys
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            ).fetchall()
        return [
            ApiKeyRecord(
                api_key_id=row["api_key_id"],
                tenant_id=row["tenant_id"],
                key_hash=row["key_hash"],
                name=row["name"] or "",
                status=row["status"],
                created_at=_parse_datetime(row["created_at"]),
                last_used_at=_parse_datetime(row["last_used_at"]) if row["last_used_at"] else None,
            )
            for row in rows
        ]

    def revoke_api_key(self, api_key_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE api_keys SET status = 'revoked' WHERE api_key_id = ?",
                (api_key_id,),
            )

    def authenticate_api_key(self, raw_api_key: str) -> ApiKeyRecord:
        key_hash = hash_api_key(raw_api_key)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT api_key_id, tenant_id, key_hash, name, status, created_at, last_used_at
                FROM api_keys
                WHERE key_hash = ?
                LIMIT 1
                """,
                (key_hash,),
            ).fetchone()
            if row is None or not verify_api_key(raw_api_key, row["key_hash"]):
                raise AuthenticationError("Invalid API key.")
            connection.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE api_key_id = ?",
                (utc_now().isoformat(), row["api_key_id"]),
            )
        return ApiKeyRecord(
            api_key_id=row["api_key_id"],
            tenant_id=row["tenant_id"],
            key_hash=row["key_hash"],
            name=row["name"] or "",
            status=row["status"],
            created_at=_parse_datetime(row["created_at"]),
            last_used_at=utc_now(),
        )

    def _migrate_legacy_schema(self, connection: sqlite3.Connection) -> None:
        node_columns = {row["name"] for row in connection.execute("PRAGMA table_info(nodes)").fetchall()}
        edge_columns = {row["name"] for row in connection.execute("PRAGMA table_info(edges)").fetchall()}
        if "tenant_id" not in node_columns:
            connection.execute(
                f"ALTER TABLE nodes ADD COLUMN tenant_id TEXT NOT NULL DEFAULT '{self.tenant_id}'"
            )
            connection.execute("UPDATE nodes SET tenant_id = ? WHERE tenant_id = ''", (self.tenant_id,))
        if "evidence_records" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN evidence_records TEXT DEFAULT '[]'")
        if "metadata" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN metadata TEXT DEFAULT '{}'")
        if "valid_from" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN valid_from TEXT DEFAULT NULL")
        if "valid_to" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN valid_to TEXT DEFAULT NULL")
        if "agent_id" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN agent_id TEXT DEFAULT ''")
        if "project" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN project TEXT DEFAULT ''")
        if "session_id" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN session_id TEXT DEFAULT ''")
        if "context_window_id" not in node_columns:
            connection.execute("ALTER TABLE nodes ADD COLUMN context_window_id TEXT DEFAULT NULL")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_nodes_context_window ON nodes(context_window_id)")
        if "tenant_id" not in edge_columns:
            connection.execute(
                f"ALTER TABLE edges ADD COLUMN tenant_id TEXT NOT NULL DEFAULT '{self.tenant_id}'"
            )
            connection.execute("UPDATE edges SET tenant_id = ? WHERE tenant_id = ''", (self.tenant_id,))
        transcript_columns = {row["name"] for row in connection.execute("PRAGMA table_info(transcript_records)").fetchall()}
        if "message_identity" not in transcript_columns:
            connection.execute(
                "ALTER TABLE transcript_records ADD COLUMN message_identity TEXT DEFAULT NULL"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_ui_state (
                tenant_id TEXT NOT NULL DEFAULT 'local-default',
                agent_id TEXT DEFAULT '',
                project TEXT DEFAULT '',
                session_id TEXT DEFAULT '',
                positions TEXT DEFAULT '{}',
                zoom REAL DEFAULT 1.0,
                viewport TEXT DEFAULT '{}',
                groups_json TEXT DEFAULT '[]',
                collapsed_groups TEXT DEFAULT '[]',
                selected_nodes TEXT DEFAULT '[]',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (tenant_id, agent_id, project, session_id)
            )
            """
        )
        # Always ensure the partial unique index exists (IF NOT EXISTS is safe for reruns).
        # Must be outside the if-block so new databases (where the column comes from CREATE TABLE)
        # also get the index, not just existing databases that went through ALTER TABLE.
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_identity
            ON transcript_records(tenant_id, session_id, message_identity)
            WHERE message_identity IS NOT NULL
            """
        )

        connection.execute(
            """
            INSERT OR IGNORE INTO schema_migrations (version, applied_at)
            VALUES (?, ?)
            """,
            (SCHEMA_VERSION, utc_now().isoformat()),
        )

    def ensure_repo(self, project: str = "") -> str:
        name = project.strip() or "default"
        repo_id = f"{self.tenant_id}:{slugify(name)}"
        now = utc_now().isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO repos (id, tenant_id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, '', ?, ?)
                ON CONFLICT(tenant_id, name) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (repo_id, self.tenant_id, name, now, now),
            )
            row = connection.execute(
                "SELECT id FROM repos WHERE tenant_id = ? AND name = ?",
                (self.tenant_id, name),
            ).fetchone()
        return str(row["id"])

    def ensure_context_window(self, session_id: str = "", repo_id: str | None = None) -> str:
        normalized_session = session_id.strip() or "default"
        resolved_repo_id = repo_id or self.ensure_repo("default")
        window_id = f"{resolved_repo_id}:{slugify(normalized_session)}"
        now = utc_now().isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO context_windows (
                    id, tenant_id, repo_id, session_id, title, status, node_count,
                    embedding_stale, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, '', 'active', 0, 1, ?, ?)
                ON CONFLICT(tenant_id, repo_id, session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (window_id, self.tenant_id, resolved_repo_id, normalized_session, now, now),
            )
            row = connection.execute(
                """
                SELECT id FROM context_windows
                WHERE tenant_id = ? AND repo_id = ? AND session_id = ?
                """,
                (self.tenant_id, resolved_repo_id, normalized_session),
            ).fetchone()
        return str(row["id"])

    def resolve_window_context(self, project: str | None = None, session_id: str | None = None) -> tuple[str, str]:
        repo_id = self.ensure_repo(project or "default")
        window_id = self.ensure_context_window(session_id or "default", repo_id)
        return repo_id, window_id

    def update_window_node_count(self, window_id: str) -> int:
        with self._lock, self._connect() as connection:
            count = self._update_window_node_count(connection, window_id)
        return count

    def mark_window_embedding_stale(self, window_id: str) -> None:
        with self._lock, self._connect() as connection:
            self._mark_window_embedding_stale(connection, window_id)

    def _update_window_node_count(self, connection: sqlite3.Connection, window_id: str) -> int:
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM nodes WHERE tenant_id = ? AND context_window_id = ?",
                (self.tenant_id, window_id),
            ).fetchone()[0]
        )
        connection.execute(
            """
            UPDATE context_windows
            SET node_count = ?, updated_at = ?
            WHERE tenant_id = ? AND id = ?
            """,
            (count, utc_now().isoformat(), self.tenant_id, window_id),
        )
        return count

    def _mark_window_embedding_stale(self, connection: sqlite3.Connection, window_id: str) -> None:
        connection.execute(
            """
            UPDATE context_windows
            SET embedding_stale = 1, updated_at = ?
            WHERE tenant_id = ? AND id = ?
            """,
            (utc_now().isoformat(), self.tenant_id, window_id),
        )

    def get_context_window(self, window_id: str) -> ContextWindow:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, tenant_id, repo_id, session_id, title, status, node_count,
                       embedding_stale, created_at, updated_at, closed_at
                FROM context_windows
                WHERE tenant_id = ? AND id = ?
                """,
                (self.tenant_id, window_id),
            ).fetchone()
        if row is None:
            raise ValueError(f"Context window not found: {window_id}")
        return self._row_to_context_window(row)

    def list_context_windows(
        self,
        *,
        project: str = "",
        status: str = "",
        limit: int = 20,
    ) -> list[ContextWindow]:
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        normalized_status = status.strip().lower()
        if normalized_status and normalized_status not in {"active", "closed", "archived"}:
            raise ValueError("status must be one of: active, closed, archived.")

        query = """
            SELECT cw.id, cw.tenant_id, cw.repo_id, cw.session_id, cw.title, cw.status, cw.node_count,
                   cw.embedding_stale, cw.created_at, cw.updated_at, cw.closed_at
            FROM context_windows cw
            JOIN repos r ON r.id = cw.repo_id
            WHERE cw.tenant_id = ?
        """
        params: list[Any] = [self.tenant_id]
        if project.strip():
            query += " AND r.name = ?"
            params.append(project.strip())
        if normalized_status:
            query += " AND cw.status = ?"
            params.append(normalized_status)
        query += " ORDER BY cw.updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_context_window(row) for row in rows]

    def get_context_window_edges(self, window_id: str) -> list[ContextWindowEdge]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, tenant_id, source_window_id, target_window_id, edge_type,
                       shared_entities, weight, metadata, created_at
                FROM context_window_edges
                WHERE tenant_id = ?
                  AND (source_window_id = ? OR target_window_id = ?)
                ORDER BY created_at DESC
                """,
                (self.tenant_id, window_id, window_id),
            ).fetchall()
        return [self._row_to_context_window_edge(row) for row in rows]

    def close_context_window(self, window_id: str) -> ContextWindow:
        embedding = self.compute_window_embedding(window_id)
        with self._lock, self._connect() as connection:
            if embedding is not None:
                self._save_window_embedding(connection, window_id, embedding)
            self._update_window_node_count(connection, window_id)
            now = utc_now().isoformat()
            connection.execute(
                """
                UPDATE context_windows
                SET status = 'closed', closed_at = COALESCE(closed_at, ?), updated_at = ?
                WHERE tenant_id = ? AND id = ?
                """,
                (now, now, self.tenant_id, window_id),
            )
            row = connection.execute(
                """
                SELECT id, tenant_id, repo_id, session_id, title, status, node_count,
                       embedding_stale, created_at, updated_at, closed_at
                FROM context_windows
                WHERE tenant_id = ? AND id = ?
                """,
                (self.tenant_id, window_id),
            ).fetchone()
        if row is None:
            raise ValueError(f"Context window not found: {window_id}")
        window = self._row_to_context_window(row)
        self.derive_context_window_edges(window.id, window.repo_id)
        return window

    def get_nodes_without_window(self) -> list[Node]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, context_window_id, label, content, node_type,
                       tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ? AND context_window_id IS NULL
                ORDER BY updated_at ASC
                """,
                (self.tenant_id,),
            ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def assign_nodes_to_window(self, node_ids: list[str], window_id: str) -> int:
        if not node_ids:
            return 0
        placeholders = ", ".join("?" for _ in node_ids)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE nodes
                SET context_window_id = ?
                WHERE tenant_id = ? AND context_window_id IS NULL AND id IN ({placeholders})
                """,
                (window_id, self.tenant_id, *node_ids),
            )
            updated = int(cursor.rowcount or 0)
            self._update_window_node_count(connection, window_id)
            self._mark_window_embedding_stale(connection, window_id)
        return updated

    def list_repos(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, tenant_id, name, description, created_at, updated_at
                FROM repos
                WHERE tenant_id = ?
                ORDER BY updated_at DESC
                """,
                (self.tenant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_repo_windows(
        self,
        repo_id: str,
        *,
        exclude: str | None = None,
        include_archived: bool = False,
    ) -> list[ContextWindow]:
        query = """
            SELECT id, tenant_id, repo_id, session_id, title, status, node_count,
                   embedding_stale, created_at, updated_at, closed_at
            FROM context_windows
            WHERE tenant_id = ? AND repo_id = ?
        """
        params: list[Any] = [self.tenant_id, repo_id]
        if exclude:
            query += " AND id != ?"
            params.append(exclude)
        if not include_archived:
            query += " AND status != 'archived'"
        query += " ORDER BY updated_at DESC"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_context_window(row) for row in rows]

    def get_window_nodes(self, window_id: str, node_types: list[NodeType] | None = None) -> list[Node]:
        query = """
            SELECT id, agent_id, project, session_id, context_window_id, label, content, node_type,
                   tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                   created_at, updated_at, access_count, tenant_id
            FROM nodes
            WHERE tenant_id = ? AND context_window_id = ?
        """
        params: list[Any] = [self.tenant_id, window_id]
        if node_types:
            placeholders = ", ".join("?" for _ in node_types)
            query += f" AND node_type IN ({placeholders})"
            params.extend(node_type.value for node_type in node_types)
        query += " ORDER BY updated_at DESC"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_node(row) for row in rows]

    def compute_window_embedding(self, window_id: str) -> np.ndarray | None:
        meaningful_types = [
            NodeType.DECISION,
            NodeType.FACT,
            NodeType.ENTITY,
            NodeType.PREFERENCE,
            NodeType.CONCEPT,
        ]
        nodes = self.get_window_nodes(window_id, node_types=meaningful_types)
        if not nodes:
            return None

        type_rank = {
            NodeType.DECISION: 0,
            NodeType.FACT: 1,
            NodeType.ENTITY: 2,
            NodeType.PREFERENCE: 3,
            NodeType.CONCEPT: 4,
        }
        nodes.sort(key=lambda node: (type_rank.get(node.node_type, 99), -node.updated_at.timestamp(), node.label.lower()))
        if len(nodes) > 100:
            LOGGER.warning("context_window_embedding_truncated", extra={"window_id": window_id, "node_count": len(nodes)})
            nodes = nodes[:100]
        window_text = " | ".join(f"{node.label}: {node.content}" for node in nodes)
        if not window_text.strip():
            return None
        return self.embedding_model.embed(window_text[:12000])

    def get_window_embedding(self, window_id: str) -> np.ndarray | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT embedding, embedding_stale
                FROM context_windows
                WHERE tenant_id = ? AND id = ?
                """,
                (self.tenant_id, window_id),
            ).fetchone()
            if row is None:
                return None
            if row["embedding"] is not None and not bool(row["embedding_stale"]):
                return self.embedding_model.from_bytes(row["embedding"])

        embedding = self.compute_window_embedding(window_id)
        if embedding is None:
            return None
        with self._lock, self._connect() as connection:
            self._save_window_embedding(connection, window_id, embedding)
        return embedding

    def _save_window_embedding(self, connection: sqlite3.Connection, window_id: str, embedding: np.ndarray) -> None:
        connection.execute(
            """
            UPDATE context_windows
            SET embedding = ?, embedding_stale = 0, updated_at = ?
            WHERE tenant_id = ? AND id = ?
            """,
            (self.embedding_model.to_bytes(embedding), utc_now().isoformat(), self.tenant_id, window_id),
        )

    def extract_window_entities(self, window_id: str) -> list[dict[str, str]]:
        nodes = self.get_window_nodes(
            window_id,
            node_types=[NodeType.ENTITY, NodeType.FACT, NodeType.DECISION, NodeType.PREFERENCE],
        )
        return [
            {
                "label": node.label,
                "node_type": node.node_type.value,
                "content": node.content,
            }
            for node in nodes
        ]

    def create_context_window_edge(
        self,
        *,
        source_window_id: str,
        target_window_id: str,
        edge_type: str,
        shared_entities: list[str] | None = None,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> ContextWindowEdge:
        entities = sorted({entity.strip().lower() for entity in (shared_entities or []) if entity.strip()})
        edge = ContextWindowEdge(
            tenant_id=self.tenant_id,
            source_window_id=source_window_id,
            target_window_id=target_window_id,
            edge_type=edge_type,
            shared_entities=entities,
            weight=max(0.0, min(1.0, weight)),
            metadata=metadata or {},
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO context_window_edges (
                    id, tenant_id, source_window_id, target_window_id, edge_type,
                    shared_entities, weight, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, source_window_id, target_window_id, edge_type, shared_entities)
                DO UPDATE SET weight = MAX(context_window_edges.weight, excluded.weight),
                              metadata = excluded.metadata
                """,
                (
                    edge.id,
                    edge.tenant_id,
                    edge.source_window_id,
                    edge.target_window_id,
                    edge.edge_type,
                    json.dumps(edge.shared_entities, sort_keys=True),
                    edge.weight,
                    _encode_metadata(edge.metadata),
                    edge.created_at.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT id, tenant_id, source_window_id, target_window_id, edge_type,
                       shared_entities, weight, metadata, created_at
                FROM context_window_edges
                WHERE tenant_id = ? AND source_window_id = ? AND target_window_id = ?
                  AND edge_type = ? AND shared_entities = ?
                """,
                (
                    self.tenant_id,
                    source_window_id,
                    target_window_id,
                    edge_type,
                    json.dumps(edge.shared_entities, sort_keys=True),
                ),
            ).fetchone()
        return self._row_to_context_window_edge(row)

    def derive_context_window_edges(self, window_id: str, repo_id: str) -> list[ContextWindowEdge]:
        current_entities = self.extract_window_entities(window_id)
        if not current_entities:
            return []

        current_by_label = {entity["label"].strip().lower(): entity for entity in current_entities if entity["label"].strip()}
        if not current_by_label:
            return []

        created_edges: list[ContextWindowEdge] = []
        other_windows = self.get_repo_windows(repo_id, exclude=window_id)
        if len(other_windows) > 200:
            other_windows = other_windows[:200]

        for other_window in other_windows:
            other_entities = self.extract_window_entities(other_window.id)
            other_by_label = {entity["label"].strip().lower(): entity for entity in other_entities if entity["label"].strip()}
            overlap = set(current_by_label) & set(other_by_label)
            if not overlap:
                continue

            has_conflict = any(
                normalize_text(current_by_label[label]["content"]) != normalize_text(other_by_label[label]["content"])
                for label in overlap
            )
            edge_type = "supersedes" if has_conflict else "entity_overlap"
            denominator = max(len(current_by_label), len(other_by_label), 1)
            created_edges.append(
                self.create_context_window_edge(
                    source_window_id=other_window.id,
                    target_window_id=window_id,
                    edge_type=edge_type,
                    shared_entities=sorted(overlap),
                    weight=len(overlap) / denominator,
                )
            )

        previous_window = next(iter(other_windows), None)
        if previous_window is not None:
            created_edges.append(
                self.create_context_window_edge(
                    source_window_id=previous_window.id,
                    target_window_id=window_id,
                    edge_type="temporal_sequence",
                    shared_entities=[],
                    weight=1.0,
                )
            )

        LOGGER.info(
            "window_edges_derived",
            extra={
                "window_id": window_id,
                "repo_id": repo_id,
                "edges_created": len(created_edges),
            },
        )
        return created_edges

    def add_node(
        self,
        *,
        label: str,
        content: str,
        node_type: NodeType,
        tags: list[str] | None = None,
        source_prompt: str = "",
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
        evidence_records: list[EvidenceRecord] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> NodeStoreResult:
        _, context_window_id = self.resolve_window_context(project=project, session_id=session_id)
        node = Node(
            tenant_id=self.tenant_id,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
            context_window_id=context_window_id,
            label=label,
            content=content,
            node_type=node_type,
            tags=tags or [],
            source_prompt=source_prompt,
            metadata={},
            evidence_records=evidence_records or [],
            valid_from=valid_from,
            valid_to=valid_to,
        )
        embedding = self.embedding_model.embed(node.content)

        with self._lock, self._connect() as connection:
            duplicate = self._find_duplicate_node(connection, node=node, embedding=embedding)
            if duplicate is not None:
                existing_node, dedup_reason, similarity = duplicate
                merged_node = self._merge_duplicate_node(
                    connection,
                    existing_node=existing_node,
                    incoming_node=node,
                )
                if merged_node.context_window_id:
                    connection.execute(
                        """
                        UPDATE nodes
                        SET context_window_id = COALESCE(context_window_id, ?)
                        WHERE tenant_id = ? AND id = ?
                        """,
                        (merged_node.context_window_id, self.tenant_id, merged_node.id),
                    )
                    self._mark_window_embedding_stale(connection, merged_node.context_window_id)
                return NodeStoreResult(
                    node=merged_node,
                    created=False,
                    dedup_reason=dedup_reason,
                    similarity=similarity,
                )

            connection.execute(
                """
                INSERT INTO nodes (
                    id, tenant_id, agent_id, project, session_id, context_window_id,
                    label, content, node_type, tags, metadata, embedding,
                    source_prompt, evidence_records, valid_from, valid_to,
                    created_at, updated_at, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.tenant_id,
                    node.agent_id,
                    node.project,
                    node.session_id,
                    node.context_window_id,
                    node.label,
                    node.content,
                    node.node_type.value,
                    json.dumps(node.tags),
                    _encode_metadata(node.metadata),
                    self.embedding_model.to_bytes(embedding),
                    node.source_prompt,
                    _encode_evidence_records(node.evidence_records),
                    node.valid_from.isoformat() if node.valid_from is not None else None,
                    node.valid_to.isoformat() if node.valid_to is not None else None,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                    node.access_count,
                ),
            )
            self._mark_window_embedding_stale(connection, context_window_id)
            self._update_window_node_count(connection, context_window_id)
            conflicts = self._register_conflicts(connection, node)
        return NodeStoreResult(node=node, created=True, conflicts=conflicts)

    def add_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        relationship: str | RelationType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
        ) -> Edge:
        edge = Edge(
            tenant_id=self.tenant_id,
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            weight=weight,
            metadata=metadata or {},
        )

        with self._lock, self._connect() as connection:
            self._require_node(connection, edge.source_id)
            self._require_node(connection, edge.target_id)
            source_node = self.get_node(edge.source_id)
            target_node = self.get_node(edge.target_id)
            existing_edge = self._find_existing_edge(
                connection,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relationship=edge.relationship,
            )
            if existing_edge is not None:
                return existing_edge
            connection.execute(
                """
                INSERT INTO edges (
                    id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    edge.tenant_id,
                    edge.source_id,
                    edge.target_id,
                    edge.relationship,
                    edge.weight,
                    json.dumps(edge.metadata),
                    edge.created_at.isoformat(),
                ),
            )
            if edge.relationship in {RelationType.UPDATES.value, RelationType.CONTRADICTS.value}:
                self._mark_node_superseded(connection, old_node=target_node, new_node=source_node, relationship=edge.relationship)
        return edge

    def get_node(self, node_id: str) -> Node:
        with self._lock, self._connect() as connection:
            row = self._fetch_node_row(connection, node_id)
            if row is None:
                raise ValueError(f"Node not found: {node_id}")
            return self._row_to_node(row)

    def get_node_history(self, *, node_id: str, max_depth: int = 2) -> NodeHistoryResult:
        node = self.get_node(node_id)
        related = self.get_related(node_id=node_id, max_depth=max_depth)
        related_nodes = [item for item in related.nodes if item.id != node_id]
        return NodeHistoryResult(node=node, related_nodes=related_nodes, edges=related.edges)

    def timeline(
        self,
        *,
        node_id: str = "",
        query: str = "",
        limit: int = 25,
        max_depth: int = 2,
        include_evidence: bool = True,
    ) -> TimelineResult:
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        if node_id.strip() and query.strip():
            raise ValueError("Provide either node_id or query, not both.")

        if node_id.strip():
            related = self.get_related(node_id=node_id, max_depth=max_depth)
            nodes = related.nodes
            edges = related.edges
            scope = f"node:{node_id.strip()}"
        elif query.strip():
            subgraph = self.query(query=query, max_nodes=max(limit, 10), max_depth=max_depth)
            nodes = subgraph.nodes
            edges = subgraph.edges
            scope = f"query:{query.strip()}"
        else:
            with self._lock, self._connect() as connection:
                nodes = self.list_recent_nodes(limit=max(limit, 10))
                edges = self._fetch_edges_for_nodes(connection, [node.id for node in nodes])
            scope = "tenant"

        items = self._build_timeline_items(
            nodes=nodes,
            edges=edges,
            include_evidence=include_evidence,
            limit=limit,
        )
        return TimelineResult(scope=scope, items=items)

    def list_conflicts(
        self,
        *,
        include_resolved: bool = False,
        limit: int = 25,
    ) -> ConflictListResult:
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        with self._lock, self._connect() as connection:
            edge_rows = connection.execute(
                """
                SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
                FROM edges
                WHERE tenant_id = ?
                  AND relationship IN (?, ?)
                ORDER BY created_at DESC
                """,
                (self.tenant_id, RelationType.CONTRADICTS.value, RelationType.UPDATES.value),
            ).fetchall()
            edges = [self._row_to_edge(row) for row in edge_rows]
            entries = self._build_conflict_entries(
                connection,
                edges=edges,
                include_resolved=include_resolved,
                limit=limit,
            )
        return ConflictListResult(conflicts=entries, include_resolved=include_resolved)

    def resolve_conflict(
        self,
        *,
        edge_id: str,
        resolution_note: str = "",
    ) -> ConflictEntry:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
                FROM edges
                WHERE tenant_id = ? AND id = ?
                LIMIT 1
                """,
                (self.tenant_id, edge_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Conflict edge not found: {edge_id}")
            edge = self._row_to_edge(row)
            if edge.relationship not in {RelationType.CONTRADICTS.value, RelationType.UPDATES.value}:
                raise ValueError("Only contradicts or updates edges can be resolved.")

            metadata = dict(edge.metadata)
            metadata["resolved"] = True
            metadata["resolved_at"] = utc_now().isoformat()
            if resolution_note.strip():
                metadata["resolution_note"] = resolution_note.strip()

            connection.execute(
                """
                UPDATE edges
                SET metadata = ?
                WHERE tenant_id = ? AND id = ?
                """,
                (json.dumps(metadata, sort_keys=True), self.tenant_id, edge_id),
            )
            self._mark_node_superseded(
                connection,
                old_node=self.get_node(edge.target_id),
                new_node=self.get_node(edge.source_id),
                relationship=edge.relationship,
            )
            updated_edge = Edge(
                id=edge.id,
                tenant_id=edge.tenant_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relationship=edge.relationship,
                weight=edge.weight,
                metadata=metadata,
                created_at=edge.created_at,
            )
            entry = self._build_conflict_entries(
                connection,
                edges=[updated_edge],
                include_resolved=True,
                limit=1,
            )
        if not entry:
            raise ValueError(f"Resolved conflict could not be loaded: {edge_id}")
        return entry[0]

    def query(
        self,
        *,
        query: str,
        max_nodes: int = 20,
        max_depth: int = 2,
        expand_depth: int = 0,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
        retrieval_mode: str = "graph",
    ) -> SubgraphResult:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Query cannot be empty.")
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        if expand_depth < 0:
            raise ValueError("expand_depth cannot be negative.")
        normalized_mode = retrieval_mode.strip().lower()
        if normalized_mode not in {"graph", "replay", "fusion"}:
            raise ValueError("retrieval_mode must be one of: graph, replay, fusion.")

        graph_result = (
            self.tiered_query(
                query=query_text,
                project=project,
                max_nodes=max_nodes,
                max_depth=max_depth,
                top_k_windows=self.tiered_retrieval_top_k_windows,
            )
            if self.tiered_retrieval and project.strip()
            else self._query_graph_only(
                    query=query_text,
                    max_nodes=max_nodes,
                    max_depth=max_depth,
                    expand_depth=expand_depth,
                    agent_id=agent_id,
                    project=project,
                    session_id=session_id,
                )
            if normalized_mode in {"graph", "fusion"}
            else None
        )
        replay_hits = (
            self._query_replay_hits(
                query=query_text,
                max_hits=max_nodes,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
            )
            if normalized_mode in {"replay", "fusion"}
            else []
        )
        if normalized_mode == "graph":
            if graph_result.retrieval_mode not in {"tiered", "flat_fallback"}:
                graph_result.retrieval_mode = "graph"
            return graph_result
        if normalized_mode == "replay":
            return SubgraphResult(
                replay_hits=replay_hits,
                retrieval_mode="replay",
                query=query_text,
                total_nodes_in_graph=graph_result.total_nodes_in_graph if graph_result is not None else 0,
            )
        fusion_hits = self._build_fusion_hits(graph_result or SubgraphResult(query=query_text), replay_hits)
        return SubgraphResult(
            nodes=graph_result.nodes if graph_result is not None else [],
            edges=graph_result.edges if graph_result is not None else [],
            replay_hits=replay_hits,
            fusion_hits=fusion_hits[:max_nodes],
            retrieval_mode="fusion",
            query=query_text,
            total_nodes_in_graph=graph_result.total_nodes_in_graph if graph_result is not None else 0,
        )

    def tiered_query(
        self,
        *,
        query: str,
        project: str = "",
        repo_id: str | None = None,
        max_nodes: int = 20,
        max_depth: int = 2,
        top_k_windows: int | None = None,
    ) -> SubgraphResult:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Query cannot be empty.")
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        resolved_repo_id = repo_id or self.ensure_repo(project or "default")
        query_embedding = self.embedding_model.embed(self._expand_query_aliases(query_text))
        windows = self.get_repo_windows(resolved_repo_id)
        now = time.time()
        replay_session_scores = self._query_replay_session_scores(
            query=query_text,
            query_embedding=query_embedding,
            agent_id="",
            project=project,
            session_id="",
        )
        window_scores: list[tuple[float, ContextWindow]] = []
        for window in windows:
            window_embedding = self.get_window_embedding(window.id)
            if window_embedding is None:
                continue
            similarity = max(self.embedding_model.cosine_similarity(query_embedding, window_embedding), 0.0)
            similarity = self._blend_session_signal(
                base_similarity=similarity,
                session_signal=replay_session_scores.get(window.session_id, 0.0),
            )
            recency = recency_weight(
                window.updated_at.timestamp(),
                now=now,
                half_life_days=self.recency_half_life_days,
            )
            window_scores.append(((0.6 * similarity) + (0.4 * recency), window))

        if not window_scores:
            fallback = self._query_graph_only(
                query=query_text,
                max_nodes=max_nodes,
                max_depth=max_depth,
                expand_depth=0,
                agent_id="",
                project=project,
                session_id="",
            )
            fallback.retrieval_mode = "flat_fallback"
            return fallback

        window_scores.sort(key=lambda item: (item[0], item[1].updated_at.timestamp()), reverse=True)
        selected_windows = [window for _, window in window_scores[: max(1, top_k_windows or self.tiered_retrieval_top_k_windows)]]
        selected_window_ids = {window.id for window in selected_windows}

        with self._lock, self._connect() as connection:
            candidate_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, context_window_id, label, content, node_type,
                       tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, embedding, tenant_id
                FROM nodes
                WHERE tenant_id = ? AND context_window_id IN ({})
                  AND embedding IS NOT NULL
                """.format(", ".join("?" for _ in selected_window_ids)),
                (self.tenant_id, *selected_window_ids),
            ).fetchall()
            total_nodes = int(
                connection.execute(
                    "SELECT COUNT(*) FROM nodes WHERE tenant_id = ?",
                    (self.tenant_id,),
                ).fetchone()[0]
            )

            if not candidate_rows:
                fallback = self._query_graph_only(
                    query=query_text,
                    max_nodes=max_nodes,
                    max_depth=max_depth,
                    expand_depth=0,
                    agent_id="",
                    project=project,
                    session_id="",
                )
                fallback.retrieval_mode = "flat_fallback"
                return fallback

            candidates: list[Node] = []
            similarity_by_id: dict[str, float] = {}
            for row in candidate_rows:
                node = self._row_to_node(row)
                semantic = max(
                    self.embedding_model.cosine_similarity(query_embedding, self.embedding_model.from_bytes(row["embedding"])),
                    0.0,
                )
                lexical = self._lexical_score_for_node(query_text, node)
                similarity = max(0.0, min(1.0, (0.8 * semantic) + (0.2 * lexical)))
                similarity = self._blend_session_signal(
                    base_similarity=similarity,
                    session_signal=replay_session_scores.get(node.session_id, 0.0),
                )
                candidates.append(node)
                similarity_by_id[node.id] = similarity

            candidate_ids = [node.id for node in candidates]
            edges = self._fetch_edges_for_nodes(connection, candidate_ids)
            scored_nodes = [
                self._apply_node_score(
                    node,
                    similarity=similarity_by_id.get(node.id, 0.0),
                    edge_weight=self._strongest_edge_weight(node.id, edges),
                    now=now,
                )
                for node in candidates
            ]
            scored_nodes.sort(
                key=lambda node: (
                    node.final_score if node.final_score is not None else 0.0,
                    node.updated_at.timestamp(),
                    node.label.lower(),
                ),
                reverse=True,
            )
            selected_nodes = scored_nodes[:max_nodes]
            selected_ids = [node.id for node in selected_nodes]
            selected_edges = self._fetch_edges_for_nodes(connection, selected_ids)
            self._increment_access_counts(connection, selected_ids)
            for node in selected_nodes:
                node.access_count += 1

        return SubgraphResult(
            nodes=selected_nodes,
            edges=selected_edges,
            retrieval_mode="tiered",
            query=query_text,
            total_nodes_in_graph=total_nodes,
        )

    def debug_retrieval(
        self,
        *,
        query: str,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        max_nodes: int = 10,
        max_depth: int = 2,
    ) -> dict[str, Any]:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Query cannot be empty.")
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        expanded_query = self._expand_query_aliases(query_text)
        query_embedding = self.embedding_model.embed(expanded_query)
        repo_id = self.ensure_repo(project or "default")
        now = time.time()

        replay_session_scores = self._query_replay_session_scores(
            query=expanded_query,
            query_embedding=query_embedding,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
        )
        windows = self.get_repo_windows(repo_id)
        window_details: list[dict[str, Any]] = []
        for window in windows:
            recency = recency_weight(
                window.updated_at.timestamp(),
                now=now,
                half_life_days=self.recency_half_life_days,
            )
            detail: dict[str, Any] = {
                "window_id": window.id,
                "repo_id": window.repo_id,
                "session_id": window.session_id,
                "title": window.title,
                "status": window.status,
                "node_count": window.node_count,
                "embedding": "missing",
                "embedding_stale": window.embedding_stale,
                "similarity": None,
                "recency": round(float(recency), 4),
                "routing_score": None,
                "updated_at": window.updated_at.isoformat(),
            }
            window_embedding = self.get_window_embedding(window.id)
            if window_embedding is not None:
                similarity = max(self.embedding_model.cosine_similarity(query_embedding, window_embedding), 0.0)
                similarity = self._blend_session_signal(
                    base_similarity=similarity,
                    session_signal=replay_session_scores.get(window.session_id, 0.0),
                )
                routing_score = (0.6 * similarity) + (0.4 * recency)
                detail.update(
                    {
                        "embedding": "ok",
                        "similarity": round(float(similarity), 4),
                        "routing_score": round(float(routing_score), 4),
                    }
                )
            window_details.append(detail)

        window_details.sort(
            key=lambda item: (
                item["routing_score"] if item["routing_score"] is not None else -1.0,
                item["updated_at"],
            ),
            reverse=True,
        )

        flat_result = self._query_graph_only(
            query=query_text,
            max_nodes=max_nodes,
            max_depth=max_depth,
            expand_depth=0,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
        )
        tiered_result = self.tiered_query(
            query=query_text,
            project=project,
            repo_id=repo_id,
            max_nodes=max_nodes,
            max_depth=max_depth,
            top_k_windows=self.tiered_retrieval_top_k_windows,
        )

        def summarize_node(node: Node) -> dict[str, Any]:
            return {
                "node_id": node.id,
                "label": node.label,
                "node_type": node.node_type.value,
                "project": node.project,
                "session_id": node.session_id,
                "context_window_id": node.context_window_id,
                "similarity_score": node.similarity_score,
                "recency_score": node.recency_score,
                "edge_score": node.edge_score,
                "final_score": node.final_score,
                "updated_at": node.updated_at.isoformat(),
            }

        return {
            "query": query_text,
            "expanded_query": expanded_query,
            "repo_id": repo_id,
            "project": project,
            "agent_id": agent_id,
            "session_id": session_id,
            "retrieval_mode": "tiered" if self.tiered_retrieval else "flat",
            "embedding_preview": [round(float(value), 6) for value in query_embedding[:5]],
            "windows_evaluated": len(window_details),
            "all_windows": window_details,
            "selected_windows": [
                window
                for window in window_details
                if window["routing_score"] is not None
            ][: max(1, self.tiered_retrieval_top_k_windows)],
            "flat_top_nodes": [summarize_node(node) for node in flat_result.nodes[:max_nodes]],
            "tiered_top_nodes": [summarize_node(node) for node in tiered_result.nodes[:max_nodes]],
            "tiered_result_mode": tiered_result.retrieval_mode,
        }

    def _query_graph_only(
        self,
        *,
        query: str,
        max_nodes: int,
        max_depth: int,
        expand_depth: int,
        agent_id: str,
        project: str,
        session_id: str,
    ) -> SubgraphResult:
        with self._lock, self._connect() as connection:
            temporal_hints = infer_temporal_hints(query)
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, context_window_id, label, content, node_type, tags,
                       source_prompt, metadata, evidence_records, valid_from, valid_to, created_at,
                       updated_at, access_count, embedding, tenant_id
                FROM nodes
                WHERE tenant_id = ? AND embedding IS NOT NULL
                """,
                (self.tenant_id,),
            ).fetchall()
            total_nodes = len(node_rows)
            if total_nodes == 0:
                return SubgraphResult(query=query, total_nodes_in_graph=0)

            def collect_scoped_nodes(active_session_id: str) -> tuple[dict[str, Node], dict[str, np.ndarray]]:
                scoped_nodes: dict[str, Node] = {}
                scoped_embeddings: dict[str, np.ndarray] = {}
                for row in node_rows:
                    node = self._row_to_node(row)
                    if not _scope_matches(node, agent_id=agent_id, project=project, session_id=active_session_id):
                        continue
                    scoped_nodes[node.id] = node
                    scoped_embeddings[node.id] = self.embedding_model.from_bytes(row["embedding"])
                return scoped_nodes, scoped_embeddings

            active_session_id = _retrieval_session_scope(
                agent_id=agent_id,
                project=project,
                session_id=session_id,
            )
            nodes_by_id, embeddings_by_id = collect_scoped_nodes(active_session_id)

            if not nodes_by_id:
                return SubgraphResult(query=query, total_nodes_in_graph=total_nodes)

            expanded_query = self._expand_query_aliases(query)
            query_embedding = self.embedding_model.embed(expanded_query)
            similarity_by_id = {
                node_id: max(self.embedding_model.cosine_similarity(query_embedding, embedding), 0.0)
                for node_id, embedding in embeddings_by_id.items()
            }
            replay_session_scores = self._query_replay_session_scores(
                query=expanded_query,
                query_embedding=query_embedding,
                agent_id=agent_id,
                project=project,
                session_id=active_session_id,
            )
            similarity_by_id = {
                node_id: self._blend_session_signal(
                    base_similarity=similarity,
                    session_signal=replay_session_scores.get(nodes_by_id[node_id].session_id, 0.0),
                )
                for node_id, similarity in similarity_by_id.items()
            }
            lexical_by_id = {
                node_id: self._lexical_score_for_node(expanded_query, node)
                for node_id, node in nodes_by_id.items()
            }
            negation_intent = self._has_negation_intent(query)
            negation_boost_by_id = {
                node_id: self._negation_boost(node) if negation_intent else 0.0
                for node_id, node in nodes_by_id.items()
            }

            seed_count = min(total_nodes, max(1, max_nodes // 2))
            seed_candidates = [
                (
                    node_id,
                    (0.7 * similarity_by_id.get(node_id, 0.0))
                    + (0.3 * lexical_by_id.get(node_id, 0.0)),
                    negation_boost_by_id.get(node_id, 0.0),
                    self._seed_temporal_order(nodes_by_id[node_id], temporal_hints),
                )
                for node_id in nodes_by_id
            ]
            if temporal_hints.recency_mode in {"latest", "oldest"}:
                temporal_seed_candidates = [
                    item
                    for item in seed_candidates
                    if item[1] >= TOPIC_RELEVANCE_THRESHOLD
                    and (
                        lexical_by_id.get(item[0], 0.0) > 0.0
                        or similarity_by_id.get(item[0], 0.0) >= TOPIC_SEMANTIC_ONLY_THRESHOLD
                    )
                ]
                if not temporal_seed_candidates:
                    temporal_seed_candidates = sorted(
                        seed_candidates,
                        key=lambda item: (-(item[1] + item[2]), nodes_by_id[item[0]].label.lower()),
                    )[: max_nodes * 2]
                ranked_seed_ids = [
                    item[0]
                    for item in sorted(
                        temporal_seed_candidates,
                        key=lambda item: (item[3], -(item[1] + item[2]), nodes_by_id[item[0]].label.lower()),
                    )[:seed_count]
                ]
            else:
                ranked_seed_ids = [
                    item[0]
                    for item in sorted(
                        seed_candidates,
                        key=lambda item: (-(item[1] + item[2]), item[3], nodes_by_id[item[0]].label.lower()),
                    )[:seed_count]
                ]
            if len(self._split_query_intents(query)) >= 2:
                ranked_seed_ids = self._add_clause_seed_ids(
                    query=query,
                    ranked_seed_ids=ranked_seed_ids,
                    nodes_by_id=nodes_by_id,
                    embeddings_by_id=embeddings_by_id,
                    max_seeds=max_nodes,
                )

            graph = self._load_graph(connection, node_ids=nodes_by_id.keys())
            expanded_depths, expansion_metadata = self._expand_node_depths_with_context(
                graph, ranked_seed_ids, max_depth
            )
            candidate_nodes = [nodes_by_id[node_id] for node_id in expanded_depths]
            temporal_candidates = [node for node in candidate_nodes if within_time_window(node, temporal_hints)]
            if temporal_candidates:
                candidate_nodes = temporal_candidates

            max_access = max((node.access_count for node in candidate_nodes), default=0)
            degree_by_id = dict(graph.degree(expanded_depths.keys()))
            max_degree = max(degree_by_id.values(), default=0)
            candidate_edges = self._fetch_edges_for_nodes(connection, [node.id for node in candidate_nodes])
            scored_nodes = self._sort_scored_nodes(
                candidate_nodes,
                max_nodes=max_nodes,
                temporal_hints=temporal_hints,
                similarity_by_id=similarity_by_id,
                lexical_by_id=lexical_by_id,
                negation_boost_by_id=negation_boost_by_id,
                degree_by_id=degree_by_id,
                max_access=max_access,
                max_degree=max_degree,
                max_depth=max_depth,
                expanded_depths=expanded_depths,
                edges=candidate_edges,
                expansion_metadata=expansion_metadata,
            )
            scored_nodes = self._diversify_multi_intent_nodes(
                query=query,
                ranked_nodes=scored_nodes,
                embeddings_by_id=embeddings_by_id,
                max_nodes=max_nodes,
            )
            result_limit = max_nodes if expand_depth == 0 else max_nodes + max(1, max_nodes // 2)
            selected_nodes = self._enforce_clause_coverage(
                query=query,
                selected_nodes=scored_nodes[:result_limit],
                ranked_nodes=scored_nodes,
                embeddings_by_id=embeddings_by_id,
                max_nodes=result_limit,
            )
            candidate_pool = {node.id: node for node in candidate_nodes}
            selected_nodes = self._ensure_support_coverage(selected_nodes, candidate_pool, graph, result_limit)
            selected_ids = [node.id for node in selected_nodes]

            edges = self._fetch_edges_for_nodes(connection, selected_ids)
            self._increment_access_counts(connection, selected_ids)
            for node in selected_nodes:
                node.access_count += 1

            return SubgraphResult(
                nodes=selected_nodes,
                edges=edges,
                retrieval_mode="graph",
                query=query,
                total_nodes_in_graph=total_nodes,
            )

    def _query_replay_hits(
        self,
        *,
        query: str,
        max_hits: int,
        agent_id: str,
        project: str,
        session_id: str,
    ) -> list[ReplayHit]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, tenant_id, agent_id, project, session_id, observed_at, turn_index, role, transcript_text, embedding, metadata
                FROM transcript_records
                WHERE tenant_id = ? AND embedding IS NOT NULL
                ORDER BY observed_at DESC, turn_index DESC
                """,
                (self.tenant_id,),
            ).fetchall()
        if not rows:
            return []
        query_embedding = self.embedding_model.embed(query)
        temporal_hints = infer_temporal_hints(query)
        timestamps = np.asarray([_parse_datetime(row["observed_at"]).timestamp() for row in rows], dtype=np.float64)
        max_timestamp = float(np.max(timestamps))
        min_timestamp = float(np.min(timestamps))
        span = max(max_timestamp - min_timestamp, 1.0)

        def build_hits(active_session_id: str) -> list[tuple[float, ReplayHit]]:
            hits: list[tuple[float, ReplayHit]] = []
            for row, raw_timestamp in zip(rows, timestamps, strict=True):
                record = self._row_to_transcript_record(row)
                if not self._transcript_scope_matches(record, agent_id=agent_id, project=project, session_id=active_session_id):
                    continue
                embedding = self.embedding_model.from_bytes(row["embedding"])
                semantic_score = max(self.embedding_model.cosine_similarity(query_embedding, embedding), 0.0)
                lexical_score = lexical_overlap(query, record.role, record.transcript_text)
                temporal_score = 0.0
                if temporal_hints.recency_mode == "latest":
                    temporal_score = float((raw_timestamp - min_timestamp) / span)
                elif temporal_hints.recency_mode == "oldest":
                    temporal_score = float((max_timestamp - raw_timestamp) / span)
                role_score = 1.0 if record.role == "user" else 0.8
                score = (0.6 * semantic_score) + (0.2 * lexical_score) + (0.1 * temporal_score) + (0.1 * role_score)
                hits.append(
                    (
                        score,
                        ReplayHit(
                            score=score,
                            session_id=record.session_id,
                            turn_index=record.turn_index,
                            role=record.role,
                            transcript_text=record.transcript_text,
                            transcript_snippet=record.transcript_text[:280],
                            observed_at=record.observed_at,
                        ),
                    )
                )
            return hits

        active_session_id = _retrieval_session_scope(
            agent_id=agent_id,
            project=project,
            session_id=session_id,
        )
        hits = build_hits(active_session_id)
        return [item[1] for item in sorted(hits, key=lambda item: (-item[0], -item[1].observed_at.timestamp(), item[1].turn_index))[:max_hits]]

    def _query_replay_session_scores(
        self,
        *,
        query: str,
        query_embedding: np.ndarray | None = None,
        agent_id: str,
        project: str,
        session_id: str,
    ) -> dict[str, float]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, tenant_id, agent_id, project, session_id, observed_at, turn_index, role, transcript_text, embedding, metadata
                FROM transcript_records
                WHERE tenant_id = ? AND embedding IS NOT NULL
                ORDER BY observed_at DESC, turn_index DESC
                """,
                (self.tenant_id,),
            ).fetchall()
        if not rows:
            return {}

        query_vector = query_embedding if query_embedding is not None else self.embedding_model.embed(query)
        active_session_id = _retrieval_session_scope(
            agent_id=agent_id,
            project=project,
            session_id=session_id,
        )
        scores_by_session: dict[str, float] = {}
        for row in rows:
            record = self._row_to_transcript_record(row)
            if not self._transcript_scope_matches(record, agent_id=agent_id, project=project, session_id=active_session_id):
                continue
            scoped_session_id = record.session_id.strip()
            if not scoped_session_id:
                continue
            embedding = self.embedding_model.from_bytes(row["embedding"])
            semantic_score = max(self.embedding_model.cosine_similarity(query_vector, embedding), 0.0)
            lexical_score = lexical_overlap(query, record.role, record.transcript_text)
            role_score = 1.0 if record.role == "user" else 0.8
            score = max(0.0, min(1.0, (0.65 * semantic_score) + (0.25 * lexical_score) + (0.10 * role_score)))
            previous = scores_by_session.get(scoped_session_id, 0.0)
            if score > previous:
                scores_by_session[scoped_session_id] = score
        return scores_by_session

    def _recent_transcript_session_scores(
        self,
        *,
        agent_id: str,
        project: str,
        session_id: str,
    ) -> dict[str, float]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, tenant_id, agent_id, project, session_id, observed_at, turn_index, role, transcript_text, metadata
                FROM transcript_records
                WHERE tenant_id = ?
                ORDER BY observed_at DESC, turn_index DESC
                """,
                (self.tenant_id,),
            ).fetchall()
        if not rows:
            return {}

        active_session_id = _retrieval_session_scope(
            agent_id=agent_id,
            project=project,
            session_id=session_id,
        )
        timestamps = [
            self._row_to_transcript_record(row).observed_at.timestamp()
            for row in rows
            if self._transcript_scope_matches(
                self._row_to_transcript_record(row),
                agent_id=agent_id,
                project=project,
                session_id=active_session_id,
            )
        ]
        if not timestamps:
            return {}
        now = max(timestamps)
        scores_by_session: dict[str, float] = {}
        for row in rows:
            record = self._row_to_transcript_record(row)
            if not self._transcript_scope_matches(record, agent_id=agent_id, project=project, session_id=active_session_id):
                continue
            scoped_session_id = record.session_id.strip()
            if not scoped_session_id:
                continue
            score = recency_weight(
                record.observed_at.timestamp(),
                now=now,
                half_life_days=self.recency_half_life_days,
            )
            previous = scores_by_session.get(scoped_session_id, 0.0)
            if score > previous:
                scores_by_session[scoped_session_id] = score
        return scores_by_session

    def _blend_session_signal(
        self,
        *,
        base_similarity: float,
        session_signal: float,
        session_weight: float = 0.25,
    ) -> float:
        base = max(0.0, min(1.0, base_similarity))
        session = max(0.0, min(1.0, session_signal))
        return max(0.0, min(1.0, ((1.0 - session_weight) * base) + (session_weight * session)))

    def _build_fusion_hits(self, graph_result: SubgraphResult, replay_hits: list[ReplayHit]) -> list[FusionHit]:
        rrf_k = 60.0
        replay_by_session = {hit.session_id for hit in replay_hits if hit.session_id}
        graph_edge_map: dict[str, list[dict[str, Any]]] = {}
        graph_nodes_by_session = {node.session_id: node for node in graph_result.nodes if node.session_id}
        combined: dict[str, FusionHit] = {}

        for edge in graph_result.edges:
            payload = {
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relationship": edge.relationship,
                "weight": edge.weight,
            }
            graph_edge_map.setdefault(edge.source_id, []).append(payload)
            graph_edge_map.setdefault(edge.target_id, []).append(payload)

        for index, node in enumerate(graph_result.nodes, start=1):
            source_lane = "both" if node.session_id and node.session_id in replay_by_session else "graph"
            combined[f"graph:{node.id}"] = FusionHit(
                content=node.content,
                score=1.0 / (rrf_k + index),
                source_lane=source_lane,
                graph_rank=index,
                replay_rank=None,
                fused_rank=0,
                node_id=node.id,
                node_type=node.node_type.value,
                edges=graph_edge_map.get(node.id, []),
                session_id=node.session_id or None,
            )

        for index, hit in enumerate(replay_hits, start=1):
            contribution = 1.0 / (rrf_k + index)
            matching_graph = graph_nodes_by_session.get(hit.session_id) if hit.session_id else None
            if matching_graph is not None:
                existing = combined.get(f"graph:{matching_graph.id}")
                if existing is not None:
                    existing.score += contribution
                    existing.source_lane = "both"
                    existing.replay_rank = index
                    existing.session_id = hit.session_id or None
                    continue
                key = f"both:{matching_graph.id}:{hit.session_id}:{hit.turn_index}"
                source_lane = "both"
            else:
                key = f"replay:{hit.session_id}:{hit.turn_index}:{index}"
                source_lane = "replay"
            combined[key] = FusionHit(
                content=hit.transcript_text,
                score=contribution,
                source_lane=source_lane,
                graph_rank=None,
                replay_rank=index,
                fused_rank=0,
                session_id=hit.session_id or None,
                transcript_snippet=hit.transcript_snippet,
                turn_index=hit.turn_index,
            )

        ordered = sorted(
            combined.values(),
            key=lambda item: (-item.score, 0 if item.source_lane in {"both", "graph"} else 1, item.content.lower()),
        )
        for index, item in enumerate(ordered, start=1):
            item.fused_rank = index
        return ordered

    def get_related(self, *, node_id: str, max_depth: int = 2) -> SubgraphResult:
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        with self._lock, self._connect() as connection:
            self._require_node(connection, node_id)
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                """
            , (self.tenant_id,)).fetchall()
            nodes_by_id = {row["id"]: self._row_to_node(row) for row in node_rows}
            graph = self._load_graph(connection, node_ids=nodes_by_id.keys())
            related_ids = list(self._expand_node_depths(graph, [node_id], max_depth))

            ordered_nodes: list[Node] = []
            seen: set[str] = set()
            for related_id in [node_id, *related_ids]:
                if related_id in seen:
                    continue
                seen.add(related_id)
                ordered_nodes.append(nodes_by_id[related_id])

            edges = self._fetch_edges_for_nodes(connection, [node.id for node in ordered_nodes])
            now = time.time()
            for node in ordered_nodes:
                distance = 0 if node.id == node_id else nx.shortest_path_length(graph, source=node_id, target=node.id)
                edge_weight = self._strongest_edge_weight(node.id, edges)
                similarity = max(0.0, 1.0 - (0.25 * distance))
                self._apply_node_score(node, similarity=similarity, edge_weight=edge_weight, now=now)
            ordered_nodes.sort(
                key=lambda node: (
                    -(node.final_score if node.final_score is not None else 0.0),
                    0 if node.id == node_id else 1,
                    -node.updated_at.timestamp(),
                    node.label.lower(),
                )
            )
            self._increment_access_counts(connection, [node.id for node in ordered_nodes])
            for node in ordered_nodes:
                node.access_count += 1

            return SubgraphResult(
                nodes=ordered_nodes,
                edges=edges,
                query=f"related:{node_id}",
                total_nodes_in_graph=len(nodes_by_id),
            )

    def update_node(
        self,
        *,
        node_id: str,
        content: str | None = None,
        label: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        project: str | None = None,
        session_id: str | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        evidence_records: list[EvidenceRecord] | None = None,
    ) -> Node:
        if (
            content is None
            and label is None
            and tags is None
            and agent_id is None
            and project is None
            and session_id is None
            and valid_from is None
            and valid_to is None
            and evidence_records is None
        ):
            raise ValueError("At least one field must be provided for update.")

        with self._lock, self._connect() as connection:
            row = self._fetch_node_row(connection, node_id)
            if row is None:
                raise ValueError(f"Node not found: {node_id}")

            node = self._row_to_node(row)
            updated_label = label if label is not None else node.label
            updated_content = content if content is not None else node.content
            updated_tags = tags if tags is not None else node.tags
            updated_at = utc_now()
            embedding_bytes = row["embedding"]
            if content is not None:
                embedding_bytes = self.embedding_model.to_bytes(self.embedding_model.embed(updated_content))

            updated_node = Node(
                id=node.id,
                tenant_id=node.tenant_id,
                agent_id=agent_id if agent_id is not None else node.agent_id,
                project=project if project is not None else node.project,
                session_id=session_id if session_id is not None else node.session_id,
                label=updated_label,
                content=updated_content,
                node_type=node.node_type,
                tags=updated_tags,
                source_prompt=node.source_prompt,
                metadata=node.metadata,
                evidence_records=evidence_records if evidence_records is not None else node.evidence_records,
                valid_from=valid_from if valid_from is not None else node.valid_from,
                valid_to=valid_to if valid_to is not None else node.valid_to,
                created_at=node.created_at,
                updated_at=updated_at,
                access_count=node.access_count,
            )

            connection.execute(
                """
                UPDATE nodes
                SET label = ?, content = ?, tags = ?, metadata = ?, embedding = ?, updated_at = ?,
                    agent_id = ?, project = ?, session_id = ?,
                    evidence_records = ?, valid_from = ?, valid_to = ?
                WHERE id = ? AND tenant_id = ?
                """,
                (
                    updated_node.label,
                    updated_node.content,
                    json.dumps(updated_node.tags),
                    _encode_metadata(updated_node.metadata),
                    embedding_bytes,
                    updated_node.updated_at.isoformat(),
                    updated_node.agent_id,
                    updated_node.project,
                    updated_node.session_id,
                    _encode_evidence_records(updated_node.evidence_records),
                    updated_node.valid_from.isoformat() if updated_node.valid_from is not None else None,
                    updated_node.valid_to.isoformat() if updated_node.valid_to is not None else None,
                    updated_node.id,
                    self.tenant_id,
                ),
            )
            return updated_node

    def update_edge(
        self,
        *,
        edge_id: str,
        source_id: str | None = None,
        target_id: str | None = None,
        relationship: str | RelationType | None = None,
        weight: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        if (
            source_id is None
            and target_id is None
            and relationship is None
            and weight is None
            and metadata is None
        ):
            raise ValueError("At least one field must be provided for edge update.")

        with self._lock, self._connect() as connection:
            row = self._fetch_edge_row(connection, edge_id)
            if row is None:
                raise ValueError(f"Edge not found: {edge_id}")
            edge = self._row_to_edge(row)
            updated_edge = Edge(
                id=edge.id,
                tenant_id=edge.tenant_id,
                source_id=source_id if source_id is not None else edge.source_id,
                target_id=target_id if target_id is not None else edge.target_id,
                relationship=relationship if relationship is not None else edge.relationship,
                weight=weight if weight is not None else edge.weight,
                metadata=metadata if metadata is not None else edge.metadata,
                created_at=edge.created_at,
            )
            self._require_node(connection, updated_edge.source_id)
            self._require_node(connection, updated_edge.target_id)
            connection.execute(
                """
                UPDATE edges
                SET source_id = ?, target_id = ?, relationship = ?, weight = ?, metadata = ?
                WHERE id = ? AND tenant_id = ?
                """,
                (
                    updated_edge.source_id,
                    updated_edge.target_id,
                    updated_edge.relationship,
                    updated_edge.weight,
                    _encode_metadata(updated_edge.metadata),
                    edge_id,
                    self.tenant_id,
                ),
            )
            return updated_edge

    def delete_edge(self, *, edge_id: str) -> Edge:
        with self._lock, self._connect() as connection:
            row = self._fetch_edge_row(connection, edge_id)
            if row is None:
                raise ValueError(f"Edge not found: {edge_id}")
            edge = self._row_to_edge(row)
            connection.execute("DELETE FROM edges WHERE id = ? AND tenant_id = ?", (edge_id, self.tenant_id))
            return edge

    def delete_node(self, *, node_id: str) -> Node:
        with self._lock, self._connect() as connection:
            row = self._fetch_node_row(connection, node_id)
            if row is None:
                raise ValueError(f"Node not found: {node_id}")
            node = self._row_to_node(row)
            connection.execute("DELETE FROM nodes WHERE id = ? AND tenant_id = ?", (node_id, self.tenant_id))
            return node

    def list_recent_nodes(
        self,
        limit: int = 10,
        *,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> list[Node]:
        limit = max(1, limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (self.tenant_id,),
            ).fetchall()
            selected: list[Node] = []
            for row in rows:
                node = self._row_to_node(row)
                if not _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id):
                    continue
                selected.append(node)
                if len(selected) >= limit:
                    break
            return selected

    def list_context_scopes(self) -> ContextScopeResult:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT agent_id, project, session_id
                FROM nodes
                WHERE tenant_id = ?
                """,
                (self.tenant_id,),
            ).fetchall()
        agent_ids = sorted({str(row["agent_id"]).strip() for row in rows if str(row["agent_id"]).strip()})
        projects = sorted({str(row["project"]).strip() for row in rows if str(row["project"]).strip()})
        session_ids = sorted({str(row["session_id"]).strip() for row in rows if str(row["session_id"]).strip()})
        return ContextScopeResult(agent_ids=agent_ids, projects=projects, session_ids=session_ids)

    def get_stats(self) -> GraphStats:
        with self._lock, self._connect() as connection:
            total_nodes = int(
                connection.execute("SELECT COUNT(*) FROM nodes WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            total_edges = int(
                connection.execute("SELECT COUNT(*) FROM edges WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            total_repos = int(
                connection.execute("SELECT COUNT(*) FROM repos WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            total_context_windows = int(
                connection.execute("SELECT COUNT(*) FROM context_windows WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            total_context_window_edges = int(
                connection.execute("SELECT COUNT(*) FROM context_window_edges WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            windows_with_embeddings = int(
                connection.execute(
                    "SELECT COUNT(*) FROM context_windows WHERE tenant_id = ? AND embedding IS NOT NULL",
                    (self.tenant_id,),
                ).fetchone()[0]
            )
            windows_with_stale_embeddings = int(
                connection.execute(
                    "SELECT COUNT(*) FROM context_windows WHERE tenant_id = ? AND embedding_stale = 1",
                    (self.tenant_id,),
                ).fetchone()[0]
            )

            counts = {
                node_type.value: 0
                for node_type in NodeType
            }
            for row in connection.execute(
                "SELECT node_type, COUNT(*) AS count FROM nodes WHERE tenant_id = ? GROUP BY node_type",
                (self.tenant_id,),
            ).fetchall():
                counts[str(row["node_type"])] = int(row["count"])
            window_status_counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM context_windows WHERE tenant_id = ? GROUP BY status",
                    (self.tenant_id,),
                ).fetchall()
            }
            window_edge_type_counts = {
                str(row["edge_type"]): int(row["count"])
                for row in connection.execute(
                    "SELECT edge_type, COUNT(*) AS count FROM context_window_edges WHERE tenant_id = ? GROUP BY edge_type",
                    (self.tenant_id,),
                ).fetchall()
            }

            most_connected_rows = connection.execute(
                """
                SELECT n.id, n.label, n.node_type,
                       COUNT(e.id) AS connection_count
                FROM nodes AS n
                LEFT JOIN edges AS e
                    ON (n.id = e.source_id OR n.id = e.target_id) AND e.tenant_id = ?
                WHERE n.tenant_id = ?
                GROUP BY n.id
                ORDER BY connection_count DESC, n.updated_at DESC
                LIMIT 5
                """
            , (self.tenant_id, self.tenant_id)).fetchall()

            most_recent_rows = connection.execute(
                """
                SELECT id, label, node_type, updated_at
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 5
                """
            , (self.tenant_id,)).fetchall()

            return GraphStats(
                total_nodes=total_nodes,
                total_edges=total_edges,
                total_repos=total_repos,
                total_context_windows=total_context_windows,
                context_window_status_breakdown=window_status_counts,
                total_context_window_edges=total_context_window_edges,
                context_window_edge_type_breakdown=window_edge_type_counts,
                windows_with_embeddings=windows_with_embeddings,
                windows_with_stale_embeddings=windows_with_stale_embeddings,
                node_type_breakdown=counts,
                most_connected_nodes=[
                    ConnectedNodeStat(
                        id=row["id"],
                        label=row["label"],
                        node_type=NodeType(row["node_type"]),
                        connection_count=int(row["connection_count"]),
                    )
                    for row in most_connected_rows
                ],
                most_recent_nodes=[
                    RecentNodeStat(
                        id=row["id"],
                        label=row["label"],
                        node_type=NodeType(row["node_type"]),
                        updated_at=_parse_datetime(row["updated_at"]),
                    )
                    for row in most_recent_rows
                ],
            )

    def export_graph_html(
        self,
        *,
        output_path: str | Path | None = None,
        include_physics: bool = True,
    ) -> Path:
        try:
            from pyvis.network import Network
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyvis is not installed. Install the project dependencies again.") from exc

        with self._lock, self._connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, label, content, node_type, tags, source_prompt, metadata,
                       created_at, updated_at, access_count
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """
            , (self.tenant_id,)).fetchall()
            edge_rows = connection.execute(
                """
                SELECT id, source_id, target_id, relationship, weight, metadata, created_at
                FROM edges
                WHERE tenant_id = ?
                ORDER BY created_at ASC
                """
            , (self.tenant_id,)).fetchall()

        if output_path is None:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = utc_now().strftime("%Y%m%d-%H%M%S")
            destination = self.export_dir / f"waggle-{timestamp}.html"
        else:
            destination = Path(output_path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)

        network = Network(
            height="800px",
            width="100%",
            directed=True,
            bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        network.barnes_hut()
        if not include_physics:
            network.toggle_physics(False)

        palette = {
            NodeType.FACT: "#38bdf8",
            NodeType.ENTITY: "#34d399",
            NodeType.CONCEPT: "#fbbf24",
            NodeType.PREFERENCE: "#fb7185",
            NodeType.DECISION: "#c084fc",
            NodeType.QUESTION: "#f97316",
            NodeType.NOTE: "#94a3b8",
        }

        nodes = [self._row_to_node(row) for row in node_rows]
        edges = [self._row_to_edge(row) for row in edge_rows]

        for node in nodes:
            title_lines = [
                f"<b>{node.label}</b>",
                f"Type: {node.node_type.value}",
                f"Created: {node.created_at.isoformat()}",
                f"Updated: {node.updated_at.isoformat()}",
                f"Access Count: {node.access_count}",
                "",
                node.content,
            ]
            if node.tags:
                title_lines.insert(4, f"Tags: {', '.join(node.tags)}")

            network.add_node(
                node.id,
                label=node.label,
                title="<br>".join(title_lines),
                color=palette[node.node_type],
                shape="dot",
                size=18 + min(node.access_count, 8) * 2,
            )

        for edge in edges:
            network.add_edge(
                edge.source_id,
                edge.target_id,
                label=edge.relationship,
                title=f"weight={edge.weight}",
                value=max(edge.weight, 0.1),
                arrows="to",
            )

        destination.write_text(network.generate_html(notebook=False), encoding="utf-8")
        return destination

    def export_window_graph_html(
        self,
        *,
        project: str = "",
        output_path: str | Path | None = None,
        include_physics: bool = True,
    ) -> Path:
        try:
            from pyvis.network import Network
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyvis is not installed. Install the project dependencies again.") from exc

        repo_id = self.ensure_repo(project or "default")
        windows = self.get_repo_windows(repo_id, include_archived=True)
        window_ids = {window.id for window in windows}
        with self._lock, self._connect() as connection:
            edge_rows = connection.execute(
                """
                SELECT id, tenant_id, source_window_id, target_window_id, edge_type,
                       shared_entities, weight, metadata, created_at
                FROM context_window_edges
                WHERE tenant_id = ?
                  AND source_window_id IN ({})
                  AND target_window_id IN ({})
                ORDER BY created_at ASC
                """.format(
                    ", ".join("?" for _ in window_ids) or "NULL",
                    ", ".join("?" for _ in window_ids) or "NULL",
                ),
                (self.tenant_id, *window_ids, *window_ids),
            ).fetchall() if window_ids else []
        edges = [self._row_to_context_window_edge(row) for row in edge_rows]

        if output_path is None:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = utc_now().strftime("%Y%m%d-%H%M%S")
            destination = self.export_dir / f"waggle-window-graph-{timestamp}.html"
        else:
            destination = Path(output_path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)

        network = Network(
            height="800px",
            width="100%",
            directed=True,
            bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        network.barnes_hut()
        if not include_physics:
            network.toggle_physics(False)

        status_colors = {
            "active": "#34d399",
            "closed": "#38bdf8",
            "archived": "#94a3b8",
        }
        edge_colors = {
            "entity_overlap": "#38bdf8",
            "supersedes": "#fb7185",
            "temporal_sequence": "#94a3b8",
            "continuation": "#fbbf24",
            "shared_scope": "#34d399",
        }

        for window in windows:
            connected_edges = [edge for edge in edges if edge.source_window_id == window.id or edge.target_window_id == window.id]
            label = window.title or window.session_id or window.id
            title_lines = [
                f"<b>{label}</b>",
                f"Window: {window.id}",
                f"Repo: {window.repo_id}",
                f"Status: {window.status}",
                f"Session: {window.session_id}",
                f"Nodes: {window.node_count}",
                f"Connected Windows: {len(connected_edges)}",
                f"Created: {window.created_at.isoformat()}",
                f"Updated: {window.updated_at.isoformat()}",
            ]
            network.add_node(
                window.id,
                label=label,
                title="<br>".join(title_lines),
                color=status_colors.get(window.status, "#94a3b8"),
                shape="dot",
                size=18 + min(max(window.node_count, 0), 50),
            )

        for edge in edges:
            shared = ", ".join(edge.shared_entities)
            network.add_edge(
                edge.source_window_id,
                edge.target_window_id,
                label=edge.edge_type,
                title=f"weight={edge.weight}" + (f"<br>shared={shared}" if shared else ""),
                value=max(edge.weight, 0.1),
                color=edge_colors.get(edge.edge_type, "#94a3b8"),
                arrows="to",
            )

        destination.write_text(network.generate_html(notebook=False), encoding="utf-8")
        return destination

    def export_graph_backup(self, *, output_path: str | Path | None = None) -> BackupResult:
        with self._lock, self._connect() as connection:
            snapshot = self._build_backup_snapshot(connection)

        if output_path is None:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = utc_now().strftime("%Y%m%d-%H%M%S")
            destination = self.export_dir / f"waggle-backup-{timestamp}.json"
        else:
            destination = Path(output_path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return BackupResult(
            output_path=str(destination),
            tenant_id=self.tenant_id,
            schema_version=SCHEMA_VERSION,
            node_count=len(snapshot["nodes"]),
            edge_count=len(snapshot["edges"]),
        )

    def export_abhi(
        self,
        *,
        output_path: str | Path | None = None,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> AbhiExportResult:
        with self._lock, self._connect() as connection:
            snapshot = self._build_backup_snapshot(connection)
        snapshot["ui"] = self.get_ui_state(project=project, agent_id=agent_id, session_id=session_id)
        filtered = filter_snapshot_by_scope(snapshot, project=project, agent_id=agent_id, session_id=session_id)
        if output_path is None:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = utc_now().strftime("%Y%m%d-%H%M%S")
            destination = self.export_dir / f"waggle-memory-{timestamp}.abhi"
        else:
            destination = Path(output_path).expanduser()
        return write_abhi_document(filtered, output_path=destination)

    def get_graph_snapshot(
        self,
        *,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            snapshot = self._build_backup_snapshot(connection)
        filtered = filter_snapshot_by_scope(snapshot, project=project, agent_id=agent_id, session_id=session_id)
        filtered["ui"] = self.get_ui_state(project=project, agent_id=agent_id, session_id=session_id)
        return filtered

    def export_context_bundle(
        self,
        *,
        mode: str = "prime",
        query: str = "",
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        max_nodes: int = 25,
        max_depth: int = 2,
        retrieval_mode: str = "graph",
        format: str = "both",
        output_path: str | Path | None = None,
        include_edges: bool = True,
        include_timestamps: bool = True,
        include_source_prompt: bool = False,
        audience: str = "llm",
    ) -> ContextBundleExportResult:
        normalized_mode = mode.strip().lower()
        normalized_format = format.strip().lower()
        normalized_audience = audience.strip().lower()
        normalized_retrieval_mode = retrieval_mode.strip().lower()
        if normalized_mode not in {"prime", "query", "graph"}:
            raise ValidationFailure("mode must be one of: prime, query, graph.")
        if normalized_format not in {"markdown", "json", "both"}:
            raise ValidationFailure("format must be one of: markdown, json, both.")
        if normalized_audience not in {"llm", "human"}:
            raise ValidationFailure("audience must be one of: llm, human.")
        if normalized_retrieval_mode not in {"graph", "replay", "fusion"}:
            raise ValidationFailure("retrieval_mode must be one of: graph, replay, fusion.")
        if normalized_mode == "query" and not query.strip():
            raise ValidationFailure("query is required when mode='query'.")
        if normalized_mode != "query" and normalized_retrieval_mode != "graph":
            raise ValidationFailure("retrieval_mode is only supported when mode='query'.")

        replay_hits: list[ReplayHit] = []
        if normalized_mode == "prime":
            selected = self.prime_context(project=project, agent_id=agent_id, session_id=session_id, max_nodes=max_nodes)
            selected_nodes = selected.nodes
            selected_edges = selected.edges if include_edges else []
            summary = selected.summary
        elif normalized_mode == "query":
            selected = self.query(
                query=query,
                max_nodes=max_nodes,
                max_depth=max_depth,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
                retrieval_mode=normalized_retrieval_mode,
            )
            selected_nodes = selected.nodes
            selected_edges = selected.edges if include_edges else []
            replay_hits = selected.replay_hits
            summary = build_query_summary(
                query=query,
                nodes=selected_nodes,
                edges=selected_edges,
                replay_hits=replay_hits,
                retrieval_mode=normalized_retrieval_mode,
            )
        else:
            with self._lock, self._connect() as connection:
                node_rows = connection.execute(
                    """
                    SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata,
                           evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
                    FROM nodes
                    WHERE tenant_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    """,
                    (self.tenant_id,),
                ).fetchall()
                edge_rows = connection.execute(
                    """
                    SELECT id, source_id, target_id, relationship, weight, metadata, created_at
                    FROM edges
                    WHERE tenant_id = ?
                    ORDER BY created_at ASC
                    """,
                    (self.tenant_id,),
                ).fetchall()
            selected_nodes = [
                node
                for row in node_rows
                for node in [self._row_to_node(row)]
                if _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id)
            ]
            selected_edges = [self._row_to_edge(row) for row in edge_rows] if include_edges else []
            if include_edges:
                selected_ids = {node.id for node in selected_nodes}
                selected_edges = [
                    edge for edge in selected_edges if edge.source_id in selected_ids and edge.target_id in selected_ids
                ]
            summary = (
                f"Full graph export for tenant '{self.tenant_id}' with {len(selected_nodes)} nodes and "
                f"{len(selected_edges)} edges."
            )

        bundle = build_context_bundle(
            tenant_id=self.tenant_id,
            project=project,
            mode=normalized_mode,
            retrieval_mode=normalized_retrieval_mode if normalized_mode == "query" else "graph",
            audience=normalized_audience,
            query=query,
            summary=summary,
            nodes=selected_nodes,
            edges=selected_edges,
            replay_hits=replay_hits,
            stats=self.get_stats(),
        )
        return export_context_bundle_files(
            bundle,
            output_path=output_path,
            export_dir=self.export_dir,
            format=normalized_format,
            include_edges=include_edges,
            include_timestamps=include_timestamps,
            include_source_prompt=include_source_prompt,
        )

    def export_markdown_vault(
        self,
        *,
        root_path: str | Path,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
    ) -> MarkdownVaultExportResult:
        root = Path(root_path).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata,
                       evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (self.tenant_id,),
            ).fetchall()
            edge_rows = connection.execute(
                """
                SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
                FROM edges
                WHERE tenant_id = ?
                ORDER BY created_at ASC
                """,
                (self.tenant_id,),
            ).fetchall()
        selected_nodes = [
            node
            for row in node_rows
            for node in [self._row_to_node(row)]
            if _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id)
        ]
        selected_ids = {node.id for node in selected_nodes}
        selected_edges = [
            self._row_to_edge(row)
            for row in edge_rows
            if row["source_id"] in selected_ids and row["target_id"] in selected_ids
        ]
        node_by_id = {node.id: node for node in selected_nodes}
        files_written: list[str] = []
        for node in selected_nodes:
            project_dir = slugify(node.project or project or "default")
            node_type_dir = slugify(node.node_type.value)
            destination = root / project_dir / node_type_dir / vault_filename(node)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                render_node_document(node, selected_edges, node_by_id),
                encoding="utf-8",
            )
            files_written.append(str(destination.relative_to(root)))
        return MarkdownVaultExportResult(
            root_path=str(root),
            tenant_id=self.tenant_id,
            project=project,
            node_count=len(selected_nodes),
            edge_count=len(selected_edges),
            files_written=files_written,
        )

    def import_markdown_vault(
        self,
        *,
        root_path: str | Path,
    ) -> MarkdownVaultImportResult:
        root = Path(root_path).expanduser()
        documents = iter_vault_documents(root)
        result = MarkdownVaultImportResult(root_path=str(root), tenant_id=self.tenant_id)
        if not documents:
            return result

        with self._lock, self._connect() as connection:
            nodes_by_id = {
                node.id: node
                for node in self._fetch_nodes_by_ids(
                    connection,
                    [str(document.frontmatter["node_id"]) for document in documents],
                )
            }
            label_index: dict[str, Node] = {}
            all_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata,
                       evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                """,
                (self.tenant_id,),
            ).fetchall()
            for row in all_rows:
                node = self._row_to_node(row)
                label_index.setdefault(node.label.strip().lower(), node)
                nodes_by_id.setdefault(node.id, node)

            imported_id_map: dict[str, str] = {}
            for document in documents:
                original_node_id = str(document.frontmatter.get("node_id", "")).strip()
                node, created = self._upsert_vault_document(connection, document)
                nodes_by_id[node.id] = node
                if original_node_id:
                    imported_id_map[original_node_id] = node.id
                    nodes_by_id[original_node_id] = node
                label_index[node.label.strip().lower()] = node
                if created:
                    result.nodes_created += 1
                else:
                    result.nodes_updated += 1

            for document in documents:
                source_node_id = str(document.frontmatter.get("node_id", "")).strip()
                source_node = nodes_by_id.get(imported_id_map.get(source_node_id, source_node_id))
                if source_node is None:
                    result.conflicts.append(f"Missing source node for document {document.path}.")
                    continue
                for relation in document.relations:
                    target_lookup_id = imported_id_map.get(relation.target_node_id, relation.target_node_id)
                    target_node = nodes_by_id.get(target_lookup_id) if target_lookup_id else None
                    if target_node is None and relation.target_label:
                        target_node = label_index.get(relation.target_label.strip().lower())
                    if target_node is None and relation.target_label:
                        target_node = self._insert_vault_stub_node(
                            connection,
                            label=relation.target_label,
                            project=source_node.project,
                            agent_id=source_node.agent_id,
                            session_id=source_node.session_id,
                        )
                        nodes_by_id[target_node.id] = target_node
                        label_index[target_node.label.strip().lower()] = target_node
                        result.stub_nodes_created += 1
                    if target_node is None:
                        result.conflicts.append(
                            f"Could not resolve relation target '{relation.target_label}' in {document.path.name}."
                        )
                        continue
                    if relation.deleted:
                        if self._delete_edge_record(
                            connection,
                            source_id=source_node.id,
                            target_id=target_node.id,
                            relationship=relation.relationship,
                        ):
                            result.edges_deleted += 1
                        continue
                    if self._find_existing_edge(
                        connection,
                        source_id=source_node.id,
                        target_id=target_node.id,
                        relationship=relation.relationship,
                    ) is None:
                        self._insert_edge_record(
                            connection,
                            source_id=source_node.id,
                            target_id=target_node.id,
                            relationship=relation.relationship,
                        )
                        result.edges_created += 1
        return result

    def import_graph_backup(self, *, input_path: str | Path) -> ImportResult:
        source = Path(input_path).expanduser()
        snapshot = json.loads(source.read_text(encoding="utf-8"))

        with self._lock, self._connect() as connection:
            snapshot_tenant = str(snapshot.get("tenant_id") or self.tenant_id)
            result = ImportResult(
                input_path=str(source),
                tenant_id=self.tenant_id,
                schema_version=int(snapshot.get("schema_version", 1)),
            )
            for raw_repo in snapshot.get("repos", []):
                self._upsert_snapshot_repo(connection, {**raw_repo, "tenant_id": self.tenant_id})

            for raw_window in snapshot.get("context_windows", []):
                self._upsert_snapshot_context_window(connection, {**raw_window, "tenant_id": self.tenant_id})

            for raw_node in snapshot.get("nodes", []):
                raw_node = {**raw_node, "tenant_id": raw_node.get("tenant_id") or snapshot_tenant}
                if raw_node["tenant_id"] != self.tenant_id:
                    raw_node["tenant_id"] = self.tenant_id
                if self._fetch_node_row(connection, raw_node["id"]) is None:
                    self._insert_snapshot_node(connection, raw_node)
                    result.nodes_created += 1
                else:
                    self._update_snapshot_node(connection, raw_node)
                    result.nodes_updated += 1

            for raw_edge in snapshot.get("edges", []):
                raw_edge = {**raw_edge, "tenant_id": raw_edge.get("tenant_id") or snapshot_tenant}
                if raw_edge["tenant_id"] != self.tenant_id:
                    raw_edge["tenant_id"] = self.tenant_id
                if self._fetch_edge_row(connection, raw_edge["id"]) is None:
                    self._insert_snapshot_edge(connection, raw_edge)
                    result.edges_created += 1
                else:
                    self._update_snapshot_edge(connection, raw_edge)
                    result.edges_updated += 1

            for raw_window_edge in snapshot.get("context_window_edges", []):
                self._upsert_snapshot_context_window_edge(
                    connection,
                    {**raw_window_edge, "tenant_id": self.tenant_id},
                )

            for raw_window in snapshot.get("context_windows", []):
                window_id = str(raw_window.get("id", "")).strip()
                if window_id:
                    self._update_window_node_count(connection, window_id)
                    self._mark_window_embedding_stale(connection, window_id)
        self.save_ui_state(
            positions=snapshot.get("ui", {}).get("positions", {}),
            zoom=snapshot.get("ui", {}).get("zoom", 1.0),
            viewport=snapshot.get("ui", {}).get("viewport", {"center_x": 0, "center_y": 0}),
            groups=snapshot.get("ui", {}).get("groups", []),
            collapsed_groups=snapshot.get("ui", {}).get("collapsed_groups", []),
            selected_nodes=snapshot.get("ui", {}).get("selected_nodes", []),
        )
        return result

    def validate_abhi(self, *, input_path: str | Path) -> AbhiValidationResult:
        document = load_abhi_document(input_path)
        return validate_abhi_document(document, input_path=input_path)

    def inspect_abhi(self, *, input_path: str | Path) -> AbhiInspectResult:
        document = load_abhi_document(input_path)
        return inspect_abhi_document(document, input_path=input_path)

    def diff_abhi(self, *, input_path_a: str | Path, input_path_b: str | Path) -> AbhiDiffResult:
        return diff_abhi_files(input_path_a=input_path_a, input_path_b=input_path_b)

    def query_abhi(self, *, input_path: str | Path, query_id: str = "", query_text: str = "") -> AbhiQueryResult:
        return query_abhi_file(input_path=input_path, query_id=query_id, query_text=query_text)

    def load_abhi_chunks(
        self,
        *,
        input_path: str | Path,
        chunk_ids: list[str] | None = None,
        query_id: str = "",
        query_text: str = "",
    ) -> AbhiChunkLoadResult:
        return load_abhi_chunk_file(
            input_path=input_path,
            chunk_ids=chunk_ids or [],
            query_id=query_id,
            query_text=query_text,
        )

    def merge_abhi(
        self,
        *,
        base_input_path: str | Path,
        left_input_path: str | Path,
        right_input_path: str | Path,
        output_path: str | Path,
        merge_strategy: str = "prefer_right",
    ) -> AbhiMergeResult:
        return merge_abhi_files(
            base_input_path=base_input_path,
            left_input_path=left_input_path,
            right_input_path=right_input_path,
            output_path=output_path,
            merge_strategy=merge_strategy,
        )

    def import_abhi(self, *, input_path: str | Path) -> AbhiImportResult:
        source = Path(input_path).expanduser()
        document = load_abhi_document(source)
        validation = validate_abhi_document(document, input_path=source)
        if not validation.valid:
            raise ValidationFailure("Invalid .abhi file: " + "; ".join(validation.errors))
        executed_actions = dispatch_abhi_event(document, event_name="on_import", persist=False, input_path=source)
        snapshot = abhi_to_snapshot(document, fallback_tenant_id=self.tenant_id)

        with self._lock, self._connect() as connection:
            snapshot_tenant = str(snapshot.get("tenant_id") or self.tenant_id)
            result = AbhiImportResult(
                input_path=str(source),
                tenant_id=self.tenant_id,
                schema_version=int(snapshot.get("schema_version", 1)),
                abhi_spec_version=validation.abhi_spec_version or ABHI_SPEC_VERSION,
                hash_verified=True,
                executed_actions=executed_actions,
            )
            for raw_repo in snapshot.get("repos", []):
                self._upsert_snapshot_repo(connection, {**raw_repo, "tenant_id": self.tenant_id})

            for raw_window in snapshot.get("context_windows", []):
                self._upsert_snapshot_context_window(connection, {**raw_window, "tenant_id": self.tenant_id})

            for raw_node in snapshot.get("nodes", []):
                raw_node = {**raw_node, "tenant_id": raw_node.get("tenant_id") or snapshot_tenant}
                if raw_node["tenant_id"] != self.tenant_id:
                    raw_node["tenant_id"] = self.tenant_id
                if self._fetch_node_row(connection, raw_node["id"]) is None:
                    self._insert_snapshot_node(connection, raw_node)
                    result.nodes_created += 1
                else:
                    self._update_snapshot_node(connection, raw_node)
                    result.nodes_updated += 1

            for raw_edge in snapshot.get("edges", []):
                raw_edge = {**raw_edge, "tenant_id": raw_edge.get("tenant_id") or snapshot_tenant}
                if raw_edge["tenant_id"] != self.tenant_id:
                    raw_edge["tenant_id"] = self.tenant_id
                if self._fetch_edge_row(connection, raw_edge["id"]) is None:
                    self._insert_snapshot_edge(connection, raw_edge)
                    result.edges_created += 1
                else:
                    self._update_snapshot_edge(connection, raw_edge)
                    result.edges_updated += 1

            for raw_window_edge in snapshot.get("context_window_edges", []):
                self._upsert_snapshot_context_window_edge(
                    connection,
                    {**raw_window_edge, "tenant_id": self.tenant_id},
                )

            for raw_window in snapshot.get("context_windows", []):
                window_id = str(raw_window.get("id", "")).strip()
                if window_id:
                    self._update_window_node_count(connection, window_id)
                    self._mark_window_embedding_stale(connection, window_id)
        self.save_ui_state(
            positions=snapshot.get("ui", {}).get("positions", {}),
            zoom=snapshot.get("ui", {}).get("zoom", 1.0),
            viewport=snapshot.get("ui", {}).get("viewport", {"center_x": 0, "center_y": 0}),
            groups=snapshot.get("ui", {}).get("groups", []),
            collapsed_groups=snapshot.get("ui", {}).get("collapsed_groups", []),
            selected_nodes=snapshot.get("ui", {}).get("selected_nodes", []),
        )
        return result

    def decompose_and_store(self, *, content: str, context: str = "") -> SubgraphResult:
        trimmed_content = content.strip()
        if not trimmed_content:
            raise ValueError("Content cannot be empty.")

        created_nodes: list[Node] = []
        created_ids: set[str] = set()
        context_node: Node | None = None
        if context.strip():
            context_result = self.add_node(
                label=infer_label(context),
                content=context.strip(),
                node_type=NodeType.CONCEPT,
                tags=["decomposition-context"],
                source_prompt=trimmed_content,
            )
            context_node = context_result.node
            created_nodes.append(context_node)
            created_ids.add(context_node.id)

        atomic_items = split_atomic_items(trimmed_content)
        item_nodes: list[Node] = []
        for item in atomic_items:
            store_result = self.add_node(
                label=infer_label(item),
                content=item,
                node_type=infer_node_type(item),
                tags=["decomposed"],
                source_prompt=context.strip() or trimmed_content,
            )
            node = store_result.node
            item_nodes.append(node)
            if node.id not in created_ids:
                created_nodes.append(node)
                created_ids.add(node.id)
            if context_node is not None:
                self.add_edge(
                    source_id=node.id,
                    target_id=context_node.id,
                    relationship=RelationType.PART_OF,
                    metadata={"origin": "decomposition"},
                )

        for index, node in enumerate(item_nodes):
            if index == 0:
                continue
            previous = item_nodes[index - 1]
            shared_tokens = tokenize_text(previous.content) & tokenize_text(node.content)
            if shared_tokens or previous.node_type == node.node_type:
                self.add_edge(
                    source_id=previous.id,
                    target_id=node.id,
                    relationship=infer_relationship(previous, node, shared_tokens=shared_tokens),
                    metadata={"origin": "decomposition"},
                )

        node_ids = [node.id for node in created_nodes]
        with self._lock, self._connect() as connection:
            edges = self._fetch_edges_for_nodes(connection, node_ids)
        return SubgraphResult(
            nodes=created_nodes,
            edges=edges,
            query=f"decomposition:{context.strip() or infer_label(trimmed_content)}",
            total_nodes_in_graph=self.get_stats().total_nodes,
        )

    def _apply_observation_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        transcript: str,
        user_turn_index: int,
        assistant_turn_index: int,
        observed_at: datetime,
        session_id: str,
        agent_id: str,
        project: str,
        edge_origin: str = "observe_conversation",
    ) -> ObservationResult:
        """Shared extraction helper used by both observe_conversation and ingest_transcript_handoff.

        Takes pre-extracted candidates and stores them as nodes with evidence,
        then links decision->rationale edges.  Both single-turn and batch paths call this
        so memory semantics stay aligned.
        """
        result = ObservationResult()
        stored_candidate_records: list[tuple[Node, list[str]]] = []
        for candidate in candidates:
            candidate_tags = list(candidate.get("tags", []))
            speaker_tag = next((tag for tag in candidate_tags if str(tag).startswith("speaker:")), "")
            speaker = speaker_tag.split(":", 1)[1] if ":" in speaker_tag else "user"
            turn_index = user_turn_index if speaker == "user" else assistant_turn_index
            evidence = build_observation_evidence(
                transcript=transcript,
                source_text=str(candidate["content"]),
                speaker=speaker,
                turn_index=turn_index,
                observed_at=observed_at,
                session_id=session_id,
            )
            store_result = self.add_node(
                label=str(candidate["label"]),
                content=str(candidate["content"]),
                node_type=candidate["node_type"],
                tags=candidate_tags,
                source_prompt=transcript,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
                evidence_records=[evidence],
                valid_from=observed_at,
            )
            result.stored_nodes.append(store_result.node)
            stored_candidate_records.append((store_result.node, candidate_tags))
            if store_result.created:
                result.created_count += 1
            else:
                result.reused_count += 1
            for conflict in store_result.conflicts:
                if conflict.other_node_id not in {item.other_node_id for item in result.conflicts}:
                    result.conflicts.append(conflict)

        decision_nodes = [
            (node, tags)
            for node, tags in stored_candidate_records
            if node.node_type == NodeType.DECISION
        ]
        rationale_nodes = [
            (node, tags)
            for node, tags in stored_candidate_records
            if "decision-rationale" in tags and node.node_type == NodeType.FACT
        ]
        for decision_node, decision_tags in decision_nodes:
            decision_categories = {tag for tag in decision_tags if tag in {"database", "backend-framework", "frontend-framework", "auth-mechanism", "api-style"}}
            for rationale_node, rationale_tags in rationale_nodes:
                rationale_categories = {tag for tag in rationale_tags if tag in {"database", "backend-framework", "frontend-framework", "auth-mechanism", "api-style"}}
                if rationale_categories and decision_categories and not (rationale_categories & decision_categories):
                    continue
                self.add_edge(
                    source_id=decision_node.id,
                    target_id=rationale_node.id,
                    relationship=RelationType.DEPENDS_ON,
                    metadata={"origin": edge_origin},
                )
        return result

    def observe_conversation(
        self,
        *,
        user_message: str,
        assistant_response: str,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> ObservationResult:
        transcript = f"user: {user_message.strip()}\nassistant: {assistant_response.strip()}".strip()
        observed_at = utc_now()
        candidates = extract_conversation_candidates(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        with self._lock, self._connect() as connection:
            next_turn_index = self._next_transcript_turn_index(connection, session_id=session_id)
            turns = [
                ("user", user_message.strip(), next_turn_index),
                ("assistant", assistant_response.strip(), next_turn_index + 1),
            ]
            for role, text, turn_index in turns:
                if not text:
                    continue
                self._store_transcript_record(
                    connection,
                    agent_id=agent_id,
                    project=project,
                    session_id=session_id,
                    observed_at=observed_at,
                    turn_index=turn_index,
                    role=role,
                    transcript_text=text,
                )

        result = self._apply_observation_candidates(
            candidates=candidates,
            transcript=transcript,
            user_turn_index=next_turn_index,
            assistant_turn_index=next_turn_index + 1,
            observed_at=observed_at,
            session_id=session_id,
            agent_id=agent_id,
            project=project,
        )
        repo_id, window_id = self.resolve_window_context(project=project, session_id=session_id)
        self.update_window_node_count(window_id)
        self.mark_window_embedding_stale(window_id)
        self.derive_context_window_edges(window_id, repo_id)
        return result

    # ---------------------------------------------------------------------------
    # Batch transcript ingestion (ingest-transcript-handoff)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _message_fingerprint(msg: TranscriptMessage, raw_position: int) -> str:
        """Compute a stable dedup identity for a transcript message.

        If the message supplies a client-side ``message_id``, use it directly.
        Otherwise compute a deterministic positional fingerprint from
        (role, content, raw_position, timestamp-or-empty).

        Positional fingerprints are idempotent only for identical reruns.
        Prepending, removing, or reordering messages in a partial resubmit
        will produce different fingerprints and be treated as new input.
        This is a documented v1 limitation.
        """
        if msg.message_id:
            return msg.message_id
        payload = "\x00".join([
            msg.role,
            msg.content,
            str(raw_position),
            msg.timestamp or "",
        ])
        return "fp:" + hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _build_extractive_blocks(
        messages: list[TranscriptMessage],
    ) -> list[tuple[str, str]]:
        """Collapse consecutive same-role extractive (user/assistant) messages into blocks.

        system and tool messages are skipped for block formation but do not
        split or interrupt blocks.  This is the v1 rule; see docs/backlog for
        the tool_boundary_splits_blocks refinement.

        Returns a list of (role, joined_content) tuples.
        """
        blocks: list[tuple[str, str]] = []
        for msg in messages:
            if msg.role not in ("user", "assistant"):
                # system/tool: skip for block purposes, no split
                continue
            if blocks and blocks[-1][0] == msg.role:
                # Collapse consecutive same-role messages
                blocks[-1] = (blocks[-1][0], blocks[-1][1] + "\n\n" + msg.content)
            else:
                blocks.append((msg.role, msg.content))
        return blocks

    @staticmethod
    def _build_session_extractive_blocks(
        rows: list[Any],
        newly_written_identities: set[str],
    ) -> list[tuple[str, str, int, bool]]:
        """Build extractive blocks from the full ordered session transcript (from DB rows).

        Each block is (role, joined_content, first_turn_index, has_new_message).
        - role: 'user' or 'assistant' (system/tool rows are skipped).
        - joined_content: consecutive same-role messages joined with '\n\n'.
        - first_turn_index: the turn_index of the first row that contributed to this block.
        - has_new_message: True if ANY message in this block was newly written this run.

        This is the correct block-scan surface for extraction: it sees the full
        session history so a previously-unpaired trailing user can be completed by
        a newly-arrived assistant message in the next ingestion call.
        """
        blocks: list[tuple[str, str, int, bool]] = []
        for row in rows:
            role: str = row["role"]
            if role not in ("user", "assistant"):
                continue
            content: str = row["transcript_text"]
            turn_index: int = row["turn_index"]
            identity: str | None = row["message_identity"]
            is_new = identity in newly_written_identities if identity else False
            if blocks and blocks[-1][0] == role:
                prev_role, prev_content, prev_turn, prev_new = blocks[-1]
                blocks[-1] = (prev_role, prev_content + "\n\n" + content, prev_turn, prev_new or is_new)
            else:
                blocks.append((role, content, turn_index, is_new))
        return blocks

    def ingest_transcript_handoff(
        self,
        payload: TranscriptIngestionInput,
        *,
        export_format: str = "both",
        output_path: str | None = None,
        max_nodes: int = 25,
    ) -> TranscriptIngestionResult:
        """Batch-ingest a full ordered transcript, extract durable memory from logical turns,
        and optionally export a session-scoped handoff bundle.

        Supported backend: SQLite only in v1.  Neo4j support is deferred.

        Algorithm (block-windowing):
        1. Persist every message to transcript_records with dedup via message_identity.
        2. Build an extractive stream keeping only user/assistant messages.
        3. Collapse consecutive same-role extractive messages into one block.
        4. Scan collapsed blocks left to right:
           - user -> assistant   => one logical turn (extract from both).
           - leading assistant   => transcript-only, skipped for extraction.
           - trailing user       => transcript-only, counted as unpaired.
        5. After consuming a u->a pair, continue from the next remaining block.

        Tool-interleaving behavior (v1 simplification, documented):
        - user -> tool -> tool -> assistant  =>  one logical turn: user -> assistant.
        - user -> assistant -> tool -> tool -> assistant => user -> (assistant + assistant).
        Tool boundary splitting is a planned v2 refinement.
        """
        result = TranscriptIngestionResult(
            project=payload.project,
            agent_id=payload.agent_id,
            session_id=payload.session_id,
        )

        if not payload.messages:
            result.export_skipped = True
            result.export_skipped_reason = "no_messages"
            return result

        result.input_message_count = len(payload.messages)
        observed_at = utc_now()

        # Step 2: Persist all messages; collect identities of newly written ones.
        newly_written_identities: set[str] = set()
        with self._lock, self._connect() as connection:
            base_turn_index = self._next_transcript_turn_index(
                connection, session_id=payload.session_id
            )
            for raw_pos, msg in enumerate(payload.messages):
                identity = self._message_fingerprint(msg, raw_pos)
                written = self._store_transcript_record(
                    connection,
                    agent_id=payload.agent_id,
                    project=payload.project,
                    session_id=payload.session_id,
                    observed_at=observed_at,
                    turn_index=base_turn_index + raw_pos,
                    role=msg.role,
                    transcript_text=msg.content,
                    message_identity=identity,
                )
                if written:
                    result.transcript_records_written += 1
                    newly_written_identities.add(identity)
                else:
                    result.transcript_records_skipped += 1

        # If every message was a duplicate (full re-run), skip extraction.
        if result.transcript_records_written == 0:
            result.export_skipped = True
            result.export_skipped_reason = "all_messages_already_ingested"
            # Still produce an export bundle from existing session memory.
            _export = self._maybe_export_bundle(
                payload=payload,
                export_format=export_format,
                output_path=output_path,
                max_nodes=max_nodes,
            )
            if _export is not None:
                result.export_skipped = False
                result.markdown_path = _export.get("markdown_path")
                result.json_path = _export.get("json_path")
                result.export_node_count = _export.get("node_count", 0)
                result.export_edge_count = _export.get("edge_count", 0)
            return result

        # Step 3: Load the FULL session transcript from the DB ordered by turn_index.
        # We must scan the full session — not just newly written messages — so that a
        # previously-unpaired trailing user block can be paired with an assistant that
        # arrives in a later ingestion call.
        with self._lock, self._connect() as connection:
            session_rows = connection.execute(
                """
                SELECT role, transcript_text, turn_index, message_identity
                FROM transcript_records
                WHERE tenant_id = ? AND session_id = ?
                ORDER BY turn_index ASC, id ASC
                """,
                (self.tenant_id, payload.session_id),
            ).fetchall()

        # Step 4: Build session-scoped extractive blocks, each tagged with
        # has_new_message=True iff any row in that block was newly written this run.
        # (role, joined_content, first_turn_index, has_new_message)
        session_blocks = self._build_session_extractive_blocks(
            session_rows, newly_written_identities
        )

        # Step 5: Scan blocks left to right; only extract turns where at least
        # one of the two blocks (user or assistant) has a new message.
        # This prevents re-extraction of already-processed turns while still
        # completing trailing-user blocks when their assistant reply arrives later.
        i = 0
        while i < len(session_blocks):
            role, content, role_turn_index, block_has_new = session_blocks[i]
            if role == "assistant":
                # Leading or orphaned assistant: transcript-only, skip.
                i += 1
                continue
            # role == "user"
            if i + 1 < len(session_blocks) and session_blocks[i + 1][0] == "assistant":
                user_content = content
                user_turn_index = role_turn_index
                user_has_new = block_has_new
                assistant_content = session_blocks[i + 1][1]
                assistant_turn_index = session_blocks[i + 1][2]
                asst_has_new = session_blocks[i + 1][3]
                if user_has_new or asst_has_new:
                    transcript = f"user: {user_content}\nassistant: {assistant_content}"
                    candidates = extract_conversation_candidates(
                        user_message=user_content,
                        assistant_response=assistant_content,
                    )
                    turn_result = self._apply_observation_candidates(
                        candidates=candidates,
                        transcript=transcript,
                        user_turn_index=user_turn_index,
                        assistant_turn_index=assistant_turn_index,
                        observed_at=observed_at,
                        session_id=payload.session_id,
                        agent_id=payload.agent_id,
                        project=payload.project,
                        edge_origin="ingest_transcript_handoff",
                    )
                    result.logical_turns_processed += 1
                    result.nodes_created += turn_result.created_count
                    result.nodes_reused += turn_result.reused_count
                    result.conflicts += len(turn_result.conflicts)
                i += 2
            else:
                # Trailing user block with no following assistant: transcript-only.
                result.unpaired_trailing_blocks += 1
                i += 1

        # Step 5: Export a session-scoped prime bundle.
        _export = self._maybe_export_bundle(
            payload=payload,
            export_format=export_format,
            output_path=output_path,
            max_nodes=max_nodes,
        )
        if _export is not None:
            result.markdown_path = _export.get("markdown_path")
            result.json_path = _export.get("json_path")
            result.export_node_count = _export.get("node_count", 0)
            result.export_edge_count = _export.get("edge_count", 0)
        else:
            result.export_skipped = True
            result.export_skipped_reason = "no_nodes_in_session"
        return result

    def _maybe_export_bundle(
        self,
        *,
        payload: TranscriptIngestionInput,
        export_format: str,
        output_path: str | None,
        max_nodes: int,
    ) -> dict[str, Any] | None:
        """Export a session-scoped context bundle after ingestion, if nodes exist."""
        stats = self.get_stats()
        if stats.total_nodes == 0:
            return None
        exported = self.export_context_bundle(
            mode="prime",
            query="",
            project=payload.project,
            agent_id=payload.agent_id,
            session_id=payload.session_id,
            max_nodes=max_nodes,
            max_depth=2,
            retrieval_mode="graph",
            format=export_format,
            output_path=output_path,
            include_edges=True,
            include_timestamps=True,
            include_source_prompt=False,
            audience="llm",
        )
        return {
            "markdown_path": exported.markdown_path,
            "json_path": exported.json_path,
            "node_count": exported.node_count,
            "edge_count": exported.edge_count,
        }

    def graph_diff(self, *, since: str = "24h") -> GraphDiffResult:
        cutoff = parse_since_value(since)
        with self._lock, self._connect() as connection:
            added_nodes = [
                self._row_to_node(row)
                for row in connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ? AND created_at >= ?
                    ORDER BY created_at DESC
                    """,
                    (self.tenant_id, cutoff.isoformat()),
                ).fetchall()
            ]
            updated_nodes = [
                self._row_to_node(row)
                for row in connection.execute(
                """
                    SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                           created_at, updated_at, access_count, tenant_id
                    FROM nodes
                    WHERE tenant_id = ?
                      AND updated_at >= ?
                      AND created_at < ?
                    ORDER BY updated_at DESC
                    """,
                    (self.tenant_id, cutoff.isoformat(), cutoff.isoformat()),
                ).fetchall()
            ]
            created_edges = [
                self._row_to_edge(row)
                for row in connection.execute(
                    """
                    SELECT id, source_id, target_id, relationship, weight, metadata, created_at
                    FROM edges
                    WHERE tenant_id = ? AND created_at >= ?
                    ORDER BY created_at DESC
                    """,
                    (self.tenant_id, cutoff.isoformat()),
                ).fetchall()
            ]
            contradiction_edges = [
                edge for edge in created_edges if edge.relationship == RelationType.CONTRADICTS.value
            ]
        return GraphDiffResult(
            since=since,
            added_nodes=added_nodes,
            updated_nodes=updated_nodes,
            created_edges=created_edges,
            contradiction_edges=contradiction_edges,
        )

    def prime_context(
        self,
        *,
        project: str = "",
        agent_id: str = "",
        session_id: str = "",
        max_nodes: int = 25,
    ) -> PrimeContextResult:
        with self._lock, self._connect() as connection:
            total_nodes = int(
                connection.execute("SELECT COUNT(*) FROM nodes WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            if total_nodes == 0:
                return PrimeContextResult(project=project, summary="No stored memory is available yet.")

            active_session_id = _retrieval_session_scope(
                agent_id=agent_id,
                project=project,
                session_id=session_id,
            )
            # Collect seed anchors from multiple sources
            seed_ids: list[str] = []
            seed_ids.extend(
                self._most_connected_node_ids(
                    connection,
                    limit=5,
                    agent_id=agent_id,
                    project=project,
                    session_id=active_session_id,
                )
            )
            seed_ids.extend(
                node.id
                for node in self.list_recent_nodes(
                    limit=5,
                    agent_id=agent_id,
                    project=project,
                    session_id=active_session_id,
                )
            )
            if project.strip():
                seed_ids.extend(
                    self._find_project_node_ids(
                        connection,
                        project=project,
                        agent_id=agent_id,
                        session_id=active_session_id,
                        limit=8,
                    )
                )
            seed_ids = list(dict.fromkeys(seed_ids))  # Deduplicate

            if not seed_ids:
                return PrimeContextResult(project=project, summary="No seed nodes found for priming.")

            # Load all embeddable nodes and build graph
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                       created_at, updated_at, access_count, embedding, tenant_id
                FROM nodes
                WHERE tenant_id = ? AND embedding IS NOT NULL
                """,
                (self.tenant_id,),
            ).fetchall()

            if not node_rows:
                return PrimeContextResult(project=project, summary="No embeddable nodes available for expansion.")

            nodes_by_id: dict[str, Node] = {}
            for row in node_rows:
                node = self._row_to_node(row)
                if not _scope_matches(node, agent_id=agent_id, project=project, session_id=active_session_id):
                    continue
                nodes_by_id[node.id] = node

            graph = self._load_graph(connection, node_ids=nodes_by_id.keys())

            if not nodes_by_id:
                return PrimeContextResult(project=project, summary="No scoped nodes found for priming.")

            # Seeds can include non-embeddable nodes (e.g., recently touched items). Filter to embeddable
            # nodes to avoid KeyError when scoring/expanding.
            scoped_seed_ids = [seed_id for seed_id in seed_ids if seed_id in nodes_by_id]
            if not scoped_seed_ids:
                # Fall back to a small set of recent embeddable nodes when none of the seeds are usable.
                scoped_seed_ids = list(nodes_by_id.keys())[:5]

            # Expand from seeds using relation-aware traversal
            max_depth = 2
            expanded_depths, expansion_metadata = self._expand_node_depths_with_context(
                graph, scoped_seed_ids, max_depth
            )

            # Build candidate nodes from expansion
            candidate_nodes = [
                nodes_by_id[nid]
                for nid in expanded_depths
                if nid in nodes_by_id
            ]
            if not candidate_nodes:
                return PrimeContextResult(project=project, summary="Expansion produced no candidate nodes.")

            # Score with relation-aware ranking (no natural language query)
            expanded_ids_in_scope = [nid for nid in expanded_depths if nid in nodes_by_id]
            similarity_by_id = {nid: 0.0 for nid in expanded_ids_in_scope}
            lexical_by_id = {nid: 0.0 for nid in expanded_ids_in_scope}
            negation_boost_by_id = {nid: 0.0 for nid in expanded_ids_in_scope}
            transcript_session_scores = self._recent_transcript_session_scores(
                agent_id=agent_id,
                project=project,
                session_id=active_session_id,
            )
            # Boost seed IDs synthetically
            for seed_id in scoped_seed_ids:
                if seed_id in similarity_by_id:
                    similarity_by_id[seed_id] = 0.5
            for node_id in list(similarity_by_id.keys()):
                node = nodes_by_id.get(node_id)
                if node is None:
                    continue
                similarity_by_id[node_id] = self._blend_session_signal(
                    base_similarity=similarity_by_id[node_id],
                    session_signal=transcript_session_scores.get(node.session_id, 0.0),
                    session_weight=0.35,
                )

            degree_by_id = dict(graph.degree(expanded_depths.keys()))
            max_access = max((node.access_count for node in candidate_nodes), default=0)
            max_degree = max(degree_by_id.values(), default=0)
            candidate_edges = self._fetch_edges_for_nodes(connection, [node.id for node in candidate_nodes])

            temporal_hints = _NeutralTemporalHints()
            scored_nodes = self._sort_scored_nodes(
                candidate_nodes,
                max_nodes=max_nodes,
                temporal_hints=temporal_hints,
                similarity_by_id=similarity_by_id,
                lexical_by_id=lexical_by_id,
                negation_boost_by_id=negation_boost_by_id,
                degree_by_id=degree_by_id,
                max_access=max_access,
                max_degree=max_degree,
                max_depth=max_depth,
                expanded_depths=expanded_depths,
                edges=candidate_edges,
                expansion_metadata=expansion_metadata,
            )

            # Apply support coverage
            selected_nodes = scored_nodes[:max_nodes]
            candidate_pool = {node.id: node for node in candidate_nodes}
            selected_nodes = self._ensure_support_coverage(
                selected_nodes, candidate_pool, graph, max_nodes
            )

            selected_ids = [node.id for node in selected_nodes]
            edges = self._fetch_edges_for_nodes(connection, selected_ids)

        # Build structured summary
        summary = self._build_prime_summary(
            selected_nodes=selected_nodes,
            edges=edges,
            total_nodes_in_graph=total_nodes,
            project=project,
        )

        return PrimeContextResult(
            project=project,
            summary=summary,
            nodes=selected_nodes,
            edges=edges,
            total_nodes_in_graph=total_nodes,
        )

    def get_topics(self) -> TopicResult:
        with self._lock, self._connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata,
                       evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
                FROM nodes
                WHERE tenant_id = ?
                """
            , (self.tenant_id,)).fetchall()
            if not node_rows:
                return TopicResult(clusters=[], total_clusters=0)
            nodes = [self._row_to_node(row) for row in node_rows]
            graph = self._load_graph(connection, node_ids=[node.id for node in nodes]).to_undirected()
            partition = self._build_topic_partition(graph, nodes)

        nodes_by_id = {node.id: node for node in nodes}
        clusters_by_id: dict[int, list[Node]] = {}
        for node_id, cluster_id in partition.items():
            clusters_by_id.setdefault(int(cluster_id), []).append(nodes_by_id[node_id])

        clusters: list[TopicCluster] = []
        for cluster_id, cluster_nodes in sorted(
            clusters_by_id.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            label, top_tags = summarize_topic(cluster_nodes)
            ordered_nodes = sorted(
                cluster_nodes,
                key=lambda node: (-node.access_count, -node.updated_at.timestamp(), node.label.lower()),
            )
            clusters.append(
                TopicCluster(
                    cluster_id=cluster_id,
                    label=label,
                    node_count=len(cluster_nodes),
                    top_tags=top_tags,
                    nodes=ordered_nodes,
                )
            )
        return TopicResult(clusters=clusters, total_clusters=len(clusters))

    def _build_prime_summary(
        self,
        *,
        selected_nodes: list[Node],
        edges: list[Edge],
        total_nodes_in_graph: int,
        project: str = "",
    ) -> str:
        """Build a structured summary of prime context with type and relationship counts."""
        # Count node types
        type_counts: dict[str, int] = {}
        for node in selected_nodes:
            type_counts[node.node_type.value] = type_counts.get(node.node_type.value, 0) + 1

        # Count edge relationships
        relationship_counts: dict[str, int] = {}
        for edge in edges:
            rel = edge.relationship
            relationship_counts[rel] = relationship_counts.get(rel, 0) + 1

        # Build type breakdown
        type_breakdown = ", ".join(
            f"{count} {ttype}" for ttype, count in sorted(type_counts.items())
        ) if type_counts else "no nodes"

        # Build relationship breakdown
        relationship_breakdown = ", ".join(
            f"{count} {rel}" for rel, count in sorted(relationship_counts.items())
        ) if relationship_counts else "no edges"

        # Check for contradictions
        has_contradictions = "contradicts" in relationship_counts
        contradiction_warning = " [⚠ Contradictions present]" if has_contradictions else ""

        # Check for questions
        has_questions = "question" in type_counts
        question_warning = " [?]" if has_questions else ""

        # Build base summary
        if project.strip():
            base = f"Prime context for project '{project}': {len(selected_nodes)} nodes ({type_breakdown}) with {len(edges)} edges ({relationship_breakdown})"
        else:
            base = f"Prime context: {len(selected_nodes)} nodes ({type_breakdown}) with {len(edges)} edges ({relationship_breakdown})"

        base += f" from {total_nodes_in_graph} total nodes"
        base += contradiction_warning + question_warning

        return base

    def _require_node(self, connection: sqlite3.Connection, node_id: str) -> None:
        if self._fetch_node_row(connection, node_id) is None:
            raise ValueError(f"Node not found: {node_id}")

    def _find_duplicate_node(
        self,
        connection: sqlite3.Connection,
        *,
        node: Node,
        embedding: np.ndarray,
    ) -> tuple[Node, str, float | None] | None:
        rows = connection.execute(
            """
            SELECT id, agent_id, project, session_id, context_window_id, label, content, node_type, tags, source_prompt, metadata, evidence_records,
                   valid_from, valid_to, created_at, updated_at, access_count, embedding, tenant_id
            FROM nodes
            WHERE tenant_id = ? AND embedding IS NOT NULL
            """,
            (self.tenant_id,),
        ).fetchall()

        normalized_label = normalize_text(node.label)
        normalized_content = normalize_text(node.content)
        # Type-aware cosine threshold — decisions merge at 0.82, facts at 0.92, etc.
        type_threshold = type_aware_dedup_threshold(node.node_type,
                                                    default=self.dedup_similarity_threshold)
        best_match: tuple[Node, float] | None = None

        for row in rows:
            existing_node = self._row_to_node(row)
            if not _scope_matches(
                existing_node,
                agent_id=node.agent_id,
                project=node.project,
                session_id=node.session_id,
            ):
                continue
            if not compatible_node_types(node.node_type, existing_node.node_type):
                continue
            existing_label = normalize_text(existing_node.label)
            existing_content = normalize_text(existing_node.content)

            # ── Layer 0: entity-key hard block ────────────────────────
            # If both nodes name a specific technology AND those technologies
            # are different (but in the same category), block the merge.
            # e.g. "use PostgreSQL" vs "use MySQL" — similar sentence, different choice.
            node_entity = extract_choice_entity(node.content)
            existing_entity = extract_choice_entity(existing_node.content)
            if (
                node_entity is not None
                and existing_entity is not None
                and node_entity[1] == existing_entity[1]   # same category
                and node_entity[0] != existing_entity[0]   # different entity
                and not describes_rejected_or_limited_option(node.content)
                and not describes_rejected_or_limited_option(existing_node.content)
            ):
                continue  # never merge "postgres" node with "mysql" node

            # ── Layer 0b: numeric-conflict guard ───────────────────────
            # Same entity BUT different critical number (e.g. JWT 15min vs 1hr).
            # Conflicting numbers signal distinct facts, not duplicates.
            # Also applies to non-entity facts that have conflicting numbers.
            if contains_conflicting_numbers(node.content, existing_node.content) and (
                node_entity is None
                or existing_entity is None
                or node_entity[0] == existing_entity[0]
            ):
                continue
            if contains_conflicting_months(node.content, existing_node.content):
                continue

            if normalized_content == existing_content:
                return existing_node, "exact_content", 1.0

            # ── Layer 2: substring containment (cheap, catches rephrased subsets)
            if len(normalized_content) >= 10 and len(existing_content) >= 10:
                if normalized_content in existing_content or existing_content in normalized_content:
                    return existing_node, "content_substring", 0.98

            # ── Layer 3: semantic similarity (expensive — compute embedding once) ─
            existing_embedding = self.embedding_model.from_bytes(row["embedding"])
            similarity = self.embedding_model.cosine_similarity(embedding, existing_embedding)
            label_score = label_similarity(node.label, existing_node.label)
            acronym_match = is_acronym_match(node.label, existing_node.label)

            if normalized_label == existing_label and similarity >= self.dedup_same_label_threshold:
                return existing_node, "same_label_high_similarity", similarity
            if acronym_match and similarity >= max(self.dedup_same_label_threshold - 0.25, 0.55):
                return existing_node, "acronym_entity_match", similarity
            if label_score >= 0.92 and similarity >= max(self.dedup_same_label_threshold - 0.2, 0.6):
                return existing_node, "label_entity_match", similarity

            # ── Layer 3b: same-entity aggressive merge ──────────────────
            # If both nodes reference the SAME named entity, lower the cosine
            # threshold significantly — "fastapi was chosen" and "we chose fastapi
            # because async" should merge even at cosine ~0.65.
            # The numeric-conflict guard (Layer 0b) already blocked cases where
            # the same entity appears with different critical numbers.
            if (
                node_entity is not None
                and existing_entity is not None
                and node_entity[0] == existing_entity[0]  # identical entity token
                and similarity >= 0.60
            ):
                return existing_node, "same_entity_merge", similarity

            # ── Layer 3c: Jaccard-boosted merge (type-aware lower threshold) ──
            # If content words overlap significantly AND cosine is high for the
            # node type, treat as duplicate — catches paraphrase true-dups.
            jaccard = content_token_jaccard(node.content, existing_node.content)
            boosted_threshold = max(type_threshold - 0.05, 0.70)
            if jaccard >= 0.35 and similarity >= boosted_threshold:
                return existing_node, "jaccard_boosted_similarity", similarity

            # ── Layer 3d: entity-less paraphrase merge ─────────────────
            # Some true duplicates share meaning but have no named entity anchor
            # and too little word overlap for the Jaccard gate above.
            if node_entity is None and existing_entity is None:
                paraphrase_score = paraphrase_dedup_score(
                    semantic_similarity=similarity,
                    lexical_overlap=jaccard,
                )
                paraphrase_threshold = max(type_threshold - 0.10, 0.72)
                if paraphrase_score >= paraphrase_threshold:
                    return existing_node, "entityless_paraphrase", paraphrase_score

            concept_overlap = canonical_concept_overlap(node.content, existing_node.content)
            if (
                node_entity is not None
                and existing_entity is not None
                and node_entity[0] == existing_entity[0]
                and concept_overlap >= 0.30
            ):
                return existing_node, "same_entity_concept_overlap", concept_overlap
            if concept_overlap >= 0.50 and similarity >= 0.35:
                return existing_node, "canonical_concept_overlap", concept_overlap

            # ── Layer 3e: pure cosine fallback (conservative global threshold) ─
            if similarity >= self.dedup_similarity_threshold:
                if best_match is None or similarity > best_match[1]:
                    best_match = (existing_node, similarity)

        if best_match is None:
            return None

        return best_match[0], "high_similarity", best_match[1]

    def _merge_duplicate_node(
        self,
        connection: sqlite3.Connection,
        *,
        existing_node: Node,
        incoming_node: Node,
    ) -> Node:
        merged_tags = list(dict.fromkeys([*existing_node.tags, *incoming_node.tags]))
        updated_source_prompt = existing_node.source_prompt or incoming_node.source_prompt
        merged_metadata = dict(existing_node.metadata)
        for key, value in incoming_node.metadata.items():
            if key not in merged_metadata:
                merged_metadata[key] = value
        merged_evidence = merge_evidence_records(existing_node.evidence_records, incoming_node.evidence_records)
        merged_valid_from, merged_valid_to = merge_validity_windows(
            existing_node.valid_from,
            incoming_node.valid_from,
            existing_node.valid_to,
            incoming_node.valid_to,
        )
        updated_at = utc_now()
        connection.execute(
            """
            UPDATE nodes
            SET agent_id = ?, project = ?, session_id = ?, context_window_id = COALESCE(context_window_id, ?),
                tags = ?, metadata = ?, source_prompt = ?, evidence_records = ?, valid_from = ?, valid_to = ?, updated_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                _merge_scope_value(existing_node.agent_id, incoming_node.agent_id),
                _merge_scope_value(existing_node.project, incoming_node.project),
                _merge_scope_value(existing_node.session_id, incoming_node.session_id),
                incoming_node.context_window_id,
                json.dumps(merged_tags),
                _encode_metadata(merged_metadata),
                updated_source_prompt,
                _encode_evidence_records(merged_evidence),
                merged_valid_from.isoformat() if merged_valid_from is not None else None,
                merged_valid_to.isoformat() if merged_valid_to is not None else None,
                updated_at.isoformat(),
                existing_node.id,
                self.tenant_id,
            ),
        )
        return Node(
            id=existing_node.id,
            tenant_id=existing_node.tenant_id,
            agent_id=_merge_scope_value(existing_node.agent_id, incoming_node.agent_id),
            project=_merge_scope_value(existing_node.project, incoming_node.project),
            session_id=_merge_scope_value(existing_node.session_id, incoming_node.session_id),
            context_window_id=existing_node.context_window_id or incoming_node.context_window_id,
            label=existing_node.label,
            content=existing_node.content,
            node_type=existing_node.node_type,
            tags=merged_tags,
            source_prompt=updated_source_prompt,
            metadata=merged_metadata,
            evidence_records=merged_evidence,
            valid_from=merged_valid_from,
            valid_to=merged_valid_to,
            created_at=existing_node.created_at,
            updated_at=updated_at,
            access_count=existing_node.access_count,
        )

    def _register_conflicts(
        self,
        connection: sqlite3.Connection,
        node: Node,
    ) -> list[ConflictRecord]:
        if node.node_type not in {NodeType.PREFERENCE, NodeType.DECISION}:
            return []

        rows = connection.execute(
            """
            SELECT id, agent_id, project, session_id, context_window_id, label, content, node_type, tags, source_prompt, metadata,
                   evidence_records, valid_from, valid_to, created_at, updated_at, access_count, embedding, tenant_id
            FROM nodes
            WHERE tenant_id = ? AND id != ?
            """,
            (self.tenant_id, node.id),
        ).fetchall()
        conflicts: list[ConflictRecord] = []
        for row in rows:
            existing_node = self._row_to_node(row)
            if not _scope_matches(
                existing_node,
                agent_id=node.agent_id,
                project=node.project,
                session_id=node.session_id,
            ):
                continue
            reason = detect_conflict_reason(existing_node, node)
            if reason is None:
                continue
            existing_edge = self._find_existing_edge(
                connection,
                source_id=node.id,
                target_id=existing_node.id,
                relationship=RelationType.CONTRADICTS,
            )
            if existing_edge is None:
                edge = Edge(
                    tenant_id=self.tenant_id,
                    source_id=node.id,
                    target_id=existing_node.id,
                    relationship=RelationType.CONTRADICTS,
                    metadata={"origin": "auto-conflict", "reason": reason},
                )
                connection.execute(
                    """
                    INSERT INTO edges (
                        id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.id,
                        edge.tenant_id,
                        edge.source_id,
                        edge.target_id,
                        edge.relationship,
                        edge.weight,
                        json.dumps(edge.metadata),
                        edge.created_at.isoformat(),
                    ),
                )
                self._mark_node_superseded(connection, old_node=existing_node, new_node=node, relationship=edge.relationship)
            conflicts.append(
                ConflictRecord(
                    other_node_id=existing_node.id,
                    other_node_label=existing_node.label,
                    reason=reason,
                )
            )
        return conflicts

    def _mark_node_superseded(
        self,
        connection: sqlite3.Connection,
        *,
        old_node: Node,
        new_node: Node,
        relationship: str,
    ) -> None:
        metadata = dict(old_node.metadata)
        metadata["superseded_by"] = new_node.id
        metadata["superseded_at"] = utc_now().isoformat()
        metadata["superseded_relationship"] = relationship
        connection.execute(
            "UPDATE nodes SET metadata = ?, updated_at = ? WHERE id = ? AND tenant_id = ?",
            (
                _encode_metadata(metadata),
                old_node.updated_at.isoformat(),
                old_node.id,
                self.tenant_id,
            ),
        )

    def _fetch_node_row(self, connection: sqlite3.Connection, node_id: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT id, agent_id, project, session_id, context_window_id, label, content, node_type, tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                   created_at, updated_at, access_count, embedding, tenant_id
            FROM nodes
            WHERE id = ? AND tenant_id = ?
            """,
            (node_id, self.tenant_id),
        ).fetchone()

    def _fetch_nodes_by_ids(
        self,
        connection: sqlite3.Connection,
        node_ids: list[str],
    ) -> list[Node]:
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        rows = connection.execute(
            f"""
            SELECT id, agent_id, project, session_id, context_window_id, label, content, node_type, tags, source_prompt, metadata, evidence_records, valid_from, valid_to,
                   created_at, updated_at, access_count, tenant_id
            FROM nodes
            WHERE tenant_id = ? AND id IN ({placeholders})
            """,
            (self.tenant_id, *node_ids),
        ).fetchall()
        rows_by_id = {row["id"]: row for row in rows}
        return [self._row_to_node(rows_by_id[node_id]) for node_id in node_ids if node_id in rows_by_id]

    def _build_timeline_items(
        self,
        *,
        nodes: list[Node],
        edges: list[Edge],
        include_evidence: bool,
        limit: int,
    ) -> list[ContextTimelineItem]:
        items: list[ContextTimelineItem] = []
        now = time.time()
        for node in nodes:
            node_recency = recency_weight(
                node.updated_at.timestamp(),
                now=now,
                half_life_days=self.recency_half_life_days,
            )
            items.append(
                ContextTimelineItem(
                    kind="node_created",
                    timestamp=node.created_at,
                    label=node.label,
                    summary=node.content,
                    node_id=node.id,
                    recency_score=node_recency,
                )
            )
            if node.updated_at != node.created_at:
                items.append(
                    ContextTimelineItem(
                        kind="node_updated",
                        timestamp=node.updated_at,
                        label=node.label,
                        summary=node.content,
                        node_id=node.id,
                        recency_score=node_recency,
                    )
                )
            if include_evidence:
                for record in node.evidence_records:
                    items.append(
                        ContextTimelineItem(
                            kind="evidence",
                            timestamp=record.observed_at,
                            label=node.label,
                            summary=f"{record.source_role or 'unknown'} turn {record.turn_index}: {record.source_text or node.content}",
                            node_id=node.id,
                            recency_score=node_recency,
                        )
                    )
        node_by_id = {node.id: node for node in nodes}
        for edge in edges:
            source_label = node_by_id.get(edge.source_id).label if edge.source_id in node_by_id else edge.source_id[:8]
            target_label = node_by_id.get(edge.target_id).label if edge.target_id in node_by_id else edge.target_id[:8]
            items.append(
                ContextTimelineItem(
                    kind=f"edge_{edge.relationship}",
                    timestamp=edge.created_at,
                    label=f"{source_label} -> {target_label}",
                    summary=edge.relationship,
                    edge_id=edge.id,
                    recency_score=recency_weight(
                        edge.created_at.timestamp(),
                        now=now,
                        half_life_days=self.recency_half_life_days,
                    ),
                )
            )
        return sorted(
            items,
            key=lambda item: (item.timestamp, item.kind, item.label),
            reverse=True,
        )[:limit]

    def _build_conflict_entries(
        self,
        connection: sqlite3.Connection,
        *,
        edges: list[Edge],
        include_resolved: bool,
        limit: int,
    ) -> list[ConflictEntry]:
        node_ids = list(dict.fromkeys([edge.source_id for edge in edges] + [edge.target_id for edge in edges]))
        nodes_by_id = {node.id: node for node in self._fetch_nodes_by_ids(connection, node_ids)}
        entries: list[ConflictEntry] = []
        for edge in edges:
            resolved, resolution_note, resolved_at = self._conflict_resolution_state(edge)
            if resolved and not include_resolved:
                continue
            source_node = nodes_by_id.get(edge.source_id)
            target_node = nodes_by_id.get(edge.target_id)
            if source_node is None or target_node is None:
                continue
            entries.append(
                ConflictEntry(
                    edge=edge,
                    source_node=source_node,
                    target_node=target_node,
                    resolved=resolved,
                    resolution_note=resolution_note,
                    resolved_at=resolved_at,
                )
            )
            if len(entries) >= limit:
                break
        return entries

    def _conflict_resolution_state(self, edge: Edge) -> tuple[bool, str, datetime | None]:
        metadata = edge.metadata or {}
        resolved = bool(metadata.get("resolved"))
        resolution_note = str(metadata.get("resolution_note", "") or "")
        resolved_at_raw = metadata.get("resolved_at")
        resolved_at = _parse_datetime(resolved_at_raw) if resolved_at_raw else None
        return resolved, resolution_note, resolved_at

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        row_keys = set(row.keys())
        return Node(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            agent_id=row["agent_id"] if "agent_id" in row_keys else "",
            project=row["project"] if "project" in row_keys else "",
            session_id=row["session_id"] if "session_id" in row_keys else "",
            context_window_id=row["context_window_id"] if "context_window_id" in row_keys else None,
            label=row["label"],
            content=row["content"],
            node_type=NodeType(row["node_type"]),
            tags=json.loads(row["tags"] or "[]"),
            source_prompt=row["source_prompt"] or "",
            metadata=_decode_metadata(row["metadata"]) if "metadata" in row_keys else {},
            evidence_records=_decode_evidence_records(row["evidence_records"]) if "evidence_records" in row_keys else [],
            valid_from=_parse_datetime(row["valid_from"]) if "valid_from" in row_keys and row["valid_from"] else None,
            valid_to=_parse_datetime(row["valid_to"]) if "valid_to" in row_keys and row["valid_to"] else None,
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
            access_count=int(row["access_count"] or 0),
        )

    def _row_to_context_window(self, row: sqlite3.Row) -> ContextWindow:
        return ContextWindow(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            repo_id=row["repo_id"],
            session_id=row["session_id"],
            title=row["title"] or "",
            status=row["status"] or "active",
            node_count=int(row["node_count"] or 0),
            embedding_stale=bool(row["embedding_stale"]),
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
            closed_at=_parse_datetime(row["closed_at"]) if row["closed_at"] else None,
        )

    def _row_to_context_window_edge(self, row: sqlite3.Row) -> ContextWindowEdge:
        return ContextWindowEdge(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            source_window_id=row["source_window_id"],
            target_window_id=row["target_window_id"],
            edge_type=row["edge_type"],
            shared_entities=json.loads(row["shared_entities"] or "[]"),
            weight=float(row["weight"] if row["weight"] is not None else 1.0),
            metadata=_decode_metadata(row["metadata"]),
            created_at=_parse_datetime(row["created_at"]),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        return Edge(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            source_id=row["source_id"],
            target_id=row["target_id"],
            relationship=row["relationship"],
            weight=float(row["weight"]),
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=_parse_datetime(row["created_at"]),
        )

    def _row_to_transcript_record(self, row: sqlite3.Row) -> TranscriptRecord:
        return TranscriptRecord(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            agent_id=row["agent_id"] if "agent_id" in row.keys() else "",
            project=row["project"] if "project" in row.keys() else "",
            session_id=row["session_id"] if "session_id" in row.keys() else "",
            observed_at=_parse_datetime(row["observed_at"]),
            turn_index=int(row["turn_index"] or 0),
            role=row["role"] or "",
            transcript_text=row["transcript_text"],
            metadata=_decode_metadata(row["metadata"]) if "metadata" in row.keys() else {},
        )

    def _transcript_scope_matches(
        self,
        record: TranscriptRecord,
        *,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> bool:
        normalized_agent = agent_id.strip().lower()
        normalized_project = project.strip().lower()
        normalized_session = session_id.strip().lower()
        if normalized_agent and record.agent_id.strip().lower() != normalized_agent:
            return False
        if normalized_project and record.project.strip().lower() != normalized_project:
            return False
        if normalized_session and record.session_id.strip().lower() != normalized_session:
            return False
        return True

    def _next_transcript_turn_index(self, connection: sqlite3.Connection, *, session_id: str) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(turn_index), -1) AS max_turn_index
            FROM transcript_records
            WHERE tenant_id = ? AND session_id = ?
            """,
            (self.tenant_id, session_id),
        ).fetchone()
        max_turn_index = row["max_turn_index"]
        return int(-1 if max_turn_index is None else max_turn_index) + 1

    def _store_transcript_record(
        self,
        connection: sqlite3.Connection,
        *,
        agent_id: str,
        project: str,
        session_id: str,
        observed_at: datetime,
        turn_index: int,
        role: str,
        transcript_text: str,
        metadata: dict[str, Any] | None = None,
        message_identity: str | None = None,
    ) -> bool:
        """Insert a transcript record.  Returns True if written, False if skipped (dedup)."""
        record = TranscriptRecord(
            tenant_id=self.tenant_id,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
            observed_at=observed_at,
            turn_index=turn_index,
            role=role,
            transcript_text=transcript_text,
            metadata=metadata or {},
        )
        embedding = self.embedding_model.embed(record.transcript_text)
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO transcript_records (
                id, tenant_id, agent_id, project, session_id, observed_at, turn_index, role,
                transcript_text, embedding, metadata, message_identity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.tenant_id,
                record.agent_id,
                record.project,
                record.session_id,
                record.observed_at.isoformat(),
                record.turn_index,
                record.role,
                record.transcript_text,
                self.embedding_model.to_bytes(embedding),
                _encode_metadata(record.metadata),
                message_identity,
            ),
        )
        return cursor.rowcount > 0

    def _upsert_vault_document(
        self,
        connection: sqlite3.Connection,
        document: Any,
    ) -> tuple[Node, bool]:
        node_id = str(document.frontmatter.get("node_id", "")).strip()
        row = self._fetch_node_row(connection, node_id)
        raw_type = str(document.frontmatter.get("node_type", "note") or "note")
        try:
            node_type = NodeType(raw_type)
        except ValueError:
            node_type = NodeType.NOTE
        tags = [str(tag) for tag in document.frontmatter.get("tags", []) or []]
        agent_id = str(document.frontmatter.get("agent_id", "") or "")
        project = str(document.frontmatter.get("project", "") or "")
        session_id = str(document.frontmatter.get("session_id", "") or "")
        valid_from = self._parse_optional_datetime(document.frontmatter.get("valid_from"))
        valid_to = self._parse_optional_datetime(document.frontmatter.get("valid_to"))
        evidence_records = evidence_from_lines(document.evidence_lines)
        content = document.content.strip() or document.label
        embedding_bytes = self.embedding_model.to_bytes(self.embedding_model.embed(content))
        if row is None:
            created_at = self._parse_optional_datetime(document.frontmatter.get("created_at")) or utc_now()
            updated_at = utc_now()
            node = Node(
                id=node_id,
                tenant_id=self.tenant_id,
                agent_id=agent_id,
                project=project,
                session_id=session_id,
                label=document.label,
                content=content,
                node_type=node_type,
                tags=tags,
                evidence_records=evidence_records,
                valid_from=valid_from,
                valid_to=valid_to,
                created_at=created_at,
                updated_at=updated_at,
            )
            connection.execute(
                """
                INSERT INTO nodes (
                    id, tenant_id, agent_id, project, session_id, label, content, node_type, tags, metadata, embedding,
                    source_prompt, evidence_records, valid_from, valid_to, created_at, updated_at, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.tenant_id,
                    node.agent_id,
                    node.project,
                    node.session_id,
                    node.label,
                    node.content,
                    node.node_type.value,
                    json.dumps(node.tags),
                    _encode_metadata(node.metadata),
                    embedding_bytes,
                    "",
                    _encode_evidence_records(node.evidence_records),
                    node.valid_from.isoformat() if node.valid_from is not None else None,
                    node.valid_to.isoformat() if node.valid_to is not None else None,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                    node.access_count,
                ),
            )
            return node, True

        existing = self._row_to_node(row)
        updated_at = utc_now()
        node = Node(
            id=existing.id,
            tenant_id=existing.tenant_id,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
            label=document.label,
            content=content,
            node_type=node_type,
            tags=tags,
            source_prompt=existing.source_prompt,
            metadata=existing.metadata,
            evidence_records=evidence_records or existing.evidence_records,
            valid_from=valid_from,
            valid_to=valid_to,
            created_at=existing.created_at,
            updated_at=updated_at,
            access_count=existing.access_count,
        )
        connection.execute(
            """
            UPDATE nodes
            SET agent_id = ?, project = ?, session_id = ?, label = ?, content = ?, node_type = ?, tags = ?,
                metadata = ?, embedding = ?, evidence_records = ?, valid_from = ?, valid_to = ?, updated_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                node.agent_id,
                node.project,
                node.session_id,
                node.label,
                node.content,
                node.node_type.value,
                json.dumps(node.tags),
                _encode_metadata(node.metadata),
                embedding_bytes,
                _encode_evidence_records(node.evidence_records),
                node.valid_from.isoformat() if node.valid_from is not None else None,
                node.valid_to.isoformat() if node.valid_to is not None else None,
                node.updated_at.isoformat(),
                node.id,
                self.tenant_id,
            ),
        )
        return node, False

    def _insert_vault_stub_node(
        self,
        connection: sqlite3.Connection,
        *,
        label: str,
        project: str,
        agent_id: str,
        session_id: str,
    ) -> Node:
        node = Node(
            tenant_id=self.tenant_id,
            agent_id=agent_id,
            project=project,
            session_id=session_id,
            label=label,
            content=f"Stub node imported from vault for {label}.",
            node_type=NodeType.NOTE,
            tags=["stub", "vault-import"],
        )
        connection.execute(
            """
            INSERT INTO nodes (
                id, tenant_id, agent_id, project, session_id, label, content, node_type, tags, metadata, embedding,
                source_prompt, evidence_records, valid_from, valid_to, created_at, updated_at, access_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.tenant_id,
                node.agent_id,
                node.project,
                node.session_id,
                node.label,
                node.content,
                node.node_type.value,
                json.dumps(node.tags),
                _encode_metadata(node.metadata),
                self.embedding_model.to_bytes(self.embedding_model.embed(node.content)),
                "",
                _encode_evidence_records([]),
                None,
                None,
                node.created_at.isoformat(),
                node.updated_at.isoformat(),
                node.access_count,
            ),
        )
        return node

    def _insert_edge_record(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        target_id: str,
        relationship: str,
    ) -> Edge:
        edge = Edge(
            tenant_id=self.tenant_id,
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
        )
        connection.execute(
            """
            INSERT INTO edges (
                id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.id,
                edge.tenant_id,
                edge.source_id,
                edge.target_id,
                edge.relationship,
                edge.weight,
                _encode_metadata(edge.metadata),
                edge.created_at.isoformat(),
            ),
        )
        return edge

    def _delete_edge_record(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        target_id: str,
        relationship: str,
    ) -> bool:
        cursor = connection.execute(
            """
            DELETE FROM edges
            WHERE tenant_id = ? AND source_id = ? AND target_id = ? AND relationship = ?
            """,
            (self.tenant_id, source_id, target_id, normalize_relationship(relationship)),
        )
        return int(cursor.rowcount or 0) > 0

    def _parse_optional_datetime(self, raw: Any) -> datetime | None:
        if raw in (None, ""):
            return None
        if isinstance(raw, datetime):
            return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        try:
            return _parse_datetime(str(raw))
        except ValueError:
            return None

    def _load_graph(
        self,
        connection: sqlite3.Connection,
        *,
        node_ids: Iterable[str],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(node_ids)
        rows = connection.execute(
            """
            SELECT source_id, target_id, relationship, weight, metadata, created_at
            FROM edges
            WHERE tenant_id = ?
            """,
            (self.tenant_id,),
        ).fetchall()

        for row in rows:
            try:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            graph.add_edge(
                row["source_id"],
                row["target_id"],
                relationship=row["relationship"] or "relates_to",
                weight=float(row["weight"]) if row["weight"] is not None else 1.0,
                metadata=metadata,
                created_at=row["created_at"],
            )
        return graph

    def _relation_priority(self, relationship: str) -> float:
        return RELATION_WEIGHTS.get(relationship, 0.50)

    def _temporal_sort_value(self, node: Node, hints: Any) -> float:
        if hints.recency_mode == "latest":
            return -node.updated_at.timestamp()
        if hints.recency_mode == "oldest":
            return node.created_at.timestamp()
        return -node.updated_at.timestamp()

    def _seed_temporal_order(self, node: Node, hints: Any) -> float:
        if hints.recency_mode == "latest":
            return -node.updated_at.timestamp()
        if hints.recency_mode == "oldest":
            return node.created_at.timestamp()
        return 0.0

    def _node_is_superseded(self, node: Node) -> bool:
        metadata = node.metadata or {}
        superseded_by = str(metadata.get("superseded_by", "") or "").strip()
        return bool(superseded_by)

    def _strongest_edge_weight(self, node_id: str, edges: list[Edge]) -> float:
        strongest = 0.0
        for edge in edges:
            if edge.source_id == node_id or edge.target_id == node_id:
                strongest = max(strongest, max(0.0, min(1.0, float(edge.weight))))
        return strongest

    def _apply_node_score(
        self,
        node: Node,
        *,
        similarity: float,
        edge_weight: float,
        now: float | None = None,
    ) -> Node:
        recency = recency_weight(
            node.updated_at.timestamp(),
            now=now,
            half_life_days=self.recency_half_life_days,
        )
        final = score_node(
            similarity,
            node.updated_at.timestamp(),
            edge_weight=edge_weight,
            now=now,
            half_life_days=self.recency_half_life_days,
            superseded=self._node_is_superseded(node),
        )
        node.similarity_score = max(0.0, min(1.0, similarity))
        node.recency_score = recency
        node.edge_score = max(0.0, min(1.0, edge_weight))
        node.final_score = final
        return node

    def _sort_scored_nodes(
        self,
        candidate_nodes: list[Node],
        *,
        max_nodes: int,
        temporal_hints: Any,
        similarity_by_id: dict[str, float],
        lexical_by_id: dict[str, float],
        negation_boost_by_id: dict[str, float],
        degree_by_id: dict[str, int],
        max_access: int,
        max_degree: int,
        max_depth: int,
        expanded_depths: dict[str, int],
        edges: list[Edge] | None = None,
        expansion_metadata: dict[str, ExpansionMeta] | None = None,
    ) -> list[Node]:
        edges = edges or []
        now = time.time()

        def combined_score(node: Node) -> float:
            semantic = similarity_by_id.get(node.id, 0.0)
            lexical = lexical_by_id.get(node.id, 0.0)
            similarity = max(0.0, min(1.0, (0.8 * semantic) + (0.2 * lexical)))
            base_edge_weight = self._strongest_edge_weight(node.id, edges)
            degree_component = degree_by_id.get(node.id, 0) / max_degree if max_degree > 0 else 0.0
            depth_component = 1.0 / (1.0 + expanded_depths.get(node.id, max_depth + 1))
            edge_weight = max(base_edge_weight, (0.6 * degree_component) + (0.4 * depth_component))
            base = score_node(
                similarity,
                node.updated_at.timestamp(),
                edge_weight=edge_weight,
                now=now,
                half_life_days=self.recency_half_life_days,
                superseded=self._node_is_superseded(node),
            ) + temporal_score_adjustment(node, temporal_hints) + negation_boost_by_id.get(node.id, 0.0)
            
            if expansion_metadata is not None and node.id in expansion_metadata:
                meta = expansion_metadata[node.id]
                base += RELATION_SCORE_BOOST.get(meta.via_relation, 0.0)
            
            self._apply_node_score(node, similarity=similarity, edge_weight=edge_weight, now=now)
            node.final_score = base
            return base

        if temporal_hints.recency_mode in {"latest", "oldest"}:
            topic_scores = {
                node.id: (0.7 * similarity_by_id.get(node.id, 0.0))
                + (0.3 * lexical_by_id.get(node.id, 0.0))
                + negation_boost_by_id.get(node.id, 0.0)
                for node in candidate_nodes
            }
            topical_nodes = [
                node
                for node in candidate_nodes
                if topic_scores.get(node.id, 0.0) >= TOPIC_RELEVANCE_THRESHOLD
                and (
                    lexical_by_id.get(node.id, 0.0) > 0.0
                    or similarity_by_id.get(node.id, 0.0) >= TOPIC_SEMANTIC_ONLY_THRESHOLD
                )
            ]
            if not topical_nodes:
                topical_nodes = sorted(
                    candidate_nodes,
                    key=lambda node: (-topic_scores.get(node.id, 0.0), node.label.lower()),
                )[: max_nodes * 2]
            else:
                best_topic_score = max(topic_scores.get(node.id, 0.0) for node in topical_nodes)
                narrowed_topical_nodes = [
                    node
                    for node in topical_nodes
                    if topic_scores.get(node.id, 0.0) >= best_topic_score - TEMPORAL_TOPIC_MARGIN
                ]
                if narrowed_topical_nodes:
                    topical_nodes = narrowed_topical_nodes
            if temporal_hints.recency_mode == "latest":
                return sorted(
                    topical_nodes,
                    key=lambda node: (
                        -node.updated_at.timestamp(),
                        -topic_scores.get(node.id, 0.0),
                        node.label.lower(),
                    ),
                )
            return sorted(
                topical_nodes,
                key=lambda node: (
                    node.created_at.timestamp(),
                    -topic_scores.get(node.id, 0.0),
                    node.label.lower(),
                ),
            )
        return sorted(
            candidate_nodes,
            key=lambda node: (-combined_score(node), -node.updated_at.timestamp(), node.label.lower()),
        )

    def _add_clause_seed_ids(
        self,
        *,
        query: str,
        ranked_seed_ids: list[str],
        nodes_by_id: dict[str, Node],
        embeddings_by_id: dict[str, np.ndarray],
        max_seeds: int,
    ) -> list[str]:
        clauses = [
            clause.strip(" ?,.;:")
            for clause in re.split(r"\b(?:and|with|plus)\b", query, flags=re.IGNORECASE)
            if len(clause.strip(" ?,.;:")) >= 4
        ]
        if len(clauses) < 2:
            return ranked_seed_ids

        expanded = list(ranked_seed_ids)
        seen = set(expanded)
        for clause in clauses[:4]:
            expanded_clause = self._expand_intent_query(clause, query)
            clause_embedding = self.embedding_model.embed(expanded_clause)
            lexical_candidates: list[tuple[float, str]] = []
            semantic_candidates: list[tuple[float, str]] = []
            for node_id, node in nodes_by_id.items():
                semantic = max(
                    self.embedding_model.cosine_similarity(clause_embedding, embeddings_by_id[node_id]),
                    0.0,
                )
                lexical = self._lexical_score_for_node(expanded_clause, node)
                score = (0.45 * semantic) + (0.55 * lexical)
                if lexical > 0.0:
                    lexical_candidates.append((score, node_id))
                elif semantic >= 0.75:
                    semantic_candidates.append((score, node_id))
            best_id = ""
            if lexical_candidates:
                best_id = max(lexical_candidates, key=lambda item: item[0])[1]
            elif semantic_candidates:
                best_id = max(semantic_candidates, key=lambda item: item[0])[1]
            if best_id and best_id not in seen:
                expanded.append(best_id)
                seen.add(best_id)
            if len(expanded) >= max_seeds:
                break

        return expanded

    def _has_negation_intent(self, query: str) -> bool:
        lowered = normalize_text(query)
        return any(term in lowered for term in NEGATION_QUERY_TERMS)

    def _negation_boost(self, node: Node) -> float:
        text = normalize_text(" ".join([node.label, node.content, *node.tags]))
        return NEGATION_SCORE_BOOST if any(term in text for term in NEGATION_NODE_TERMS) else 0.0

    def _split_query_intents(self, query: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", query.strip())
        parts = [
            part.strip(" ?,.;:")
            for part in re.split(r"\b(?:and|plus|with|or|because|justified by|supported by|due to)\b", normalized, flags=re.IGNORECASE)
            if len(part.strip(" ?,.;:")) >= 4
        ]
        if len(parts) < 2:
            return []
        return parts[:4]

    def _expand_query_aliases(self, query: str) -> str:
        normalized = query.lower()
        aliases = [
            alias
            for trigger, alias in QUERY_ALIAS_TERMS
            if trigger in normalized
        ]
        if not aliases:
            return query
        return " ".join([query, *aliases])

    def _expand_intent_query(self, intent: str, full_query: str) -> str:
        return self._expand_query_aliases(f"{intent} {full_query}".strip())

    def _lexical_score_for_node(self, query: str, node: Node) -> float:
        tag_text = " ".join(
            tag.replace(":", " ").replace("_", " ").replace("-", " ")
            for tag in node.tags
        )
        content_score = lexical_overlap(query, node.label, node.content)
        if not tag_text:
            return content_score
        tag_score = lexical_overlap(query, tag_text, "")
        return max(content_score, tag_score)

    def _diversify_multi_intent_nodes(
        self,
        *,
        query: str,
        ranked_nodes: list[Node],
        embeddings_by_id: dict[str, np.ndarray],
        max_nodes: int,
    ) -> list[Node]:
        intents = self._split_query_intents(query)
        if len(intents) < 2 or max_nodes < 2:
            return ranked_nodes

        selected: list[Node] = []
        selected_ids: set[str] = set()
        for intent in intents:
            expanded_intent = self._expand_intent_query(intent, query)
            intent_embedding = self.embedding_model.embed(expanded_intent)
            lexical_scored: list[tuple[float, Node]] = []
            semantic_scored: list[tuple[float, Node]] = []
            for node in ranked_nodes:
                embedding = embeddings_by_id.get(node.id)
                if embedding is None:
                    continue
                semantic = max(self.embedding_model.cosine_similarity(intent_embedding, embedding), 0.0)
                lexical = self._lexical_score_for_node(expanded_intent, node)
                score = (0.35 * semantic) + (0.65 * lexical)
                if lexical > 0.0:
                    lexical_scored.append((score, node))
                elif semantic >= 0.75:
                    semantic_scored.append((score, node))
            scored = lexical_scored or semantic_scored
            if not scored:
                continue
            score, node = max(scored, key=lambda item: (item[0], item[1].updated_at.timestamp()))
            if score >= 0.18 and node.id not in selected_ids:
                selected.append(node)
                selected_ids.add(node.id)
            if len(selected) >= max_nodes:
                return selected

        for node in ranked_nodes:
            if node.id not in selected_ids:
                selected.append(node)
                selected_ids.add(node.id)
            if len(selected) >= len(ranked_nodes):
                break
        return selected

    def _enforce_clause_coverage(
        self,
        *,
        query: str,
        selected_nodes: list[Node],
        ranked_nodes: list[Node],
        embeddings_by_id: dict[str, np.ndarray],
        max_nodes: int,
    ) -> list[Node]:
        intents = self._split_query_intents(query)
        if len(intents) < 2 or not ranked_nodes:
            return selected_nodes[:max_nodes]

        selected = list(selected_nodes[:max_nodes])
        selected_ids = {node.id for node in selected}
        if not selected:
            return selected

        clause_candidates: list[Node] = []
        for intent in intents:
            expanded_intent = self._expand_intent_query(intent, query)
            intent_embedding = self.embedding_model.embed(expanded_intent)
            lexical_scored: list[tuple[float, Node]] = []
            semantic_scored: list[tuple[float, Node]] = []
            for node in ranked_nodes:
                embedding = embeddings_by_id.get(node.id)
                if embedding is None:
                    continue
                semantic = max(self.embedding_model.cosine_similarity(intent_embedding, embedding), 0.0)
                lexical = self._lexical_score_for_node(expanded_intent, node)
                score = (0.35 * semantic) + (0.65 * lexical)
                if lexical > 0.0:
                    lexical_scored.append((score, node))
                elif semantic >= 0.75:
                    semantic_scored.append((score, node))
            scored = lexical_scored or semantic_scored
            if not scored:
                continue
            score, node = max(scored, key=lambda item: (item[0], item[1].updated_at.timestamp()))
            if score >= 0.20:
                clause_candidates.append(node)

        for node in clause_candidates:
            if node.id in selected_ids:
                continue
            if len(selected) < max_nodes:
                selected.append(node)
                selected_ids.add(node.id)
                continue
            replacement_index = len(selected) - 1
            if replacement_index < 0:
                break
            selected_ids.remove(selected[replacement_index].id)
            selected[replacement_index] = node
            selected_ids.add(node.id)

        return selected[:max_nodes]

    def _expand_node_depths_with_context(
        self,
        graph: nx.DiGraph,
        seed_ids: list[str],
        max_depth: int,
        *,
        min_priority: float = 0.20,
        decay: float = 0.70,
    ) -> tuple[dict[str, int], dict[str, ExpansionMeta]]:
        ordered: dict[str, int] = {}
        metadata: dict[str, ExpansionMeta] = {}
        seen: set[str] = set()

        # Heap entries: (neg_priority, tiebreaker, node_id, depth, via_relation, from_node, effective_priority)
        _counter = 0
        heap: list[tuple[float, int, str, int, str, str, float]] = []

        for seed_id in seed_ids:
            heapq.heappush(heap, (0.0, _counter, seed_id, 0, "seed", "", 0.0))
            _counter += 1

        while heap:
            neg_pri, _, node_id, depth, via_relation, from_node, effective_priority = heapq.heappop(heap)

            if node_id in seen:
                continue
            seen.add(node_id)
            ordered[node_id] = depth
            if via_relation != "seed":
                metadata[node_id] = ExpansionMeta(
                    via_relation=via_relation,
                    from_node=from_node,
                    effective_priority=effective_priority,
                )

            if depth >= max_depth:
                continue

            neighbors_with_data: list[tuple[str, dict]] = []

            if graph.has_node(node_id):
                for _, neighbor, data in graph.edges(node_id, data=True):
                    if neighbor not in seen:
                        neighbors_with_data.append((neighbor, data))

                for predecessor, _, data in graph.in_edges(node_id, data=True):
                    if predecessor not in seen:
                        neighbors_with_data.append((predecessor, data))

            for neighbor, data in neighbors_with_data:
                relationship = data.get("relationship", "relates_to")
                weight = float(data.get("weight", 1.0))

                effective = (
                    self._relation_priority(relationship)
                    * weight
                    * (decay ** depth)
                )

                if effective < min_priority:
                    continue

                heapq.heappush(
                    heap,
                    (-effective, _counter, neighbor, depth + 1, relationship, node_id, effective),
                )
                _counter += 1

        return ordered, metadata

    def _expand_node_depths(
        self,
        graph: nx.DiGraph,
        seed_ids: list[str],
        max_depth: int,
        *,
        min_priority: float = 0.20,
        decay: float = 0.70,
    ) -> dict[str, int]:
        ordered, _ = self._expand_node_depths_with_context(
            graph, seed_ids, max_depth, min_priority=min_priority, decay=decay
        )
        return ordered

    def _ensure_support_coverage(
        self,
        selected_nodes: list[Node],
        candidate_pool: dict[str, Node],
        graph: nx.DiGraph,
        max_nodes: int,
    ) -> list[Node]:
        """Augment selected nodes with supporting context for contradictions, updates, and dependencies."""
        if len(selected_nodes) >= max_nodes:
            return selected_nodes

        coverage_nodes: list[Node] = []
        seen = {node.id for node in selected_nodes}

        for node in selected_nodes:
            if len(selected_nodes) + len(coverage_nodes) >= max_nodes:
                break

            if graph.has_node(node.id):
                # Find supporting edges via MUST_PAIR_RELATIONS
                for _, neighbor, data in graph.edges(node.id, data=True):
                    if neighbor in seen or neighbor not in candidate_pool:
                        continue
                    relationship = data.get("relationship", "relates_to")
                    if relationship in MUST_PAIR_RELATIONS:
                        support_node = candidate_pool[neighbor]
                        if support_node not in coverage_nodes:
                            coverage_nodes.append(support_node)
                            seen.add(neighbor)
                        if len(selected_nodes) + len(coverage_nodes) >= max_nodes:
                            break

                # Find incoming edges (predecessors) with strong relationships
                for predecessor, _, data in graph.in_edges(node.id, data=True):
                    if predecessor in seen or predecessor not in candidate_pool:
                        continue
                    relationship = data.get("relationship", "relates_to")
                    if relationship in MUST_PAIR_RELATIONS:
                        support_node = candidate_pool[predecessor]
                        if support_node not in coverage_nodes:
                            coverage_nodes.append(support_node)
                            seen.add(predecessor)
                        if len(selected_nodes) + len(coverage_nodes) >= max_nodes:
                            break

        return selected_nodes + coverage_nodes[: max_nodes - len(selected_nodes)]

    def _build_topic_partition(self, graph: nx.Graph, nodes: list[Node]) -> dict[str, int]:
        if graph.number_of_edges() == 0:
            return {node.id: index for index, node in enumerate(nodes)}
        try:
            import community  # type: ignore[import-not-found]

            return community.best_partition(graph)
        except ImportError:  # pragma: no cover
            communities = nx.algorithms.community.greedy_modularity_communities(graph)
            partition: dict[str, int] = {}
            for cluster_id, members in enumerate(communities):
                for member in members:
                    partition[str(member)] = cluster_id
            return partition

    

    def _fetch_edges_for_nodes(
        self,
        connection: sqlite3.Connection,
        node_ids: list[str],
    ) -> list[Edge]:
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        rows = connection.execute(
            f"""
            SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
            FROM edges
            WHERE tenant_id = ?
              AND source_id IN ({placeholders})
              AND target_id IN ({placeholders})
            ORDER BY created_at ASC
            """,
            (self.tenant_id, *node_ids, *node_ids),
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def _increment_access_counts(self, connection: sqlite3.Connection, node_ids: list[str]) -> None:
        if not node_ids:
            return
        placeholders = ", ".join("?" for _ in node_ids)
        connection.execute(
            f"""
            UPDATE nodes
            SET access_count = access_count + 1
            WHERE tenant_id = ? AND id IN ({placeholders})
            """,
            (self.tenant_id, *node_ids),
        )

    def _find_existing_edge(
        self,
        connection: sqlite3.Connection,
        *,
        source_id: str,
        target_id: str,
        relationship: str | RelationType,
    ) -> Edge | None:
        row = connection.execute(
            """
            SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
            FROM edges
            WHERE tenant_id = ? AND source_id = ? AND target_id = ? AND relationship = ?
            LIMIT 1
            """,
            (self.tenant_id, source_id, target_id, normalize_relationship(relationship)),
        ).fetchone()
        return self._row_to_edge(row) if row is not None else None

    def _most_connected_node_ids(
        self,
        connection: sqlite3.Connection,
        *,
        limit: int,
        agent_id: str = "",
        project: str = "",
        session_id: str = "",
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT n.id, n.agent_id, n.project, n.session_id, n.label, n.content, n.node_type, n.tags, n.source_prompt, n.metadata,
                   n.evidence_records, n.valid_from, n.valid_to, n.created_at, n.updated_at, n.access_count, n.tenant_id,
                   COUNT(e.id) AS connection_count
            FROM nodes AS n
            LEFT JOIN edges AS e ON (n.id = e.source_id OR n.id = e.target_id) AND e.tenant_id = ?
            WHERE n.tenant_id = ?
            GROUP BY n.id
            ORDER BY connection_count DESC, n.updated_at DESC
            """,
            (self.tenant_id, self.tenant_id),
        ).fetchall()
        selected: list[str] = []
        for row in rows:
            node = self._row_to_node(row)
            if not _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id):
                continue
            selected.append(str(row["id"]))
            if len(selected) >= limit:
                break
        return selected

    def _find_project_node_ids(
        self,
        connection: sqlite3.Connection,
        *,
        project: str,
        agent_id: str = "",
        session_id: str = "",
        limit: int,
    ) -> list[str]:
        project_lower = project.strip().lower()
        rows = connection.execute(
            """
            SELECT id, agent_id, project, session_id, label, content, node_type, tags, source_prompt, metadata,
                   evidence_records, valid_from, valid_to, created_at, updated_at, access_count, tenant_id
            FROM nodes
            WHERE tenant_id = ?
            ORDER BY updated_at DESC
            """
        , (self.tenant_id,)).fetchall()
        scored: list[tuple[str, float, str]] = []
        for row in rows:
            node = self._row_to_node(row)
            if not _scope_matches(node, agent_id=agent_id, project=project, session_id=session_id):
                continue

            tags = json.loads(row["tags"] or "[]")
            tag_match = 1.0 if any(project_lower in {str(tag).lower(), f"project:{str(tag).lower()}"} for tag in tags) else 0.0
            explicit_match = 1.0 if str(row["project"] or "").strip().lower() == project_lower else 0.0
            lexical = lexical_overlap(project, row["label"], row["content"])
            score = max(explicit_match, tag_match, lexical)
            if score <= 0.0:
                continue
            scored.append((row["id"], score, row["updated_at"]))
        scored.sort(key=lambda item: (-item[1], item[2]), reverse=False)
        return [node_id for node_id, _, _ in scored[:limit]]

    def _fetch_edge_row(self, connection: sqlite3.Connection, edge_id: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
            FROM edges
            WHERE id = ? AND tenant_id = ?
            """,
            (edge_id, self.tenant_id),
        ).fetchone()

    def _build_backup_snapshot(self, connection: sqlite3.Connection) -> dict[str, Any]:
        node_rows = connection.execute(
            """
            SELECT id, tenant_id, agent_id, project, session_id, context_window_id, label, content, node_type, tags, source_prompt, metadata,
                   evidence_records, valid_from, valid_to, created_at, updated_at, access_count
            FROM nodes
            WHERE tenant_id = ?
            ORDER BY created_at ASC
            """
        , (self.tenant_id,)).fetchall()
        edge_rows = connection.execute(
            """
            SELECT id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
            FROM edges
            WHERE tenant_id = ?
            ORDER BY created_at ASC
            """
        , (self.tenant_id,)).fetchall()
        repo_rows = connection.execute(
            """
            SELECT id, tenant_id, name, description, created_at, updated_at
            FROM repos
            WHERE tenant_id = ?
            ORDER BY created_at ASC
            """,
            (self.tenant_id,),
        ).fetchall()
        window_rows = connection.execute(
            """
            SELECT id, tenant_id, repo_id, session_id, title, status, node_count,
                   embedding_stale, created_at, updated_at, closed_at
            FROM context_windows
            WHERE tenant_id = ?
            ORDER BY created_at ASC
            """,
            (self.tenant_id,),
        ).fetchall()
        window_edge_rows = connection.execute(
            """
            SELECT id, tenant_id, source_window_id, target_window_id, edge_type,
                   shared_entities, weight, metadata, created_at
            FROM context_window_edges
            WHERE tenant_id = ?
            ORDER BY created_at ASC
            """,
            (self.tenant_id,),
        ).fetchall()
        return {
            "schema_version": SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "repos": [
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "name": row["name"],
                    "description": row["description"] or "",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in repo_rows
            ],
            "context_windows": [
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "repo_id": row["repo_id"],
                    "session_id": row["session_id"],
                    "title": row["title"] or "",
                    "status": row["status"] or "active",
                    "node_count": int(row["node_count"] or 0),
                    "embedding_stale": True,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "closed_at": row["closed_at"],
                }
                for row in window_rows
            ],
            "context_window_edges": [
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "source_window_id": row["source_window_id"],
                    "target_window_id": row["target_window_id"],
                    "edge_type": row["edge_type"],
                    "shared_entities": json.loads(row["shared_entities"] or "[]"),
                    "weight": float(row["weight"] if row["weight"] is not None else 1.0),
                    "metadata": _decode_metadata(row["metadata"]),
                    "created_at": row["created_at"],
                }
                for row in window_edge_rows
            ],
            "nodes": [
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "agent_id": row["agent_id"] or "",
                    "project": row["project"] or "",
                    "session_id": row["session_id"] or "",
                    "context_window_id": row["context_window_id"],
                    "label": row["label"],
                    "content": row["content"],
                    "node_type": row["node_type"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "source_prompt": row["source_prompt"] or "",
                    "metadata": _decode_metadata(row["metadata"]),
                    "evidence_records": [record.model_dump(mode="json") for record in _decode_evidence_records(row["evidence_records"])],
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "access_count": int(row["access_count"] or 0),
                }
                for row in node_rows
            ],
            "edges": [
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "relationship": row["relationship"],
                    "weight": float(row["weight"]),
                    "metadata": json.loads(row["metadata"] or "{}"),
                    "created_at": row["created_at"],
                }
                for row in edge_rows
            ],
        }

    def _insert_snapshot_node(self, connection: sqlite3.Connection, raw_node: dict[str, Any]) -> None:
        embedding = self.embedding_model.to_bytes(self.embedding_model.embed(raw_node["content"]))
        connection.execute(
            """
            INSERT INTO nodes (
                id, tenant_id, agent_id, project, session_id, context_window_id, label, content, node_type, tags, metadata, embedding,
                source_prompt, evidence_records, valid_from, valid_to, created_at, updated_at, access_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_node["id"],
                raw_node.get("tenant_id", self.tenant_id),
                raw_node.get("agent_id", ""),
                raw_node.get("project", ""),
                raw_node.get("session_id", ""),
                raw_node.get("context_window_id"),
                raw_node["label"],
                raw_node["content"],
                raw_node["node_type"],
                json.dumps(raw_node.get("tags", [])),
                _encode_metadata(raw_node.get("metadata", {})),
                embedding,
                raw_node.get("source_prompt", ""),
                _encode_evidence_records([EvidenceRecord.model_validate(item) for item in raw_node.get("evidence_records", [])]),
                raw_node.get("valid_from"),
                raw_node.get("valid_to"),
                raw_node["created_at"],
                raw_node["updated_at"],
                int(raw_node.get("access_count", 0)),
            ),
        )

    def _upsert_snapshot_repo(self, connection: sqlite3.Connection, raw_repo: dict[str, Any]) -> None:
        now = utc_now().isoformat()
        connection.execute(
            """
            INSERT INTO repos (id, tenant_id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tenant_id = excluded.tenant_id,
                name = excluded.name,
                description = excluded.description,
                updated_at = excluded.updated_at
            """,
            (
                raw_repo["id"],
                raw_repo.get("tenant_id", self.tenant_id),
                raw_repo.get("name", raw_repo["id"]),
                raw_repo.get("description", ""),
                raw_repo.get("created_at") or now,
                raw_repo.get("updated_at") or now,
            ),
        )

    def _upsert_snapshot_context_window(self, connection: sqlite3.Connection, raw_window: dict[str, Any]) -> None:
        now = utc_now().isoformat()
        connection.execute(
            """
            INSERT INTO context_windows (
                id, tenant_id, repo_id, session_id, title, status, node_count,
                embedding, embedding_stale, created_at, updated_at, closed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tenant_id = excluded.tenant_id,
                repo_id = excluded.repo_id,
                session_id = excluded.session_id,
                title = excluded.title,
                status = excluded.status,
                node_count = excluded.node_count,
                embedding = NULL,
                embedding_stale = 1,
                updated_at = excluded.updated_at,
                closed_at = excluded.closed_at
            """,
            (
                raw_window["id"],
                raw_window.get("tenant_id", self.tenant_id),
                raw_window["repo_id"],
                raw_window.get("session_id", "default"),
                raw_window.get("title", ""),
                raw_window.get("status", "active"),
                int(raw_window.get("node_count", 0)),
                raw_window.get("created_at") or now,
                raw_window.get("updated_at") or now,
                raw_window.get("closed_at"),
            ),
        )

    def _upsert_snapshot_context_window_edge(self, connection: sqlite3.Connection, raw_edge: dict[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO context_window_edges (
                id, tenant_id, source_window_id, target_window_id, edge_type,
                shared_entities, weight, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tenant_id = excluded.tenant_id,
                source_window_id = excluded.source_window_id,
                target_window_id = excluded.target_window_id,
                edge_type = excluded.edge_type,
                shared_entities = excluded.shared_entities,
                weight = excluded.weight,
                metadata = excluded.metadata,
                created_at = excluded.created_at
            """,
            (
                raw_edge["id"],
                raw_edge.get("tenant_id", self.tenant_id),
                raw_edge["source_window_id"],
                raw_edge["target_window_id"],
                raw_edge["edge_type"],
                json.dumps(raw_edge.get("shared_entities", []), sort_keys=True),
                float(raw_edge.get("weight", 1.0)),
                _encode_metadata(raw_edge.get("metadata", {})),
                raw_edge["created_at"],
            ),
        )

    def _update_snapshot_node(self, connection: sqlite3.Connection, raw_node: dict[str, Any]) -> None:
        embedding = self.embedding_model.to_bytes(self.embedding_model.embed(raw_node["content"]))
        connection.execute(
            """
            UPDATE nodes
            SET tenant_id = ?, agent_id = ?, project = ?, session_id = ?, context_window_id = ?, label = ?, content = ?, node_type = ?, tags = ?, metadata = ?, embedding = ?,
                source_prompt = ?, evidence_records = ?, valid_from = ?, valid_to = ?,
                created_at = ?, updated_at = ?, access_count = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                raw_node.get("tenant_id", self.tenant_id),
                raw_node.get("agent_id", ""),
                raw_node.get("project", ""),
                raw_node.get("session_id", ""),
                raw_node.get("context_window_id"),
                raw_node["label"],
                raw_node["content"],
                raw_node["node_type"],
                json.dumps(raw_node.get("tags", [])),
                _encode_metadata(raw_node.get("metadata", {})),
                embedding,
                raw_node.get("source_prompt", ""),
                _encode_evidence_records([EvidenceRecord.model_validate(item) for item in raw_node.get("evidence_records", [])]),
                raw_node.get("valid_from"),
                raw_node.get("valid_to"),
                raw_node["created_at"],
                raw_node["updated_at"],
                int(raw_node.get("access_count", 0)),
                raw_node["id"],
                self.tenant_id,
            ),
        )

    def _insert_snapshot_edge(self, connection: sqlite3.Connection, raw_edge: dict[str, Any]) -> None:
        self._require_node(connection, raw_edge["source_id"])
        self._require_node(connection, raw_edge["target_id"])
        connection.execute(
            """
            INSERT INTO edges (id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_edge["id"],
                raw_edge.get("tenant_id", self.tenant_id),
                raw_edge["source_id"],
                raw_edge["target_id"],
                raw_edge["relationship"],
                float(raw_edge.get("weight", 1.0)),
                json.dumps(raw_edge.get("metadata", {})),
                raw_edge["created_at"],
            ),
        )

    def _update_snapshot_edge(self, connection: sqlite3.Connection, raw_edge: dict[str, Any]) -> None:
        self._require_node(connection, raw_edge["source_id"])
        self._require_node(connection, raw_edge["target_id"])
        connection.execute(
            """
            UPDATE edges
            SET tenant_id = ?, source_id = ?, target_id = ?, relationship = ?, weight = ?, metadata = ?, created_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                raw_edge.get("tenant_id", self.tenant_id),
                raw_edge["source_id"],
                raw_edge["target_id"],
                raw_edge["relationship"],
                float(raw_edge.get("weight", 1.0)),
                json.dumps(raw_edge.get("metadata", {})),
                raw_edge["created_at"],
                raw_edge["id"],
                self.tenant_id,
            ),
        )
