from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import networkx as nx
import numpy as np

from waggle.auth import generate_api_key, hash_api_key, verify_api_key
from waggle.embeddings import EmbeddingModel
from waggle.errors import AuthenticationError, ValidationFailure
from waggle.extractor import EXTRACT_BACKEND, extract_with_llm
from waggle.intelligence import (
    compatible_node_types,
    content_token_jaccard,
    detect_conflict_reason,
    extract_conversation_candidates,
    infer_label,
    infer_node_type,
    infer_relationship,
    infer_temporal_hints,
    is_acronym_match,
    label_similarity,
    lexical_overlap,
    normalize_text,
    parse_since_value,
    score_node,
    split_atomic_items,
    summarize_topic,
    temporal_score_adjustment,
    tokenize_text,
    type_aware_dedup_threshold,
    within_time_window,
)
from waggle.models import (
    ApiKeyCreateResult,
    ApiKeyRecord,
    BackupResult,
    ConflictRecord,
    ConnectedNodeStat,
    Edge,
    GraphDiffResult,
    GraphStats,
    ImportResult,
    Node,
    NodeStoreResult,
    NodeType,
    ObservationResult,
    PrimeContextResult,
    RecentNodeStat,
    RelationType,
    SubgraphResult,
    TenantRecord,
    TopicCluster,
    TopicResult,
    utc_now,
)

SCHEMA_VERSION = 2


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
    label TEXT NOT NULL,
    content TEXT NOT NULL,
    node_type TEXT NOT NULL CHECK(
        node_type IN ('fact', 'entity', 'concept', 'preference', 'decision', 'question', 'note')
    ),
    tags TEXT DEFAULT '[]',
    embedding BLOB,
    source_prompt TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'local-default',
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT NOT NULL CHECK(
        relationship IN (
            'relates_to',
            'contradicts',
            'depends_on',
            'part_of',
            'updates',
            'derived_from',
            'similar_to'
        )
    ),
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_tenant_type ON nodes(tenant_id, node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_tenant_updated ON nodes(tenant_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_relationship ON edges(relationship);
CREATE INDEX IF NOT EXISTS idx_edges_tenant_relationship ON edges(tenant_id, relationship);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
"""


def _parse_datetime(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
        export_dir: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.embedding_model = embedding_model
        self.tenant_id = tenant_id.strip() or "local-default"
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.dedup_same_label_threshold = dedup_same_label_threshold
        self.export_dir = Path(export_dir).expanduser() if export_dir is not None else self.db_path.parent / "exports"
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(SCHEMA_SQL)
            self._migrate_legacy_schema(connection)
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
        if "tenant_id" not in edge_columns:
            connection.execute(
                f"ALTER TABLE edges ADD COLUMN tenant_id TEXT NOT NULL DEFAULT '{self.tenant_id}'"
            )
            connection.execute("UPDATE edges SET tenant_id = ? WHERE tenant_id = ''", (self.tenant_id,))
        connection.execute(
            """
            INSERT OR IGNORE INTO schema_migrations (version, applied_at)
            VALUES (?, ?)
            """,
            (SCHEMA_VERSION, utc_now().isoformat()),
        )

    def add_node(
        self,
        *,
        label: str,
        content: str,
        node_type: NodeType,
        tags: list[str] | None = None,
        source_prompt: str = "",
    ) -> NodeStoreResult:
        node = Node(
            tenant_id=self.tenant_id,
            label=label,
            content=content,
            node_type=node_type,
            tags=tags or [],
            source_prompt=source_prompt,
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
                return NodeStoreResult(
                    node=merged_node,
                    created=False,
                    dedup_reason=dedup_reason,
                    similarity=similarity,
                )

            connection.execute(
                """
                INSERT INTO nodes (
                    id, tenant_id, label, content, node_type, tags, embedding,
                    source_prompt, created_at, updated_at, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.tenant_id,
                    node.label,
                    node.content,
                    node.node_type.value,
                    json.dumps(node.tags),
                    self.embedding_model.to_bytes(embedding),
                    node.source_prompt,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                    node.access_count,
                ),
            )
            conflicts = self._register_conflicts(connection, node)
        return NodeStoreResult(node=node, created=True, conflicts=conflicts)

    def add_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        relationship: RelationType,
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
                    edge.relationship.value,
                    edge.weight,
                    json.dumps(edge.metadata),
                    edge.created_at.isoformat(),
                ),
            )
        return edge

    def get_node(self, node_id: str) -> Node:
        with self._lock, self._connect() as connection:
            row = self._fetch_node_row(connection, node_id)
            if row is None:
                raise ValueError(f"Node not found: {node_id}")
            return self._row_to_node(row)

    def query(self, *, query: str, max_nodes: int = 20, max_depth: int = 2) -> SubgraphResult:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Query cannot be empty.")
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1.")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        with self._lock, self._connect() as connection:
            temporal_hints = infer_temporal_hints(query_text)
            node_rows = connection.execute(
                """
                SELECT id, label, content, node_type, tags, source_prompt,
                       created_at, updated_at, access_count, embedding
                FROM nodes
                WHERE tenant_id = ? AND embedding IS NOT NULL
                """
            , (self.tenant_id,)).fetchall()
            total_nodes = len(node_rows)
            if total_nodes == 0:
                return SubgraphResult(query=query_text, total_nodes_in_graph=0)

            nodes_by_id: dict[str, Node] = {}
            embeddings_by_id: dict[str, np.ndarray] = {}
            for row in node_rows:
                node = self._row_to_node(row)
                nodes_by_id[node.id] = node
                embeddings_by_id[node.id] = self.embedding_model.from_bytes(row["embedding"])

            query_embedding = self.embedding_model.embed(query_text)
            similarity_by_id = {
                node_id: max(self.embedding_model.cosine_similarity(query_embedding, embedding), 0.0)
                for node_id, embedding in embeddings_by_id.items()
            }
            lexical_by_id = {
                node_id: lexical_overlap(query_text, node.label, node.content)
                for node_id, node in nodes_by_id.items()
            }

            seed_count = min(total_nodes, max(1, max_nodes // 2))
            seed_candidates = [
                (
                    node_id,
                    (0.72 * similarity_by_id.get(node_id, 0.0))
                    + (0.28 * lexical_by_id.get(node_id, 0.0)),
                    self._seed_temporal_order(nodes_by_id[node_id], temporal_hints),
                )
                for node_id in nodes_by_id
            ]
            if temporal_hints.recency_mode in {"latest", "oldest"}:
                ranked_seed_ids = [
                    item[0]
                    for item in sorted(
                        seed_candidates,
                        key=lambda item: (item[2], -item[1], nodes_by_id[item[0]].label.lower()),
                    )[:seed_count]
                ]
            else:
                ranked_seed_ids = [
                    item[0]
                    for item in sorted(
                        seed_candidates,
                        key=lambda item: (-item[1], item[2], nodes_by_id[item[0]].label.lower()),
                    )[:seed_count]
                ]

            graph = self._load_graph(connection, node_ids=nodes_by_id.keys())
            expanded_depths = self._expand_node_depths(graph, ranked_seed_ids, max_depth)
            candidate_nodes = [nodes_by_id[node_id] for node_id in expanded_depths]
            temporal_candidates = [
                node for node in candidate_nodes if within_time_window(node, temporal_hints)
            ]
            if temporal_candidates:
                candidate_nodes = temporal_candidates

            max_access = max((node.access_count for node in candidate_nodes), default=0)
            degree_by_id = dict(graph.degree(expanded_depths.keys()))
            max_degree = max(degree_by_id.values(), default=0)
            scored_nodes = self._sort_scored_nodes(
                candidate_nodes,
                temporal_hints=temporal_hints,
                similarity_by_id=similarity_by_id,
                lexical_by_id=lexical_by_id,
                degree_by_id=degree_by_id,
                max_access=max_access,
                max_degree=max_degree,
                max_depth=max_depth,
                expanded_depths=expanded_depths,
            )
            selected_nodes = scored_nodes[:max_nodes]
            selected_ids = [node.id for node in selected_nodes]

            edges = self._fetch_edges_for_nodes(connection, selected_ids)
            self._increment_access_counts(connection, selected_ids)
            for node in selected_nodes:
                node.access_count += 1

            return SubgraphResult(
                nodes=selected_nodes,
                edges=edges,
                query=query_text,
                total_nodes_in_graph=total_nodes,
            )

    def get_related(self, *, node_id: str, max_depth: int = 2) -> SubgraphResult:
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        with self._lock, self._connect() as connection:
            self._require_node(connection, node_id)
            node_rows = connection.execute(
                """
                SELECT id, label, content, node_type, tags, source_prompt,
                       created_at, updated_at, access_count
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
    ) -> Node:
        if content is None and label is None and tags is None:
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
                label=updated_label,
                content=updated_content,
                node_type=node.node_type,
                tags=updated_tags,
                source_prompt=node.source_prompt,
                created_at=node.created_at,
                updated_at=updated_at,
                access_count=node.access_count,
            )

            connection.execute(
                """
                UPDATE nodes
                SET label = ?, content = ?, tags = ?, embedding = ?, updated_at = ?
                WHERE id = ? AND tenant_id = ?
                """,
                (
                    updated_node.label,
                    updated_node.content,
                    json.dumps(updated_node.tags),
                    embedding_bytes,
                    updated_node.updated_at.isoformat(),
                    updated_node.id,
                    self.tenant_id,
                ),
            )
            return updated_node

    def delete_node(self, *, node_id: str) -> Node:
        with self._lock, self._connect() as connection:
            row = self._fetch_node_row(connection, node_id)
            if row is None:
                raise ValueError(f"Node not found: {node_id}")
            node = self._row_to_node(row)
            connection.execute("DELETE FROM nodes WHERE id = ? AND tenant_id = ?", (node_id, self.tenant_id))
            return node

    def list_recent_nodes(self, limit: int = 10) -> list[Node]:
        limit = max(1, limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, label, content, node_type, tags, source_prompt,
                       created_at, updated_at, access_count
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (self.tenant_id, limit),
            ).fetchall()
            return [self._row_to_node(row) for row in rows]

    def get_stats(self) -> GraphStats:
        with self._lock, self._connect() as connection:
            total_nodes = int(
                connection.execute("SELECT COUNT(*) FROM nodes WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            total_edges = int(
                connection.execute("SELECT COUNT(*) FROM edges WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
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
                SELECT id, label, content, node_type, tags, source_prompt,
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
                label=edge.relationship.value,
                title=f"weight={edge.weight}",
                value=max(edge.weight, 0.1),
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

    def observe_conversation(self, *, user_message: str, assistant_response: str) -> ObservationResult:
        transcript = f"user: {user_message.strip()}\nassistant: {assistant_response.strip()}".strip()
        candidates = None
        
        if EXTRACT_BACKEND in ("auto", "llm"):
            candidates = extract_with_llm(user_message, assistant_response)
            
        if candidates is None:
            candidates = extract_conversation_candidates(
                user_message=user_message,
                assistant_response=assistant_response,
            )
        
        result = ObservationResult()
        for candidate in candidates:
            store_result = self.add_node(
                label=str(candidate["label"]),
                content=str(candidate["content"]),
                node_type=candidate["node_type"],
                tags=list(candidate.get("tags", [])),
                source_prompt=transcript,
            )
            result.stored_nodes.append(store_result.node)
            if store_result.created:
                result.created_count += 1
            else:
                result.reused_count += 1
            result.conflicts.extend(store_result.conflicts)
        return result

    def graph_diff(self, *, since: str = "24h") -> GraphDiffResult:
        cutoff = parse_since_value(since)
        with self._lock, self._connect() as connection:
            added_nodes = [
                self._row_to_node(row)
                for row in connection.execute(
                    """
                    SELECT id, label, content, node_type, tags, source_prompt, created_at, updated_at, access_count
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
                    SELECT id, label, content, node_type, tags, source_prompt, created_at, updated_at, access_count
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
                edge for edge in created_edges if edge.relationship == RelationType.CONTRADICTS
            ]
        return GraphDiffResult(
            since=since,
            added_nodes=added_nodes,
            updated_nodes=updated_nodes,
            created_edges=created_edges,
            contradiction_edges=contradiction_edges,
        )

    def prime_context(self, *, project: str = "") -> PrimeContextResult:
        with self._lock, self._connect() as connection:
            total_nodes = int(
                connection.execute("SELECT COUNT(*) FROM nodes WHERE tenant_id = ?", (self.tenant_id,)).fetchone()[0]
            )
            if total_nodes == 0:
                return PrimeContextResult(project=project, summary="No stored memory is available yet.")

            selected_ids: list[str] = []
            selected_ids.extend(self._most_connected_node_ids(connection, limit=5))
            selected_ids.extend(node.id for node in self.list_recent_nodes(limit=5))
            if project.strip():
                selected_ids.extend(self._find_project_node_ids(connection, project=project, limit=8))

            unique_ids = list(dict.fromkeys(selected_ids))
            nodes = self._fetch_nodes_by_ids(connection, unique_ids)
            edges = self._fetch_edges_for_nodes(connection, [node.id for node in nodes])

        summary = (
            f"Prime context for '{project}' with {len(nodes)} nodes selected from {total_nodes} total nodes."
            if project.strip()
            else f"Prime context with {len(nodes)} nodes selected from {total_nodes} total nodes."
        )
        return PrimeContextResult(
            project=project,
            summary=summary,
            nodes=nodes,
            edges=edges,
            total_nodes_in_graph=total_nodes,
        )

    def get_topics(self) -> TopicResult:
        with self._lock, self._connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, label, content, node_type, tags, source_prompt, created_at, updated_at, access_count
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
            SELECT id, label, content, node_type, tags, source_prompt,
                   created_at, updated_at, access_count, embedding
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
            if not compatible_node_types(node.node_type, existing_node.node_type):
                continue
            existing_label = normalize_text(existing_node.label)
            existing_content = normalize_text(existing_node.content)

            # ── Layer 1: exact string matches (free, always run) ─────────────
            if normalized_content == existing_content:
                return existing_node, "exact_content", 1.0
            if normalized_label == existing_label:
                return existing_node, "exact_label", 1.0

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

            # ── Layer 3b: Jaccard-boosted merge (type-aware lower threshold) ──
            # If content words overlap significantly AND cosine is high for the
            # node type, treat as duplicate — catches paraphrase true-dups.
            jaccard = content_token_jaccard(node.content, existing_node.content)
            boosted_threshold = max(type_threshold - 0.05, 0.70)
            if jaccard >= 0.35 and similarity >= boosted_threshold:
                return existing_node, "jaccard_boosted_similarity", similarity

            # ── Layer 3c: pure cosine fallback (conservative global threshold) ─
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
        updated_at = utc_now()
        connection.execute(
            """
            UPDATE nodes
            SET tags = ?, source_prompt = ?, updated_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                json.dumps(merged_tags),
                updated_source_prompt,
                updated_at.isoformat(),
                existing_node.id,
                self.tenant_id,
            ),
        )
        return Node(
            id=existing_node.id,
            tenant_id=existing_node.tenant_id,
            label=existing_node.label,
            content=existing_node.content,
            node_type=existing_node.node_type,
            tags=merged_tags,
            source_prompt=updated_source_prompt,
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
            SELECT id, label, content, node_type, tags, source_prompt,
                   created_at, updated_at, access_count, embedding
            FROM nodes
            WHERE tenant_id = ? AND id != ?
            """,
            (self.tenant_id, node.id),
        ).fetchall()
        conflicts: list[ConflictRecord] = []
        for row in rows:
            existing_node = self._row_to_node(row)
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
                        edge.relationship.value,
                        edge.weight,
                        json.dumps(edge.metadata),
                        edge.created_at.isoformat(),
                    ),
                )
            conflicts.append(
                ConflictRecord(
                    other_node_id=existing_node.id,
                    other_node_label=existing_node.label,
                    reason=reason,
                )
            )
        return conflicts

    def _fetch_node_row(self, connection: sqlite3.Connection, node_id: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT id, label, content, node_type, tags, source_prompt,
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
            SELECT id, label, content, node_type, tags, source_prompt, created_at, updated_at, access_count, tenant_id
            FROM nodes
            WHERE tenant_id = ? AND id IN ({placeholders})
            """,
            (self.tenant_id, *node_ids),
        ).fetchall()
        rows_by_id = {row["id"]: row for row in rows}
        return [self._row_to_node(rows_by_id[node_id]) for node_id in node_ids if node_id in rows_by_id]

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        return Node(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            label=row["label"],
            content=row["content"],
            node_type=NodeType(row["node_type"]),
            tags=json.loads(row["tags"] or "[]"),
            source_prompt=row["source_prompt"] or "",
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
            access_count=int(row["access_count"] or 0),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        return Edge(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else self.tenant_id,
            source_id=row["source_id"],
            target_id=row["target_id"],
            relationship=RelationType(row["relationship"]),
            weight=float(row["weight"]),
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=_parse_datetime(row["created_at"]),
        )

    def _load_graph(
        self,
        connection: sqlite3.Connection,
        *,
        node_ids: Iterable[str],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(node_ids)
        for row in connection.execute(
            "SELECT source_id, target_id FROM edges WHERE tenant_id = ?",
            (self.tenant_id,),
        ).fetchall():
            graph.add_edge(row["source_id"], row["target_id"])
        return graph

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

    def _sort_scored_nodes(
        self,
        candidate_nodes: list[Node],
        *,
        temporal_hints: Any,
        similarity_by_id: dict[str, float],
        lexical_by_id: dict[str, float],
        degree_by_id: dict[str, int],
        max_access: int,
        max_degree: int,
        max_depth: int,
        expanded_depths: dict[str, int],
    ) -> list[Node]:
        def combined_score(node: Node) -> float:
            return score_node(
                node=node,
                semantic_similarity=similarity_by_id.get(node.id, 0.0),
                lexical_score=lexical_by_id.get(node.id, 0.0),
                max_access=max_access,
                degree_score=(degree_by_id.get(node.id, 0) / max_degree if max_degree > 0 else 0.0),
                depth=expanded_depths.get(node.id, max_depth + 1),
            ) + temporal_score_adjustment(node, temporal_hints)

        if temporal_hints.recency_mode == "latest":
            return sorted(
                candidate_nodes,
                key=lambda node: (-node.updated_at.timestamp(), -combined_score(node), node.label.lower()),
            )
        if temporal_hints.recency_mode == "oldest":
            return sorted(
                candidate_nodes,
                key=lambda node: (node.created_at.timestamp(), -combined_score(node), node.label.lower()),
            )
        return sorted(
            candidate_nodes,
            key=lambda node: (-combined_score(node), -node.updated_at.timestamp(), node.label.lower()),
        )

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

    def _expand_node_depths(self, graph: nx.DiGraph, seed_ids: list[str], max_depth: int) -> dict[str, int]:
        ordered: dict[str, int] = {}
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque((seed_id, 0) for seed_id in seed_ids)

        while queue:
            node_id, depth = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered[node_id] = depth
            if depth >= max_depth:
                continue
            neighbors = list(graph.predecessors(node_id)) + list(graph.successors(node_id))
            for neighbor in neighbors:
                if neighbor not in seen:
                    queue.append((neighbor, depth + 1))

        return ordered

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
        relationship: RelationType,
    ) -> Edge | None:
        row = connection.execute(
            """
            SELECT id, source_id, target_id, relationship, weight, metadata, created_at, tenant_id
            FROM edges
            WHERE tenant_id = ? AND source_id = ? AND target_id = ? AND relationship = ?
            LIMIT 1
            """,
            (self.tenant_id, source_id, target_id, relationship.value),
        ).fetchone()
        return self._row_to_edge(row) if row is not None else None

    def _most_connected_node_ids(self, connection: sqlite3.Connection, *, limit: int) -> list[str]:
        rows = connection.execute(
            """
            SELECT n.id, COUNT(e.id) AS connection_count, n.updated_at
            FROM nodes AS n
            LEFT JOIN edges AS e ON (n.id = e.source_id OR n.id = e.target_id) AND e.tenant_id = ?
            WHERE n.tenant_id = ?
            GROUP BY n.id
            ORDER BY connection_count DESC, n.updated_at DESC
            LIMIT ?
            """,
            (self.tenant_id, self.tenant_id, limit),
        ).fetchall()
        return [str(row["id"]) for row in rows]

    def _find_project_node_ids(
        self,
        connection: sqlite3.Connection,
        *,
        project: str,
        limit: int,
    ) -> list[str]:
        project_lower = project.strip().lower()
        rows = connection.execute(
            """
            SELECT id, label, content, tags, updated_at
            FROM nodes
            WHERE tenant_id = ?
            ORDER BY updated_at DESC
            """
        , (self.tenant_id,)).fetchall()
        scored: list[tuple[str, float, str]] = []
        for row in rows:
            tags = json.loads(row["tags"] or "[]")
            tag_match = 1.0 if any(project_lower == str(tag).lower() for tag in tags) else 0.0
            lexical = lexical_overlap(project, row["label"], row["content"])
            score = max(tag_match, lexical)
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
            SELECT id, tenant_id, label, content, node_type, tags, source_prompt, created_at, updated_at, access_count
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
        return {
            "schema_version": SCHEMA_VERSION,
            "tenant_id": self.tenant_id,
            "nodes": [
                {
                    "id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "label": row["label"],
                    "content": row["content"],
                    "node_type": row["node_type"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "source_prompt": row["source_prompt"] or "",
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
                id, tenant_id, label, content, node_type, tags, embedding,
                source_prompt, created_at, updated_at, access_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_node["id"],
                raw_node.get("tenant_id", self.tenant_id),
                raw_node["label"],
                raw_node["content"],
                raw_node["node_type"],
                json.dumps(raw_node.get("tags", [])),
                embedding,
                raw_node.get("source_prompt", ""),
                raw_node["created_at"],
                raw_node["updated_at"],
                int(raw_node.get("access_count", 0)),
            ),
        )

    def _update_snapshot_node(self, connection: sqlite3.Connection, raw_node: dict[str, Any]) -> None:
        embedding = self.embedding_model.to_bytes(self.embedding_model.embed(raw_node["content"]))
        connection.execute(
            """
            UPDATE nodes
            SET tenant_id = ?, label = ?, content = ?, node_type = ?, tags = ?, embedding = ?,
                source_prompt = ?, created_at = ?, updated_at = ?, access_count = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (
                raw_node.get("tenant_id", self.tenant_id),
                raw_node["label"],
                raw_node["content"],
                raw_node["node_type"],
                json.dumps(raw_node.get("tags", [])),
                embedding,
                raw_node.get("source_prompt", ""),
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
