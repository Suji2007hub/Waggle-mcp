from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import networkx as nx
import numpy as np

from waggle.abhi import (
    diff_abhi_files,
    execute_abhi_query,
    load_abhi_chunk_file,
    load_abhi_document,
    merge_abhi_files,
    query_abhi_file,
)
from waggle.graph import MemoryGraph
from waggle.models import NodeType, RelationType


class FakeEmbeddingModel:
    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            index = sum(ord(character) for character in token) % len(vector)
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm

    def to_bytes(self, embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    def from_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))


def make_graph(tmp_path: Path) -> MemoryGraph:
    return MemoryGraph(tmp_path / "memory.db", FakeEmbeddingModel())


def test_memory_graph_migrates_legacy_nodes_before_creating_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-memory.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL DEFAULT 'local-default',
            agent_id TEXT DEFAULT '',
            project TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            label TEXT NOT NULL,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            embedding BLOB,
            source_prompt TEXT DEFAULT '',
            evidence_records TEXT DEFAULT '[]',
            valid_from TEXT DEFAULT NULL,
            valid_to TEXT DEFAULT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE edges (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL DEFAULT 'local-default',
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relationship TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE transcript_records (
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
            metadata TEXT DEFAULT '{}'
        );
        """
    )
    connection.close()

    graph = MemoryGraph(db_path, FakeEmbeddingModel())

    with graph._lock, graph._connect() as migrated:
        node_columns = {row["name"] for row in migrated.execute("PRAGMA table_info(nodes)").fetchall()}
        node_indexes = {row["name"] for row in migrated.execute("PRAGMA index_list(nodes)").fetchall()}

    assert "context_window_id" in node_columns
    assert "idx_nodes_context_window" in node_indexes


def _set_node_timestamp(graph: MemoryGraph, node_id: str, timestamp: datetime) -> None:
    with graph._lock, graph._connect() as connection:
        connection.execute(
            "UPDATE nodes SET created_at = ?, updated_at = ? WHERE id = ?",
            (timestamp.isoformat(), timestamp.isoformat(), node_id),
        )


def _set_node_embedding_null(graph: MemoryGraph, node_id: str) -> None:
    with graph._lock, graph._connect() as connection:
        connection.execute(
            "UPDATE nodes SET embedding = NULL WHERE id = ?",
            (node_id,),
        )


def _insert_transcript_record(
    graph: MemoryGraph,
    *,
    session_id: str,
    project: str,
    transcript_text: str,
    observed_at: datetime,
    role: str = "user",
) -> None:
    embedding = graph.embedding_model.to_bytes(graph.embedding_model.embed(transcript_text))
    with graph._lock, graph._connect() as connection:
        connection.execute(
            """
            INSERT INTO transcript_records (
                id, tenant_id, agent_id, project, session_id, observed_at, turn_index, role,
                transcript_text, embedding, metadata, message_identity
            )
            VALUES (?, ?, '', ?, ?, ?, 0, ?, ?, ?, '{}', NULL)
            """,
            (
                str(uuid4()),
                graph.tenant_id,
                project,
                session_id,
                observed_at.isoformat(),
                role,
                transcript_text,
                embedding,
            ),
        )


def test_add_query_and_related(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    project = graph.add_node(
        label="FastAPI Project",
        content="User is building a FastAPI backend service",
        node_type=NodeType.ENTITY,
        tags=["backend"],
    ).node
    preference = graph.add_node(
        label="Python Preference",
        content="User strongly prefers Python for backend work",
        node_type=NodeType.PREFERENCE,
    ).node
    graph.add_edge(
        source_id=project.id,
        target_id=preference.id,
        relationship=RelationType.RELATES_TO,
    )

    result = graph.query(query="python backend", max_nodes=5, max_depth=2)
    labels = {node.label for node in result.nodes}
    assert "FastAPI Project" in labels
    assert "Python Preference" in labels
    assert len(result.edges) == 1

    related = graph.get_related(node_id=project.id, max_depth=1)
    related_labels = {node.label for node in related.nodes}
    assert related_labels == {"FastAPI Project", "Python Preference"}


def test_update_delete_and_stats(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    node = graph.add_node(
        label="Deployment Note",
        content="Initial deployment uses SQLite",
        node_type=NodeType.NOTE,
    ).node
    updated = graph.update_node(
        node_id=node.id,
        content="Deployment now uses SQLite and nightly backups",
        tags=["ops", "database"],
    )
    assert updated.tags == ["ops", "database"]
    assert "nightly backups" in updated.content

    stats = graph.get_stats()
    assert stats.total_nodes == 1
    assert stats.node_type_breakdown["note"] == 1

    deleted = graph.delete_node(node_id=node.id)
    assert deleted.id == node.id
    assert graph.get_stats().total_nodes == 0


def test_exact_duplicate_nodes_are_reused_and_tags_are_merged(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    first = graph.add_node(
        label="Python Preference",
        content="User prefers Python for backend work",
        node_type=NodeType.PREFERENCE,
        tags=["python"],
    )
    second = graph.add_node(
        label="Python Preference",
        content="User prefers Python for backend work",
        node_type=NodeType.PREFERENCE,
        tags=["backend"],
        source_prompt="duplicate check",
    )

    assert first.created is True
    assert second.created is False
    assert second.node.id == first.node.id
    assert second.dedup_reason == "exact_content"
    assert second.node.tags == ["python", "backend"]
    assert graph.get_stats().total_nodes == 1


def test_semantic_duplicate_nodes_reuse_existing_entry(tmp_path: Path) -> None:
    graph = MemoryGraph(
        tmp_path / "semantic-memory.db",
        FakeEmbeddingModel(),
        dedup_similarity_threshold=0.75,
        dedup_same_label_threshold=0.75,
    )
    first = graph.add_node(
        label="FastAPI Project",
        content="User is building a FastAPI backend service",
        node_type=NodeType.ENTITY,
    )
    second = graph.add_node(
        label="FastAPI Project",
        content="User is building a FastAPI backend app service",
        node_type=NodeType.ENTITY,
    )

    assert first.created is True
    assert second.created is False
    assert second.node.id == first.node.id
    assert second.dedup_reason == "same_label_high_similarity"


def test_entity_resolution_reuses_acronym_matches(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    first = graph.add_node(
        label="Large Language Model",
        content="Large language model memory is important for this project",
        node_type=NodeType.ENTITY,
    )
    second = graph.add_node(
        label="LLM",
        content="LLM memory is important for this project",
        node_type=NodeType.ENTITY,
    )

    assert first.created is True
    assert second.created is False
    assert second.node.id == first.node.id
    assert second.dedup_reason == "acronym_entity_match"


def test_entityless_paraphrase_nodes_can_be_reused(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    first = graph.add_node(
        label="Dark interface preference",
        content="The user wants the app to default to a dark interface",
        node_type=NodeType.PREFERENCE,
    )
    second = graph.add_node(
        label="Low-light UI preference",
        content="They prefer a low-light theme whenever the product opens",
        node_type=NodeType.PREFERENCE,
    )

    assert first.created is True
    assert second.created is False
    assert second.node.id == first.node.id
    assert second.dedup_reason == "canonical_concept_overlap"


def test_temporal_near_duplicates_do_not_merge_across_months(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    first = graph.add_node(
        label="April database choice",
        content="In April, the team chose PostgreSQL for the production database",
        node_type=NodeType.DECISION,
    )
    second = graph.add_node(
        label="May database status",
        content="In May, the team confirmed PostgreSQL is still used in production",
        node_type=NodeType.DECISION,
    )

    assert first.created is True
    assert second.created is True
    assert graph.get_stats().total_nodes == 2


def test_query_ranking_uses_label_lexical_overlap(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="JWT Auth",
        content="Authentication subsystem for the service",
        node_type=NodeType.CONCEPT,
    )
    graph.add_node(
        label="Session Storage",
        content="Stores session data for authenticated users",
        node_type=NodeType.CONCEPT,
    )

    result = graph.query(query="jwt", max_nodes=2, max_depth=0)

    assert result.nodes[0].label == "JWT Auth"


def test_decompose_and_store_creates_nodes_and_edges(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    result = graph.decompose_and_store(
        content="- User prefers Python for backend work\n- Project uses FastAPI for the API",
        context="Backend preferences and stack",
    )

    labels = {node.label for node in result.nodes}
    assert "Backend preferences and stack" in labels
    assert any("User prefers Python" in node.content for node in result.nodes)
    assert any("Project uses FastAPI" in node.content for node in result.nodes)
    assert len(result.edges) >= 2


def test_export_and_import_backup_round_trip(tmp_path: Path) -> None:
    source = make_graph(tmp_path / "source")
    imported = make_graph(tmp_path / "target")
    first = source.add_node(
        label="Backup Node",
        content="This node should survive a backup round trip",
        node_type=NodeType.NOTE,
    ).node
    second = source.add_node(
        label="Backup Concept",
        content="This concept is connected to the backup node",
        node_type=NodeType.CONCEPT,
    ).node
    source.add_edge(
        source_id=first.id,
        target_id=second.id,
        relationship=RelationType.RELATES_TO,
    )

    backup = source.export_graph_backup(output_path=tmp_path / "backup.json")
    imported_result = imported.import_graph_backup(input_path=backup.output_path)

    assert backup.node_count == 2
    assert backup.edge_count == 1
    assert imported_result.nodes_created == 2
    assert imported_result.edges_created == 1
    assert imported.get_stats().total_nodes == 2
    assert imported.get_stats().total_edges == 1


def test_export_validate_and_import_abhi_round_trip(tmp_path: Path) -> None:
    source = make_graph(tmp_path / "source")
    target = make_graph(tmp_path / "target")
    decision = source.add_node(
        label="Use PostgreSQL",
        content="Use PostgreSQL for production.",
        node_type=NodeType.DECISION,
    ).node
    reason = source.add_node(
        label="Replication pain",
        content="MySQL replication has been painful.",
        node_type=NodeType.FACT,
    ).node
    source.add_edge(
        source_id=decision.id,
        target_id=reason.id,
        relationship=RelationType.DEPENDS_ON,
    )

    exported = source.export_abhi(output_path=tmp_path / "memory.abhi")
    validation = source.validate_abhi(input_path=exported.output_path)
    imported = target.import_abhi(input_path=exported.output_path)

    payload = json.loads(Path(exported.output_path).read_text(encoding="utf-8"))

    assert payload["graph"]["nodes"]
    assert payload["schema"]["node_types"]["decision"]["must_have"] == ["content", "ts"]
    assert payload["integrity"]["content_hash"].startswith("sha256:")
    assert validation.valid is True
    assert imported.hash_verified is True
    assert imported.nodes_created == 2
    assert imported.edges_created == 1
    assert target.get_stats().total_nodes == 2


def test_update_and_delete_edge(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    source = graph.add_node(
        label="Source",
        content="Source node",
        node_type=NodeType.NOTE,
    ).node
    target = graph.add_node(
        label="Target",
        content="Target node",
        node_type=NodeType.NOTE,
    ).node
    replacement = graph.add_node(
        label="Replacement",
        content="Replacement node",
        node_type=NodeType.NOTE,
    ).node
    edge = graph.add_edge(
        source_id=source.id,
        target_id=target.id,
        relationship=RelationType.RELATES_TO,
    )

    updated = graph.update_edge(
        edge_id=edge.id,
        target_id=replacement.id,
        relationship=RelationType.DEPENDS_ON,
        weight=0.4,
    )
    deleted = graph.delete_edge(edge_id=edge.id)

    assert updated.target_id == replacement.id
    assert updated.relationship == RelationType.DEPENDS_ON.value
    assert updated.weight == 0.4
    assert deleted.id == edge.id
    assert graph.get_stats().total_edges == 0


def test_ui_state_persists_and_round_trips_through_abhi(tmp_path: Path) -> None:
    graph = make_graph(tmp_path / "source")
    imported = make_graph(tmp_path / "target")
    node = graph.add_node(
        label="Canvas Node",
        content="Node with saved position",
        node_type=NodeType.NOTE,
        project="studio",
    ).node
    graph.save_ui_state(
        project="studio",
        positions={node.id: {"x": 111, "y": 222}},
        zoom=1.25,
        viewport={"center_x": 111, "center_y": 222},
        selected_nodes=[node.id],
    )

    snapshot = graph.get_graph_snapshot(project="studio")
    exported = graph.export_abhi(output_path=tmp_path / "memory.abhi", project="studio")
    imported.import_abhi(input_path=exported.output_path)
    imported_ui = imported.get_ui_state()

    assert snapshot["ui"]["positions"][node.id] == {"x": 111, "y": 222}
    payload = json.loads(Path(exported.output_path).read_text(encoding="utf-8"))
    assert payload["ui"]["positions"][node.id] == {"x": 111, "y": 222}
    assert imported_ui["positions"]
    assert imported_ui["zoom"] == 1.25


def test_execute_abhi_query_matches_recent_and_filtered_nodes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Database decision",
        content="Use PostgreSQL for the main database.",
        node_type=NodeType.DECISION,
        project="studio",
    )
    graph.add_node(
        label="Frontend note",
        content="Keep the browser graph editor responsive.",
        node_type=NodeType.NOTE,
        project="studio",
    )
    exported = graph.export_abhi(output_path=tmp_path / "memory.abhi", project="studio")
    document = load_abhi_document(exported.output_path)

    filtered = execute_abhi_query(document, query_text="FIND nodes WHERE type='decision' AND content CONTAINS 'database'")
    recent = execute_abhi_query(document, query_id="q1")

    assert len(filtered["nodes"]) == 1
    assert filtered["nodes"][0]["type"] == "decision"
    assert recent["query_id"] == "q1"
    assert len(recent["nodes"]) >= 1


def test_diff_and_merge_abhi_files(tmp_path: Path) -> None:
    base = make_graph(tmp_path / "base")
    left = make_graph(tmp_path / "left")
    right = make_graph(tmp_path / "right")

    for graph in (base, left, right):
        graph.add_node(
            label="Decision",
            content="Use PostgreSQL",
            node_type=NodeType.DECISION,
            project="studio",
        )

    left.add_node(
        label="Reason",
        content="Operational familiarity matters.",
        node_type=NodeType.NOTE,
        project="studio",
    )
    right_node = right.list_recent_nodes(limit=1, project="studio")[0]
    right.update_node(node_id=right_node.id, content="Use PostgreSQL with managed backups")

    base_file = base.export_abhi(output_path=tmp_path / "base.abhi", project="studio")
    left_file = left.export_abhi(output_path=tmp_path / "left.abhi", project="studio")
    right_file = right.export_abhi(output_path=tmp_path / "right.abhi", project="studio")

    diff = diff_abhi_files(input_path_a=left_file.output_path, input_path_b=right_file.output_path)
    merged = merge_abhi_files(
        base_input_path=base_file.output_path,
        left_input_path=left_file.output_path,
        right_input_path=right_file.output_path,
        output_path=tmp_path / "merged.abhi",
    )

    assert diff.nodes_added or diff.nodes_updated
    assert Path(merged.output_path).exists()
    merged_doc = load_abhi_document(merged.output_path)
    assert merged_doc["integrity"]["content_hash"].startswith("sha256:")
    assert merged.nodes_merged >= 1


def test_query_abhi_file_updates_event_log_and_relevance_hits(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Decision",
        content="Use PostgreSQL for the main database.",
        node_type=NodeType.DECISION,
        project="studio",
    )
    exported = graph.export_abhi(output_path=tmp_path / "memory.abhi", project="studio")

    result = query_abhi_file(input_path=exported.output_path, query_text="FIND nodes WHERE type='decision'")
    document = load_abhi_document(exported.output_path)
    event_log = document["waggle"]["event_log"]
    matched_node = document["graph"]["nodes"][0]

    assert result.node_count == 1
    assert "log_access" in result.executed_actions
    assert event_log
    assert matched_node["metadata"]["relevance_hits"] == 1


def test_export_abhi_builds_semantic_chunks_and_inspect_reports_them(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    for index in range(70):
        graph.add_node(
            label=f"Decision {index}",
            content=f"Use PostgreSQL for service {index}",
            node_type=NodeType.DECISION,
            project="studio",
        )
    exported = graph.export_abhi(output_path=tmp_path / "chunked.abhi", project="studio")
    inspected = graph.inspect_abhi(input_path=exported.output_path)
    document = load_abhi_document(exported.output_path)
    chunk_index = document["chunks"]["chunk_index"]

    assert inspected.chunk_count >= 2
    assert inspected.load_strategy == "on_demand"
    assert inspected.preload_chunks
    assert document["chunks"]["chunk_payloads"]
    assert all(entry["byte_length"] > 0 for entry in chunk_index.values())


def test_load_abhi_chunk_file_and_query_use_relevant_chunks(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    for index in range(70):
        graph.add_node(
            label=f"Decision {index}",
            content=f"Use PostgreSQL for service {index}",
            node_type=NodeType.DECISION,
            project="studio",
        )
    for index in range(12):
        graph.add_node(
            label=f"Fact {index}",
            content=f"MySQL replication issue {index}",
            node_type=NodeType.FACT,
            project="studio",
        )
    exported = graph.export_abhi(output_path=tmp_path / "chunked-query.abhi", project="studio")

    loaded = load_abhi_chunk_file(
        input_path=exported.output_path,
        query_text="FIND nodes WHERE type='decision' AND content CONTAINS 'PostgreSQL'",
    )
    queried = query_abhi_file(
        input_path=exported.output_path,
        query_text="FIND nodes WHERE type='decision' AND content CONTAINS 'PostgreSQL'",
    )

    assert loaded.chunk_ids
    assert loaded.available_chunk_count >= 2
    assert loaded.node_count >= queried.node_count
    assert queried.chunk_ids
    assert queried.scanned_chunk_count == len(queried.chunk_ids)
    assert all(chunk_id.startswith("decision_") for chunk_id in queried.chunk_ids)


def test_export_graph_html_creates_visualization_file(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    first = graph.add_node(
        label="Visualization Project",
        content="The memory graph should support HTML export.",
        node_type=NodeType.CONCEPT,
    ).node
    second = graph.add_node(
        label="Visualization Edge",
        content="The exported graph should include connected nodes.",
        node_type=NodeType.FACT,
    ).node
    graph.add_edge(
        source_id=second.id,
        target_id=first.id,
        relationship=RelationType.PART_OF,
    )

    output_path = graph.export_graph_html(output_path=tmp_path / "graph.html", include_physics=False)
    html = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "Visualization Project" in html
    assert "Visualization Edge" in html
    assert "part_of" in html


def test_export_window_graph_html_creates_visualization_file(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Deployment",
        content="Deployment uses Kubernetes.",
        node_type=NodeType.FACT,
        project="alpha",
        session_id="chat-a",
    )
    graph.add_node(
        label="Deployment",
        content="Deployment uses Kubernetes.",
        node_type=NodeType.FACT,
        project="alpha",
        session_id="chat-b",
    )
    repo_id = graph.ensure_repo("alpha")
    windows = graph.get_repo_windows(repo_id)
    for window in windows:
        graph.derive_context_window_edges(window.id, repo_id)

    output_path = graph.export_window_graph_html(
        project="alpha",
        output_path=tmp_path / "window-graph.html",
        include_physics=False,
    )
    html = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "chat-a" in html
    assert "chat-b" in html
    assert "entity_overlap" in html


def test_export_context_bundle_query_writes_markdown_and_json(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    decision = graph.add_node(
        label="Use PostgreSQL",
        content="We decided to use PostgreSQL for production.",
        node_type=NodeType.DECISION,
    ).node
    reason = graph.add_node(
        label="MySQL replication pain",
        content="MySQL replication was painful to operate.",
        node_type=NodeType.FACT,
    ).node
    graph.add_edge(
        source_id=decision.id,
        target_id=reason.id,
        relationship=RelationType.DEPENDS_ON,
    )

    exported = graph.export_context_bundle(
        mode="query",
        query="what database did we decide on",
        format="both",
        output_path=tmp_path / "handoff",
        include_source_prompt=False,
    )

    assert exported.markdown_path is not None
    assert exported.json_path is not None
    markdown = Path(exported.markdown_path).read_text(encoding="utf-8")
    payload = json.loads(Path(exported.json_path).read_text(encoding="utf-8"))

    assert "## Decisions With Reasons" in markdown
    assert "Use PostgreSQL" in markdown
    assert "MySQL replication pain" in markdown
    assert payload["export_type"] == "context_bundle"
    assert payload["mode"] == "query"
    assert payload["query"] == "what database did we decide on"
    assert "We decided to use PostgreSQL for production." in payload["summary"]
    assert "MySQL replication pain" in payload["summary"]
    assert payload["nodes"]
    assert payload["edges"]
    assert payload["render_hints"]["token_estimate"] > 0


def test_export_context_bundle_prime_uses_prime_context_summary(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Alpha Project",
        content="Alpha uses FastAPI and SQLite for local development.",
        node_type=NodeType.ENTITY,
        tags=["alpha"],
    )
    graph.add_node(
        label="Alpha Decision",
        content="We decided to keep Alpha on SQLite locally.",
        node_type=NodeType.DECISION,
        tags=["alpha"],
    )

    exported = graph.export_context_bundle(
        mode="prime",
        project="alpha",
        format="markdown",
        output_path=tmp_path / "prime-context.md",
    )

    assert exported.markdown_path is not None
    markdown = Path(exported.markdown_path).read_text(encoding="utf-8")
    assert "## Memory Summary" in markdown
    assert "Prime context" in markdown
    assert "Alpha Decision" in markdown


def test_export_context_bundle_graph_chunks_large_appendix(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    previous = None
    for index in range(45):
        node = graph.add_node(
            label=f"Fact {index}",
            content=f"Portable export item {index}",
            node_type=NodeType.FACT,
        ).node
        if previous is not None:
            graph.add_edge(
                source_id=previous.id,
                target_id=node.id,
                relationship=RelationType.RELATES_TO,
            )
        previous = node

    exported = graph.export_context_bundle(
        mode="graph",
        format="both",
        output_path=tmp_path / "full-graph",
    )

    markdown = Path(exported.markdown_path).read_text(encoding="utf-8")
    payload = json.loads(Path(exported.json_path).read_text(encoding="utf-8"))

    assert "Appendix Chunk 1/" in markdown
    assert exported.bundle.render_hints.chunk_count >= 2
    assert "large_graph" in payload["render_hints"]["truncation_flags"]
    assert payload["stats"]["total_nodes"] == 45


def test_query_replay_mode_returns_transcript_hits(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.observe_conversation(
        user_message="We switched production to PostgreSQL last Friday.",
        assistant_response="I will remember the PostgreSQL migration.",
        session_id="sess-db",
        project="alpha",
    )

    result = graph.query(
        query="what database did we switch production to",
        retrieval_mode="replay",
        session_id="sess-db",
    )

    assert result.retrieval_mode == "replay"
    assert result.replay_hits
    assert result.replay_hits[0].session_id == "sess-db"
    assert "PostgreSQL" in result.replay_hits[0].transcript_text


def test_query_fusion_mode_includes_graph_and_replay_provenance(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.observe_conversation(
        user_message="We switched production to PostgreSQL last Friday.",
        assistant_response="I will remember the PostgreSQL migration.",
        session_id="sess-db",
    )

    result = graph.query(
        query="latest production database",
        retrieval_mode="fusion",
        max_nodes=5,
    )

    assert result.retrieval_mode == "fusion"
    assert result.replay_hits
    assert result.fusion_hits
    assert result.fusion_hits[0].source_lane in {"graph", "replay", "both"}
    assert result.fusion_hits[0].fused_rank >= 1


def test_query_graph_mode_uses_transcript_session_signal_for_node_ranking(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    node_a = graph.add_node(
        label="Session A Note",
        content="Shared planning note.",
        node_type=NodeType.NOTE,
        project="alpha",
        session_id="sess-a",
    ).node
    node_b = graph.add_node(
        label="Session B Note",
        content="Shared planning note.",
        node_type=NodeType.NOTE,
        project="alpha",
        session_id="sess-b",
    ).node
    _set_node_timestamp(graph, node_a.id, timestamp)
    _set_node_timestamp(graph, node_b.id, timestamp)

    _insert_transcript_record(
        graph,
        session_id="sess-a",
        project="alpha",
        transcript_text="We chose PostgreSQL for production.",
        observed_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )

    result = graph.query(
        query="what database did we choose for production",
        max_nodes=2,
        max_depth=0,
        project="alpha",
    )

    assert result.nodes
    assert result.nodes[0].label == "Session A Note"
    assert result.nodes[0].similarity_score is not None
    assert result.nodes[0].final_score is not None


def test_prime_context_prefers_nodes_from_recent_transcript_sessions(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    active = graph.add_node(
        label="Active Session Decision",
        content="Shared context note.",
        node_type=NodeType.DECISION,
        project="alpha",
        session_id="sess-active",
        tags=["alpha"],
    ).node
    quiet = graph.add_node(
        label="Quiet Session Decision",
        content="Shared context note.",
        node_type=NodeType.DECISION,
        project="alpha",
        session_id="sess-quiet",
        tags=["alpha"],
    ).node
    _set_node_timestamp(graph, active.id, timestamp)
    _set_node_timestamp(graph, quiet.id, timestamp)

    _insert_transcript_record(
        graph,
        session_id="sess-active",
        project="alpha",
        transcript_text="We are actively working on the rollout plan today.",
        observed_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )

    result = graph.prime_context(project="alpha", max_nodes=2)

    assert result.nodes
    assert result.nodes[0].label == "Active Session Decision"


def test_prime_context_ignores_non_embeddable_seed_nodes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    seed = graph.add_node(
        label="Non-Embeddable Seed",
        content="This node should be ignored during prime context expansion.",
        node_type=NodeType.DECISION,
        project="alpha",
        session_id="sess-seed",
    ).node
    anchor = graph.add_node(
        label="Embeddable Anchor",
        content="This node should still allow prime context to succeed.",
        node_type=NodeType.DECISION,
        project="alpha",
        session_id="sess-anchor",
    ).node
    _set_node_timestamp(graph, seed.id, timestamp)
    _set_node_timestamp(graph, anchor.id, timestamp)
    _set_node_embedding_null(graph, seed.id)

    result = graph.prime_context(project="alpha", max_nodes=5)

    assert "Prime context" in result.summary or result.summary
    assert all(node.id != seed.id for node in result.nodes)


def test_export_context_bundle_fusion_includes_replay_hits(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.observe_conversation(
        user_message="We switched production to PostgreSQL last Friday.",
        assistant_response="I will remember the PostgreSQL migration.",
        session_id="sess-db",
    )

    exported = graph.export_context_bundle(
        mode="query",
        query="latest production database",
        retrieval_mode="fusion",
        format="both",
        output_path=tmp_path / "fusion-context",
    )
    markdown = Path(exported.markdown_path).read_text(encoding="utf-8")
    payload = json.loads(Path(exported.json_path).read_text(encoding="utf-8"))

    assert exported.retrieval_mode == "fusion"
    assert "## Replay Evidence" in markdown
    assert payload["replay_hits"]


def test_markdown_vault_export_and_import_round_trip(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    decision = graph.add_node(
        label="Use PostgreSQL",
        content="We decided to use PostgreSQL for production.",
        node_type=NodeType.DECISION,
        project="alpha",
    ).node
    reason = graph.add_node(
        label="Need ACID",
        content="ACID compliance matters.",
        node_type=NodeType.FACT,
        project="alpha",
    ).node
    graph.add_edge(
        source_id=decision.id,
        target_id=reason.id,
        relationship=RelationType.DEPENDS_ON,
    )

    exported = graph.export_markdown_vault(root_path=tmp_path / "vault")
    assert exported.files_written

    decision_file_rel = next(path for path in exported.files_written if decision.id in path)
    decision_file = Path(exported.root_path) / decision_file_rel
    updated_text = decision_file.read_text(encoding="utf-8").replace(
        "We decided to use PostgreSQL for production.",
        "We decided to use PostgreSQL 16 for production.",
    )
    updated_text = updated_text.replace(
        "## Relations\n- [[depends_on::Need ACID]] <!-- node_id:"
        f"{reason.id} -->",
        "## Relations\n"
        f"- [[depends_on::Need ACID]] <!-- node_id:{reason.id} -->\n"
        "- [[relates_to::Operational Runbook]]",
    )
    decision_file.write_text(updated_text, encoding="utf-8")

    imported = graph.import_markdown_vault(root_path=tmp_path / "vault")
    updated = graph.get_node(decision.id)
    replay = graph.get_related(node_id=decision.id, max_depth=1)

    assert imported.nodes_updated >= 1
    assert imported.stub_nodes_created == 1
    assert updated.content == "We decided to use PostgreSQL 16 for production."
    assert any(node.label == "Operational Runbook" for node in replay.nodes)
    assert any(edge.relationship == "relates_to" for edge in replay.edges)


def test_markdown_vault_import_explicit_relation_deletion(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    decision = graph.add_node(
        label="Use PostgreSQL",
        content="We decided to use PostgreSQL for production.",
        node_type=NodeType.DECISION,
    ).node
    reason = graph.add_node(
        label="Need ACID",
        content="ACID compliance matters.",
        node_type=NodeType.FACT,
    ).node
    graph.add_edge(source_id=decision.id, target_id=reason.id, relationship="depends_on")

    exported = graph.export_markdown_vault(root_path=tmp_path / "vault-delete")
    decision_file_rel = next(path for path in exported.files_written if decision.id in path)
    decision_file = Path(exported.root_path) / decision_file_rel
    updated_text = decision_file.read_text(encoding="utf-8").replace(
        f"- [[depends_on::Need ACID]] <!-- node_id:{reason.id} -->",
        f"- ~~[[depends_on::Need ACID]]~~ <!-- node_id:{reason.id} -->",
    )
    decision_file.write_text(updated_text, encoding="utf-8")

    imported = graph.import_markdown_vault(root_path=tmp_path / "vault-delete")
    related = graph.get_related(node_id=decision.id, max_depth=1)

    assert imported.edges_deleted == 1
    assert not any(edge.relationship == "depends_on" and edge.target_id == reason.id for edge in related.edges)


def test_conflict_detection_creates_contradiction_edge(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    first = graph.add_node(
        label="REST Preference",
        content="User prefers REST APIs for backend services",
        node_type=NodeType.PREFERENCE,
    )
    second = graph.add_node(
        label="GraphQL Preference",
        content="User prefers GraphQL APIs for backend services",
        node_type=NodeType.PREFERENCE,
    )

    assert second.created is True
    assert second.conflicts
    related = graph.get_related(node_id=second.node.id, max_depth=1)
    assert any(edge.relationship == RelationType.CONTRADICTS for edge in related.edges)
    assert first.node.id in {node.id for node in related.nodes}


def test_conflict_detection_skips_meta_policy_example_nodes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    first = graph.add_node(
        label="Hedged statements policy",
        content='Hedged or conditional turns such as "I think we should probably go with Redis" are stored as note nodes instead of hard decisions.',
        node_type=NodeType.DECISION,
        tags=["extraction", "policy"],
    )
    second = graph.add_node(
        label="Negated tool choice policy",
        content='Negated statements such as "We are not using MongoDB anymore" are stored as decision nodes with negated tags so the graph preserves polarity.',
        node_type=NodeType.DECISION,
        tags=["extraction", "policy"],
    )

    assert second.conflicts == []
    related = graph.get_related(node_id=second.node.id, max_depth=1)
    assert not any(edge.relationship == RelationType.CONTRADICTS for edge in related.edges)


def test_load_graph_preserves_edge_attributes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    a = graph.add_node(label="A", content="A", node_type=NodeType.FACT).node
    b = graph.add_node(label="B", content="B", node_type=NodeType.FACT).node

    graph.add_edge(
        source_id=a.id,
        target_id=b.id,
        relationship=RelationType.CONTRADICTS,
        weight=0.9,
    )

    with graph._connect() as connection:
        g = graph._load_graph(connection, node_ids=[a.id, b.id])

    edge_data = g.edges[a.id, b.id]
    assert edge_data["relationship"] == "contradicts"
    assert edge_data["weight"] == 0.9
    assert edge_data["metadata"] == {}


def test_expand_node_depths_prioritizes_stronger_relations(tmp_path: Path) -> None:
    graph_store = make_graph(tmp_path)
    graph = nx.DiGraph()
    graph.add_edge("seed", "conflict", relationship="contradicts", weight=1.0)
    graph.add_edge("seed", "support", relationship="depends_on", weight=1.0)
    graph.add_edge("seed", "weak", relationship="similar_to", weight=1.0)

    ordered = graph_store._expand_node_depths(graph, ["seed"], max_depth=1)

    keys = list(ordered.keys())

    # seed must always be first
    assert keys[0] == "seed"

    # Now check PRIORITY ORDER
    conflict_idx = keys.index("conflict")
    support_idx = keys.index("support")
    weak_idx = keys.index("weak")

    assert conflict_idx < support_idx < weak_idx


def test_expand_node_depths_prunes_weak_paths(tmp_path: Path) -> None:
    graph_store = make_graph(tmp_path)
    graph = nx.DiGraph()
    graph.add_edge("seed", "a", relationship="depends_on", weight=1.0)
    graph.add_edge("a", "b", relationship="similar_to", weight=1.0)

    ordered = graph_store._expand_node_depths(
        graph,
        ["seed"],
        max_depth=2,
        min_priority=0.25,
    )

    assert ordered["seed"] == 0
    assert ordered["a"] == 1
    assert "b" not in ordered


def test_observe_conversation_extracts_nodes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    result = graph.observe_conversation(
        user_message="I prefer Python for backend work. Can we use FastAPI?",
        assistant_response="Let's use FastAPI and store the API in src/server.py.",
    )

    assert result.created_count >= 3
    labels = {node.label for node in result.stored_nodes}
    assert "I prefer Python for backend work" in labels
    assert "Can we use FastAPI?" in labels
    assert "src/server.py" in labels
    assert all(node.evidence_records for node in result.stored_nodes)
    assert all(node.valid_from is not None for node in result.stored_nodes)


def test_observe_conversation_extracts_favorite_preference_statement(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    result = graph.observe_conversation(
        user_message="Remember that my favorite programming language is Python.",
        assistant_response="I'll remember that your favorite programming language is Python.",
    )

    preference_nodes = [node for node in result.stored_nodes if node.node_type == NodeType.PREFERENCE]

    assert preference_nodes
    assert any("favorite programming language is Python" in node.content for node in preference_nodes)
    assert any("speaker:user" in node.tags for node in preference_nodes)


def test_observe_conversation_extracts_common_preference_and_decision_phrasings(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    cases = [
        (
            "I always use PostgreSQL for this kind of project.",
            "Understood. I'll treat PostgreSQL as the default database preference.",
            NodeType.PREFERENCE,
        ),
        (
            "PostgreSQL is my go-to database.",
            "Got it. I'll remember PostgreSQL as your default database choice.",
            NodeType.PREFERENCE,
        ),
        (
            "I've switched to FastAPI for backend services.",
            "Okay. I'll remember that FastAPI is the current backend decision.",
            NodeType.DECISION,
        ),
        (
            "We should stick with Next.js for the frontend.",
            "Agreed. I'll treat Next.js as the frontend decision.",
            NodeType.DECISION,
        ),
    ]

    for user_message, assistant_response, expected_type in cases:
        result = graph.observe_conversation(
            user_message=user_message,
            assistant_response=assistant_response,
        )
        assert any(node.node_type == expected_type for node in result.stored_nodes)


def test_duplicate_nodes_accumulate_evidence_records(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    first = graph.observe_conversation(
        user_message="We chose PostgreSQL for production.",
        assistant_response="Understood.",
    )
    second = graph.observe_conversation(
        user_message="Reminder: we chose PostgreSQL for production.",
        assistant_response="Got it.",
    )

    node = next(item for item in second.stored_nodes if item.label == "Database decision")

    assert second.reused_count >= 1
    assert len(node.evidence_records) >= 2


def test_get_node_history_returns_evidence_and_related_nodes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    decision = graph.add_node(
        label="Use PostgreSQL",
        content="We chose PostgreSQL.",
        node_type=NodeType.DECISION,
    ).node
    fact = graph.add_node(
        label="Need ACID",
        content="ACID compliance matters.",
        node_type=NodeType.FACT,
    ).node
    graph.add_edge(
        source_id=decision.id,
        target_id=fact.id,
        relationship=RelationType.DEPENDS_ON,
    )

    observed = graph.observe_conversation(
        user_message="We chose PostgreSQL.",
        assistant_response="I will remember that decision.",
    )
    assert observed.stored_nodes
    history = graph.get_node_history(node_id=decision.id, max_depth=1)

    assert history.node.evidence_records
    assert any(node.id != history.node.id for node in history.related_nodes)
    assert any(edge.relationship == RelationType.DEPENDS_ON for edge in history.edges)


def test_timeline_includes_evidence_and_edges(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    decision = graph.add_node(
        label="Use PostgreSQL",
        content="We chose PostgreSQL.",
        node_type=NodeType.DECISION,
    ).node
    fact = graph.add_node(
        label="Need ACID",
        content="ACID compliance matters.",
        node_type=NodeType.FACT,
    ).node
    graph.add_edge(
        source_id=decision.id,
        target_id=fact.id,
        relationship=RelationType.DEPENDS_ON,
    )
    graph.observe_conversation(
        user_message="We chose PostgreSQL.",
        assistant_response="I will remember that decision.",
    )

    timeline = graph.timeline(node_id=decision.id, max_depth=1, include_evidence=True, limit=10)
    kinds = {item.kind for item in timeline.items}

    assert timeline.scope == f"node:{decision.id}"
    assert "node_created" in kinds
    assert "evidence" in kinds
    assert "edge_depends_on" in kinds


def test_list_conflicts_and_resolve_conflict(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="REST Preference",
        content="User prefers REST APIs for backend services",
        node_type=NodeType.PREFERENCE,
    )
    graph.add_node(
        label="GraphQL Preference",
        content="User prefers GraphQL APIs for backend services",
        node_type=NodeType.PREFERENCE,
    )

    conflicts = graph.list_conflicts()

    assert len(conflicts.conflicts) == 1
    assert conflicts.conflicts[0].resolved is False

    resolved = graph.resolve_conflict(
        edge_id=conflicts.conflicts[0].edge.id,
        resolution_note="Superseded by the newer API decision.",
    )

    assert resolved.resolved is True
    assert resolved.resolution_note == "Superseded by the newer API decision."
    assert graph.list_conflicts().conflicts == []

    resolved_conflicts = graph.list_conflicts(include_resolved=True)
    assert len(resolved_conflicts.conflicts) == 1
    assert resolved_conflicts.conflicts[0].resolved is True


def test_query_can_filter_by_explicit_scopes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="DB choice alpha",
        content="We chose PostgreSQL for production.",
        node_type=NodeType.DECISION,
        project="alpha",
        agent_id="codex",
        session_id="sess-alpha",
    )
    graph.add_node(
        label="DB choice beta",
        content="We chose MySQL for production.",
        node_type=NodeType.DECISION,
        project="beta",
        agent_id="claude",
        session_id="sess-beta",
    )

    alpha = graph.query(query="production database", project="alpha", agent_id="codex")
    beta = graph.query(query="production database", session_id="sess-beta")
    new_session = graph.query(
        query="production database",
        project="alpha",
        agent_id="codex",
        session_id="sess-new",
    )
    missing_session_only = graph.query(query="production database", session_id="sess-new")
    scopes = graph.list_context_scopes()

    assert [node.label for node in alpha.nodes] == ["DB choice alpha"]
    assert [node.label for node in beta.nodes] == ["DB choice beta"]
    assert [node.label for node in new_session.nodes] == ["DB choice alpha"]
    assert missing_session_only.nodes == []
    assert scopes.agent_ids == ["claude", "codex"]
    assert scopes.projects == ["alpha", "beta"]
    assert scopes.session_ids == ["sess-alpha", "sess-beta"]


def test_observe_conversation_extracts_clean_database_and_auth_facts(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    result = graph.observe_conversation(
        user_message=(
            "We chose PostgreSQL over MySQL because MySQL replication has been painful. "
            "We are using FastAPI for the backend. JWT tokens expire in 15 minutes."
        ),
        assistant_response=(
            "Understood. I'll remember that PostgreSQL was chosen, the reason was MySQL replication pain, "
            "FastAPI is the backend, and JWT expiry is 15 minutes."
        ),
    )

    labels = {node.label for node in result.stored_nodes}
    assert "Database decision" in labels
    assert "Backend framework" in labels
    assert "JWT expiry" in labels
    assert "MySQL replication has been painful" in labels
    assert "I'll remember that PostgreSQL was chosen," not in labels
    assert "FastAPI" not in labels


def test_observe_conversation_splits_multi_clause_turns_into_multiple_nodes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    result = graph.observe_conversation(
        user_message="I switched from VS Code to Neovim, and I'm using tmux now too.",
        assistant_response="Understood. I'll remember both tool choices.",
    )

    decision_contents = {node.content for node in result.stored_nodes if node.node_type == NodeType.DECISION}

    assert "I switched from VS Code to Neovim." in decision_contents
    assert "I'm using tmux now too." in decision_contents


def test_observe_conversation_extracts_causal_fact_and_decision_with_dependency_edge(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    result = graph.observe_conversation(
        user_message="The deadline moved to March, so we dropped the GraphQL migration.",
        assistant_response="Understood. I'll remember the reason and the decision.",
    )

    fact_node = next(node for node in result.stored_nodes if node.node_type == NodeType.FACT)
    decision_node = next(node for node in result.stored_nodes if node.node_type == NodeType.DECISION)

    assert fact_node.content == "The deadline moved to March."
    assert decision_node.content == "we dropped the GraphQL migration."
    related = graph.get_related(node_id=decision_node.id, max_depth=1)
    assert any(
        edge.relationship == RelationType.DEPENDS_ON and edge.target_id == fact_node.id
        for edge in related.edges
    )


def test_observe_conversation_stores_hedged_and_conditional_turns_as_notes(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    hedged = graph.observe_conversation(
        user_message="I think we should probably go with Redis.",
        assistant_response="Understood.",
    )
    conditional = graph.observe_conversation(
        user_message="Unless the team objects, let's use Terraform.",
        assistant_response="Understood.",
    )
    revisit = graph.observe_conversation(
        user_message="We might need to revisit this if latency gets worse.",
        assistant_response="Understood.",
    )

    hedged_node = hedged.stored_nodes[0]
    conditional_node = conditional.stored_nodes[0]
    revisit_node = revisit.stored_nodes[0]

    assert hedged_node.node_type == NodeType.NOTE
    assert "hedged" in hedged_node.tags
    assert conditional_node.node_type == NodeType.NOTE
    assert "conditional" in conditional_node.tags
    assert revisit_node.node_type == NodeType.NOTE


def test_observe_conversation_preserves_negated_tool_choices(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)

    result = graph.observe_conversation(
        user_message="We're not using MongoDB anymore.",
        assistant_response="Understood. I'll remember that.",
    )

    decision_node = next(node for node in result.stored_nodes if node.node_type == NodeType.DECISION)

    assert decision_node.content == "We're not using MongoDB anymore."
    assert "negated" in decision_node.tags
    assert "choice:mongodb" in decision_node.tags


def test_observe_conversation_creates_database_contradiction_edges(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.observe_conversation(
        user_message="We chose PostgreSQL over MySQL because MySQL replication has been painful.",
        assistant_response="Understood.",
    )

    result = graph.observe_conversation(
        user_message="The team is more familiar with MySQL, so we may switch to MySQL.",
        assistant_response="Understood. I'll note that.",
    )

    assert result.conflicts
    decision_node = next(node for node in result.stored_nodes if node.label == "Database decision")
    related = graph.get_related(node_id=decision_node.id, max_depth=1)
    assert any(edge.relationship == RelationType.CONTRADICTS for edge in related.edges)


def test_query_supports_temporal_latest_and_oldest_bias(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Auth v1",
        content="Auth architecture originally used JWT sessions",
        node_type=NodeType.CONCEPT,
    )
    graph.add_node(
        label="Auth v2",
        content="Auth architecture now uses rotating JWT tokens",
        node_type=NodeType.CONCEPT,
    )

    latest = graph.query(query="latest auth architecture", max_nodes=1, max_depth=0)
    originally = graph.query(query="originally auth architecture", max_nodes=1, max_depth=0)

    assert latest.nodes[0].label == "Auth v2"
    assert originally.nodes[0].label == "Auth v1"


def test_temporal_latest_is_gated_to_query_topic(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Auth rejected",
        content="Auth request rejected by admin",
        node_type=NodeType.FACT,
        tags=["security-review"],
    )
    graph.add_node(
        label="Auth expired",
        content="Auth token expired at 10am",
        node_type=NodeType.FACT,
        tags=["security-review"],
    )
    graph.add_node(
        label="Privacy export",
        content="Privacy export completed",
        node_type=NodeType.FACT,
        tags=["privacy-export"],
    )
    graph.add_node(
        label="Model staging",
        content="Model deployed to staging",
        node_type=NodeType.FACT,
        tags=["model-ops"],
    )

    result = graph.query(query="latest auth token", max_nodes=1, max_depth=0)

    assert result.nodes[0].label == "Auth expired"


def test_negation_query_prefers_rejected_node(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Model canary required",
        content="Model releases require offline evaluation plus a canary window.",
        node_type=NodeType.DECISION,
    )
    graph.add_node(
        label="PM approval gate",
        content="Full model rollout now requires product-manager approval after the canary.",
        node_type=NodeType.DECISION,
    )
    graph.add_node(
        label="No model auto-promotion",
        content="Evaluation winners must not be auto-promoted to production.",
        node_type=NodeType.DECISION,
    )

    result = graph.query(
        query="which model deployment shortcut remains forbidden even when evaluation looks good",
        max_nodes=1,
        max_depth=0,
    )

    assert result.nodes[0].label == "No model auto-promotion"


def test_implicit_reference_security_review_emergency_access_prefers_break_glass(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="RBAC only",
        content="Access control started as role-based access only.",
        node_type=NodeType.DECISION,
        tags=["scenario:access_control"],
    )
    graph.add_node(
        label="Audited break-glass access",
        content="Break-glass access now uses per-user accounts with audit trails.",
        node_type=NodeType.DECISION,
        tags=["scenario:security_review_actions"],
    )

    result = graph.query(
        query="what was the final answer to that security review item about emergency access",
        max_nodes=1,
        max_depth=0,
    )

    assert result.nodes[0].label == "Audited break-glass access"


def test_implicit_reference_pm_gate_prefers_no_auto_promote(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Canary required",
        content="Model releases require offline evaluation plus a canary window.",
        node_type=NodeType.DECISION,
        tags=["scenario:model_ops_rollout"],
    )
    graph.add_node(
        label="PM approval gate",
        content="Full model rollout now requires product-manager approval after the canary.",
        node_type=NodeType.DECISION,
        tags=["scenario:model_ops_rollout"],
    )
    graph.add_node(
        label="No model auto-promotion",
        content="Evaluation winners must not be auto-promoted to production.",
        node_type=NodeType.DECISION,
        tags=["scenario:model_ops_rollout"],
    )

    result = graph.query(
        query="what rejected model rollout behavior came before the PM gate",
        max_nodes=1,
        max_depth=0,
    )

    assert result.nodes[0].label == "No model auto-promotion"


def test_temporal_current_phrase_prefers_latest_state(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="JWT expiry 15 minutes",
        content="JWT tokens previously expired after 15 minutes.",
        node_type=NodeType.FACT,
    )
    graph.add_node(
        label="JWT expiry 1 hour",
        content="JWT tokens now expire after 1 hour.",
        node_type=NodeType.FACT,
    )

    result = graph.query(query="what is the current jwt expiry", max_nodes=1, max_depth=0)

    assert result.nodes[0].label == "JWT expiry 1 hour"


def test_temporal_latest_database_choice_prefers_database_fact(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="PostgreSQL production",
        content="PostgreSQL is the production database for parity and safer migrations.",
        node_type=NodeType.DECISION,
    )
    graph.add_node(
        label="PostgreSQL updated choice",
        content="Updated to PostgreSQL for production deployment and concurrent write support.",
        node_type=NodeType.DECISION,
    )

    result = graph.query(query="what is the latest production database choice", max_nodes=1, max_depth=0)

    assert result.nodes[0].label == "PostgreSQL production"


def test_temporal_original_phrase_prefers_oldest_state(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="CSV only export",
        content="CSV was the only export format initially.",
        node_type=NodeType.FACT,
    )
    graph.add_node(
        label="CSV and Parquet export",
        content="Exports now support both CSV and Parquet for data warehouse sync.",
        node_type=NodeType.FACT,
    )

    result = graph.query(query="what was the original export format", max_nodes=1, max_depth=0)

    assert result.nodes[0].label == "CSV only export"


def test_now_phrase_prefers_current_backend_choice(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Redis session cache",
        content="Redis handles session caching because TTL support is simple.",
        node_type=NodeType.FACT,
    )
    graph.add_node(
        label="KeyDB session cache",
        content="KeyDB now handles session caching for active-active failover.",
        node_type=NodeType.FACT,
    )

    result = graph.query(query="which cache backend handles sessions now", max_nodes=1, max_depth=0)

    assert result.nodes[0].label == "KeyDB session cache"


def test_temporal_latest_privacy_export_policy_prefers_approval_fact(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="CSV and Parquet export",
        content="Exports now support both CSV and Parquet for data warehouse sync.",
        node_type=NodeType.FACT,
    )
    graph.add_node(
        label="Enterprise export approval",
        content="Enterprise data exports now require admin approval and signed download links.",
        node_type=NodeType.FACT,
    )

    result = graph.query(query="what is the latest enterprise data export policy", max_nodes=1, max_depth=0)

    assert result.nodes[0].label == "Enterprise export approval"


def test_graph_diff_and_prime_context(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Project Alpha",
        content="Project Alpha uses FastAPI",
        node_type=NodeType.ENTITY,
        tags=["alpha"],
    )
    note = graph.add_node(
        label="Alpha Decision",
        content="We decided to use SQLite for Alpha",
        node_type=NodeType.DECISION,
        tags=["alpha"],
    ).node
    graph.update_node(node_id=note.id, tags=["alpha", "updated"])

    diff = graph.graph_diff(since="24h")
    prime = graph.prime_context(project="alpha")
    new_session_prime = graph.prime_context(project="alpha", session_id="fresh-session")

    assert diff.added_nodes
    assert prime.nodes
    assert new_session_prime.nodes
    assert any("alpha" in node.tags for node in prime.nodes)
    assert any("alpha" in node.tags for node in new_session_prime.nodes)


def test_get_topics_returns_clusters(tmp_path: Path) -> None:
    graph = make_graph(tmp_path)
    graph.add_node(
        label="Auth REST",
        content="User prefers REST APIs for auth",
        node_type=NodeType.PREFERENCE,
        tags=["auth", "api"],
    )
    graph.add_node(
        label="Auth JWT",
        content="Project uses JWT authentication",
        node_type=NodeType.CONCEPT,
        tags=["auth"],
    )
    graph.add_node(
        label="Database Neo4j",
        content="Project uses Neo4j for memory storage",
        node_type=NodeType.ENTITY,
        tags=["database"],
    )

    topics = graph.get_topics()

    assert topics.total_clusters >= 1
    assert topics.clusters[0].nodes
