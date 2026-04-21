from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import waggle
from waggle.graph import MemoryGraph
from waggle.models import NodeType, RelationType
from waggle.config import AppConfig
from waggle.server import (
    WaggleServer,
    _assert_runtime_feature_parity,
    _build_parser,
    _default_graph,
    _run_admin_command,
    _write_codex,
    _write_other,
)


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


def make_app(tmp_path: Path) -> WaggleServer:
    graph = MemoryGraph(tmp_path / "server-memory.db", FakeEmbeddingModel())
    config = AppConfig(
        backend="sqlite",
        transport="stdio",
        model_name="fake-model",
        db_path=str(tmp_path / "server-memory.db"),
        default_tenant_id="local-default",
        http_host="127.0.0.1",
        http_port=8080,
        log_level="INFO",
        rate_limit_rpm=120,
        write_rate_limit_rpm=60,
        max_concurrent_requests=8,
        max_payload_bytes=1024 * 1024,
        request_timeout_seconds=30,
        export_dir=None,
        neo4j_uri="",
        neo4j_username="",
        neo4j_password="",
        neo4j_database="",
    )
    return WaggleServer(graph=graph, config=config)


def test_store_node_and_stats_tool(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    result = app.handle_tool_call(
        "store_node",
        {
            "label": "User Preference",
            "content": "User prefers Python for backend development",
            "node_type": NodeType.PREFERENCE.value,
            "tags": ["python"],
        },
    )
    assert "Stored node" in result.content[0].text

    stats_result = app.handle_tool_call("get_stats", {})
    assert "Memory Graph Stats" in stats_result.content[0].text
    assert stats_result.structuredContent["total_nodes"] == 1


def test_tool_schemas_are_glama_friendly(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    for tool in app.build_tools():
        assert tool.description
        assert tool.inputSchema["type"] == "object"
        assert tool.inputSchema["additionalProperties"] is False
        assert isinstance(tool.inputSchema["properties"], dict)
        for field_name, field_schema in tool.inputSchema["properties"].items():
            assert field_schema.get("description"), f"{tool.name}.{field_name} is missing a description"


def test_memory_policy_prompt_and_resource(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    prompts = app.build_prompts()
    assert [prompt.name for prompt in prompts] == ["waggle_memory_policy"]

    prompt_result = app.get_prompt_result(
        "waggle_memory_policy",
        {"project": "MCP", "agent_id": "codex", "session_id": "thread-1"},
    )
    prompt_text = prompt_result.messages[0].content.text

    assert "The user should not manually manage memory" in prompt_text
    assert "Use query_graph before answering" in prompt_text
    assert "Use observe_conversation after completed turns" in prompt_text
    assert "project: MCP" in prompt_text

    resource_text = app.read_resource_text("graph://memory-policy")
    assert "Waggle automatic memory policy" in resource_text
    assert "Do not call store_node for normal conversation memory" in resource_text


def test_memory_tools_describe_automatic_usage(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    tools = {tool.name: tool for tool in app.build_tools()}

    assert "Automatically search the memory graph before answering" in tools["query_graph"].description
    assert "Do not ask the user to trigger this" in tools["observe_conversation"].description
    assert "Automatically build a compact context brief" in tools["prime_context"].description


def test_export_graph_html_tool(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.handle_tool_call(
        "store_node",
        {
            "label": "Visual Node",
            "content": "The graph should be exportable as HTML.",
            "node_type": NodeType.CONCEPT.value,
        },
    )

    result = app.handle_tool_call(
        "export_graph_html",
        {
            "output_path": str(tmp_path / "visualization.html"),
            "include_physics": False,
        },
    )

    assert result.isError is False
    assert result.structuredContent["total_nodes"] == 1
    assert Path(result.structuredContent["output_path"]).exists()
    assert "Exported graph visualization" in result.content[0].text


def test_decompose_and_store_tool_persists_subgraph(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    result = app.handle_tool_call(
        "decompose_and_store",
        {
            "content": "- User prefers Python\n- Project uses FastAPI",
            "context": "Backend memory",
        },
    )

    assert result.isError is False
    assert result.structuredContent["total_nodes_in_graph"] >= 3
    assert len(result.structuredContent["edges"]) >= 2
    assert "Memory Graph Results" in result.content[0].text


def test_export_and_import_backup_tools(tmp_path: Path) -> None:
    source = make_app(tmp_path / "source")
    target = make_app(tmp_path / "target")
    source.handle_tool_call(
        "store_node",
        {
            "label": "Backup Tool Node",
            "content": "This node should appear after import.",
            "node_type": NodeType.NOTE.value,
        },
    )

    backup = source.handle_tool_call(
        "export_graph_backup",
        {"output_path": str(tmp_path / "graph-backup.json")},
    )
    imported = target.handle_tool_call(
        "import_graph_backup",
        {"input_path": backup.structuredContent["output_path"]},
    )

    assert backup.isError is False
    assert Path(backup.structuredContent["output_path"]).exists()
    assert imported.isError is False
    assert imported.structuredContent["nodes_created"] == 1
    assert target.graph.get_stats().total_nodes == 1


def test_export_context_bundle_tool(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    decision = app.graph.add_node(
        label="Ship portable export",
        content="We decided to ship portable context export first.",
        node_type=NodeType.DECISION,
    ).node
    reason = app.graph.add_node(
        label="Rate limits block handoff",
        content="Rate limits force users to move context across AIs.",
        node_type=NodeType.FACT,
    ).node
    app.graph.add_edge(
        source_id=decision.id,
        target_id=reason.id,
        relationship=RelationType.DEPENDS_ON,
    )

    result = app.handle_tool_call(
        "export_context_bundle",
        {
            "mode": "query",
            "query": "why did we ship portable export",
            "format": "both",
            "output_path": str(tmp_path / "handoff"),
        },
    )

    assert result.isError is False
    assert result.structuredContent["mode"] == "query"
    assert result.structuredContent["node_count"] >= 1
    assert Path(result.structuredContent["markdown_path"]).exists()
    assert Path(result.structuredContent["json_path"]).exists()
    assert result.structuredContent["render_hints"]["token_estimate"] > 0
    assert "Context Bundle Export" in result.content[0].text


def test_get_node_history_tool(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    observed = app.handle_tool_call(
        "observe_conversation",
        {
            "user_message": "We chose PostgreSQL over MySQL because ACID matters.",
            "assistant_response": "I'll remember that decision.",
        },
    )
    decision = next(
        node for node in observed.structuredContent["stored_nodes"]
        if node["label"] == "Database decision"
    )

    result = app.handle_tool_call("get_node_history", {"node_id": decision["id"], "max_depth": 1})

    assert result.isError is False
    assert result.structuredContent["node"]["evidence_records"]
    assert "Node History" in result.content[0].text


def test_timeline_tool(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    decision = app.graph.add_node(
        label="Use PostgreSQL",
        content="We chose PostgreSQL.",
        node_type=NodeType.DECISION,
    ).node
    app.graph.observe_conversation(
        user_message="We chose PostgreSQL.",
        assistant_response="I'll remember that decision.",
    )

    result = app.handle_tool_call(
        "timeline",
        {"node_id": decision.id, "limit": 10, "include_evidence": True},
    )

    assert result.isError is False
    assert result.structuredContent["scope"] == f"node:{decision.id}"
    assert any(item["kind"] == "evidence" for item in result.structuredContent["items"])
    assert "Timeline" in result.content[0].text


def test_list_conflicts_and_resolve_conflict_tools(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.handle_tool_call(
        "store_node",
        {
            "label": "REST Preference",
            "content": "User prefers REST APIs for backend services",
            "node_type": NodeType.PREFERENCE.value,
        },
    )
    app.handle_tool_call(
        "store_node",
        {
            "label": "GraphQL Preference",
            "content": "User prefers GraphQL APIs for backend services",
            "node_type": NodeType.PREFERENCE.value,
        },
    )

    listed = app.handle_tool_call("list_conflicts", {})
    edge_id = listed.structuredContent["conflicts"][0]["edge"]["id"]
    resolved = app.handle_tool_call(
        "resolve_conflict",
        {"edge_id": edge_id, "resolution_note": "Superseded by the newer API decision."},
    )
    unresolved_after = app.handle_tool_call("list_conflicts", {})
    resolved_after = app.handle_tool_call("list_conflicts", {"include_resolved": True})

    assert listed.isError is False
    assert len(listed.structuredContent["conflicts"]) == 1
    assert resolved.isError is False
    assert resolved.structuredContent["resolved"] is True
    assert unresolved_after.structuredContent["conflicts"] == []
    assert resolved_after.structuredContent["conflicts"][0]["resolved"] is True


def test_list_context_scopes_tool(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.handle_tool_call(
        "store_node",
        {
            "label": "Scoped node",
            "content": "This belongs to the alpha workspace.",
            "node_type": NodeType.NOTE.value,
            "agent_id": "codex",
            "project": "alpha",
            "session_id": "sess-1",
        },
    )

    result = app.handle_tool_call("list_context_scopes", {})

    assert result.isError is False
    assert result.structuredContent["agent_ids"] == ["codex"]
    assert result.structuredContent["projects"] == ["alpha"]
    assert result.structuredContent["session_ids"] == ["sess-1"]


def test_export_context_bundle_cli_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    app = make_app(tmp_path)
    app.graph.add_node(
        label="CLI Export Decision",
        content="Use the CLI export command for handoff workflows.",
        node_type=NodeType.DECISION,
    )

    args = SimpleNamespace(
        command="export-context-bundle",
        mode="graph",
        query="",
        project="",
        max_nodes=25,
        max_depth=2,
        format="both",
        output_path=str(tmp_path / "cli-handoff"),
        include_edges=True,
        include_timestamps=True,
        include_source_prompt=False,
        audience="llm",
    )
    exit_code = _run_admin_command(app.config, args)
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert exit_code == 0
    assert payload["mode"] == "graph"
    assert Path(payload["markdown_path"]).exists()
    assert Path(payload["json_path"]).exists()


def test_markdown_vault_tool_and_cli_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    app = make_app(tmp_path)
    app.graph.add_node(
        label="Vault Decision",
        content="Export this node to a markdown vault.",
        node_type=NodeType.DECISION,
        project="alpha",
    )

    tool_result = app.handle_tool_call(
        "export_markdown_vault",
        {"root_path": str(tmp_path / "vault"), "project": "alpha"},
    )
    assert tool_result.isError is False
    assert tool_result.structuredContent["files_written"]

    args = SimpleNamespace(command="export-markdown-vault", root_path=str(tmp_path / "vault-cli"), project="", agent_id="", session_id="")
    exit_code = _run_admin_command(app.config, args)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["files_written"]


def test_runtime_feature_parity_check_passes_for_current_memory_graph() -> None:
    _assert_runtime_feature_parity()


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert f"waggle-mcp {waggle.__version__}" in captured.out


def test_store_node_reports_deduplication(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    first = app.handle_tool_call(
        "store_node",
        {
            "label": "Session Preference",
            "content": "This session prefers persistent graph memory.",
            "node_type": NodeType.PREFERENCE.value,
        },
    )
    second = app.handle_tool_call(
        "store_node",
        {
            "label": "Session Preference",
            "content": "This session prefers persistent graph memory.",
            "node_type": NodeType.PREFERENCE.value,
            "tags": ["deduped"],
        },
    )

    assert first.structuredContent["created"] is True
    assert second.structuredContent["created"] is False
    assert second.structuredContent["dedup_reason"] == "exact_content"
    assert "Reused existing node" in second.content[0].text


def test_store_node_reports_conflicts(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.handle_tool_call(
        "store_node",
        {
            "label": "REST Preference",
            "content": "User prefers REST APIs for backend work",
            "node_type": NodeType.PREFERENCE.value,
        },
    )

    result = app.handle_tool_call(
        "store_node",
        {
            "label": "GraphQL Preference",
            "content": "User prefers GraphQL APIs for backend work",
            "node_type": NodeType.PREFERENCE.value,
        },
    )

    assert result.isError is False
    assert result.structuredContent["conflicts"]


def test_observe_conversation_tool(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    result = app.handle_tool_call(
        "observe_conversation",
        {
            "user_message": "I prefer Python for backend work.",
            "assistant_response": "Let's use FastAPI and update src/server.py.",
        },
    )

    assert result.isError is False
    assert result.structuredContent["created_count"] >= 2
    assert any(node["evidence_records"] for node in result.structuredContent["stored_nodes"])
    assert "Conversation Observation" in result.content[0].text


def test_observe_conversation_tool_reports_database_conflicts(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.handle_tool_call(
        "observe_conversation",
        {
            "user_message": "We chose PostgreSQL over MySQL because MySQL replication has been painful.",
            "assistant_response": "Understood.",
        },
    )

    result = app.handle_tool_call(
        "observe_conversation",
        {
            "user_message": "The team is more familiar with MySQL, so we may switch to MySQL.",
            "assistant_response": "Understood.",
        },
    )

    assert result.isError is False
    assert result.structuredContent["conflicts"]


def test_graph_diff_prime_context_and_topics_tools(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.handle_tool_call(
        "store_node",
        {
            "label": "Alpha Project",
            "content": "Project Alpha uses FastAPI",
            "node_type": NodeType.ENTITY.value,
            "tags": ["alpha"],
        },
    )
    diff = app.handle_tool_call("graph_diff", {"since": "24h"})
    prime = app.handle_tool_call("prime_context", {"project": "alpha"})
    topics = app.handle_tool_call("get_topics", {})

    assert diff.isError is False
    assert diff.structuredContent["added_nodes"]
    assert prime.isError is False
    assert prime.structuredContent["nodes"]
    assert topics.isError is False
    assert topics.structuredContent["total_clusters"] >= 1


def test_recent_resource_serialization(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.graph.add_node(
        label="Architecture",
        content="The project uses SQLite and NetworkX",
        node_type=NodeType.CONCEPT,
    )

    resource_text = app.read_resource_text("graph://recent")
    assert "Recent Memory Nodes" in resource_text
    assert "Architecture" in resource_text


def test_unknown_tool_raises(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    result = app.handle_tool_call("does_not_exist", {})
    assert result.isError is True
    assert result.structuredContent["error_type"] == "ValidationFailure"


def test_invalid_tool_inputs_return_structured_errors(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    empty_query = app.handle_tool_call("query_graph", {"query": ""})
    assert empty_query.isError is True
    assert "Query cannot be empty" in empty_query.content[0].text

    missing_node_edge = app.handle_tool_call(
        "store_edge",
        {
            "source_id": "missing-a",
            "target_id": "missing-b",
            "relationship": "relates_to",
        },
    )
    assert missing_node_edge.isError is True
    assert missing_node_edge.structuredContent["error_type"] == "ValueError"


def test_tool_payload_limit_is_enforced(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.config.max_payload_bytes = 8

    result = app.handle_tool_call(
        "store_node",
        {
            "label": "Too Large",
            "content": "this payload is definitely larger than eight bytes",
            "node_type": NodeType.NOTE.value,
        },
    )

    assert result.isError is True
    assert result.structuredContent["error_code"] == "payload_too_large"


def test_default_graph_uses_sqlite_backend_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WAGGLE_BACKEND", raising=False)
    monkeypatch.setenv("WAGGLE_DB_PATH", str(tmp_path / "sqlite-memory.db"))

    graph = _default_graph()

    assert isinstance(graph, MemoryGraph)
    assert graph.db_path == tmp_path / "sqlite-memory.db"


def test_default_graph_can_build_neo4j_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeNeo4jGraph:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    import waggle.neo4j_graph as neo4j_graph_module

    monkeypatch.setattr(neo4j_graph_module, "Neo4jMemoryGraph", FakeNeo4jGraph)
    monkeypatch.setenv("WAGGLE_BACKEND", "neo4j")
    monkeypatch.setenv("WAGGLE_MODEL", "fake-model")
    monkeypatch.setenv("WAGGLE_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("WAGGLE_NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("WAGGLE_NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("WAGGLE_NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("WAGGLE_NEO4J_DATABASE", "memory")

    graph = _default_graph()

    assert isinstance(graph, FakeNeo4jGraph)
    assert captured["uri"] == "bolt://localhost:7687"
    assert captured["username"] == "neo4j"
    assert captured["password"] == "secret"
    assert captured["database"] == "memory"
    assert captured["export_dir"] == str(tmp_path / "exports")


def test_default_graph_requires_neo4j_connection_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAGGLE_BACKEND", "neo4j")
    monkeypatch.delenv("WAGGLE_NEO4J_URI", raising=False)
    monkeypatch.delenv("WAGGLE_NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("WAGGLE_NEO4J_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="Neo4j backend requires"):
        _default_graph()


def test_write_other_config_no_longer_uses_pythonpath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    config_path = _write_other(str(tmp_path / "memory.db"), "/tmp/fake-python")
    contents = config_path.read_text()
    payload = json.loads(contents)

    assert "PYTHONPATH" not in contents
    assert payload["args"] == ["-m", "waggle.server"]


def test_write_codex_config_no_longer_uses_pythonpath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    config_path = _write_codex(str(tmp_path / "memory.db"), "/tmp/fake-python")
    contents = config_path.read_text()

    assert "PYTHONPATH" not in contents
    assert 'args = ["-m", "waggle.server"]' in contents
