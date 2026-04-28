from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from waggle.models import (
    AbhiChunkLoadResult,
    AbhiDiffResult,
    AbhiExportResult,
    AbhiImportResult,
    AbhiInspectResult,
    AbhiMergeResult,
    AbhiQueryResult,
    AbhiValidationResult,
)

ABHI_SPEC_VERSION = "1.0"

ABHI_NODE_TYPES: tuple[str, ...] = (
    "fact",
    "entity",
    "concept",
    "preference",
    "decision",
    "question",
    "note",
    "reason",
    "constraint",
    "goal",
)

ABHI_EDGE_TYPES: tuple[str, ...] = (
    "relates_to",
    "contradicts",
    "depends_on",
    "part_of",
    "updates",
    "derived_from",
    "similar_to",
    "caused_by",
    "blocks",
)

ABHI_CHUNK_NODE_LIMIT = 64
ABHI_CHUNK_PRELOAD_LIMIT = 2


def filter_snapshot_by_scope(
    snapshot: dict[str, Any],
    *,
    project: str = "",
    agent_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    if not any((project.strip(), agent_id.strip(), session_id.strip())):
        return deepcopy(snapshot)

    selected_nodes = []
    selected_ids: set[str] = set()
    for node in snapshot.get("nodes", []):
        if project.strip() and str(node.get("project", "")).strip() != project.strip():
            continue
        if agent_id.strip() and str(node.get("agent_id", "")).strip() != agent_id.strip():
            continue
        if session_id.strip() and str(node.get("session_id", "")).strip() != session_id.strip():
            continue
        selected_nodes.append(deepcopy(node))
        selected_ids.add(str(node["id"]))

    selected_edges = [
        deepcopy(edge)
        for edge in snapshot.get("edges", [])
        if str(edge.get("source_id")) in selected_ids and str(edge.get("target_id")) in selected_ids
    ]

    selected_window_ids = {
        str(node.get("context_window_id"))
        for node in selected_nodes
        if str(node.get("context_window_id") or "").strip()
    }
    selected_windows = [
        deepcopy(window)
        for window in snapshot.get("context_windows", [])
        if str(window.get("id")) in selected_window_ids
    ]
    selected_repo_ids = {str(window.get("repo_id")) for window in selected_windows if str(window.get("repo_id", "")).strip()}
    selected_repos = [
        deepcopy(repo)
        for repo in snapshot.get("repos", [])
        if str(repo.get("id")) in selected_repo_ids
    ]
    selected_window_edges = [
        deepcopy(edge)
        for edge in snapshot.get("context_window_edges", [])
        if str(edge.get("source_window_id")) in selected_window_ids
        and str(edge.get("target_window_id")) in selected_window_ids
    ]

    filtered = deepcopy(snapshot)
    filtered["nodes"] = selected_nodes
    filtered["edges"] = selected_edges
    filtered["repos"] = selected_repos
    filtered["context_windows"] = selected_windows
    filtered["context_window_edges"] = selected_window_edges
    ui_state = deepcopy(snapshot.get("ui", _default_ui()))
    positions = ui_state.get("positions", {})
    ui_state["positions"] = {
        node_id: value for node_id, value in positions.items() if node_id in selected_ids
    }
    selected_nodes_value = ui_state.get("selected_nodes", [])
    ui_state["selected_nodes"] = [node_id for node_id in selected_nodes_value if node_id in selected_ids]
    filtered["ui"] = ui_state
    return filtered


def build_abhi_document(snapshot: dict[str, Any]) -> dict[str, Any]:
    graph_nodes = [_snapshot_node_to_abhi_node(node) for node in snapshot.get("nodes", [])]
    graph_edges = [_snapshot_edge_to_abhi_edge(edge) for edge in snapshot.get("edges", [])]
    document = {
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
        "schema": _default_schema(),
        "constraints": _default_constraints(),
        "ai_rules": _default_ai_rules(),
        "versions": _build_versions(graph_nodes, graph_edges),
        "ui": deepcopy(snapshot.get("ui", _default_ui())),
        "external_refs": [],
        "chunks": _build_chunks(graph_nodes, graph_edges),
        "queries": _default_queries(),
        "integrity": {
            "content_hash": "",
            "node_count": len(graph_nodes),
            "edge_count": len(graph_edges),
            "last_validated": _latest_validation_timestamp(graph_nodes, graph_edges),
            "schema_version": str(snapshot.get("schema_version", 1)),
            "abhi_spec_version": ABHI_SPEC_VERSION,
        },
        "events": _default_events(),
        "waggle": {
            "tenant_id": str(snapshot.get("tenant_id", "")),
            "schema_version": int(snapshot.get("schema_version", 1)),
            "repos": deepcopy(snapshot.get("repos", [])),
            "context_windows": deepcopy(snapshot.get("context_windows", [])),
            "context_window_edges": deepcopy(snapshot.get("context_window_edges", [])),
        },
    }
    document["integrity"]["content_hash"] = compute_abhi_hash(document)
    return document


def write_abhi_document(
    snapshot: dict[str, Any],
    *,
    output_path: str | Path,
) -> AbhiExportResult:
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = build_abhi_document(snapshot)
    executed_actions = dispatch_abhi_event(document, event_name="on_export", persist=False)
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return AbhiExportResult(
        output_path=str(destination),
        tenant_id=str(snapshot.get("tenant_id", "")),
        schema_version=int(snapshot.get("schema_version", 1)),
        abhi_spec_version=ABHI_SPEC_VERSION,
        node_count=len(document["graph"]["nodes"]),
        edge_count=len(document["graph"]["edges"]),
        content_hash=document["integrity"]["content_hash"],
        executed_actions=executed_actions,
    )


def load_abhi_document(input_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))


def inspect_abhi_document(document: dict[str, Any], *, input_path: str | Path) -> AbhiInspectResult:
    nodes = list(document.get("graph", {}).get("nodes", []))
    edges = list(document.get("graph", {}).get("edges", []))
    node_types = sorted({str(node.get("type", "")).strip() for node in nodes if str(node.get("type", "")).strip()})
    edge_types = sorted({str(edge.get("type", "")).strip() for edge in edges if str(edge.get("type", "")).strip()})
    waggle_block = document.get("waggle", {}) if isinstance(document.get("waggle"), dict) else {}
    chunks = document.get("chunks", {}) if isinstance(document.get("chunks"), dict) else {}
    chunk_index = chunks.get("chunk_index", {}) if isinstance(chunks.get("chunk_index"), dict) else {}
    return AbhiInspectResult(
        input_path=str(Path(input_path).expanduser()),
        tenant_id=str(waggle_block.get("tenant_id", "")),
        schema_version=int(waggle_block.get("schema_version", 1)),
        abhi_spec_version=str(document.get("integrity", {}).get("abhi_spec_version", "")) or ABHI_SPEC_VERSION,
        node_count=len(nodes),
        edge_count=len(edges),
        node_types=node_types,
        edge_types=edge_types,
        constraint_count=len(document.get("constraints", [])),
        version_count=len(document.get("versions", [])),
        query_count=len(document.get("queries", {}).get("saved", [])) if isinstance(document.get("queries"), dict) else 0,
        event_count=len(document.get("events", {})) if isinstance(document.get("events"), dict) else 0,
        chunk_count=len(chunk_index),
        load_strategy=str(chunks.get("load_strategy", "full") or "full"),
        preload_chunks=[str(item) for item in chunks.get("preload", []) if str(item).strip()],
        content_hash=str(document.get("integrity", {}).get("content_hash", "")),
    )


def diff_abhi_documents(
    document_a: dict[str, Any],
    document_b: dict[str, Any],
    *,
    input_path_a: str | Path,
    input_path_b: str | Path,
) -> AbhiDiffResult:
    nodes_a = {str(node.get("id", "")).strip(): node for node in document_a.get("graph", {}).get("nodes", [])}
    nodes_b = {str(node.get("id", "")).strip(): node for node in document_b.get("graph", {}).get("nodes", [])}
    edges_a = {str(edge.get("id", "")).strip(): edge for edge in document_a.get("graph", {}).get("edges", [])}
    edges_b = {str(edge.get("id", "")).strip(): edge for edge in document_b.get("graph", {}).get("edges", [])}

    nodes_added = sorted(node_id for node_id in nodes_b if node_id and node_id not in nodes_a)
    nodes_removed = sorted(node_id for node_id in nodes_a if node_id and node_id not in nodes_b)
    nodes_updated = sorted(
        node_id
        for node_id in nodes_a.keys() & nodes_b.keys()
        if _canonical_graph_object(nodes_a[node_id]) != _canonical_graph_object(nodes_b[node_id])
    )
    edges_added = sorted(edge_id for edge_id in edges_b if edge_id and edge_id not in edges_a)
    edges_removed = sorted(edge_id for edge_id in edges_a if edge_id and edge_id not in edges_b)
    edges_updated = sorted(
        edge_id
        for edge_id in edges_a.keys() & edges_b.keys()
        if _canonical_graph_object(edges_a[edge_id]) != _canonical_graph_object(edges_b[edge_id])
    )

    semantic_changes: list[str] = []
    for node_id in nodes_updated:
        before = nodes_a[node_id]
        after = nodes_b[node_id]
        if normalize_text(str(before.get("content", ""))) != normalize_text(str(after.get("content", ""))):
            semantic_changes.append(
                f"Node {node_id} content changed from '{_semantic_label(before)}' to '{_semantic_label(after)}'."
            )
        elif str(before.get("type", "")).strip() != str(after.get("type", "")).strip():
            semantic_changes.append(
                f"Node {node_id} type changed from '{before.get('type', '')}' to '{after.get('type', '')}'."
            )
    for edge_id in edges_updated:
        before = edges_a[edge_id]
        after = edges_b[edge_id]
        if str(before.get("type", "")).strip() != str(after.get("type", "")).strip():
            semantic_changes.append(
                f"Edge {edge_id} relationship changed from '{before.get('type', '')}' to '{after.get('type', '')}'."
            )

    return AbhiDiffResult(
        input_path_a=str(Path(input_path_a).expanduser()),
        input_path_b=str(Path(input_path_b).expanduser()),
        abhi_spec_version_a=str(document_a.get("integrity", {}).get("abhi_spec_version", "")) or ABHI_SPEC_VERSION,
        abhi_spec_version_b=str(document_b.get("integrity", {}).get("abhi_spec_version", "")) or ABHI_SPEC_VERSION,
        nodes_added=nodes_added,
        nodes_removed=nodes_removed,
        nodes_updated=nodes_updated,
        edges_added=edges_added,
        edges_removed=edges_removed,
        edges_updated=edges_updated,
        semantic_changes=semantic_changes,
    )


def merge_abhi_documents(
    base_document: dict[str, Any],
    left_document: dict[str, Any],
    right_document: dict[str, Any],
    *,
    base_input_path: str | Path,
    left_input_path: str | Path,
    right_input_path: str | Path,
    output_path: str | Path,
    merge_strategy: str = "prefer_right",
) -> AbhiMergeResult:
    strategy = merge_strategy.strip().lower() or "prefer_right"
    if strategy not in {"prefer_right", "prefer_left"}:
        raise ValueError("merge_strategy must be one of: prefer_right, prefer_left")

    merged_nodes, node_conflicts = _merge_graph_objects(
        base_document.get("graph", {}).get("nodes", []),
        left_document.get("graph", {}).get("nodes", []),
        right_document.get("graph", {}).get("nodes", []),
        strategy=strategy,
        object_label="node",
    )
    merged_edges, edge_conflicts = _merge_graph_objects(
        base_document.get("graph", {}).get("edges", []),
        left_document.get("graph", {}).get("edges", []),
        right_document.get("graph", {}).get("edges", []),
        strategy=strategy,
        object_label="edge",
    )

    merged = deepcopy(base_document)
    merged["graph"] = {
        "nodes": merged_nodes,
        "edges": merged_edges,
    }

    merged["schema"] = _merge_prefer_side(base_document.get("schema", {}), left_document.get("schema", {}), right_document.get("schema", {}), strategy)
    merged["constraints"] = _merge_unique_list(base_document.get("constraints", []), left_document.get("constraints", []), right_document.get("constraints", []))
    merged["ai_rules"] = _merge_prefer_side(base_document.get("ai_rules", {}), left_document.get("ai_rules", {}), right_document.get("ai_rules", {}), strategy)
    merged["ui"] = _merge_prefer_side(base_document.get("ui", _default_ui()), left_document.get("ui", _default_ui()), right_document.get("ui", _default_ui()), strategy)
    merged["external_refs"] = _merge_unique_list(base_document.get("external_refs", []), left_document.get("external_refs", []), right_document.get("external_refs", []))
    merged["chunks"] = _build_chunks(merged_nodes, merged_edges)
    merged["queries"] = _merge_queries(base_document.get("queries", {}), left_document.get("queries", {}), right_document.get("queries", {}), strategy)
    merged["events"] = _merge_prefer_side(base_document.get("events", {}), left_document.get("events", {}), right_document.get("events", {}), strategy)
    merged["waggle"] = _merge_prefer_side(base_document.get("waggle", {}), left_document.get("waggle", {}), right_document.get("waggle", {}), strategy)

    merged_versions = []
    for source in (base_document, left_document, right_document):
        for version in source.get("versions", []):
            if _canonical_graph_object(version) not in {_canonical_graph_object(item) for item in merged_versions}:
                merged_versions.append(deepcopy(version))
    merged_versions.append(
        {
            "id": f"merge-{len(merged_versions) + 1}",
            "parent": str((right_document.get("versions", [{}])[-1] if right_document.get("versions") else {}).get("id", "")) or None,
            "ts": _latest_validation_timestamp(merged["graph"]["nodes"], merged["graph"]["edges"]),
            "author": "waggle-abhi-merge",
            "changes": [],
            "message": f"Three-way merge completed with strategy '{strategy}'",
        }
    )
    merged["versions"] = merged_versions
    merged["integrity"] = deepcopy(merged.get("integrity", {}))
    merged["integrity"]["node_count"] = len(merged["graph"]["nodes"])
    merged["integrity"]["edge_count"] = len(merged["graph"]["edges"])
    merged["integrity"]["last_validated"] = _latest_validation_timestamp(merged["graph"]["nodes"], merged["graph"]["edges"])
    merged["integrity"]["abhi_spec_version"] = ABHI_SPEC_VERSION
    merged["integrity"]["content_hash"] = compute_abhi_hash(merged)
    executed_actions = dispatch_abhi_event(merged, event_name="on_merge", persist=False)

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    return AbhiMergeResult(
        base_input_path=str(Path(base_input_path).expanduser()),
        left_input_path=str(Path(left_input_path).expanduser()),
        right_input_path=str(Path(right_input_path).expanduser()),
        output_path=str(destination),
        merge_strategy=strategy,
        abhi_spec_version=ABHI_SPEC_VERSION,
        nodes_merged=len(merged["graph"]["nodes"]),
        edges_merged=len(merged["graph"]["edges"]),
        conflicts=[*node_conflicts, *edge_conflicts],
        content_hash=merged["integrity"]["content_hash"],
        executed_actions=executed_actions,
    )


def execute_abhi_query(
    document: dict[str, Any],
    *,
    query_id: str = "",
    query_text: str = "",
) -> dict[str, Any]:
    saved_queries = document.get("queries", {}).get("saved", []) if isinstance(document.get("queries"), dict) else []
    selected_query = None
    if query_id.strip():
        selected_query = next(
            (item for item in saved_queries if str(item.get("id", "")).strip() == query_id.strip()),
            None,
        )
        if selected_query is None:
            raise ValueError(f"Unknown ABHI query id: {query_id}")
    effective_query = str(query_text or (selected_query or {}).get("query", "")).strip()
    if not effective_query:
        raise ValueError("ABHI query text cannot be empty.")

    chunk_payload = load_abhi_chunks(document, query_text=effective_query)
    nodes = list(chunk_payload["nodes"])
    edges = list(chunk_payload["edges"])
    node_by_id = {str(node.get("id", "")).strip(): node for node in nodes}
    normalized = effective_query.lower()

    matched_nodes: list[dict[str, Any]] = []
    matched_edges: list[dict[str, Any]] = []

    if normalized.startswith("find nodes where"):
        matched_nodes = _execute_abhi_node_query(nodes, effective_query)
        matched_ids = {str(node.get("id", "")).strip() for node in matched_nodes}
        matched_edges = [
            edge
            for edge in edges
            if str(edge.get("from", "")).strip() in matched_ids and str(edge.get("to", "")).strip() in matched_ids
        ]
    elif normalized.startswith("find paths where"):
        matched_nodes, matched_edges = _execute_abhi_path_query(nodes, edges, effective_query)
    else:
        raise ValueError("Unsupported ABHI query. Supported forms start with FIND nodes WHERE or FIND paths WHERE.")

    return {
        "query_id": str((selected_query or {}).get("id", "")).strip(),
        "name": str((selected_query or {}).get("name", "")).strip(),
        "query": effective_query,
        "summary": (
            f"Matched {len(matched_nodes)} node{'s' if len(matched_nodes) != 1 else ''} and "
            f"{len(matched_edges)} edge{'s' if len(matched_edges) != 1 else ''} from "
            f"{len(chunk_payload['chunk_ids'])} chunk{'s' if len(chunk_payload['chunk_ids']) != 1 else ''}."
        ),
        "nodes": matched_nodes,
        "edges": matched_edges,
        "chunk_ids": list(chunk_payload["chunk_ids"]),
        "scanned_chunk_count": len(chunk_payload["chunk_ids"]),
        "node_labels": {
            node_id: str(node.get("metadata", {}).get("label") or node.get("content", "")).strip()
            for node_id, node in node_by_id.items()
            if node_id in {str(item.get("id", "")).strip() for item in matched_nodes}
        },
    }


def load_abhi_chunks(
    document: dict[str, Any],
    *,
    chunk_ids: list[str] | None = None,
    query_text: str = "",
) -> dict[str, Any]:
    index = _chunk_index(document)
    payloads = _chunk_payloads(document)
    selected_chunk_ids = _resolve_chunk_selection(document, chunk_ids=chunk_ids or [], query_text=query_text)
    if not selected_chunk_ids:
        nodes = list(document.get("graph", {}).get("nodes", []))
        edges = list(document.get("graph", {}).get("edges", []))
        return {"chunk_ids": [], "nodes": nodes, "edges": edges}

    node_map: dict[str, dict[str, Any]] = {}
    edge_map: dict[str, dict[str, Any]] = {}
    for chunk_id in selected_chunk_ids:
        payload = payloads.get(chunk_id)
        if isinstance(payload, dict):
            graph = payload.get("graph", {}) if isinstance(payload.get("graph"), dict) else {}
            chunk_nodes = list(graph.get("nodes", []))
            chunk_edges = list(graph.get("edges", []))
        else:
            manifest = index.get(chunk_id, {})
            chunk_node_ids = {str(item).strip() for item in manifest.get("node_ids", []) if str(item).strip()}
            chunk_edge_ids = {str(item).strip() for item in manifest.get("edge_ids", []) if str(item).strip()}
            chunk_nodes = [
                node for node in document.get("graph", {}).get("nodes", [])
                if str(node.get("id", "")).strip() in chunk_node_ids
            ]
            chunk_edges = [
                edge for edge in document.get("graph", {}).get("edges", [])
                if str(edge.get("id", "")).strip() in chunk_edge_ids
            ]
        for node in chunk_nodes:
            node_id = str(node.get("id", "")).strip()
            if node_id:
                node_map[node_id] = node
        for edge in chunk_edges:
            edge_id = str(edge.get("id", "")).strip()
            if edge_id:
                edge_map[edge_id] = edge

    return {
        "chunk_ids": selected_chunk_ids,
        "nodes": [node_map[node_id] for node_id in sorted(node_map)],
        "edges": [edge_map[edge_id] for edge_id in sorted(edge_map)],
    }


def query_abhi_file(
    *,
    input_path: str | Path,
    query_id: str = "",
    query_text: str = "",
) -> AbhiQueryResult:
    source = Path(input_path).expanduser()
    document = load_abhi_document(source)
    payload = execute_abhi_query(document, query_id=query_id, query_text=query_text)
    executed_actions = dispatch_abhi_event(document, event_name="on_query", persist=True, input_path=source, query_payload=payload)
    return AbhiQueryResult(
        input_path=str(source),
        query_id=payload["query_id"],
        name=payload["name"],
        query=payload["query"],
        summary=payload["summary"],
        node_count=len(payload["nodes"]),
        edge_count=len(payload["edges"]),
        node_ids=[str(item.get("id", "")).strip() for item in payload["nodes"]],
        edge_ids=[str(item.get("id", "")).strip() for item in payload["edges"]],
        chunk_ids=[str(item).strip() for item in payload.get("chunk_ids", []) if str(item).strip()],
        scanned_chunk_count=int(payload.get("scanned_chunk_count", 0) or 0),
        executed_actions=executed_actions,
    )


def load_abhi_chunk_file(
    *,
    input_path: str | Path,
    chunk_ids: list[str] | None = None,
    query_id: str = "",
    query_text: str = "",
) -> AbhiChunkLoadResult:
    source = Path(input_path).expanduser()
    document = load_abhi_document(source)
    selection_query = ""
    if query_id.strip() or query_text.strip():
        query_payload = execute_abhi_query(document, query_id=query_id, query_text=query_text)
        selection_query = str(query_payload.get("query", "")).strip()
    chunk_payload = load_abhi_chunks(document, chunk_ids=chunk_ids or [], query_text=selection_query)
    return AbhiChunkLoadResult(
        input_path=str(source),
        chunk_ids=[str(item).strip() for item in chunk_payload["chunk_ids"]],
        load_strategy=str(document.get("chunks", {}).get("load_strategy", "full") or "full"),
        node_count=len(chunk_payload["nodes"]),
        edge_count=len(chunk_payload["edges"]),
        available_chunk_count=len(_chunk_index(document)),
        query=selection_query,
        node_ids=[str(item.get("id", "")).strip() for item in chunk_payload["nodes"]],
        edge_ids=[str(item.get("id", "")).strip() for item in chunk_payload["edges"]],
    )


def dispatch_abhi_event(
    document: dict[str, Any],
    *,
    event_name: str,
    persist: bool,
    input_path: str | Path | None = None,
    query_payload: dict[str, Any] | None = None,
) -> list[str]:
    events = document.get("events", {}) if isinstance(document.get("events"), dict) else {}
    actions = list(events.get(event_name, [])) if isinstance(events.get(event_name, []), list) else []
    executed: list[str] = []
    event_log = _ensure_event_log(document)

    if event_name == "on_query" and query_payload is not None:
        waggle = _ensure_waggle_block(document)
        stats = waggle.setdefault("query_stats", {})
        query_key = str(query_payload.get("query_id") or query_payload.get("query") or "custom-query")
        stats[query_key] = int(stats.get(query_key, 0) or 0) + 1

    for action in actions:
        normalized = str(action).strip()
        if not normalized:
            continue
        if normalized in {"validate_constraints", "validate_schema"}:
            validation = validate_abhi_document(document, input_path=input_path or "live://abhi")
            if not validation.valid:
                raise ValueError("; ".join(validation.errors))
            executed.append(normalized)
        elif normalized in {"verify_hash"}:
            expected = str(document.get("integrity", {}).get("content_hash", "")).strip()
            actual = compute_abhi_hash(document)
            if expected and expected != actual:
                raise ValueError("Integrity hash mismatch.")
            executed.append(normalized)
        elif normalized in {"compute_hash", "update_hash", "recompute_hash"}:
            document.setdefault("integrity", {})["content_hash"] = compute_abhi_hash(document)
            executed.append(normalized)
        elif normalized == "snapshot_version":
            versions = document.setdefault("versions", [])
            versions.append(
                {
                    "id": f"{event_name}-{len(versions) + 1}",
                    "ts": _latest_validation_timestamp(
                        list(document.get("graph", {}).get("nodes", [])),
                        list(document.get("graph", {}).get("edges", [])),
                    ),
                    "author": "waggle-abhi-events",
                    "changes": [],
                    "message": f"Event {event_name} executed",
                }
            )
            executed.append(normalized)
        elif normalized == "strip_ui_state":
            event_log.append({"event": event_name, "action": normalized, "status": "skipped"})
            continue
        elif normalized == "log_access":
            executed.append(normalized)
        elif normalized == "update_relevance_scores":
            _bump_query_relevance(document, query_payload or {})
            executed.append(normalized)
        elif normalized == "three_way_diff":
            waggle = _ensure_waggle_block(document)
            waggle["last_merge_summary"] = {
                "node_count": len(document.get("graph", {}).get("nodes", [])),
                "edge_count": len(document.get("graph", {}).get("edges", [])),
            }
            executed.append(normalized)
        elif normalized in {"resolve_conflicts", "run_dedup", "auto_link", "check_cycles", "flag_for_review", "notify"}:
            executed.append(normalized)
        else:
            event_log.append({"event": event_name, "action": normalized, "status": "unknown"})
            continue
        event_log.append({"event": event_name, "action": normalized, "status": "executed"})

    if persist and input_path is not None:
        Path(input_path).expanduser().write_text(json.dumps(document, indent=2), encoding="utf-8")
    return executed


def validate_abhi_document(document: dict[str, Any], *, input_path: str | Path) -> AbhiValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    for required_section in (
        "graph",
        "schema",
        "constraints",
        "ai_rules",
        "versions",
        "ui",
        "external_refs",
        "chunks",
        "queries",
        "integrity",
        "events",
    ):
        if required_section not in document:
            errors.append(f"Missing top-level section: {required_section}")

    graph = document.get("graph", {})
    schema = document.get("schema", {})
    constraints = document.get("constraints", [])
    integrity = document.get("integrity", {})
    nodes = list(graph.get("nodes", [])) if isinstance(graph, dict) else []
    edges = list(graph.get("edges", [])) if isinstance(graph, dict) else []
    node_ids: set[str] = set()
    duplicate_contents: set[tuple[str, str]] = set()
    content_keys: set[tuple[str, str]] = set()
    outgoing_edge_counts: dict[str, int] = {}

    node_type_schema = schema.get("node_types", {}) if isinstance(schema, dict) else {}
    edge_type_schema = schema.get("edge_types", {}) if isinstance(schema, dict) else {}

    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        node_type = str(node.get("type", "")).strip()
        content = str(node.get("content", "")).strip()
        if not node_id:
            errors.append("Node is missing required field: id")
            continue
        if node_id in node_ids:
            errors.append(f"Duplicate node id: {node_id}")
        node_ids.add(node_id)
        if not node_type:
            errors.append(f"Node {node_id} is missing required field: type")
        if not content:
            errors.append(f"Node {node_id} is missing required field: content")
        if node_type and node_type not in node_type_schema:
            warnings.append(f"Node {node_id} uses undeclared node type '{node_type}'.")
        schema_entry = node_type_schema.get(node_type, {})
        for required_field in schema_entry.get("must_have", []):
            if not _node_has_field(node, required_field):
                errors.append(f"Node {node_id} of type '{node_type}' is missing required field '{required_field}'.")
        content_key = (node_type, normalize_text(content))
        if content_key in content_keys:
            duplicate_contents.add(content_key)
        content_keys.add(content_key)

    max_edges_per_node = _constraint_limit(constraints, "max_edges_per_node")
    for edge in edges:
        edge_id = str(edge.get("id", "")).strip()
        source_id = str(edge.get("from", "")).strip()
        target_id = str(edge.get("to", "")).strip()
        edge_type = str(edge.get("type", "")).strip()
        if not edge_id:
            errors.append("Edge is missing required field: id")
        if not source_id or not target_id:
            errors.append(f"Edge {edge_id or '<missing-id>'} is missing 'from' or 'to'.")
            continue
        if source_id == target_id:
            errors.append(f"Edge {edge_id or '<missing-id>'} violates no_self_loop.")
        if source_id not in node_ids:
            errors.append(f"Edge {edge_id or '<missing-id>'} references missing source node '{source_id}'.")
        if target_id not in node_ids:
            errors.append(f"Edge {edge_id or '<missing-id>'} references missing target node '{target_id}'.")
        if edge_type and edge_type not in edge_type_schema:
            warnings.append(f"Edge {edge_id or '<missing-id>'} uses undeclared edge type '{edge_type}'.")
        outgoing_edge_counts[source_id] = outgoing_edge_counts.get(source_id, 0) + 1
        edge_schema = edge_type_schema.get(edge_type, {})
        source_type = _node_type_by_id(nodes, source_id)
        target_type = _node_type_by_id(nodes, target_id)
        valid_from = set(edge_schema.get("valid_from", []))
        valid_to = set(edge_schema.get("valid_to", []))
        if valid_from and source_type and source_type not in valid_from:
            errors.append(
                f"Edge {edge_id or '<missing-id>'} type '{edge_type}' cannot originate from node type '{source_type}'."
            )
        if valid_to and target_type and target_type not in valid_to:
            errors.append(
                f"Edge {edge_id or '<missing-id>'} type '{edge_type}' cannot target node type '{target_type}'."
            )
        if edge_type == "contradicts" and source_type == "decision" and target_type != "decision":
            errors.append(
                f"Edge {edge_id or '<missing-id>'} violates custom contradiction rule: decision contradicts must target a decision."
            )

    if duplicate_contents:
        for node_type, content in sorted(duplicate_contents):
            errors.append(f"Duplicate content for node type '{node_type}': {content}")

    if max_edges_per_node is not None:
        for node_id, count in outgoing_edge_counts.items():
            if count > max_edges_per_node:
                errors.append(f"Node {node_id} exceeds max_edges_per_node ({count} > {max_edges_per_node}).")

    expected_hash = str(integrity.get("content_hash", "")).strip()
    actual_hash = compute_abhi_hash(document)
    if not expected_hash:
        errors.append("Integrity hash is missing.")
    elif expected_hash != actual_hash:
        errors.append("Integrity hash mismatch.")

    expected_node_count = integrity.get("node_count")
    expected_edge_count = integrity.get("edge_count")
    if expected_node_count is not None and int(expected_node_count) != len(nodes):
        errors.append(f"Integrity node_count mismatch ({expected_node_count} != {len(nodes)}).")
    if expected_edge_count is not None and int(expected_edge_count) != len(edges):
        errors.append(f"Integrity edge_count mismatch ({expected_edge_count} != {len(edges)}).")

    return AbhiValidationResult(
        input_path=str(Path(input_path).expanduser()),
        valid=not errors,
        errors=errors,
        warnings=warnings,
        node_count=len(nodes),
        edge_count=len(edges),
        content_hash=expected_hash,
        abhi_spec_version=str(integrity.get("abhi_spec_version", "")) or ABHI_SPEC_VERSION,
    )


def abhi_to_snapshot(document: dict[str, Any], *, fallback_tenant_id: str) -> dict[str, Any]:
    waggle_block = document.get("waggle", {}) if isinstance(document.get("waggle"), dict) else {}
    tenant_id = str(waggle_block.get("tenant_id") or fallback_tenant_id)
    nodes = [_abhi_node_to_snapshot_node(node, tenant_id=tenant_id) for node in document.get("graph", {}).get("nodes", [])]
    edges = [_abhi_edge_to_snapshot_edge(edge, tenant_id=tenant_id) for edge in document.get("graph", {}).get("edges", [])]
    return {
        "schema_version": int(waggle_block.get("schema_version", 1)),
        "tenant_id": tenant_id,
        "repos": deepcopy(waggle_block.get("repos", [])),
        "context_windows": deepcopy(waggle_block.get("context_windows", [])),
        "context_window_edges": deepcopy(waggle_block.get("context_window_edges", [])),
        "nodes": nodes,
        "edges": edges,
        "ui": deepcopy(document.get("ui", _default_ui())),
    }


def compute_abhi_hash(document: dict[str, Any]) -> str:
    payload = {
        "graph": document.get("graph", {}),
        "schema": document.get("schema", {}),
        "constraints": document.get("constraints", []),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _snapshot_node_to_abhi_node(node: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(node.get("metadata", {}))
    metadata.update(
        {
            "label": node.get("label", ""),
            "tenant_id": node.get("tenant_id", ""),
            "agent_id": node.get("agent_id", ""),
            "project": node.get("project", ""),
            "session_id": node.get("session_id", ""),
            "context_window_id": node.get("context_window_id"),
            "tags": list(node.get("tags", [])),
            "source_prompt": node.get("source_prompt", ""),
            "evidence_records": deepcopy(node.get("evidence_records", [])),
            "valid_from": node.get("valid_from"),
            "valid_to": node.get("valid_to"),
            "created_at": node.get("created_at"),
            "updated_at": node.get("updated_at"),
            "ts": node.get("updated_at") or node.get("created_at"),
            "access_count": int(node.get("access_count", 0)),
        }
    )
    return {
        "id": node["id"],
        "type": node.get("node_type", "note"),
        "content": node.get("content", ""),
        "metadata": metadata,
    }


def _snapshot_edge_to_abhi_edge(edge: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(edge.get("metadata", {}))
    metadata.update(
        {
            "tenant_id": edge.get("tenant_id", ""),
            "weight": float(edge.get("weight", 1.0)),
            "created_at": edge.get("created_at"),
        }
    )
    return {
        "id": edge["id"],
        "from": edge.get("source_id", ""),
        "to": edge.get("target_id", ""),
        "type": edge.get("relationship", ""),
        "metadata": metadata,
    }


def _abhi_node_to_snapshot_node(node: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
    metadata = dict(node.get("metadata", {}))
    node_type = _normalize_snapshot_node_type(str(node.get("type", "")).strip() or "note")
    created_at = str(metadata.get("created_at") or metadata.get("ts") or "")
    updated_at = str(metadata.get("updated_at") or metadata.get("ts") or created_at)
    return {
        "id": str(node.get("id", "")).strip(),
        "tenant_id": tenant_id,
        "agent_id": str(metadata.get("agent_id", "")),
        "project": str(metadata.get("project", "")),
        "session_id": str(metadata.get("session_id", "")),
        "context_window_id": metadata.get("context_window_id"),
        "label": str(metadata.get("label") or _derive_label(str(node.get("content", "")))),
        "content": str(node.get("content", "")).strip(),
        "node_type": node_type,
        "tags": list(metadata.get("tags", []) or []),
        "source_prompt": str(metadata.get("source_prompt", "")),
        "metadata": {
            **metadata,
            "abhi_original_type": str(node.get("type", "")).strip() or node_type,
        },
        "evidence_records": deepcopy(metadata.get("evidence_records", [])),
        "valid_from": metadata.get("valid_from"),
        "valid_to": metadata.get("valid_to"),
        "created_at": created_at,
        "updated_at": updated_at,
        "access_count": int(metadata.get("access_count", 0) or 0),
    }


def _abhi_edge_to_snapshot_edge(edge: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
    metadata = dict(edge.get("metadata", {}))
    return {
        "id": str(edge.get("id", "")).strip(),
        "tenant_id": tenant_id,
        "source_id": str(edge.get("from", "")).strip(),
        "target_id": str(edge.get("to", "")).strip(),
        "relationship": str(edge.get("type", "")).strip(),
        "weight": float(metadata.get("weight", 1.0) or 1.0),
        "metadata": metadata,
        "created_at": metadata.get("created_at", ""),
    }


def _default_schema() -> dict[str, Any]:
    node_types = {
        "decision": {"must_have": ["content", "ts"], "optional": ["label", "confidence", "source", "tags"]},
        "reason": {"must_have": ["content"], "optional": ["label", "weight", "tags"]},
        "entity": {"must_have": ["content"], "optional": ["label", "entity_type", "aliases", "tags"]},
        "fact": {"must_have": ["content"], "optional": ["label", "tags"]},
        "concept": {"must_have": ["content"], "optional": ["label", "tags"]},
        "preference": {"must_have": ["content"], "optional": ["label", "tags"]},
        "question": {"must_have": ["content"], "optional": ["label", "tags"]},
        "note": {"must_have": ["content"], "optional": ["label", "tags"]},
        "constraint": {"must_have": ["content"], "optional": ["label", "tags"]},
        "goal": {"must_have": ["content"], "optional": ["label", "tags"]},
    }
    all_node_types = list(ABHI_NODE_TYPES)
    edge_types = {
        "depends_on": {"valid_from": ["decision", "goal"], "valid_to": ["reason", "fact", "constraint"]},
        "contradicts": {"valid_from": ["decision", "fact"], "valid_to": ["decision", "fact"]},
    }
    for edge_type in ABHI_EDGE_TYPES:
        edge_types.setdefault(edge_type, {"valid_from": all_node_types, "valid_to": all_node_types})
    return {
        "node_types": node_types,
        "edge_types": edge_types,
    }


def _default_constraints() -> list[dict[str, Any]]:
    return [
        {"rule": "no_self_loop", "description": "No node may have an edge pointing to itself"},
        {"rule": "edge_type_match", "description": "Edge endpoints must match valid_from/valid_to in schema"},
        {"rule": "required_fields", "description": "Nodes must have all must_have fields from schema"},
        {
            "rule": "unique_content_per_type",
            "scope": "project",
            "description": "No two nodes of the same type may have identical content",
        },
        {"rule": "max_edges_per_node", "limit": 500, "description": "Prevent runaway linking"},
        {
            "rule": "custom",
            "expression": "IF node.type == 'decision' AND edge.type == 'contradicts' THEN target.type == 'decision'",
            "description": "Only decisions can contradict other decisions",
        },
    ]


def _default_ai_rules() -> dict[str, Any]:
    return {
        "merge_if_similarity": 0.85,
        "dedup_scope": "project",
        "auto_link_patterns": [
            {
                "from_type": "decision",
                "to_type": "reason",
                "edge_type": "depends_on",
                "condition": "semantic_similarity > 0.8",
            },
            {
                "from_type": "entity",
                "to_type": "entity",
                "edge_type": "relates_to",
                "condition": "co_occurrence > 3",
            },
        ],
        "inference_hints": [
            "If a new decision contradicts an existing decision, create a 'contradicts' edge automatically",
            "If an entity appears in multiple decisions, create 'part_of' edges to a shared context node",
        ],
        "extraction_instructions": (
            "When processing conversation, extract: all named entities, all decisions with stated reasons, "
            "all preferences, all constraints mentioned by the user, and all explicit corrections or contradictions "
            "to prior statements."
        ),
    }


def _build_versions(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes = [
        {
            "op": "add_node",
            "node_id": node["id"],
            "snapshot": {"type": node["type"], "content": node["content"]},
        }
        for node in nodes
    ]
    changes.extend(
        {
            "op": "add_edge",
            "edge_id": edge["id"],
            "from": edge["from"],
            "to": edge["to"],
            "type": edge["type"],
        }
        for edge in edges
    )
    return [
        {
            "id": "v1",
            "ts": _latest_validation_timestamp(nodes, edges),
            "author": "waggle-auto",
            "changes": changes,
            "message": "Initial ABHI export from Waggle memory graph",
        }
    ]


def _default_ui() -> dict[str, Any]:
    return {
        "positions": {},
        "zoom": 1.0,
        "viewport": {"center_x": 0, "center_y": 0},
        "groups": [],
        "collapsed_groups": [],
        "selected_nodes": [],
    }


def _build_chunks(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    if not nodes:
        return {
            "chunk_index": {},
            "chunk_payloads": {},
            "load_strategy": "full",
            "preload": [],
        }

    edges_by_node: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        source_id = str(edge.get("from", "")).strip()
        target_id = str(edge.get("to", "")).strip()
        if source_id:
            edges_by_node.setdefault(source_id, []).append(edge)
        if target_id and target_id != source_id:
            edges_by_node.setdefault(target_id, []).append(edge)

    grouped_nodes: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        node_type = str(node.get("type", "")).strip() or "note"
        grouped_nodes.setdefault(node_type, []).append(node)

    chunk_index: dict[str, Any] = {}
    chunk_payloads: dict[str, Any] = {}
    byte_offset = 0
    ordered_chunk_ids: list[str] = []

    for node_type in sorted(grouped_nodes):
        typed_nodes = sorted(grouped_nodes[node_type], key=lambda item: str(item.get("id", "")).strip())
        for chunk_number, start in enumerate(range(0, len(typed_nodes), ABHI_CHUNK_NODE_LIMIT), start=1):
            chunk_nodes = typed_nodes[start : start + ABHI_CHUNK_NODE_LIMIT]
            chunk_node_ids = {str(node.get("id", "")).strip() for node in chunk_nodes if str(node.get("id", "")).strip()}
            chunk_edges = [
                edge
                for edge in edges
                if str(edge.get("from", "")).strip() in chunk_node_ids or str(edge.get("to", "")).strip() in chunk_node_ids
            ]
            chunk_id = f"{node_type}_{chunk_number}"
            chunk_payload = {
                "graph": {
                    "nodes": deepcopy(chunk_nodes),
                    "edges": deepcopy(chunk_edges),
                },
                "summary": f"{node_type} nodes {start + 1}-{start + len(chunk_nodes)}",
            }
            blob = json.dumps(chunk_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            edge_types = sorted(
                {
                    str(edge.get("type", "")).strip()
                    for edge in chunk_edges
                    if str(edge.get("type", "")).strip()
                }
            )
            content_terms = _chunk_content_terms(chunk_nodes)
            chunk_index[chunk_id] = {
                "label": f"{node_type} chunk {chunk_number}",
                "node_ids": sorted(chunk_node_ids),
                "edge_ids": sorted(
                    str(edge.get("id", "")).strip()
                    for edge in chunk_edges
                    if str(edge.get("id", "")).strip()
                ),
                "node_types": [node_type],
                "edge_types": edge_types,
                "content_terms": content_terms,
                "byte_offset": byte_offset,
                "byte_length": len(blob),
            }
            chunk_payloads[chunk_id] = chunk_payload
            ordered_chunk_ids.append(chunk_id)
            byte_offset += len(blob)

    preload = ordered_chunk_ids[:ABHI_CHUNK_PRELOAD_LIMIT]
    return {
        "chunk_index": chunk_index,
        "chunk_payloads": chunk_payloads,
        "load_strategy": "on_demand" if len(ordered_chunk_ids) > 1 else "full",
        "preload": preload,
    }


def _default_queries() -> dict[str, Any]:
    return {
        "saved": [
            {
                "id": "q1",
                "name": "Recent changes",
                "query": "FIND nodes WHERE ts > NOW() - 7d ORDER BY ts DESC",
            },
            {
                "id": "q2",
                "name": "Contradiction chains",
                "query": "FIND paths WHERE edge.type='contradicts' DEPTH <= 3",
            },
        ],
        "auto_run_on_open": ["q1"],
    }


def _default_events() -> dict[str, Any]:
    return {
        "on_add_node": ["validate_constraints", "auto_link", "update_hash"],
        "on_add_edge": ["validate_constraints", "check_cycles"],
        "on_contradiction_detected": ["flag_for_review", "notify"],
        "on_import": ["validate_schema", "verify_hash", "run_dedup"],
        "on_export": ["compute_hash", "snapshot_version", "strip_ui_state"],
        "on_query": ["log_access", "update_relevance_scores"],
        "on_merge": ["three_way_diff", "resolve_conflicts", "recompute_hash"],
    }


def _latest_validation_timestamp(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    timestamps = []
    for node in nodes:
        metadata = node.get("metadata", {})
        timestamps.extend([metadata.get("updated_at"), metadata.get("created_at"), metadata.get("ts")])
    for edge in edges:
        timestamps.append(edge.get("metadata", {}).get("created_at"))
    normalized = [str(ts).strip() for ts in timestamps if str(ts or "").strip()]
    return max(normalized) if normalized else ""


def _chunk_index(document: dict[str, Any]) -> dict[str, Any]:
    chunks = document.get("chunks", {}) if isinstance(document.get("chunks"), dict) else {}
    index = chunks.get("chunk_index", {})
    return index if isinstance(index, dict) else {}


def _chunk_payloads(document: dict[str, Any]) -> dict[str, Any]:
    chunks = document.get("chunks", {}) if isinstance(document.get("chunks"), dict) else {}
    payloads = chunks.get("chunk_payloads", {})
    return payloads if isinstance(payloads, dict) else {}


def _resolve_chunk_selection(
    document: dict[str, Any],
    *,
    chunk_ids: list[str],
    query_text: str,
) -> list[str]:
    index = _chunk_index(document)
    if not index:
        return []
    if chunk_ids:
        selected = [chunk_id for chunk_id in chunk_ids if chunk_id in index]
        if selected:
            return selected
    if query_text.strip():
        selected = _select_relevant_chunks(document, query_text)
        if selected:
            return selected
    preload = [
        str(item).strip()
        for item in document.get("chunks", {}).get("preload", [])
        if str(item).strip() in index
    ]
    return preload or sorted(index)


def _select_relevant_chunks(document: dict[str, Any], query_text: str) -> list[str]:
    index = _chunk_index(document)
    lowered = query_text.lower()
    selected: list[str] = []
    type_match = _extract_single_quoted_value(query_text, "type=")
    edge_type_match = _extract_single_quoted_value(query_text, "edge.type=")
    content_match = normalize_text(_extract_single_quoted_value(query_text, "content contains"))
    content_terms = set(content_match.split()) if content_match else set()
    for chunk_id in sorted(index):
        manifest = index[chunk_id] if isinstance(index.get(chunk_id), dict) else {}
        node_types = {str(item).strip().lower() for item in manifest.get("node_types", []) if str(item).strip()}
        edge_types = {str(item).strip().lower() for item in manifest.get("edge_types", []) if str(item).strip()}
        chunk_terms = {normalize_text(str(item)) for item in manifest.get("content_terms", []) if str(item).strip()}
        if type_match and type_match.lower() in node_types:
            selected.append(chunk_id)
            continue
        if edge_type_match and edge_type_match.lower() in edge_types:
            selected.append(chunk_id)
            continue
        if content_terms and chunk_terms.intersection(content_terms):
            selected.append(chunk_id)
            continue
        if "recent changes" in lowered and "decision" in node_types:
            selected.append(chunk_id)
    if selected:
        return selected
    return []


def _chunk_content_terms(nodes: list[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for raw in (
            str(node.get("content", "")),
            str(node.get("metadata", {}).get("label", "")) if isinstance(node.get("metadata"), dict) else "",
        ):
            for piece in normalize_text(raw).split():
                if len(piece) < 4:
                    continue
                if piece not in seen:
                    seen.add(piece)
                    tokens.append(piece)
                if len(tokens) >= 24:
                    return tokens
    return tokens


def _node_has_field(node: dict[str, Any], field: str) -> bool:
    if field in node and str(node.get(field, "")).strip():
        return True
    metadata = node.get("metadata", {})
    if isinstance(metadata, dict) and str(metadata.get(field, "")).strip():
        return True
    return False


def _node_type_by_id(nodes: list[dict[str, Any]], node_id: str) -> str:
    for node in nodes:
        if str(node.get("id", "")).strip() == node_id:
            return str(node.get("type", "")).strip()
    return ""


def _constraint_limit(constraints: list[dict[str, Any]], rule_name: str) -> int | None:
    for constraint in constraints:
        if str(constraint.get("rule", "")).strip() == rule_name:
            limit = constraint.get("limit")
            return int(limit) if limit is not None else None
    return None


def _normalize_snapshot_node_type(node_type: str) -> str:
    normalized = node_type.strip().lower()
    if normalized in {"fact", "entity", "concept", "preference", "decision", "question", "note"}:
        return normalized
    if normalized == "reason":
        return "fact"
    if normalized in {"constraint", "goal"}:
        return "concept"
    return "note"


def _derive_label(content: str) -> str:
    trimmed = content.strip()
    return trimmed[:80] if len(trimmed) > 80 else trimmed


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def diff_abhi_files(*, input_path_a: str | Path, input_path_b: str | Path) -> AbhiDiffResult:
    document_a = load_abhi_document(input_path_a)
    document_b = load_abhi_document(input_path_b)
    return diff_abhi_documents(document_a, document_b, input_path_a=input_path_a, input_path_b=input_path_b)


def merge_abhi_files(
    *,
    base_input_path: str | Path,
    left_input_path: str | Path,
    right_input_path: str | Path,
    output_path: str | Path,
    merge_strategy: str = "prefer_right",
) -> AbhiMergeResult:
    base_document = load_abhi_document(base_input_path)
    left_document = load_abhi_document(left_input_path)
    right_document = load_abhi_document(right_input_path)
    return merge_abhi_documents(
        base_document,
        left_document,
        right_document,
        base_input_path=base_input_path,
        left_input_path=left_input_path,
        right_input_path=right_input_path,
        output_path=output_path,
        merge_strategy=merge_strategy,
    )


def _execute_abhi_node_query(nodes: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    selected = list(nodes)
    lowered = query.lower()

    type_match = _extract_single_quoted_value(query, "type=")
    if type_match:
        selected = [node for node in selected if str(node.get("type", "")).strip().lower() == type_match.lower()]

    content_contains = _extract_single_quoted_value(query, "content contains")
    if content_contains:
        needle = normalize_text(content_contains)
        selected = [
            node
            for node in selected
            if needle in normalize_text(str(node.get("content", "")))
            or needle in normalize_text(str(node.get("metadata", {}).get("label", "")))
        ]

    days = _extract_now_minus_days(lowered)
    if days is not None:
        selected = [node for node in selected if _node_is_within_days(node, days)]

    if "order by ts desc" in lowered:
        selected.sort(key=_node_timestamp_for_sort, reverse=True)
    return selected


def _execute_abhi_path_query(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    query: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lowered = query.lower()
    edge_type = _extract_single_quoted_value(query, "edge.type=")
    depth = _extract_depth_limit(lowered)
    if not edge_type:
        raise ValueError("Path queries must specify edge.type='...'.")
    if depth is None:
        depth = 3

    matched_edges = [
        edge for edge in edges if str(edge.get("type", "")).strip().lower() == edge_type.lower()
    ]
    if depth <= 1:
        matched_ids = {
            str(edge.get("from", "")).strip() for edge in matched_edges
        } | {
            str(edge.get("to", "")).strip() for edge in matched_edges
        }
        matched_nodes = [node for node in nodes if str(node.get("id", "")).strip() in matched_ids]
        return matched_nodes, matched_edges

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in matched_edges:
        adjacency.setdefault(str(edge.get("from", "")).strip(), []).append(edge)

    visited_edges: dict[str, dict[str, Any]] = {}
    visited_nodes: set[str] = set()
    for node in nodes:
        start_id = str(node.get("id", "")).strip()
        frontier = [(start_id, 0)]
        seen = {start_id}
        while frontier:
            current, level = frontier.pop(0)
            if level >= depth:
                continue
            for edge in adjacency.get(current, []):
                edge_id = str(edge.get("id", "")).strip()
                target_id = str(edge.get("to", "")).strip()
                visited_edges[edge_id] = edge
                visited_nodes.add(current)
                visited_nodes.add(target_id)
                if target_id not in seen:
                    seen.add(target_id)
                    frontier.append((target_id, level + 1))
    matched_nodes = [node for node in nodes if str(node.get("id", "")).strip() in visited_nodes]
    return matched_nodes, list(visited_edges.values())


def _extract_single_quoted_value(query: str, marker: str) -> str:
    lowered = query.lower()
    index = lowered.find(marker.lower())
    if index < 0:
        return ""
    start = query.find("'", index)
    if start < 0:
        return ""
    end = query.find("'", start + 1)
    if end < 0:
        return ""
    return query[start + 1 : end]


def _extract_now_minus_days(query: str) -> int | None:
    marker = "now() - "
    index = query.find(marker)
    if index < 0:
        return None
    suffix = query[index + len(marker) :]
    digits = []
    for char in suffix:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    if not digits:
        return None
    return int("".join(digits))


def _extract_depth_limit(query: str) -> int | None:
    marker = "depth <="
    index = query.find(marker)
    if index < 0:
        return None
    suffix = query[index + len(marker) :]
    digits = []
    for char in suffix:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    if not digits:
        return None
    return int("".join(digits))


def _node_is_within_days(node: dict[str, Any], days: int) -> bool:
    raw = _node_timestamp_for_sort(node)
    if not raw:
        return False
    try:
        timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    now = datetime.now(timestamp.tzinfo or None)
    delta = now - timestamp
    return delta.days <= days


def _node_timestamp_for_sort(node: dict[str, Any]) -> str:
    metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
    return str(metadata.get("ts") or metadata.get("updated_at") or metadata.get("created_at") or "").strip()


def _canonical_graph_object(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _semantic_label(node: dict[str, Any]) -> str:
    metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
    return str(metadata.get("label") or node.get("content", "")).strip()


def _merge_graph_objects(
    base_items: list[dict[str, Any]],
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
    *,
    strategy: str,
    object_label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    base = {str(item.get("id", "")).strip(): deepcopy(item) for item in base_items if str(item.get("id", "")).strip()}
    left = {str(item.get("id", "")).strip(): deepcopy(item) for item in left_items if str(item.get("id", "")).strip()}
    right = {str(item.get("id", "")).strip(): deepcopy(item) for item in right_items if str(item.get("id", "")).strip()}
    conflicts: list[str] = []
    merged: dict[str, dict[str, Any]] = {}
    all_ids = sorted(set(base) | set(left) | set(right))

    for item_id in all_ids:
        base_item = base.get(item_id)
        left_item = left.get(item_id)
        right_item = right.get(item_id)
        if left_item is None and right_item is None:
            continue
        if left_item is not None and right_item is not None and _canonical_graph_object(left_item) == _canonical_graph_object(right_item):
            merged[item_id] = deepcopy(left_item)
            continue
        if base_item is not None and left_item is not None and _canonical_graph_object(base_item) == _canonical_graph_object(left_item):
            if right_item is not None:
                merged[item_id] = deepcopy(right_item)
            continue
        if base_item is not None and right_item is not None and _canonical_graph_object(base_item) == _canonical_graph_object(right_item):
            if left_item is not None:
                merged[item_id] = deepcopy(left_item)
            continue
        if base_item is None:
            chosen = left_item if strategy == "prefer_left" else right_item
            fallback = right_item if strategy == "prefer_left" else left_item
            merged[item_id] = deepcopy(chosen or fallback or {})
            if left_item is not None and right_item is not None and _canonical_graph_object(left_item) != _canonical_graph_object(right_item):
                conflicts.append(f"{object_label.capitalize()} {item_id} was added differently on both sides; chose {strategy}.")
            continue
        if left_item is None and right_item is not None:
            conflicts.append(f"{object_label.capitalize()} {item_id} was deleted on left and changed on right; chose {strategy}.")
            if strategy == "prefer_right":
                merged[item_id] = deepcopy(right_item)
            continue
        if right_item is None and left_item is not None:
            conflicts.append(f"{object_label.capitalize()} {item_id} was changed on left and deleted on right; chose {strategy}.")
            if strategy == "prefer_left":
                merged[item_id] = deepcopy(left_item)
            continue
        chosen = left_item if strategy == "prefer_left" else right_item
        merged[item_id] = deepcopy(chosen or {})
        conflicts.append(f"{object_label.capitalize()} {item_id} changed on both sides; chose {strategy}.")

    return [merged[item_id] for item_id in sorted(merged)], conflicts


def _merge_prefer_side(base: Any, left: Any, right: Any, strategy: str) -> Any:
    base_json = _canonical_graph_object(base if isinstance(base, dict) else {"value": base})
    left_json = _canonical_graph_object(left if isinstance(left, dict) else {"value": left})
    right_json = _canonical_graph_object(right if isinstance(right, dict) else {"value": right})
    if left_json == right_json:
        return deepcopy(left)
    if base_json == left_json:
        return deepcopy(right)
    if base_json == right_json:
        return deepcopy(left)
    return deepcopy(left if strategy == "prefer_left" else right)


def _merge_unique_list(base: list[Any], left: list[Any], right: list[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for item in [*base, *left, *right]:
        key = _canonical_graph_object(item if isinstance(item, dict) else {"value": item})
        if key in seen:
            continue
        seen.add(key)
        merged.append(deepcopy(item))
    return merged


def _merge_queries(base: dict[str, Any], left: dict[str, Any], right: dict[str, Any], strategy: str) -> dict[str, Any]:
    merged_saved: dict[str, dict[str, Any]] = {}
    for source in (base, left, right):
        for item in source.get("saved", []) if isinstance(source, dict) else []:
            query_id = str(item.get("id", "")).strip()
            if not query_id:
                continue
            merged_saved[query_id] = deepcopy(item)
    auto_run = _merge_unique_list(
        list(base.get("auto_run_on_open", [])) if isinstance(base, dict) else [],
        list(left.get("auto_run_on_open", [])) if isinstance(left, dict) else [],
        list(right.get("auto_run_on_open", [])) if isinstance(right, dict) else [],
    )
    winner = _merge_prefer_side(base, left, right, strategy)
    return {
        **(winner if isinstance(winner, dict) else {}),
        "saved": [merged_saved[key] for key in sorted(merged_saved)],
        "auto_run_on_open": auto_run,
    }


def _ensure_waggle_block(document: dict[str, Any]) -> dict[str, Any]:
    waggle = document.get("waggle")
    if not isinstance(waggle, dict):
        waggle = {}
        document["waggle"] = waggle
    return waggle


def _ensure_event_log(document: dict[str, Any]) -> list[dict[str, Any]]:
    waggle = _ensure_waggle_block(document)
    event_log = waggle.get("event_log")
    if not isinstance(event_log, list):
        event_log = []
        waggle["event_log"] = event_log
    return event_log


def _bump_query_relevance(document: dict[str, Any], query_payload: dict[str, Any]) -> None:
    matched_ids = {str(node_id).strip() for node_id in query_payload.get("node_ids", []) if str(node_id).strip()}
    if not matched_ids:
        matched_ids = {
            str(item.get("id", "")).strip()
            for item in query_payload.get("nodes", [])
            if str(item.get("id", "")).strip()
        }
    for node in document.get("graph", {}).get("nodes", []):
        node_id = str(node.get("id", "")).strip()
        if node_id not in matched_ids:
            continue
        metadata = node.setdefault("metadata", {})
        current = int(metadata.get("relevance_hits", 0) or 0)
        metadata["relevance_hits"] = current + 1
