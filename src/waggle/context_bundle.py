from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from waggle.evidence import rank_node_evidence
from waggle.models import (
    ContextBundle,
    ContextBundleExportResult,
    ContextRenderHints,
    ContextTimelineItem,
    Edge,
    GraphStats,
    Node,
    ReplayHit,
)

APPENDIX_CHUNK_SIZE = 40
TIMELINE_LIMIT = 25


def _estimate_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


def _resolve_output_paths(
    *,
    output_path: str | Path | None,
    export_dir: Path,
    format: str,
    mode: str,
) -> tuple[Path | None, Path | None]:
    if output_path is None:
        export_dir.mkdir(parents=True, exist_ok=True)
        stem = export_dir / f"waggle-context-{mode}-{bundle_timestamp()}"
    else:
        requested = Path(output_path).expanduser()
        requested.parent.mkdir(parents=True, exist_ok=True)
        stem = requested.with_suffix("") if requested.suffix else requested

    markdown_path = stem.with_suffix(".md") if format in {"markdown", "both"} else None
    json_path = stem.with_suffix(".json") if format in {"json", "both"} else None
    return markdown_path, json_path


def bundle_timestamp() -> str:
    from waggle.models import utc_now

    return utc_now().strftime("%Y%m%d-%H%M%S")


def _node_to_export_dict(
    node: Node,
    *,
    include_timestamps: bool,
    include_source_prompt: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": node.id,
        "tenant_id": node.tenant_id,
        "agent_id": node.agent_id,
        "project": node.project,
        "session_id": node.session_id,
        "label": node.label,
        "content": node.content,
        "node_type": node.node_type.value,
        "tags": list(node.tags),
        "access_count": node.access_count,
        "evidence_records": [record.model_dump(mode="json") for record in node.evidence_records],
        "valid_from": node.valid_from.isoformat() if node.valid_from is not None else None,
        "valid_to": node.valid_to.isoformat() if node.valid_to is not None else None,
    }
    if include_timestamps:
        payload["created_at"] = node.created_at.isoformat()
        payload["updated_at"] = node.updated_at.isoformat()
    if include_source_prompt and node.source_prompt:
        payload["source_prompt"] = node.source_prompt
    return payload


def _edge_to_export_dict(edge: Edge, *, include_timestamps: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": edge.id,
        "tenant_id": edge.tenant_id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relationship": edge.relationship,
        "weight": edge.weight,
        "metadata": dict(edge.metadata),
    }
    if include_timestamps:
        payload["created_at"] = edge.created_at.isoformat()
    return payload


def _timeline_from_nodes_and_edges(nodes: list[Node], edges: list[Edge]) -> list[ContextTimelineItem]:
    items: list[ContextTimelineItem] = []
    for node in nodes:
        kind = "node_updated" if node.updated_at != node.created_at else "node_created"
        timestamp = node.updated_at if kind == "node_updated" else node.created_at
        items.append(
            ContextTimelineItem(
                kind=kind,
                timestamp=timestamp,
                label=node.label,
                summary=node.content,
                node_id=node.id,
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
            )
        )
    return sorted(items, key=lambda item: (item.timestamp, item.kind, item.label), reverse=True)[:TIMELINE_LIMIT]


def build_context_bundle(
    *,
    tenant_id: str,
    project: str,
    mode: str,
    retrieval_mode: str,
    audience: str,
    query: str,
    summary: str,
    nodes: list[Node],
    edges: list[Edge],
    replay_hits: list[ReplayHit],
    stats: GraphStats,
) -> ContextBundle:
    timeline = _timeline_from_nodes_and_edges(nodes, edges)
    chunk_count = max(1, math.ceil(len(nodes) / APPENDIX_CHUNK_SIZE))
    render_hints = ContextRenderHints(
        recommended_paste_order=[
            "Paste the summary and key facts first.",
            "Paste decisions, contradictions, and timeline next.",
            "Paste appendix chunks only if the receiving AI needs more detail.",
        ],
        truncation_flags=["large_graph"] if mode == "graph" and chunk_count > 1 else [],
        chunk_count=chunk_count,
    )
    return ContextBundle(
        tenant_id=tenant_id,
        project=project,
        mode=mode,
        retrieval_mode=retrieval_mode,
        audience=audience,
        query=query,
        summary=summary,
        nodes=nodes,
        edges=edges,
        replay_hits=replay_hits,
        timeline=timeline,
        stats=stats,
        render_hints=render_hints,
    )


def _shorten_summary_text(text: str, *, max_chars: int = 160) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    clipped = normalized[: max_chars - 1].rstrip(" ,;:-")
    return f"{clipped}…"


def _support_labels_for_node(node: Node, edges: list[Edge], node_by_id: dict[str, Node]) -> list[str]:
    labels: list[str] = []
    for edge in edges:
        if edge.relationship not in {"depends_on", "derived_from", "part_of", "relates_to"}:
            continue
        if edge.source_id == node.id and edge.target_id in node_by_id:
            labels.append(node_by_id[edge.target_id].label)
        elif edge.target_id == node.id and edge.source_id in node_by_id:
            labels.append(node_by_id[edge.source_id].label)
    return list(dict.fromkeys(labels))


def build_query_summary(
    *,
    query: str,
    nodes: list[Node],
    edges: list[Edge],
    replay_hits: list[ReplayHit],
    retrieval_mode: str,
) -> str:
    if not nodes:
        if replay_hits:
            snippets = "; ".join(
                _shorten_summary_text(hit.transcript_snippet or hit.transcript_text, max_chars=100)
                for hit in replay_hits[:2]
            )
            return f"For '{query}', no structured nodes matched. Top replay evidence: {snippets}."
        return f"No memory matched '{query}'."

    node_by_id = {node.id: node for node in nodes}
    parts: list[str] = []

    fact_like_nodes = [
        node for node in nodes if node.node_type.value in {"fact", "preference", "concept", "entity"}
    ]
    if fact_like_nodes:
        parts.append(
            "Key context: "
            + "; ".join(_shorten_summary_text(node.content) for node in fact_like_nodes[:2])
            + "."
        )

    decision_nodes = [node for node in nodes if node.node_type.value == "decision"]
    if decision_nodes:
        decision_summaries: list[str] = []
        for node in decision_nodes[:2]:
            support_labels = _support_labels_for_node(node, edges, node_by_id)
            primary_text = _shorten_summary_text(node.content, max_chars=120)
            if support_labels:
                decision_summaries.append(
                    f"{primary_text} Supported by {', '.join(support_labels[:2])}"
                )
            else:
                decision_summaries.append(primary_text)
        parts.append("Decisions: " + "; ".join(decision_summaries) + ".")

    note_nodes = [node for node in nodes if node.node_type.value == "note"]
    if note_nodes:
        parts.append("Open items: " + "; ".join(node.label for node in note_nodes[:2]) + ".")

    contradiction_count = sum(1 for edge in edges if edge.relationship == "contradicts")
    update_count = sum(1 for edge in edges if edge.relationship == "updates")
    if contradiction_count:
        parts.append(f"{contradiction_count} contradiction edge{'s' if contradiction_count != 1 else ''} included.")
    elif update_count:
        parts.append(f"{update_count} update edge{'s' if update_count != 1 else ''} included.")

    if replay_hits and retrieval_mode in {"replay", "fusion", "verbatim", "hybrid"}:
        parts.append(f"{len(replay_hits)} replay hit{'s' if len(replay_hits) != 1 else ''} included for provenance.")

    if not parts:
        return f"Query context for '{query}' returned {len(nodes)} nodes."
    return " ".join(parts)


def _render_instruction_note(bundle: ContextBundle) -> str:
    if bundle.audience == "human":
        return (
            "Use this as a portable snapshot of the memory graph. Read the summary first, then scan "
            "decisions, contradictions, and the timeline before diving into the appendix."
        )
    return (
        "Use this as authoritative imported context. Prefer the summary, decisions, contradictions, "
        "timeline, and relationship map before consulting the appendix for raw node details."
    )


def _group_nodes(nodes: list[Node]) -> dict[str, list[Node]]:
    grouped: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        grouped[node.node_type.value].append(node)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _decision_support_lines(bundle: ContextBundle) -> list[str]:
    node_by_id = {node.id: node for node in bundle.nodes}
    support_relationships = {"depends_on", "derived_from", "part_of", "relates_to"}
    lines: list[str] = []
    for node in bundle.nodes:
        if node.node_type.value != "decision":
            continue
        support_nodes: list[str] = []
        for edge in bundle.edges:
            if edge.relationship not in support_relationships:
                continue
            if edge.source_id == node.id and edge.target_id in node_by_id:
                support_nodes.append(f"{node_by_id[edge.target_id].label} ({edge.relationship})")
            elif edge.target_id == node.id and edge.source_id in node_by_id:
                support_nodes.append(f"{node_by_id[edge.source_id].label} ({edge.relationship})")
        if support_nodes:
            lines.append(f'- "{node.label}"')
            for support in support_nodes:
                lines.append(f"  - support: {support}")
        else:
            lines.append(f'- "{node.label}"')
    return lines


def _conflict_lines(bundle: ContextBundle) -> list[str]:
    node_by_id = {node.id: node for node in bundle.nodes}
    lines: list[str] = []
    for edge in bundle.edges:
        if edge.relationship not in {"contradicts", "updates"}:
            continue
        source_label = node_by_id.get(edge.source_id).label if edge.source_id in node_by_id else edge.source_id[:8]
        target_label = node_by_id.get(edge.target_id).label if edge.target_id in node_by_id else edge.target_id[:8]
        lines.append(f'- "{source_label}" --[{edge.relationship}]--> "{target_label}"')
    return lines


def _relationship_lines(bundle: ContextBundle) -> list[str]:
    if not bundle.edges:
        return ["- No relationships included in this bundle."]
    node_by_id = {node.id: node for node in bundle.nodes}
    lines: list[str] = []
    for edge in bundle.edges:
        source_label = node_by_id.get(edge.source_id).label if edge.source_id in node_by_id else edge.source_id[:8]
        target_label = node_by_id.get(edge.target_id).label if edge.target_id in node_by_id else edge.target_id[:8]
        lines.append(
            f'- "{source_label}" --[{edge.relationship}, weight={edge.weight:.2f}]--> "{target_label}"'
        )
    return lines


def _estimate_bundle_tokens(
    bundle: ContextBundle,
    *,
    include_edges: bool,
    include_timestamps: bool,
    include_source_prompt: bool,
) -> int:
    parts: list[str] = [
        bundle.tenant_id,
        bundle.project,
        bundle.mode,
        bundle.retrieval_mode,
        bundle.audience,
        bundle.query,
        bundle.summary,
        bundle.generated_at.isoformat(),
    ]
    for node in bundle.nodes:
        parts.extend(
            [
                node.id,
                node.agent_id,
                node.project,
                node.session_id,
                node.label,
                node.content,
                node.node_type.value,
                " ".join(node.tags),
            ]
        )
        if include_timestamps:
            parts.extend(
                [
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                    node.valid_from.isoformat() if node.valid_from is not None else "",
                    node.valid_to.isoformat() if node.valid_to is not None else "",
                ]
            )
        if include_source_prompt:
            parts.append(node.source_prompt)
        for evidence in rank_node_evidence(node, query=bundle.query, limit=2):
            parts.extend(
                [
                    evidence.source_role,
                    str(evidence.turn_index),
                    evidence.source_text,
                ]
            )
    if include_edges:
        for edge in bundle.edges:
            parts.extend(
                [
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.relationship,
                    f"{edge.weight:.4f}",
                    json.dumps(edge.metadata, sort_keys=True),
                ]
            )
            if include_timestamps:
                parts.append(edge.created_at.isoformat())
    for hit in bundle.replay_hits:
        parts.extend(
            [
                str(hit.score),
                hit.session_id,
                str(hit.turn_index),
                hit.role,
                hit.transcript_text,
                hit.transcript_snippet,
            ]
        )
        if include_timestamps:
            parts.append(hit.observed_at.isoformat())
    for item in bundle.timeline:
        parts.extend([item.kind, item.label, item.summary])
        if include_timestamps:
            parts.append(item.timestamp.isoformat())
    return _estimate_tokens("\n".join(part for part in parts if part))


def render_context_bundle_markdown(
    bundle: ContextBundle,
    *,
    include_edges: bool,
    include_timestamps: bool,
    include_source_prompt: bool,
) -> tuple[str, int]:
    grouped_nodes = _group_nodes(bundle.nodes)
    lines = [
        "# Waggle Context Bundle",
        "",
        f"- Tenant: `{bundle.tenant_id}`",
        f"- Project: `{bundle.project or 'n/a'}`",
        f"- Mode: `{bundle.mode}`",
        f"- Query: `{bundle.query or 'n/a'}`",
        f"- Generated at: `{bundle.generated_at.isoformat()}`",
        "",
        "## How To Use",
        "",
        _render_instruction_note(bundle),
        "",
        "## Memory Summary",
        "",
        bundle.summary or "No summary available.",
        "",
        "## Key Facts By Node Type",
        "",
    ]

    for node_type, nodes in grouped_nodes.items():
        lines.append(f"### {node_type.replace('_', ' ').title()}")
        for node in nodes:
            suffix = ""
            if include_timestamps:
                suffix = f" (created {node.created_at.date()}, updated {node.updated_at.date()})"
            lines.append(f'- "{node.label}"{suffix}: {node.content}')
        lines.append("")

    lines.extend(["## Decisions With Reasons", ""])
    decision_lines = _decision_support_lines(bundle)
    lines.extend(decision_lines or ["- No decision nodes in this bundle."])
    lines.extend(["", "## Contradictions And Updates", ""])
    lines.extend(_conflict_lines(bundle) or ["- No contradiction or update edges in this bundle."])
    lines.extend(["", "## Timeline Of Recent Changes", ""])
    if bundle.timeline:
        for item in bundle.timeline:
            timestamp = f" `{item.timestamp.isoformat()}`" if include_timestamps else ""
            lines.append(f'- [{item.kind}]{timestamp} {item.label} — {item.summary}')
    else:
        lines.append("- No recent changes available.")

    lines.extend(["", "## Relationship Map", ""])
    lines.extend(_relationship_lines(bundle) if include_edges else ["- Edge export disabled for this bundle."])
    lines.extend(["", "## Replay Evidence", ""])
    if bundle.replay_hits:
        for hit in bundle.replay_hits[:10]:
            timestamp = f" `{hit.observed_at.isoformat()}`" if include_timestamps else ""
            lines.append(
                f'- [session `{hit.session_id or "n/a"}` turn {hit.turn_index} role `{hit.role or "unknown"}`]{timestamp} '
                f'{hit.transcript_snippet or hit.transcript_text}'
            )
    else:
        lines.append("- No replay evidence included in this bundle.")
    lines.extend(["", "## Full Node Appendix", ""])

    for chunk_index, start in enumerate(range(0, len(bundle.nodes), APPENDIX_CHUNK_SIZE), start=1):
        chunk = bundle.nodes[start:start + APPENDIX_CHUNK_SIZE]
        if bundle.render_hints.chunk_count > 1:
            lines.extend([f"### Appendix Chunk {chunk_index}/{bundle.render_hints.chunk_count}", ""])
        for node in chunk:
            lines.append(f"- id: `{node.id}`")
            lines.append(f"  - type: `{node.node_type.value}`")
            if node.agent_id:
                lines.append(f"  - agent_id: {node.agent_id}")
            if node.project:
                lines.append(f"  - project: {node.project}")
            if node.session_id:
                lines.append(f"  - session_id: {node.session_id}")
            lines.append(f"  - label: {node.label}")
            lines.append(f"  - content: {node.content}")
            if node.tags:
                lines.append(f"  - tags: {', '.join(node.tags)}")
            if include_timestamps:
                lines.append(f"  - created_at: {node.created_at.isoformat()}")
                lines.append(f"  - updated_at: {node.updated_at.isoformat()}")
                if node.valid_from or node.valid_to:
                    lines.append(
                        "  - validity: "
                        f"{node.valid_from.isoformat() if node.valid_from else 'open'} -> "
                        f"{node.valid_to.isoformat() if node.valid_to else 'open'}"
                    )
            if node.evidence_records:
                top_evidence = rank_node_evidence(node, query=bundle.query, limit=2)
                for evidence in top_evidence:
                    lines.append(
                        "  - evidence: "
                        f"[{evidence.source_role or 'unknown'} turn {evidence.turn_index}] {evidence.source_text}"
                    )
            if include_source_prompt and node.source_prompt:
                lines.append(f"  - source_prompt: {node.source_prompt}")
        lines.append("")

    markdown = "\n".join(lines).strip() + "\n"
    return markdown, _estimate_tokens(markdown)


def render_context_bundle_json(
    bundle: ContextBundle,
    *,
    include_edges: bool,
    include_timestamps: bool,
    include_source_prompt: bool,
) -> str:
    payload = {
        "schema_version": bundle.schema_version,
        "export_type": bundle.export_type,
        "generated_at": bundle.generated_at.isoformat(),
        "tenant_id": bundle.tenant_id,
        "project": bundle.project,
        "mode": bundle.mode,
        "audience": bundle.audience,
        "query": bundle.query,
        "summary": bundle.summary,
        "nodes": [
            _node_to_export_dict(
                node,
                include_timestamps=include_timestamps,
                include_source_prompt=include_source_prompt,
            )
            for node in bundle.nodes
        ],
        "edges": [
            _edge_to_export_dict(edge, include_timestamps=include_timestamps)
            for edge in bundle.edges
        ] if include_edges else [],
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
            for hit in bundle.replay_hits
        ],
        "timeline": [
            {
                "kind": item.kind,
                "timestamp": item.timestamp.isoformat(),
                "label": item.label,
                "summary": item.summary,
                "node_id": item.node_id,
                "edge_id": item.edge_id,
            }
            for item in bundle.timeline
        ],
        "stats": {
            "total_nodes": bundle.stats.total_nodes,
            "total_edges": bundle.stats.total_edges,
            "node_type_breakdown": bundle.stats.node_type_breakdown,
        },
        "render_hints": {
            "token_estimate": bundle.render_hints.token_estimate,
            "recommended_paste_order": bundle.render_hints.recommended_paste_order,
            "truncation_flags": bundle.render_hints.truncation_flags,
            "chunk_count": bundle.render_hints.chunk_count,
        },
    }
    return json.dumps(payload, indent=2)


def export_context_bundle_files(
    bundle: ContextBundle,
    *,
    output_path: str | Path | None,
    export_dir: str | Path,
    format: str,
    include_edges: bool,
    include_timestamps: bool,
    include_source_prompt: bool,
) -> ContextBundleExportResult:
    export_directory = Path(export_dir).expanduser()
    markdown_path, json_path = _resolve_output_paths(
        output_path=output_path,
        export_dir=export_directory,
        format=format,
        mode=bundle.mode,
    )

    if markdown_path is not None:
        markdown, token_estimate = render_context_bundle_markdown(
            bundle,
            include_edges=include_edges,
            include_timestamps=include_timestamps,
            include_source_prompt=include_source_prompt,
        )
        bundle.render_hints.token_estimate = token_estimate
        markdown_path.write_text(
            markdown,
            encoding="utf-8",
        )

    if json_path is not None:
        if bundle.render_hints.token_estimate == 0:
            bundle.render_hints.token_estimate = _estimate_bundle_tokens(
                bundle,
                include_edges=include_edges,
                include_timestamps=include_timestamps,
                include_source_prompt=include_source_prompt,
            )
        json_path.write_text(
            render_context_bundle_json(
                bundle,
                include_edges=include_edges,
                include_timestamps=include_timestamps,
                include_source_prompt=include_source_prompt,
            ),
            encoding="utf-8",
        )

    return ContextBundleExportResult(
        tenant_id=bundle.tenant_id,
        project=bundle.project,
        mode=bundle.mode,
        retrieval_mode=getattr(bundle, "retrieval_mode", "graph"),
        query=bundle.query,
        summary=bundle.summary,
        markdown_path=str(markdown_path) if markdown_path is not None else None,
        json_path=str(json_path) if json_path is not None else None,
        node_count=len(bundle.nodes),
        edge_count=len(bundle.edges) if include_edges else 0,
        bundle=bundle,
    )
