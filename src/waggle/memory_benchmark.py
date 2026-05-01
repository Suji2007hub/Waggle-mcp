from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from waggle.benchmark_cache import BenchmarkCache, make_cache_key
from waggle.benchmark_harness import BenchmarkRuntimeError
from waggle.graph import MemoryGraph
from waggle.models import Node, NodeType, RelationType, SubgraphResult

BenchmarkName = Literal["longmemeval", "locomo"]
EvalArm = Literal["waggle_graph", "naive_rag", "full_context", "no_context"]
HARNESS_VERSION = "memory-benchmark-v1"


@dataclass
class MemoryCase:
    case_id: str
    question: str
    gold_answer: str
    question_type: str
    sessions: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedCase:
    case: MemoryCase
    project_scope: str
    chunk_node_ids: list[str]


@dataclass
class CaseResult:
    case_id: str
    question_type: str
    arm: EvalArm
    question: str
    gold_answer: str
    prediction: str
    correct: bool
    judge_reason: str
    retrieved_count: int
    retrieved_chars: int
    prompt_chars: int
    latency_seconds: float
    retrieved_context: str


def load_longmemeval(path: Path) -> list[MemoryCase]:
    cases: list[MemoryCase] = []
    rows = _read_jsonl_or_json(path)
    for row_index, row in enumerate(rows):
        sessions: list[dict[str, Any]] = []
        haystack_sessions = row.get("haystack_sessions", [])
        if haystack_sessions and isinstance(haystack_sessions[0], dict):
            for session in haystack_sessions:
                sessions.append(
                    {
                        "session_id": str(session.get("session_id", f"session-{len(sessions)}")),
                        "timestamp": str(session.get("session_date") or session.get("timestamp") or ""),
                        "turns": [
                            {
                                "speaker": str(turn.get("role", "user")),
                                "text": str(turn.get("content", "")),
                            }
                            for turn in session.get("turns", [])
                        ],
                    }
                )
        else:
            session_ids = row.get("haystack_session_ids", [])
            dates = row.get("haystack_dates", [])
            for index, turns in enumerate(haystack_sessions):
                sessions.append(
                    {
                        "session_id": str(session_ids[index] if index < len(session_ids) else f"session-{index}"),
                        "timestamp": str(dates[index] if index < len(dates) else ""),
                        "turns": [
                            {
                                "speaker": str(turn.get("role", "user")),
                                "text": str(turn.get("content", "")),
                            }
                            for turn in turns
                        ],
                    }
                )
        cases.append(
            MemoryCase(
                case_id=str(row.get("question_id") or row.get("id") or f"case-{row_index}"),
                question=str(row.get("question", "")).strip(),
                gold_answer=_normalize_answer(row.get("answer", "")),
                question_type=str(row.get("question_type", "unknown")),
                sessions=sessions,
                metadata={
                    "source": "longmemeval",
                    "gold_support_ids": _extract_support_ids_from_row(row),
                },
            )
        )
    return cases


def load_locomo(path: Path) -> list[MemoryCase]:
    category_names = {
        1: "single-hop",
        2: "multi-hop",
        3: "temporal",
        4: "open-domain",
        5: "adversarial",
    }
    cases: list[MemoryCase] = []
    rows = _read_jsonl_or_json(path)
    for row_index, row in enumerate(rows):
        conversation = row.get("conversation", {})
        sessions: list[dict[str, Any]] = []
        session_keys = sorted(
            [
                key
                for key in conversation.keys()
                if key.startswith("session_") and not key.endswith("date_time")
            ],
            key=lambda key: int(key.split("_")[1]),
        )
        for key in session_keys:
            turns = [
                {
                    "speaker": str(turn.get("speaker", "")),
                    "text": str(turn.get("text", "")),
                }
                for turn in conversation.get(key, [])
            ]
            sessions.append(
                {
                    "session_id": key,
                    "timestamp": str(conversation.get(f"{key}_date_time", "")),
                    "turns": turns,
                }
            )
        for qa_index, qa in enumerate(row.get("qa", [])):
            category = int(qa.get("category", 0) or 0)
            cases.append(
                MemoryCase(
                    case_id=f"{row.get('sample_id', f'row-{row_index}')}_q{qa_index}",
                    question=str(qa.get("question", "")).strip(),
                    gold_answer=_normalize_answer(qa.get("answer", "")),
                    question_type=category_names.get(category, f"category_{category}"),
                    sessions=sessions,
                    metadata={"source": "locomo", "evidence": qa.get("evidence", [])},
                )
            )
    return cases


def _read_jsonl_or_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise BenchmarkRuntimeError("Expected a top-level JSON array.")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _normalize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return " | ".join(str(item).strip() for item in answer)
    return str(answer).strip()


def _extract_support_ids_from_row(row: dict[str, Any]) -> list[str]:
    for key in (
        "correct_session_ids",
        "answer_session_ids",
        "needle_session_ids",
        "ground_truth_session_ids",
        "support_session_ids",
    ):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [str(item) for item in value]
    for key in ("correct_session_id", "answer_session_id", "needle_session_id"):
        value = row.get(key)
        if value:
            return [str(value)]
    return []


def _parse_timestamp(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    for candidate in (text, text.split(" (", 1)[0]):
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        try:
            parsed = datetime.strptime(candidate, "%Y/%m/%d")
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        try:
            parsed = datetime.strptime(candidate, "%Y/%m/%d %H:%M")
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _set_node_timestamp(graph: MemoryGraph, node_id: str, timestamp: datetime | None) -> None:
    if timestamp is None:
        return
    iso = timestamp.astimezone(timezone.utc).isoformat()
    with graph._lock, graph._connect() as connection:  # noqa: SLF001 - benchmark-only timestamp fixup
        connection.execute(
            """
            UPDATE nodes
            SET valid_from = COALESCE(valid_from, ?), created_at = ?, updated_at = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (iso, iso, iso, node_id, graph.tenant_id),
        )


def _set_node_timestamp_with_connection(
    graph: MemoryGraph,
    connection: sqlite3.Connection,
    node_id: str,
    timestamp: datetime | None,
) -> None:
    if timestamp is None:
        return
    iso = timestamp.astimezone(timezone.utc).isoformat()
    connection.execute(
        """
        UPDATE nodes
        SET valid_from = COALESCE(valid_from, ?), created_at = ?, updated_at = ?
        WHERE id = ? AND tenant_id = ?
        """,
        (iso, iso, iso, node_id, graph.tenant_id),
    )


def _insert_benchmark_edges_with_connection(
    graph: MemoryGraph,
    connection: sqlite3.Connection,
    inserted_ids: list[str],
    pending_edges: list[tuple[int, int, RelationType]],
) -> None:
    if not pending_edges:
        return
    unique_edges: list[tuple[int, int, RelationType]] = []
    seen: set[tuple[str, str, str]] = set()
    for source_index, target_index, relationship in pending_edges:
        key = (
            inserted_ids[source_index],
            inserted_ids[target_index],
            relationship.value,
        )
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append((source_index, target_index, relationship))
    rows = [
        (
            str(uuid4()),
            graph.tenant_id,
            inserted_ids[source_index],
            inserted_ids[target_index],
            relationship.value,
            1.0,
            "{}",
            datetime.now(timezone.utc).isoformat(),
        )
        for source_index, target_index, relationship in unique_edges
    ]
    connection.executemany(
        """
        INSERT INTO edges (
            id, tenant_id, source_id, target_id, relationship, weight, metadata, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _session_anchor_content(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in turns:
        speaker = str(turn.get("speaker", "speaker")).strip() or "speaker"
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"[{speaker}] {text}")
    return "\n".join(lines)


def index_case_into_waggle(case: MemoryCase, graph: MemoryGraph) -> IndexedCase:
    project_scope = f"memory-benchmark::{case.metadata.get('source', 'unknown')}::{case.case_id}"
    chunk_node_ids: list[str] = []
    previous_session_anchor_index: int | None = None
    local_turn_link_span = 2
    pending_nodes: list[dict[str, Any]] = []
    pending_edges: list[tuple[str, str, RelationType]] = []
    for session_index, session in enumerate(case.sessions):
        session_id = str(session.get("session_id", f"session-{session_index}"))
        timestamp = _parse_timestamp(str(session.get("timestamp", "")))
        _, context_window_id = graph.resolve_window_context(project=project_scope, session_id=session_id)
        turns = session.get("turns", [])
        session_anchor_content = _session_anchor_content(turns)
        if not session_anchor_content.strip():
            continue
        pending_nodes.append(
            {
                "label": f"Session {session_id}",
                "content": session_anchor_content,
                "node_type": NodeType.NOTE,
                "tags": [
                    "benchmark",
                    f"benchmark:{case.metadata.get('source', 'unknown')}",
                    f"case:{case.case_id}",
                    f"session:{session_id}",
                    "session-anchor",
                ],
                "project": project_scope,
                "session_id": session_id,
                "valid_from": timestamp,
                "context_window_id": context_window_id,
            }
        )
        session_anchor_index = len(pending_nodes) - 1
        recent_turn_indices: list[int] = []
        for turn_index, turn in enumerate(turns):
            speaker = str(turn.get("speaker", "speaker")).strip() or "speaker"
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            pending_nodes.append(
                {
                    "label": f"{speaker.title()} Turn {turn_index + 1}",
                    "content": f"[{speaker}] {text}",
                    "node_type": NodeType.NOTE,
                    "tags": [
                        "benchmark",
                        f"benchmark:{case.metadata.get('source', 'unknown')}",
                        f"case:{case.case_id}",
                        f"session:{session_id}",
                        f"turn:{turn_index + 1}",
                        ],
                        "project": project_scope,
                        "session_id": session_id,
                        "valid_from": timestamp,
                        "context_window_id": context_window_id,
                    }
                )
            turn_node_index = len(pending_nodes) - 1
            pending_edges.append((turn_node_index, session_anchor_index, RelationType.PART_OF))
            pending_edges.append((session_anchor_index, turn_node_index, RelationType.RELATES_TO))
            for prior_turn_index in recent_turn_indices[-local_turn_link_span:]:
                pending_edges.append((prior_turn_index, turn_node_index, RelationType.RELATES_TO))
                pending_edges.append((turn_node_index, prior_turn_index, RelationType.RELATES_TO))
            recent_turn_indices.append(turn_node_index)
        if previous_session_anchor_index is not None:
            pending_edges.append((previous_session_anchor_index, session_anchor_index, RelationType.RELATES_TO))
            pending_edges.append((session_anchor_index, previous_session_anchor_index, RelationType.RELATES_TO))
        previous_session_anchor_index = session_anchor_index

    if pending_nodes:
        texts = [node["content"] for node in pending_nodes]
        if hasattr(graph.embedding_model, "embed_batch"):
            embeddings = graph.embedding_model.embed_batch(texts)
        else:
            embeddings = [graph.embedding_model.embed(text) for text in texts]
    else:
        embeddings = []
    inserted_ids: list[str] = []
    with graph._lock, graph._connect() as connection:  # noqa: SLF001 - benchmark batching path
        for node_spec, embedding in zip(pending_nodes, embeddings, strict=True):
            stored = graph.add_node(**node_spec, embedding=embedding, connection=connection).node
            _set_node_timestamp_with_connection(graph, connection, stored.id, node_spec["valid_from"])
            inserted_ids.append(stored.id)
            chunk_node_ids.append(stored.id)
        _insert_benchmark_edges_with_connection(graph, connection, inserted_ids, pending_edges)
    return IndexedCase(case=case, project_scope=project_scope, chunk_node_ids=chunk_node_ids)


def retrieve_waggle_graph(indexed: IndexedCase, graph: MemoryGraph, *, limit: int = 15, hops: int = 1) -> SubgraphResult:
    return graph.query(
        query=indexed.case.question,
        max_nodes=limit,
        max_depth=hops,
        expand_depth=hops,
        project=indexed.project_scope,
        retrieval_mode="graph",
    )


def retrieve_naive_rag(indexed: IndexedCase, graph: MemoryGraph, *, limit: int = 15) -> SubgraphResult:
    return graph.query(
        query=indexed.case.question,
        max_nodes=limit,
        max_depth=0,
        expand_depth=0,
        project=indexed.project_scope,
        retrieval_mode="graph",
    )


def retrieve_full_context(indexed: IndexedCase, graph: MemoryGraph) -> SubgraphResult:
    return SubgraphResult(
        nodes=[graph.get_node(node_id) for node_id in indexed.chunk_node_ids],
        retrieval_mode="full_context",
        query=indexed.case.question,
    )


def retrieve_no_context(indexed: IndexedCase, graph: MemoryGraph) -> SubgraphResult:
    return SubgraphResult(retrieval_mode="no_context", query=indexed.case.question)


def format_context(result: SubgraphResult, *, sort_oldest_first: bool = False) -> str:
    if not result.nodes:
        return "(no retrieved memory)"
    nodes = list(result.nodes)
    if sort_oldest_first:
        nodes.sort(key=_node_sort_key)
    lines: list[str] = []
    for node in nodes:
        timestamp = _format_node_timestamp(node)
        lines.append(f"[{timestamp} | {node.session_id or '?'}] {node.content}")
    if result.edges:
        edge_lines: list[str] = []
        indexed_nodes = {node.id: node for node in nodes}
        for edge in result.edges:
            source = indexed_nodes.get(edge.source_id)
            target = indexed_nodes.get(edge.target_id)
            if source is None or target is None:
                continue
            edge_lines.append(
                f"  -> {edge.relationship}: [{_format_node_timestamp(target)} | {target.session_id or '?'}] {target.content}"
            )
        if edge_lines:
            lines.append("")
            lines.append("GRAPH LINKS:")
            lines.extend(edge_lines[:20])
    return "\n".join(lines).strip()


def _node_sort_key(node: Node) -> tuple[str, str]:
    timestamp = node.valid_from or node.created_at
    return (timestamp.isoformat() if timestamp else "", node.id)


def _format_node_timestamp(node: Node) -> str:
    timestamp = node.valid_from or node.created_at
    if timestamp is None:
        return "?"
    normalized = timestamp.astimezone(timezone.utc) if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
    return normalized.isoformat()


LONGMEMEVAL_DEFAULT = """You are answering a question based on memory retrieved from prior conversations between a user and an assistant.

The retrieved memory below contains facts, decisions, preferences, and events from earlier sessions. Each item may include a timestamp.

RETRIEVED MEMORY:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer ONLY using information present in the retrieved memory above.
- If the memory contains conflicting information, prefer the most recent entry based on timestamps.
- If the memory does not contain enough information to answer, respond exactly: "I don't have enough information to answer."
- Be concise. Direct answer first, then one short justification sentence.
- Do not invent dates, names, numbers, or facts.

ANSWER:"""

LONGMEMEVAL_TEMPORAL = """You are answering a temporal-reasoning question from memory of prior conversations.

RETRIEVED MEMORY (each entry includes a timestamp):
{context}

QUESTION:
{question}

REASONING STEPS:
1. Identify all memory entries relevant to the entities or topics in the question.
2. For each relevant entry, note its timestamp.
3. Apply the temporal condition (e.g., "before", "after", "most recent", "between X and Y").
4. Select only entries that satisfy the condition.
5. Produce the final answer from those selected entries.

OUTPUT FORMAT:
Relevant entries: <list timestamps and one-line summaries>
Filtered by temporal condition: <list the subset>
Final answer: <the direct answer>

If no entries satisfy the condition, the final answer must be: "I don't have enough information to answer."
Do not invent timestamps or facts."""

LONGMEMEVAL_KNOWLEDGE_UPDATE = """You are answering a question where the user's information may have changed over time.

RETRIEVED MEMORY (sorted oldest to newest):
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Memory entries about the same fact may contradict each other because the user updated their information.
- The MOST RECENT entry is the correct current answer.
- Older entries are historical context, not the current truth.
- If no relevant entry exists, respond: "I don't have enough information to answer."

OUTPUT FORMAT:
Most recent relevant entry: <timestamp + content>
Superseded entries (if any): <list>
Final answer: <direct answer based on the most recent entry>"""

LOCOMO_DEFAULT = """You are an assistant answering a question about a long-running conversation between two speakers.

The retrieved memory below contains dialogue snippets from prior sessions. Each entry includes the speaker, the session date, and the content.

RETRIEVED MEMORY:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer ONLY from the retrieved memory.
- Pay attention to WHO said or did what.
- If the question requires combining facts from multiple memory entries, reason step by step.
- If the retrieved memory does not contain the answer, respond exactly: "Not mentioned in the conversation."
- Do not invent details. Do not assume based on stereotypes or external knowledge.
- One or two sentences maximum.

ANSWER:"""

LOCOMO_MULTIHOP = """You are answering a multi-hop question that requires combining information from multiple parts of a long conversation.

RETRIEVED MEMORY:
{context}

QUESTION:
{question}

REASONING STEPS:
1. Identify the entities or events in the question.
2. For each one, find the relevant memory entry.
3. Identify the connection (causal, temporal, or relational) between them.
4. Combine into a single answer.

OUTPUT FORMAT:
Hop 1: <memory entry + brief description>
Hop 2: <memory entry + brief description>
Connection: <how they relate>
Final answer: <one to two sentences>

If any hop cannot be filled from the retrieved memory, the final answer must be: Not mentioned in the conversation."""

LOCOMO_ADVERSARIAL = """You are answering a question about a long-running conversation. Some questions deliberately ask about things that were NEVER discussed, to test whether you make things up.

RETRIEVED MEMORY:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Search the retrieved memory carefully for direct or strongly implied answers.
- "Strongly implied" means derivable from explicit statements. Plausible guesses do NOT count.
- If the memory does not contain or strongly imply an answer, respond EXACTLY: "Not mentioned in the conversation."
- Do not guess. Do not use general world knowledge.

OUTPUT FORMAT:
Evidence found: <quote relevant memory entries, or write "None">
Final answer: <direct answer, or "Not mentioned in the conversation.">"""

LOCOMO_TEMPORAL = """You are answering a temporal question about a multi-session conversation.

RETRIEVED MEMORY (with session dates):
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Identify the temporal anchor in the question (e.g., "last month", "before X", "the first time").
- Match it against the session dates in the retrieved memory.
- Only use entries within the temporal window.
- If the temporal window is empty or ambiguous, respond: "Not mentioned in the conversation."

Final answer:"""

JUDGE_PROMPT = """You are grading whether a model's answer matches the ground-truth answer for a memory benchmark question.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{gold}

MODEL ANSWER:
{prediction}

GRADING RULES:
- CORRECT if the model expresses the same factual content as the ground truth, even with different wording.
- CORRECT if the model is more specific than the ground truth but does not contradict it.
- INCORRECT if the model adds facts not in the ground truth (hallucination), contradicts the ground truth, or is vaguer in a way that loses the answer.
- For "I don't have enough information" / "Not mentioned" answers: CORRECT only if the ground truth is also that no answer exists.
- Ignore differences in formatting, punctuation, or filler.

OUTPUT EXACTLY:
VERDICT: CORRECT
or
VERDICT: INCORRECT

Then on the next line:
REASON: <one short sentence>"""


def select_prompt(benchmark: BenchmarkName, question_type: str) -> str:
    lowered = question_type.lower()
    if benchmark == "longmemeval":
        if "temporal" in lowered:
            return LONGMEMEVAL_TEMPORAL
        if "knowledge-update" in lowered or "knowledge_update" in lowered:
            return LONGMEMEVAL_KNOWLEDGE_UPDATE
        return LONGMEMEVAL_DEFAULT
    if lowered == "multi-hop":
        return LOCOMO_MULTIHOP
    if lowered == "adversarial":
        return LOCOMO_ADVERSARIAL
    if lowered == "temporal":
        return LOCOMO_TEMPORAL
    return LOCOMO_DEFAULT


def parse_verdict(text: str) -> tuple[bool, str]:
    verdict_match = re.search(r"VERDICT:\s*(CORRECT|INCORRECT)", text, re.IGNORECASE)
    correct = verdict_match is not None and verdict_match.group(1).upper() == "CORRECT"
    reason_match = re.search(r"REASON:\s*(.+)", text)
    return correct, (reason_match.group(1).strip() if reason_match else "")


def run_case(
    indexed: IndexedCase,
    graph: MemoryGraph,
    *,
    benchmark: BenchmarkName,
    arm: EvalArm,
    answer_model_call: Callable[[str], str],
    judge_model_call: Callable[[str], str],
    retrieval_limit: int = 15,
) -> CaseResult:
    if arm == "waggle_graph":
        retrieved = retrieve_waggle_graph(indexed, graph, limit=retrieval_limit, hops=1)
    elif arm == "naive_rag":
        retrieved = retrieve_naive_rag(indexed, graph, limit=retrieval_limit)
    elif arm == "full_context":
        retrieved = retrieve_full_context(indexed, graph)
    elif arm == "no_context":
        retrieved = retrieve_no_context(indexed, graph)
    else:
        raise ValueError(f"unknown arm: {arm}")

    template = select_prompt(benchmark, indexed.case.question_type)
    sort_oldest_first = benchmark == "longmemeval" and "knowledge-update" in indexed.case.question_type.lower()
    context = format_context(retrieved, sort_oldest_first=sort_oldest_first)
    prompt = template.format(context=context, question=indexed.case.question)

    started = time.time()
    prediction = answer_model_call(prompt).strip()
    latency = time.time() - started

    verdict_text = judge_model_call(
        JUDGE_PROMPT.format(
            question=indexed.case.question,
            gold=indexed.case.gold_answer,
            prediction=prediction,
        )
    )
    correct, reason = parse_verdict(verdict_text)

    return CaseResult(
        case_id=indexed.case.case_id,
        question_type=indexed.case.question_type,
        arm=arm,
        question=indexed.case.question,
        gold_answer=indexed.case.gold_answer,
        prediction=prediction,
        correct=correct,
        judge_reason=reason,
        retrieved_count=len(retrieved.nodes),
        retrieved_chars=len(context),
        prompt_chars=len(prompt),
        latency_seconds=latency,
        retrieved_context=context,
    )


def run_case_cached(
    indexed: IndexedCase,
    graph: MemoryGraph,
    *,
    benchmark: BenchmarkName,
    arm: EvalArm,
    answer_model_call: Callable[[str], str],
    judge_model_call: Callable[[str], str],
    answer_model_name: str,
    judge_model_name: str,
    retrieval_limit: int,
    cache: BenchmarkCache | None,
    cache_extra: dict[str, Any] | None = None,
) -> tuple[CaseResult, bool]:
    template = select_prompt(benchmark, indexed.case.question_type)
    key = make_cache_key(
        benchmark=benchmark,
        case_id=indexed.case.case_id,
        arm=arm,
        answer_model=answer_model_name,
        judge_model=judge_model_name,
        prompt_template=template,
        retrieval_limit=retrieval_limit,
        extra={"harness_version": HARNESS_VERSION, **(cache_extra or {})},
    )
    if cache is not None and cache.has(key):
        cached = cache.get(key)
        if cached is not None:
            return CaseResult(**cached), True
    result = run_case(
        indexed,
        graph,
        benchmark=benchmark,
        arm=arm,
        answer_model_call=answer_model_call,
        judge_model_call=judge_model_call,
        retrieval_limit=retrieval_limit,
    )
    if cache is not None:
        cache.put(
            key,
            result,
            meta={
                "benchmark": benchmark,
                "case_id": indexed.case.case_id,
                "question_type": indexed.case.question_type,
                "arm": arm,
                "answer_model": answer_model_name,
                "judge_model": judge_model_name,
            },
        )
    return result, False


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = by_arm.setdefault(result.arm, {"total": 0, "correct": 0, "by_type": {}})
        bucket["total"] += 1
        bucket["correct"] += int(result.correct)
        type_bucket = bucket["by_type"].setdefault(result.question_type, {"total": 0, "correct": 0})
        type_bucket["total"] += 1
        type_bucket["correct"] += int(result.correct)
    summary: dict[str, Any] = {}
    for arm, bucket in by_arm.items():
        summary[arm] = {
            "accuracy": (bucket["correct"] / bucket["total"]) if bucket["total"] else 0.0,
            "correct": bucket["correct"],
            "total": bucket["total"],
            "by_question_type": {
                question_type: {
                    "accuracy": (entry["correct"] / entry["total"]) if entry["total"] else 0.0,
                    "correct": entry["correct"],
                    "total": entry["total"],
                }
                for question_type, entry in bucket["by_type"].items()
            },
        }
    return summary


def write_report(path: Path, results: list[CaseResult], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": [asdict(result) for result in results]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
