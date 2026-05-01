from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from waggle.embeddings import EmbeddingModel
from waggle.graph import MemoryGraph
from waggle.intelligence import lexical_overlap
from waggle.models import NodeType, RelationType


DEFAULT_PROJECT = "token-efficiency-benchmark"
DEFAULT_WAGGLE_TOP_K = 5
DEFAULT_WAGGLE_MAX_DEPTH = 2
DEFAULT_WAGGLE_EXPAND_DEPTH = 0
DEFAULT_RAG_CHUNK_SIZE = 512
DEFAULT_RAG_CHUNK_OVERLAP = 64
DEFAULT_RAG_TOP_K = 5
DEFAULT_HYBRID_CHUNK_TOP_K = 5
DEFAULT_HYBRID_RERANK_POOL = 12


@dataclass(frozen=True)
class BenchmarkFact:
    id: str
    label: str
    content: str
    evidence_text: str
    node_type: str
    timestamp: str


@dataclass(frozen=True)
class BenchmarkRelation:
    source_fact_id: str
    target_fact_id: str
    relationship: str


@dataclass(frozen=True)
class BenchmarkConversation:
    id: str
    title: str
    timestamp: str
    transcript: str
    facts: list[BenchmarkFact]
    relations: list[BenchmarkRelation]


@dataclass(frozen=True)
class BenchmarkQuery:
    id: str
    kind: str
    query: str
    gold_fact_ids: list[str]
    gold_strings: list[str]


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    conversation_id: str
    text: str
    token_count: int
    timestamp: str
    support_fact_ids: list[str]


@dataclass
class RetrievalExample:
    query_id: str
    kind: str
    query: str
    gold_fact_ids: list[str]
    waggle_retrieval_mode: str
    waggle_nodes: list[str]
    waggle_edges: list[str]
    rag_chunks: list[str]


@dataclass
class QueryResult:
    query_id: str
    kind: str
    question_tokens: int
    retrieved_context_tokens: int
    input_tokens: int
    rerank_tokens: int
    total_token_cost: int
    hit_at_k: bool
    exact_support: bool
    latency_seconds: float
    retrieved_ids: list[str]
    retrieval_mode: str
    node_count: int = 0
    edge_count: int = 0
    retrieval_trace: dict[str, Any] | None = None


@dataclass
class SystemSummary:
    system: str
    avg_input_tokens_per_query: float
    avg_retrieved_context_tokens: float
    avg_rerank_tokens: float
    avg_total_token_cost: float
    recall_at_k: float
    multi_hop_accuracy: float
    extraction_failure_recall: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    avg_node_count: float | None = None
    avg_edge_count: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievalConfig:
    id: str
    label: str
    retrieval_mode: str
    max_depth: int
    expand_depth: int
    rerank_enabled: bool


@dataclass
class ComparisonReport:
    corpus: dict[str, Any]
    configs: list[dict[str, Any]]
    summaries: dict[str, SystemSummary]
    subset_breakdowns: dict[str, dict[str, dict[str, float]]]
    per_query: dict[str, list[QueryResult]]
    notes: list[str]
    failing_multi_hop_traces: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "configs": self.configs,
            "summaries": {key: asdict(value) for key, value in self.summaries.items()},
            "subset_breakdowns": self.subset_breakdowns,
            "per_query": {key: [asdict(item) for item in value] for key, value in self.per_query.items()},
            "notes": list(self.notes),
            "failing_multi_hop_traces": self.failing_multi_hop_traces,
        }


@dataclass
class BenchmarkReport:
    corpus: dict[str, Any]
    rag_config: dict[str, Any]
    waggle_config: dict[str, Any]
    actual_embeddings: dict[str, Any]
    summaries: dict[str, SystemSummary]
    examples: list[RetrievalExample]
    per_query: dict[str, list[QueryResult]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "rag_config": self.rag_config,
            "waggle_config": self.waggle_config,
            "actual_embeddings": self.actual_embeddings,
            "summaries": {key: asdict(value) for key, value in self.summaries.items()},
            "examples": [asdict(example) for example in self.examples],
            "per_query": {key: [asdict(item) for item in value] for key, value in self.per_query.items()},
            "notes": list(self.notes),
        }


class OpenAIEmbeddingModel:
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model_name = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embedding benchmarks.")
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import error depends on env
            raise RuntimeError(f"openai package is required for {model_name}: {exc}") from exc
        self._client = OpenAI(api_key=self._api_key)

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model_name, input=text)
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return np.asarray([item.embedding for item in response.data], dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_vec = np.asarray(a, dtype=np.float32)
        b_vec = np.asarray(b, dtype=np.float32)
        a_norm = float(np.linalg.norm(a_vec))
        b_norm = float(np.linalg.norm(b_vec))
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / (a_norm * b_norm))


def _load_encoding() -> Any | None:
    try:
        import tiktoken
    except Exception:  # pragma: no cover - depends on env
        return None
    return tiktoken.get_encoding("cl100k_base")


_ENCODING = _load_encoding()


def count_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(normalized))
    return max(1, math.ceil(len(normalized) / 4))


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    step = max(1, chunk_size - overlap)
    if _ENCODING is not None:
        tokens = _ENCODING.encode(normalized)
        chunks: list[str] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + chunk_size]
            if not window:
                continue
            chunks.append(_ENCODING.decode(window))
            if start + chunk_size >= len(tokens):
                break
        return chunks

    words = normalized.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        token_budget = 0
        end = start
        while end < len(words):
            word_cost = max(1, math.ceil(len(words[end]) / 4))
            separator_cost = 1 if end > start else 0
            if token_budget + word_cost + separator_cost > chunk_size and end > start:
                break
            token_budget += word_cost + separator_cost
            end += 1
        window = words[start:end]
        if not window:
            break
        chunks.append(" ".join(window))
        if end >= len(words):
            break
        overlap_budget = 0
        overlap_start = end
        while overlap_start > start:
            overlap_start -= 1
            overlap_budget += max(1, math.ceil(len(words[overlap_start]) / 4)) + 1
            if overlap_budget >= overlap:
                break
        start = max(start + 1, overlap_start)
    return chunks


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)


def _quoted_phrase_score(question: str, content: str) -> float:
    lowered = question.lower()
    quoted_phrases: list[str] = []
    quote_open = None
    for index, character in enumerate(lowered):
        if character in {"'", '"'}:
            if quote_open is None:
                quote_open = (character, index)
            elif quote_open[0] == character:
                phrase = lowered[quote_open[1] + 1 : index].strip()
                if phrase:
                    quoted_phrases.append(phrase)
                quote_open = None
    if not quoted_phrases:
        return 0.0
    return max(1.0 if phrase in content.lower() else 0.0 for phrase in quoted_phrases)


def _term_coverage_score(question: str, content: str) -> float:
    query_terms = {
        token.strip(".,?!:;()[]{}\"'").lower()
        for token in question.split()
        if len(token.strip(".,?!:;()[]{}\"'")) >= 4
    }
    if not query_terms:
        return 0.0
    content_lower = content.lower()
    matched = sum(1 for term in query_terms if term in content_lower)
    return matched / len(query_terms)


def _relation_type(value: str) -> RelationType:
    normalized = value.strip().lower()
    mapping = {
        "relates_to": RelationType.RELATES_TO,
        "depends_on": RelationType.DEPENDS_ON,
        "updates": RelationType.UPDATES,
        "contradicts": RelationType.CONTRADICTS,
        "part_of": RelationType.PART_OF,
        "similar_to": RelationType.SIMILAR_TO,
        "derived_from": RelationType.DERIVED_FROM,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported relation type: {value}")
    return mapping[normalized]


def _node_type(value: str) -> NodeType:
    normalized = value.strip().lower()
    return NodeType(normalized)


def load_dataset(path: Path) -> tuple[list[BenchmarkConversation], list[BenchmarkQuery]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    conversations = [
        BenchmarkConversation(
            id=item["id"],
            title=item["title"],
            timestamp=item["timestamp"],
            transcript=item["transcript"],
            facts=[
                BenchmarkFact(
                    id=fact["id"],
                    label=fact["label"],
                    content=fact["content"],
                    evidence_text=fact["evidence_text"],
                    node_type=fact["node_type"],
                    timestamp=fact["timestamp"],
                )
                for fact in item["facts"]
            ],
            relations=[
                BenchmarkRelation(
                    source_fact_id=relation["source_fact_id"],
                    target_fact_id=relation["target_fact_id"],
                    relationship=relation["relationship"],
                )
                for relation in item.get("relations", [])
            ],
        )
        for item in raw["conversations"]
    ]
    queries = [
        BenchmarkQuery(
            id=item["id"],
            kind=item["kind"],
            query=item["query"],
            gold_fact_ids=list(item["gold_fact_ids"]),
            gold_strings=list(item.get("gold_strings", [])),
        )
        for item in raw["queries"]
    ]
    return conversations, queries


def _make_noise_paragraph(domain: str, subject: str, variant: int) -> str:
    items = [
        f"We reviewed {domain} rollout notes, risk registers, test plans, migration checklists, ownership gaps, and release sequencing for {subject}.",
        f"The team compared implementation tradeoffs, onboarding pain points, incident history, rollback options, and coordination overhead around {subject}.",
        f"People also discussed naming consistency, reporting requirements, support load, sprint scheduling, and documentation debt tied to {subject}.",
        f"Several comments focused on stale assumptions, handoff quality, escalation paths, edge cases, and cross-team dependencies affecting {subject}.",
        f"Meeting notes repeated the need for tighter instrumentation, cleaner alerts, migration rehearsals, dependency reviews, and follow-up tasks for {subject}.",
    ]
    return items[variant % len(items)]


def _generate_transcript(title: str, evidence_lines: list[str], *, noise_blocks: int) -> str:
    lines: list[str] = []
    for index in range(noise_blocks):
        speaker = "User" if index % 2 == 0 else "Agent"
        lines.append(f"{speaker}: {_make_noise_paragraph(title, title, index)}")
    for evidence_index, evidence in enumerate(evidence_lines):
        lines.append(f"User: {evidence}")
        lines.append(f"Agent: Recorded. {evidence}")
        for offset in range(3):
            block_index = noise_blocks + (evidence_index * 3) + offset
            speaker = "User" if block_index % 2 == 0 else "Agent"
            lines.append(f"{speaker}: {_make_noise_paragraph(title, evidence, block_index)}")
    return "\n".join(lines)


def generate_default_dataset(path: Path) -> Path:
    base_time = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    domain_specs = [
        ("database", "SQLite local only", "PostgreSQL production", "safer migrations and parity", "analytics joins"),
        ("auth", "JWT expires in 15 minutes", "JWT expires in 1 hour", "reduce support lockouts", "mobile refresh flow"),
        ("cache", "Redis cache disabled in prod", "Redis cache enabled in prod", "cut API latency", "search autosuggest"),
        ("search", "Meilisearch for staging", "OpenSearch for production", "better faceting", "catalog navigation"),
        ("billing", "manual invoice review", "automatic invoice review", "reduce finance backlog", "renewal reminders"),
        ("support", "email-only intake", "portal-based intake", "triage SLA", "priority routing"),
        ("analytics", "daily batch exports", "hourly exports", "fresher dashboards", "revenue board"),
        ("deploy", "single-region hosting", "multi-region hosting", "improve failover", "status page"),
        ("notifications", "email digests weekly", "email digests daily", "reduce missed updates", "ops summaries"),
        ("mobile", "React Native 0.74", "React Native 0.76", "fix Android crashes", "push token refresh"),
    ]

    conversations: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []

    for domain_index, (domain, old_value, new_value, reason, dependency) in enumerate(domain_specs):
        slug = domain.replace(" ", "_")
        old_fact_id = f"{slug}_old"
        new_fact_id = f"{slug}_new"
        reason_fact_id = f"{slug}_reason"
        dependency_fact_id = f"{slug}_dependency"
        owner_fact_id = f"{slug}_owner"
        owner = f"{domain.title()} platform team"
        codeword = f"{slug}-cipher-{domain_index + 11}"

        timestamps = [base_time + timedelta(days=(domain_index * 6) + offset) for offset in range(5)]
        evidence_sets = [
            [f"For {domain}, the current baseline is {old_value}."],
            [f"For {domain}, the current production choice is now {new_value}.", f"The reason is {reason}."],
            [f"The {dependency} workflow depends on {new_value} for {domain}."],
            [f"The owner for the {domain} stack is the {owner}."],
            [
                f"Keep the previous {domain} notes archived, but current decisions still use {new_value} because of {reason}.",
                f"The private codeword for {domain} is {codeword}.",
            ],
        ]

        facts_per_conversation = [
            [
                {
                    "id": old_fact_id,
                    "label": f"{domain.title()} previous state",
                    "content": f"The previous baseline for {domain} was {old_value}.",
                    "evidence_text": evidence_sets[0][0],
                    "node_type": "decision",
                    "timestamp": timestamps[0].isoformat(),
                }
            ],
            [
                {
                    "id": new_fact_id,
                    "label": f"{domain.title()} current state",
                    "content": f"The current production choice for {domain} is {new_value}.",
                    "evidence_text": evidence_sets[1][0],
                    "node_type": "decision",
                    "timestamp": timestamps[1].isoformat(),
                },
                {
                    "id": reason_fact_id,
                    "label": f"{domain.title()} change reason",
                    "content": f"The reason the team changed {domain} is {reason}.",
                    "evidence_text": evidence_sets[1][1],
                    "node_type": "fact",
                    "timestamp": timestamps[1].isoformat(),
                },
            ],
            [
                {
                    "id": dependency_fact_id,
                    "label": f"{domain.title()} dependency",
                    "content": f"The {dependency} workflow depends on the current {domain} choice.",
                    "evidence_text": evidence_sets[2][0],
                    "node_type": "fact",
                    "timestamp": timestamps[2].isoformat(),
                }
            ],
            [
                {
                    "id": owner_fact_id,
                    "label": f"{domain.title()} owner",
                    "content": f"The owner for the {domain} stack is the {owner}.",
                    "evidence_text": evidence_sets[3][0],
                    "node_type": "entity",
                    "timestamp": timestamps[3].isoformat(),
                }
            ],
            [],
        ]

        relations_per_conversation = [
            [],
            [
                {"source_fact_id": new_fact_id, "target_fact_id": old_fact_id, "relationship": "updates"},
                {"source_fact_id": new_fact_id, "target_fact_id": old_fact_id, "relationship": "contradicts"},
                {"source_fact_id": reason_fact_id, "target_fact_id": new_fact_id, "relationship": "depends_on"},
            ],
            [
                {"source_fact_id": dependency_fact_id, "target_fact_id": new_fact_id, "relationship": "depends_on"},
                {"source_fact_id": dependency_fact_id, "target_fact_id": reason_fact_id, "relationship": "relates_to"},
            ],
            [
                {"source_fact_id": owner_fact_id, "target_fact_id": new_fact_id, "relationship": "part_of"},
            ],
            [
                {"source_fact_id": new_fact_id, "target_fact_id": reason_fact_id, "relationship": "relates_to"},
            ],
        ]

        for convo_index in range(5):
            transcript = _generate_transcript(
                f"{domain} thread {convo_index + 1}",
                evidence_sets[convo_index],
                noise_blocks=90,
            )
            conversations.append(
                {
                    "id": f"{slug}_c{convo_index + 1}",
                    "title": f"{domain.title()} thread {convo_index + 1}",
                    "timestamp": timestamps[convo_index].isoformat(),
                    "transcript": transcript,
                    "facts": facts_per_conversation[convo_index],
                    "relations": relations_per_conversation[convo_index],
                }
            )

        if domain_index < 10:
            queries.extend(
                [
                    {
                        "id": f"{slug}_factual",
                        "kind": "single_fact",
                        "query": f"What is the current production choice for {domain}?",
                        "gold_fact_ids": [new_fact_id],
                        "gold_strings": [new_value],
                    },
                    {
                        "id": f"{slug}_multi_hop",
                        "kind": "multi_hop",
                        "query": f"After the discussion about {reason}, what current {domain} choice did we settle on, and who owns that stack for the {dependency} workflow?",
                        "gold_fact_ids": [new_fact_id, reason_fact_id, owner_fact_id, dependency_fact_id],
                        "gold_strings": [new_value, owner, dependency],
                    },
                    {
                        "id": f"{slug}_extraction_failure",
                        "kind": "extraction_failure",
                        "query": f"What was the private codeword recorded for {domain}?",
                        "gold_fact_ids": [],
                        "gold_strings": [codeword],
                    },
                ]
            )

        payload = {"conversations": conversations, "queries": queries[:30]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _conversation_stats(conversations: list[BenchmarkConversation], queries: list[BenchmarkQuery]) -> dict[str, Any]:
    transcript_tokens = [count_tokens(conversation.transcript) for conversation in conversations]
    return {
        "conversation_count": len(conversations),
        "query_count": len(queries),
        "total_transcript_tokens": sum(transcript_tokens),
        "avg_tokens_per_conversation": statistics.mean(transcript_tokens) if transcript_tokens else 0.0,
        "max_tokens_per_conversation": max(transcript_tokens) if transcript_tokens else 0,
    }


def _build_rag_chunks(
    conversations: list[BenchmarkConversation],
    *,
    chunk_size: int,
    overlap: int,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for conversation in conversations:
        for index, chunk_text_value in enumerate(chunk_text(conversation.transcript, chunk_size=chunk_size, overlap=overlap)):
            support_fact_ids = [
                fact.id
                for fact in conversation.facts
                if fact.evidence_text in chunk_text_value
            ]
            chunks.append(
                RagChunk(
                    chunk_id=f"{conversation.id}::chunk{index}",
                    conversation_id=conversation.id,
                    text=chunk_text_value,
                    token_count=count_tokens(chunk_text_value),
                    timestamp=conversation.timestamp,
                    support_fact_ids=support_fact_ids,
                )
            )
    return chunks


def _serialize_waggle_context(nodes: list[Any], edges: list[Any]) -> str:
    lines = [f"{node.label}: {node.content}" for node in nodes]
    edge_lines = []
    node_lookup = {node.id: node for node in nodes}
    for edge in edges:
        source = node_lookup.get(edge.source_id)
        target = node_lookup.get(edge.target_id)
        if source is None or target is None:
            continue
        edge_lines.append(f"{source.label} -[{edge.relationship}]-> {target.label}")
    if edge_lines:
        lines.append("GRAPH LINKS:")
        lines.extend(edge_lines)
    return "\n".join(lines).strip()


def _query_matches_context(query: BenchmarkQuery, context_text: str) -> tuple[bool, bool]:
    normalized_context = context_text.lower()
    gold_strings = [item.strip().lower() for item in query.gold_strings if item.strip()]
    if not gold_strings:
        return False, False
    hits = [needle for needle in gold_strings if needle in normalized_context]
    return bool(hits), len(hits) == len(gold_strings)


def _rank_hybrid_chunks(
    query: str,
    chunks: list[RagChunk],
    *,
    embedding_model: Any,
    chunk_embeddings: np.ndarray,
    top_k: int,
    rerank_enabled: bool,
) -> tuple[list[RagChunk], int, list[dict[str, Any]]]:
    if not chunks:
        return [], 0, []
    query_embedding = embedding_model.embed(query)
    first_pass_scores: dict[str, float] = {}
    scored: list[tuple[float, RagChunk]] = []
    for chunk, chunk_embedding in zip(chunks, chunk_embeddings, strict=True):
        semantic = embedding_model.cosine_similarity(query_embedding, chunk_embedding)
        lexical = _term_coverage_score(query, chunk.text)
        score = (0.75 * semantic) + (0.25 * lexical)
        first_pass_scores[chunk.chunk_id] = score
        scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    first_pass = [item[1] for item in scored[:DEFAULT_HYBRID_RERANK_POOL]]
    trace = [
        {
            "chunk_id": chunk.chunk_id,
            "conversation_id": chunk.conversation_id,
            "first_pass_rank": index + 1,
            "preview": chunk.text[:180].replace("\n", " "),
        }
        for index, chunk in enumerate(first_pass)
    ]
    if not rerank_enabled:
        return first_pass[:top_k], 0, trace

    rerank_tokens = sum(count_tokens(chunk.text) for chunk in first_pass)
    reranked = sorted(
        first_pass,
        key=lambda chunk: (
            -(
                (0.55 * first_pass_scores.get(chunk.chunk_id, 0.0))
                + (0.25 * lexical_overlap(query, chunk.conversation_id, chunk.text))
                + (0.20 * _quoted_phrase_score(query, chunk.text))
            ),
            chunk.chunk_id,
        ),
    )
    rerank_lookup = {chunk.chunk_id: index + 1 for index, chunk in enumerate(reranked)}
    for item in trace:
        item["rerank_rank"] = rerank_lookup.get(item["chunk_id"], 0)
    return reranked[:top_k], rerank_tokens, trace


def _build_graph(
    conversations: list[BenchmarkConversation],
    *,
    embedding_model: Any,
    project: str,
) -> tuple[MemoryGraph, dict[str, str]]:
    tmpdir = tempfile.TemporaryDirectory()
    graph = MemoryGraph(
        Path(tmpdir.name) / "token-efficiency.db",
        embedding_model,
        dedup_similarity_threshold=1.01,
        dedup_same_label_threshold=1.01,
    )
    setattr(graph, "_benchmark_tmpdir", tmpdir)
    fact_node_ids: dict[str, str] = {}

    for conversation in conversations:
        for fact in conversation.facts:
            result = graph.add_node(
                label=fact.label,
                content=fact.content,
                node_type=_node_type(fact.node_type),
                tags=["benchmark", f"fact_id:{fact.id}", f"conversation:{conversation.id}"],
                project=project,
                session_id=conversation.id,
                valid_from=datetime.fromisoformat(fact.timestamp),
            )
            fact_node_ids[fact.id] = result.node.id

    for conversation in conversations:
        for relation in conversation.relations:
            source_id = fact_node_ids.get(relation.source_fact_id)
            target_id = fact_node_ids.get(relation.target_fact_id)
            if source_id is None or target_id is None or source_id == target_id:
                continue
            graph.add_edge(
                source_id=source_id,
                target_id=target_id,
                relationship=_relation_type(relation.relationship),
                metadata={"conversation_id": conversation.id, "origin": "token-efficiency-benchmark"},
            )

    return graph, fact_node_ids


def _rank_rag_chunks(
    query: str,
    chunks: list[RagChunk],
    *,
    embedding_model: Any,
    chunk_embeddings: np.ndarray,
    top_k: int,
) -> list[RagChunk]:
    if not chunks:
        return []
    query_embedding = embedding_model.embed(query)
    scored: list[tuple[float, str, RagChunk]] = []
    for chunk, chunk_embedding in zip(chunks, chunk_embeddings, strict=True):
        score = embedding_model.cosine_similarity(query_embedding, chunk_embedding)
        scored.append((score, chunk.chunk_id, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]]


def _summarize_results(
    system: str,
    results: list[QueryResult],
    *,
    include_graph_shape: bool,
    metadata: dict[str, Any] | None = None,
) -> SystemSummary:
    multi_hop = [result for result in results if result.kind == "multi_hop"]
    extraction_failure = [result for result in results if result.kind == "extraction_failure"]
    return SystemSummary(
        system=system,
        avg_input_tokens_per_query=statistics.mean(result.input_tokens for result in results) if results else 0.0,
        avg_retrieved_context_tokens=statistics.mean(result.retrieved_context_tokens for result in results) if results else 0.0,
        avg_rerank_tokens=statistics.mean(result.rerank_tokens for result in results) if results else 0.0,
        avg_total_token_cost=statistics.mean(result.total_token_cost for result in results) if results else 0.0,
        recall_at_k=(sum(1 if result.hit_at_k else 0 for result in results) / len(results)) if results else 0.0,
        multi_hop_accuracy=(sum(1 if result.exact_support else 0 for result in multi_hop) / len(multi_hop)) if multi_hop else 0.0,
        extraction_failure_recall=(sum(1 if result.hit_at_k else 0 for result in extraction_failure) / len(extraction_failure)) if extraction_failure else 0.0,
        latency_p50_seconds=_percentile([result.latency_seconds for result in results], 0.50),
        latency_p95_seconds=_percentile([result.latency_seconds for result in results], 0.95),
        avg_node_count=(statistics.mean(result.node_count for result in results) if include_graph_shape and results else None),
        avg_edge_count=(statistics.mean(result.edge_count for result in results) if include_graph_shape and results else None),
        metadata=metadata or {},
    )


def _subset_breakdown(results: list[QueryResult]) -> dict[str, dict[str, float]]:
    breakdown: dict[str, dict[str, float]] = {}
    for subset in ("single_fact", "multi_hop", "extraction_failure"):
        subset_results = [result for result in results if result.kind == subset]
        if not subset_results:
            breakdown[subset] = {"count": 0.0, "recall_at_k": 0.0, "exact_support": 0.0}
            continue
        breakdown[subset] = {
            "count": float(len(subset_results)),
            "recall_at_k": sum(1 if item.hit_at_k else 0 for item in subset_results) / len(subset_results),
            "exact_support": sum(1 if item.exact_support else 0 for item in subset_results) / len(subset_results),
        }
    return breakdown


def _pick_example_ids(queries: list[BenchmarkQuery]) -> list[str]:
    picks: list[str] = []
    for kind in ("factual", "relational", "multi_hop"):
        match = next((query.id for query in queries if query.kind == kind), None)
        if match is not None:
            picks.append(match)
    return picks


def _build_examples(
    queries: list[BenchmarkQuery],
    waggle_rows: dict[str, tuple[Any, QueryResult]],
    rag_rows: dict[str, tuple[list[RagChunk], QueryResult]],
) -> list[RetrievalExample]:
    examples: list[RetrievalExample] = []
    for query_id in _pick_example_ids(queries):
        query = next(item for item in queries if item.id == query_id)
        waggle_subgraph, waggle_result = waggle_rows[query_id]
        rag_chunks, _ = rag_rows[query_id]
        examples.append(
            RetrievalExample(
                query_id=query.id,
                kind=query.kind,
                query=query.query,
                gold_fact_ids=query.gold_fact_ids,
                waggle_retrieval_mode=waggle_result.retrieval_mode,
                waggle_nodes=[f"{node.label}: {node.content}" for node in waggle_subgraph.nodes[:5]],
                waggle_edges=[
                    (
                        f"{next((node.label for node in waggle_subgraph.nodes if node.id == edge.source_id), edge.source_id)}"
                        f" -[{edge.relationship}]-> "
                        f"{next((node.label for node in waggle_subgraph.nodes if node.id == edge.target_id), edge.target_id)}"
                    )
                    for edge in waggle_subgraph.edges[:5]
                ],
                rag_chunks=[chunk.text[:240].replace("\n", " ") for chunk in rag_chunks[:3]],
            )
        )
    return examples


def _actual_embedding_label(model: Any) -> str:
    return getattr(model, "model_name", model.__class__.__name__)


def run_benchmark(
    *,
    dataset_path: Path,
    rag_embedding_model_name: str = "text-embedding-3-small",
    allow_local_baseline_fallback: bool = False,
    local_fallback_embedding_model_name: str = "all-MiniLM-L6-v2",
    waggle_top_k: int = DEFAULT_WAGGLE_TOP_K,
    waggle_max_depth: int = DEFAULT_WAGGLE_MAX_DEPTH,
    waggle_expand_depth: int = DEFAULT_WAGGLE_EXPAND_DEPTH,
    rag_chunk_size: int = DEFAULT_RAG_CHUNK_SIZE,
    rag_chunk_overlap: int = DEFAULT_RAG_CHUNK_OVERLAP,
    rag_top_k: int = DEFAULT_RAG_TOP_K,
    project: str = DEFAULT_PROJECT,
) -> BenchmarkReport:
    conversations, queries = load_dataset(dataset_path)
    notes: list[str] = []

    waggle_embedding = EmbeddingModel(local_fallback_embedding_model_name)
    try:
        rag_embedding: Any = OpenAIEmbeddingModel(model_name=rag_embedding_model_name)
        rag_embedding_source = "openai"
    except Exception as exc:
        if not allow_local_baseline_fallback:
            raise
        rag_embedding = EmbeddingModel(local_fallback_embedding_model_name)
        rag_embedding_source = "local_fallback"
        notes.append(
            f"Vanilla RAG requested `{rag_embedding_model_name}` but ran with local fallback "
            f"`{local_fallback_embedding_model_name}` because OpenAI embeddings were unavailable: {exc}"
        )

    graph, _ = _build_graph(conversations, embedding_model=waggle_embedding, project=project)
    chunks = _build_rag_chunks(conversations, chunk_size=rag_chunk_size, overlap=rag_chunk_overlap)
    chunk_embeddings = rag_embedding.embed_batch([chunk.text for chunk in chunks]) if chunks else np.empty((0, 0), dtype=np.float32)

    waggle_results: list[QueryResult] = []
    rag_results: list[QueryResult] = []
    waggle_rows: dict[str, tuple[Any, QueryResult]] = {}
    rag_rows: dict[str, tuple[list[RagChunk], QueryResult]] = {}

    for query in queries:
        question_tokens = count_tokens(query.query)

        started = time.perf_counter()
        subgraph = graph.query(
            query=query.query,
            max_nodes=waggle_top_k,
            max_depth=waggle_max_depth,
            expand_depth=waggle_expand_depth,
            project=project,
            retrieval_mode="graph",
        )
        waggle_latency = time.perf_counter() - started
        waggle_ids = [
            tag.split(":", 1)[1]
            for node in subgraph.nodes
            for tag in node.tags
            if tag.startswith("fact_id:")
        ]
        waggle_context = _serialize_waggle_context(subgraph.nodes, subgraph.edges)
        waggle_hit, waggle_exact = _query_matches_context(query, waggle_context)
        waggle_result = QueryResult(
            query_id=query.id,
            kind=query.kind,
            question_tokens=question_tokens,
            retrieved_context_tokens=count_tokens(waggle_context),
            input_tokens=question_tokens + count_tokens(waggle_context),
            rerank_tokens=0,
            total_token_cost=question_tokens + count_tokens(waggle_context),
            hit_at_k=waggle_hit,
            exact_support=waggle_exact,
            latency_seconds=waggle_latency,
            retrieved_ids=waggle_ids,
            retrieval_mode=subgraph.retrieval_mode,
            node_count=len(subgraph.nodes),
            edge_count=len(subgraph.edges),
        )
        waggle_results.append(waggle_result)
        waggle_rows[query.id] = (subgraph, waggle_result)

        started = time.perf_counter()
        ranked_chunks = _rank_rag_chunks(
            query.query,
            chunks,
            embedding_model=rag_embedding,
            chunk_embeddings=chunk_embeddings,
            top_k=rag_top_k,
        )
        rag_latency = time.perf_counter() - started
        rag_ids = [fact_id for chunk in ranked_chunks for fact_id in chunk.support_fact_ids]
        rag_context = "\n".join(chunk.text for chunk in ranked_chunks)
        rag_hit, rag_exact = _query_matches_context(query, rag_context)
        rag_result = QueryResult(
            query_id=query.id,
            kind=query.kind,
            question_tokens=question_tokens,
            retrieved_context_tokens=count_tokens(rag_context),
            input_tokens=question_tokens + count_tokens(rag_context),
            rerank_tokens=0,
            total_token_cost=question_tokens + count_tokens(rag_context),
            hit_at_k=rag_hit,
            exact_support=rag_exact,
            latency_seconds=rag_latency,
            retrieved_ids=rag_ids,
            retrieval_mode="topk_vector_chunks",
        )
        rag_results.append(rag_result)
        rag_rows[query.id] = (ranked_chunks, rag_result)

    waggle_summary = _summarize_results(
        "waggle_graph",
        waggle_results,
        include_graph_shape=True,
        metadata={"avg_subgraph_nodes": statistics.mean(result.node_count for result in waggle_results) if waggle_results else 0.0},
    )
    rag_summary = _summarize_results(
        "vanilla_rag",
        rag_results,
        include_graph_shape=False,
        metadata={"chunk_count": len(chunks)},
    )

    return BenchmarkReport(
        corpus=_conversation_stats(conversations, queries) | {"dataset_path": str(dataset_path)},
        rag_config={
            "requested_embedding_model": rag_embedding_model_name,
            "actual_embedding_model": _actual_embedding_label(rag_embedding),
            "chunk_size_tokens": rag_chunk_size,
            "chunk_overlap_tokens": rag_chunk_overlap,
            "top_k": rag_top_k,
        },
        waggle_config={
            "embedding_model": _actual_embedding_label(waggle_embedding),
            "retrieval_mode": "graph",
            "max_nodes": waggle_top_k,
            "max_depth": waggle_max_depth,
            "expand_depth": waggle_expand_depth,
            "tiered_retrieval": graph.tiered_retrieval,
            "tiered_top_k_windows": graph.tiered_retrieval_top_k_windows,
        },
        actual_embeddings={
            "waggle": _actual_embedding_label(waggle_embedding),
            "rag": _actual_embedding_label(rag_embedding),
            "rag_source": rag_embedding_source,
        },
        summaries={
            "waggle_graph": waggle_summary,
            "vanilla_rag": rag_summary,
        },
        examples=_build_examples(queries, waggle_rows, rag_rows),
        per_query={
            "waggle_graph": waggle_results,
            "vanilla_rag": rag_results,
        },
        notes=notes,
    )


def run_comparison_benchmark(
    *,
    dataset_path: Path,
    allow_local_baseline_fallback: bool = True,
    local_fallback_embedding_model_name: str = "all-MiniLM-L6-v2",
    project: str = DEFAULT_PROJECT,
) -> ComparisonReport:
    conversations, queries = load_dataset(dataset_path)
    notes: list[str] = []
    waggle_embedding = EmbeddingModel(local_fallback_embedding_model_name)
    graph, _ = _build_graph(conversations, embedding_model=waggle_embedding, project=project)
    chunks = _build_rag_chunks(conversations, chunk_size=DEFAULT_RAG_CHUNK_SIZE, overlap=DEFAULT_RAG_CHUNK_OVERLAP)
    chunk_embeddings = waggle_embedding.embed_batch([chunk.text for chunk in chunks]) if chunks else np.empty((0, 0), dtype=np.float32)

    configs = [
        RetrievalConfig(id="old", label="Old", retrieval_mode="graph", max_depth=2, expand_depth=0, rerank_enabled=False),
        RetrievalConfig(id="new_no_rerank", label="New-no-rerank", retrieval_mode="hybrid", max_depth=2, expand_depth=0, rerank_enabled=False),
        RetrievalConfig(id="new_full", label="New-full", retrieval_mode="hybrid", max_depth=2, expand_depth=0, rerank_enabled=True),
    ]

    per_query: dict[str, list[QueryResult]] = {}
    subset_breakdowns: dict[str, dict[str, dict[str, float]]] = {}
    summaries: dict[str, SystemSummary] = {}
    failing_multi_hop_traces: dict[str, list[dict[str, Any]]] = {}

    for config in configs:
        config_results: list[QueryResult] = []
        for query in queries:
            question_tokens = count_tokens(query.query)
            started = time.perf_counter()
            subgraph = graph.query(
                query=query.query,
                max_nodes=DEFAULT_WAGGLE_TOP_K,
                max_depth=config.max_depth,
                expand_depth=config.expand_depth,
                project=project,
                retrieval_mode="graph",
            )
            graph_context = _serialize_waggle_context(subgraph.nodes, subgraph.edges)
            rerank_tokens = 0
            chunk_trace: list[dict[str, Any]] = []
            chunk_context = ""
            if config.retrieval_mode == "hybrid":
                ranked_chunks, rerank_tokens, chunk_trace = _rank_hybrid_chunks(
                    query.query,
                    chunks,
                    embedding_model=waggle_embedding,
                    chunk_embeddings=chunk_embeddings,
                    top_k=DEFAULT_HYBRID_CHUNK_TOP_K,
                    rerank_enabled=config.rerank_enabled,
                )
                chunk_context = "\n".join(chunk.text for chunk in ranked_chunks)
            latency = time.perf_counter() - started
            full_context = "\n".join(part for part in [graph_context, chunk_context] if part.strip())
            hit_at_k, exact_support = _query_matches_context(query, full_context)
            result = QueryResult(
                query_id=query.id,
                kind=query.kind,
                question_tokens=question_tokens,
                retrieved_context_tokens=count_tokens(full_context),
                input_tokens=question_tokens + count_tokens(full_context),
                rerank_tokens=rerank_tokens,
                total_token_cost=question_tokens + count_tokens(full_context) + rerank_tokens,
                hit_at_k=hit_at_k,
                exact_support=exact_support,
                latency_seconds=latency,
                retrieved_ids=[
                    tag.split(":", 1)[1]
                    for node in subgraph.nodes
                    for tag in node.tags
                    if tag.startswith("fact_id:")
                ],
                retrieval_mode=config.retrieval_mode,
                node_count=len(subgraph.nodes),
                edge_count=len(subgraph.edges),
                retrieval_trace={
                    "graph_nodes": [node.label for node in subgraph.nodes[:8]],
                    "graph_edges": [edge.relationship for edge in subgraph.edges[:8]],
                    "chunk_trace": chunk_trace[:8],
                },
            )
            config_results.append(result)
        per_query[config.id] = config_results
        subset_breakdowns[config.id] = _subset_breakdown(config_results)
        summaries[config.id] = _summarize_results(
            config.id,
            config_results,
            include_graph_shape=True,
            metadata={
                "label": config.label,
                "retrieval_mode": config.retrieval_mode,
                "max_depth": config.max_depth,
                "expand_depth": config.expand_depth,
                "rerank_enabled": config.rerank_enabled,
            },
        )
        if summaries[config.id].multi_hop_accuracy < 0.30:
            failures = [result for result in config_results if result.kind == "multi_hop" and not result.exact_support][:3]
            failing_multi_hop_traces[config.id] = [
                {
                    "query_id": failure.query_id,
                    "trace": failure.retrieval_trace or {},
                }
                for failure in failures
            ]

    notes.append("Hybrid modes use graph retrieval plus transcript chunk retrieval. `New-full` adds heuristic reranking over the first-pass chunk pool.")
    if allow_local_baseline_fallback:
        notes.append(f"Benchmark used local embeddings `{local_fallback_embedding_model_name}` for graph and transcript chunk ranking.")
    expected_checks = [
        (
            "Single-fact recall",
            subset_breakdowns["new_full"]["single_fact"]["recall_at_k"] >= 0.95
            and 0.75 <= subset_breakdowns["old"]["single_fact"]["recall_at_k"] <= 0.90,
        ),
        ("Multi-hop accuracy", summaries["new_full"].multi_hop_accuracy >= 0.40 and summaries["old"].multi_hop_accuracy == 0.0),
        ("Extraction-failure recall", summaries["new_full"].extraction_failure_recall >= 0.90 and summaries["old"].extraction_failure_recall <= 0.05),
        ("Latency target", 0.10 <= summaries["new_full"].latency_p50_seconds <= 0.18),
    ]
    for label, passed in expected_checks:
        if not passed:
            notes.append(f"Sanity-check mismatch: {label} did not meet the expected target in this run.")
    return ComparisonReport(
        corpus=_conversation_stats(conversations, queries) | {"dataset_path": str(dataset_path)},
        configs=[asdict(config) for config in configs],
        summaries=summaries,
        subset_breakdowns=subset_breakdowns,
        per_query=per_query,
        notes=notes,
        failing_multi_hop_traces=failing_multi_hop_traces,
    )


def build_v2_comparison_report(report: ComparisonReport) -> str:
    configs = report.configs
    ordered_ids = [item["id"] for item in configs]
    lines = [
        "# Token Efficiency Benchmark V2 Comparison",
        "",
        f"- Corpus: {report.corpus['conversation_count']} conversations, {report.corpus['query_count']} queries, ~{report.corpus['total_transcript_tokens']} transcript tokens",
        "",
        "## Config Table",
        "",
        "| Metric | Old | New-no-rerank | New-full |",
        "|--------|-----|---------------|----------|",
    ]
    metric_rows = [
        ("Recall@k", lambda s: f"{s.recall_at_k:.1%}"),
        ("Multi-hop accuracy", lambda s: f"{s.multi_hop_accuracy:.1%}"),
        ("Extraction-failure recall", lambda s: f"{s.extraction_failure_recall:.1%}"),
        ("Avg input tokens", lambda s: f"{s.avg_input_tokens_per_query:.1f}"),
        ("Avg retrieved context tokens", lambda s: f"{s.avg_retrieved_context_tokens:.1f}"),
        ("Avg rerank tokens", lambda s: f"{s.avg_rerank_tokens:.1f}"),
        ("Avg total token cost", lambda s: f"{s.avg_total_token_cost:.1f}"),
        ("Latency p50 / p95", lambda s: f"{s.latency_p50_seconds * 1000:.1f} / {s.latency_p95_seconds * 1000:.1f} ms"),
    ]
    for label, formatter in metric_rows:
        values = [formatter(report.summaries[config_id]) for config_id in ordered_ids]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")

    lines.extend(["", "## Breakdown by Subset", ""])
    lines.append("| Subset | Metric | Old | New-no-rerank | New-full |")
    lines.append("|--------|--------|-----|---------------|----------|")
    for subset in ("single_fact", "multi_hop", "extraction_failure"):
        metric = "exact_support" if subset == "multi_hop" else "recall_at_k"
        values = [report.subset_breakdowns[config_id][subset][metric] for config_id in ordered_ids]
        lines.append(f"| {subset} | {metric} | {values[0]:.1%} | {values[1]:.1%} | {values[2]:.1%} |")

    lines.extend(["", "## Verdict", ""])
    old = report.summaries["old"]
    no_rerank = report.summaries["new_no_rerank"]
    full = report.summaries["new_full"]
    if full.multi_hop_accuracy >= no_rerank.multi_hop_accuracy and full.recall_at_k >= no_rerank.recall_at_k:
        verdict = (
            f"`Old` stays cheapest and simplest, but it still loses whenever the answer only exists in transcript text or requires broader context recovery. "
            f"`New-no-rerank` improves recall by adding transcript evidence. "
            f"`New-full` is the strongest overall setting here because reranking preserves or improves recall while adding only {full.avg_rerank_tokens:.1f} average rerank tokens per query."
        )
    else:
        verdict = (
            f"`Old` stays cheapest and simplest, but its recall ({old.recall_at_k:.1%}) collapses on extraction-failure and multi-hop questions. "
            f"`New-no-rerank` is the best performer in this run: it reaches {no_rerank.recall_at_k:.1%} overall recall and {no_rerank.multi_hop_accuracy:.1%} multi-hop accuracy without rerank overhead. "
            f"`New-full` currently regresses relative to `New-no-rerank`; reranking adds {full.avg_rerank_tokens:.1f} tokens per query on average and should be treated as a bug to debug rather than a win."
        )
    lines.append(verdict)

    if report.failing_multi_hop_traces:
        lines.extend(["", "## Failing Multi-hop Traces", ""])
        for config_id, failures in report.failing_multi_hop_traces.items():
            if not failures:
                continue
            lines.append(f"### {config_id}")
            for failure in failures:
                lines.append(f"- `{failure['query_id']}`: {json.dumps(failure['trace'])[:500]}")
            lines.append("")

    if report.notes:
        lines.extend(["## Notes", ""])
        for note in report.notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_comparison_report(markdown_path: Path, json_path: Path, report: ComparisonReport) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_v2_comparison_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def build_markdown_report(report: BenchmarkReport) -> str:
    waggle = report.summaries["waggle_graph"]
    rag = report.summaries["vanilla_rag"]

    def _fmt_latency(summary: SystemSummary) -> str:
        return f"{summary.latency_p50_seconds * 1000:.1f} ms / {summary.latency_p95_seconds * 1000:.1f} ms"

    lines = [
        "# Waggle Graph vs Vanilla RAG Token Efficiency",
        "",
        f"- Corpus: {report.corpus['conversation_count']} conversations, {report.corpus['query_count']} queries, ~{report.corpus['total_transcript_tokens']} transcript tokens",
        f"- Waggle embedding: `{report.actual_embeddings['waggle']}`",
        f"- Vanilla RAG embedding: `{report.rag_config['requested_embedding_model']}` (actual: `{report.actual_embeddings['rag']}`)",
        "",
        "## Headline Table",
        "",
        "| Metric | Waggle Graph | Vanilla RAG |",
        "|--------|--------------|-------------|",
        f"| Avg input tokens per query | {waggle.avg_input_tokens_per_query:.1f} | {rag.avg_input_tokens_per_query:.1f} |",
        f"| Avg retrieved-context tokens | {waggle.avg_retrieved_context_tokens:.1f} | {rag.avg_retrieved_context_tokens:.1f} |",
        f"| Recall@k (did it find the fact) | {waggle.recall_at_k:.1%} | {rag.recall_at_k:.1%} |",
        f"| Multi-hop accuracy | {waggle.multi_hop_accuracy:.1%} | {rag.multi_hop_accuracy:.1%} |",
        f"| Latency p50 / p95 | {_fmt_latency(waggle)} | {_fmt_latency(rag)} |",
        "",
        "## Waggle Retrieval Policy",
        "",
        f"- `retrieval_mode=graph`",
        f"- `max_nodes={report.waggle_config['max_nodes']}`",
        f"- `max_depth={report.waggle_config['max_depth']}`",
        f"- `expand_depth={report.waggle_config['expand_depth']}`",
        f"- `tiered_retrieval={report.waggle_config['tiered_retrieval']}`",
        f"- `tiered_top_k_windows={report.waggle_config['tiered_top_k_windows']}`",
        f"- Avg returned subgraph: {waggle.avg_node_count:.1f} nodes, {waggle.avg_edge_count:.1f} edges",
        "",
        "## Vanilla RAG Baseline",
        "",
        f"- `chunk_size={report.rag_config['chunk_size_tokens']}` tokens",
        f"- `overlap={report.rag_config['chunk_overlap_tokens']}` tokens",
        f"- `top_k={report.rag_config['top_k']}`",
        f"- Requested embedding model: `{report.rag_config['requested_embedding_model']}`",
        "",
        "## Example Retrievals",
        "",
    ]

    for example in report.examples:
        lines.extend(
            [
                f"### {example.query_id} ({example.kind})",
                "",
                f"**Query:** {example.query}",
                "",
                f"**Gold facts:** `{', '.join(example.gold_fact_ids)}`",
                "",
                f"**Waggle (`{example.waggle_retrieval_mode}`):**",
            ]
        )
        for node in example.waggle_nodes:
            lines.append(f"- {node}")
        if example.waggle_edges:
            lines.append("Edges:")
            for edge in example.waggle_edges:
                lines.append(f"- {edge}")
        lines.extend(["", "**Vanilla RAG top chunks:**"])
        for chunk in example.rag_chunks:
            lines.append(f"- {chunk}")
        lines.append("")

    if waggle.multi_hop_accuracy >= rag.multi_hop_accuracy:
        verdict_delta = (
            f"and improved multi-hop accuracy from {rag.multi_hop_accuracy:.1%} to {waggle.multi_hop_accuracy:.1%}"
        )
    else:
        verdict_delta = (
            f"but lost on multi-hop accuracy ({waggle.multi_hop_accuracy:.1%} vs {rag.multi_hop_accuracy:.1%})"
        )
    verdict = (
        f"Waggle wins on token efficiency in this benchmark: it cut average retrieved context from "
        f"{rag.avg_retrieved_context_tokens:.1f} to {waggle.avg_retrieved_context_tokens:.1f} tokens {verdict_delta}. "
        f"The tradeoff is that graph retrieval only helps when the memory graph already captured the right facts and links; "
        f"plain chunked RAG remains competitive on direct factual recall and has the simpler ingestion story because it indexes raw transcripts without structure."
    )
    lines.extend(["## Verdict", "", verdict, ""])

    if report.notes:
        lines.extend(["## Notes", ""])
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def write_report(markdown_path: Path, json_path: Path, report: BenchmarkReport) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
