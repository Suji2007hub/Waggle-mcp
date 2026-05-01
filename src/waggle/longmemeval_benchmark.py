from __future__ import annotations

import argparse
import json
import time
from hashlib import sha256
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from waggle.benchmark_harness import BenchmarkRuntimeError
from waggle.embeddings import EmbeddingModel
from waggle.intelligence import infer_temporal_hints, lexical_overlap


@dataclass
class LongMemEvalCaseResult:
    query_id: str
    question: str
    correct_session_ids: list[str]
    retrieved_session_ids: list[str]
    hit_at_5: bool
    exact_at_5: bool
    exact_at_10: bool
    exact_at_20: bool


@dataclass(frozen=True)
class LongMemEvalCardinalityMetrics:
    count: int
    recall_at_5: float
    exact_at_5: float
    exact_at_10: float
    exact_at_20: float


@dataclass(frozen=True)
class LongMemEvalDivergenceExample:
    case_id: str
    gold_set: list[str]
    retrieved_top5: list[str]
    missing: list[str]


@dataclass(frozen=True)
class PreparedLongMemEvalSession:
    session_id: str
    label: str
    content: str
    updated_at: datetime


@dataclass(frozen=True)
class PreparedLongMemEvalChunk:
    session_id: str
    chunk_id: str
    content: str


@dataclass
class PreparedLongMemEvalEntry:
    query_id: str
    question: str
    correct_session_ids: list[str]
    sessions: list[PreparedLongMemEvalSession]
    embedding_matrix: np.ndarray
    chunks: list[PreparedLongMemEvalChunk]


@dataclass
class LongMemEvalReport:
    dataset_path: str
    mode: str
    case_count: int
    cache_status: str
    cache_path: str
    prepared_entry_count: int
    prepared_session_count: int
    cache_key: str
    r_at_5: float
    exact_at_5: float
    by_gold_cardinality: dict[str, LongMemEvalCardinalityMetrics]
    divergence_examples: list[LongMemEvalDivergenceExample]
    per_case: list[LongMemEvalCaseResult]
    split_type: str = "full"
    split_seed: int | None = None
    profile: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "mode": self.mode,
            "case_count": self.case_count,
            "split_type": self.split_type,
            "split_seed": self.split_seed,
            "cache_status": self.cache_status,
            "cache_path": self.cache_path,
            "prepared_entry_count": self.prepared_entry_count,
            "prepared_session_count": self.prepared_session_count,
            "cache_key": self.cache_key,
            "r_at_5": self.r_at_5,
            "exact_at_5": self.exact_at_5,
            "summary": {
                "case_count": self.case_count,
                "recall_at_5": self.r_at_5,
                "exact_at_5": self.exact_at_5,
                "exact_at_10": (
                    sum(1 if case.exact_at_10 else 0 for case in self.per_case) / self.case_count if self.case_count else 0.0
                ),
                "exact_at_20": (
                    sum(1 if case.exact_at_20 else 0 for case in self.per_case) / self.case_count if self.case_count else 0.0
                ),
                "by_gold_cardinality": {
                    cardinality: asdict(metrics)
                    for cardinality, metrics in sorted(self.by_gold_cardinality.items(), key=lambda item: int(item[0]))
                },
            },
            "divergence_examples": [asdict(example) for example in self.divergence_examples],
            "per_case": [asdict(case) for case in self.per_case],
            "profile": self.profile,
        }


@dataclass
class LongMemEvalProfile:
    cache_lookup_seconds: float = 0.0
    prepare_seconds: float = 0.0
    query_embed_seconds: float = 0.0
    raw_rank_seconds: float = 0.0
    hybrid_rank_seconds: float = 0.0
    case_total_seconds: float = 0.0
    cases_profiled: int = 0

    def to_dict(self) -> dict[str, Any]:
        cases = max(self.cases_profiled, 1)
        return {
            "cases_profiled": self.cases_profiled,
            "cache_lookup_seconds": self.cache_lookup_seconds,
            "prepare_seconds": self.prepare_seconds,
            "query_embed_seconds": self.query_embed_seconds,
            "raw_rank_seconds": self.raw_rank_seconds,
            "hybrid_rank_seconds": self.hybrid_rank_seconds,
            "case_total_seconds": self.case_total_seconds,
            "avg_query_embed_seconds": self.query_embed_seconds / cases,
            "avg_raw_rank_seconds": self.raw_rank_seconds / cases,
            "avg_hybrid_rank_seconds": self.hybrid_rank_seconds / cases,
            "avg_case_total_seconds": self.case_total_seconds / cases,
        }


@dataclass
class PreparedLongMemEvalCache:
    prepared_entries: list[PreparedLongMemEvalEntry]
    question_embeddings: np.ndarray


def _normalized_question_terms(question: str) -> list[str]:
    return [
        token
        for token in "".join(character.lower() if character.isalnum() else " " for character in question).split()
        if len(token) > 2
    ]


def _quoted_phrase_score(question: str, content: str) -> float:
    lowered_content = content.lower()
    phrases = []
    for quote in ("'", '"'):
        parts = question.split(quote)
        phrases.extend(parts[index].strip().lower() for index in range(1, len(parts), 2))
    if not phrases:
        return 0.0
    matches = sum(1 for phrase in phrases if len(phrase) >= 3 and phrase in lowered_content)
    return matches / len(phrases)


def _term_coverage_score(question: str, content: str) -> float:
    terms = _normalized_question_terms(question)
    if not terms:
        return 0.0
    lowered_content = content.lower()
    hits = sum(1 for term in terms if term in lowered_content)
    return hits / len(terms)


def _temporal_alignment_scores(question: str, sessions: list[PreparedLongMemEvalSession]) -> np.ndarray:
    temporal_hints = infer_temporal_hints(question)
    temporal_scores = np.zeros(len(sessions), dtype=np.float32)
    if temporal_hints.recency_mode == "default" or not sessions:
        return temporal_scores
    timestamps = np.asarray([session.updated_at.timestamp() for session in sessions], dtype=np.float64)
    max_timestamp = float(np.max(timestamps))
    min_timestamp = float(np.min(timestamps))
    span = max(max_timestamp - min_timestamp, 1.0)
    if temporal_hints.recency_mode == "latest":
        temporal_scores = np.asarray((timestamps - min_timestamp) / span, dtype=np.float32)
    elif temporal_hints.recency_mode == "oldest":
        temporal_scores = np.asarray((max_timestamp - timestamps) / span, dtype=np.float32)
    return temporal_scores


def _combined_candidate_scores(
    question: str,
    entry: PreparedLongMemEvalEntry,
    question_embedding: np.ndarray,
    embedding_model: Any,
    *,
    semantic_weight: float,
    lexical_weight: float,
    temporal_weight: float,
    coverage_weight: float,
    quoted_weight: float,
) -> np.ndarray:
    semantic_scores = _vector_similarity_matrix(question_embedding, entry, embedding_model)
    if semantic_scores.size == 0:
        return np.empty(0, dtype=np.float32)
    lexical_scores = np.asarray(
        [lexical_overlap(question, session.label, session.content) for session in entry.sessions],
        dtype=np.float32,
    )
    temporal_scores = _temporal_alignment_scores(question, entry.sessions)
    coverage_scores = np.asarray(
        [_term_coverage_score(question, session.content) for session in entry.sessions],
        dtype=np.float32,
    )
    quoted_scores = np.asarray(
        [_quoted_phrase_score(question, session.content) for session in entry.sessions],
        dtype=np.float32,
    )
    return (
        (semantic_weight * semantic_scores)
        + (lexical_weight * lexical_scores)
        + (temporal_weight * temporal_scores)
        + (coverage_weight * coverage_scores)
        + (quoted_weight * quoted_scores)
    )


def _dataset_sha256(dataset_path: str | Path) -> str:
    digest = sha256()
    with Path(dataset_path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _case_id(entry: dict[str, Any], index: int, question: str) -> str:
    raw_id = str(entry.get("id", "")).strip()
    if raw_id and raw_id.lower() not in {"entry", "question", "case"}:
        return raw_id
    question_digest = sha256(question.encode("utf-8")).hexdigest()[:8]
    return f"case_{index:03d}_{question_digest}"


def _load_entries(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("entries", "data", "questions"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise BenchmarkRuntimeError("Unsupported LongMemEval file shape. Expected a list or dict with entries/data/questions.")


def _extract_correct_session_ids(entry: dict[str, Any]) -> list[str]:
    for key in (
        "correct_session_ids",
        "answer_session_ids",
        "needle_session_ids",
        "ground_truth_session_ids",
        "support_session_ids",
    ):
        value = entry.get(key)
        if isinstance(value, list) and value:
            return [str(item) for item in value]
    for key in ("correct_session_id", "answer_session_id", "needle_session_id"):
        value = entry.get(key)
        if value:
            return [str(value)]
    raise BenchmarkRuntimeError("Could not find ground-truth session IDs in LongMemEval entry.")


def _normalize_timestamp(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        return datetime.now(timezone.utc).isoformat()
    try:
        if "/" in text and " (" in text:
            parsed = datetime.strptime(text.split(" (", 1)[0], "%Y/%m/%d")
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc).isoformat()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _session_text(session: list[dict[str, Any]], *, include_assistant: bool) -> str:
    lines: list[str] = []
    for turn in session:
        role = str(turn.get("role", "unknown")).strip()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if include_assistant or role == "user":
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _embed_texts(embedding_model: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if hasattr(embedding_model, "embed_batch"):
        return np.asarray(embedding_model.embed_batch(texts), dtype=np.float32)
    return np.asarray([embedding_model.embed(text) for text in texts], dtype=np.float32)


def _embed_texts_in_chunks(embedding_model: Any, texts: list[str], *, chunk_size: int = 256) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    chunks = [
        _embed_texts(embedding_model, texts[index : index + chunk_size])
        for index in range(0, len(texts), chunk_size)
    ]
    if len(chunks) == 1:
        return chunks[0]
    return np.vstack(chunks)


def _rank_candidates_heuristic(question: str, sessions: list[PreparedLongMemEvalSession], *, top_k: int) -> list[PreparedLongMemEvalSession]:
    temporal_scores = _temporal_alignment_scores(question, sessions)
    scored: list[tuple[float, int, PreparedLongMemEvalSession]] = []
    for index, session in enumerate(sessions):
        base_score = 1.0 / (index + 1)
        lexical_score = lexical_overlap(question, session.label, session.content)
        temporal_score = float(temporal_scores[index]) if len(temporal_scores) > index else 0.0
        coverage_score = _term_coverage_score(question, session.content)
        quoted_score = _quoted_phrase_score(question, session.content)
        score = (
            (0.35 * base_score)
            + (0.25 * lexical_score)
            + (0.15 * temporal_score)
            + (0.20 * coverage_score)
            + (0.05 * quoted_score)
        )
        scored.append((score, -index, session))
    return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]]


def _session_chunks(session: PreparedLongMemEvalSession) -> list[PreparedLongMemEvalChunk]:
    lines = [line.strip() for line in session.content.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) == 1:
        return [PreparedLongMemEvalChunk(session_id=session.session_id, chunk_id=f"{session.session_id}:0", content=lines[0])]
    chunks: list[PreparedLongMemEvalChunk] = [
        PreparedLongMemEvalChunk(
            session_id=session.session_id,
            chunk_id=f"{session.session_id}:full",
            content=session.content,
        )
    ]
    for index, line in enumerate(lines):
        chunks.append(
            PreparedLongMemEvalChunk(
                session_id=session.session_id,
                chunk_id=f"{session.session_id}:line:{index}",
                content=line,
            )
        )
    return chunks[:5]


def _chunk_rerank_sessions(
    question: str,
    sessions: list[PreparedLongMemEvalSession],
    *,
    top_k: int,
) -> list[PreparedLongMemEvalSession]:
    if not sessions:
        return []
    chunk_pool: list[PreparedLongMemEvalChunk] = []
    for session in sessions:
        for chunk in _session_chunks(session):
            chunk_pool.append(chunk)
    if not chunk_pool:
        return sessions[:top_k]

    chunk_scores: dict[str, float] = {}
    for chunk in chunk_pool:
        lexical = lexical_overlap(question, chunk.session_id, chunk.content)
        coverage = _term_coverage_score(question, chunk.content)
        quoted = _quoted_phrase_score(question, chunk.content)
        score = (0.45 * lexical) + (0.45 * coverage) + (0.10 * quoted)
        previous = chunk_scores.get(chunk.session_id, 0.0)
        if score > previous:
            chunk_scores[chunk.session_id] = score

    ranked = sorted(
        sessions,
        key=lambda session: (
            -chunk_scores.get(session.session_id, 0.0),
            -session.updated_at.timestamp(),
            session.session_id,
        ),
    )
    return ranked[:top_k]


def _prepare_entry_specs(entry: dict[str, Any], *, mode: str) -> tuple[str, str, list[str], list[PreparedLongMemEvalSession]]:
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]
    include_assistant = mode == "graph_hybrid"
    prepared_sessions: list[PreparedLongMemEvalSession] = []
    for session, session_id, raw_date in zip(sessions, session_ids, dates, strict=True):
        content = _session_text(session, include_assistant=include_assistant)
        if not content.strip():
            continue
        prepared_sessions.append(
            PreparedLongMemEvalSession(
                session_id=str(session_id),
                label=f"LongMemEval Session {session_id}",
                content=content,
                updated_at=datetime.fromisoformat(_normalize_timestamp(str(raw_date))),
            )
        )
    return (
        str(entry.get("id", "entry")),
        str(entry["question"]),
        _extract_correct_session_ids(entry),
        prepared_sessions,
    )


def _prepare_entries(entries: list[dict[str, Any]], *, mode: str, embedding_model: Any) -> list[PreparedLongMemEvalEntry]:
    entry_specs = [_prepare_entry_specs(entry, mode=mode) for entry in entries]
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    entry_chunks: list[list[PreparedLongMemEvalChunk]] = []
    for _, _, _, sessions in entry_specs:
        for session in sessions:
            if session.content not in seen_texts:
                seen_texts.add(session.content)
                unique_texts.append(session.content)
        chunks = [chunk for session in sessions for chunk in _session_chunks(session)]
        entry_chunks.append(chunks)
    embedding_cache: dict[str, np.ndarray] = {}
    if unique_texts:
        for text, embedding in zip(unique_texts, _embed_texts_in_chunks(embedding_model, unique_texts), strict=True):
            embedding_cache[text] = embedding
    prepared_entries: list[PreparedLongMemEvalEntry] = []
    for (query_id, question, correct_session_ids, sessions), chunks in zip(entry_specs, entry_chunks, strict=True):
        if sessions:
            embedding_matrix = np.asarray([embedding_cache[session.content] for session in sessions], dtype=np.float32)
        else:
            embedding_matrix = np.empty((0, 0), dtype=np.float32)
        prepared_entries.append(
            PreparedLongMemEvalEntry(
                query_id=query_id,
                question=question,
                correct_session_ids=correct_session_ids,
                sessions=sessions,
                embedding_matrix=embedding_matrix,
                chunks=chunks,
            )
        )
    return prepared_entries


def _cache_dir_for_dataset(dataset_path: str | Path, cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(dataset_path).resolve().parent / ".cache"


def _embedding_model_version(embedding_model: Any) -> str:
    return str(getattr(embedding_model, "model_version", "") or embedding_model.__class__.__name__)


def _cache_key(
    dataset_path: str | Path,
    *,
    mode: str,
    embedding_model: Any,
    limit: int | None,
    dataset_digest: str,
) -> str:
    model_name = getattr(embedding_model, "model_name", embedding_model.__class__.__name__)
    return sha256(
        json.dumps(
            {
                "dataset_sha256": dataset_digest,
                "mode": mode,
                "limit": limit if limit is not None else "full",
                "embedding_model": str(model_name),
                "embedding_model_version": _embedding_model_version(embedding_model),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def _cache_file_stem(
    dataset_path: str | Path,
    *,
    cache_key: str,
    cache_dir: str | Path | None,
) -> Path:
    return _cache_dir_for_dataset(dataset_path, cache_dir) / f"longmemeval-{cache_key}"


def _cache_file_paths(
    dataset_path: str | Path,
    *,
    cache_key: str,
    cache_dir: str | Path | None,
) -> tuple[Path, Path]:
    stem = _cache_file_stem(dataset_path, cache_key=cache_key, cache_dir=cache_dir)
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _serialize_prepared_entries(prepared_entries: list[PreparedLongMemEvalEntry]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    payload: list[dict[str, Any]] = []
    for index, entry in enumerate(prepared_entries):
        embedding_key = f"entry_{index}_embedding_matrix"
        arrays[embedding_key] = np.asarray(entry.embedding_matrix, dtype=np.float32)
        payload.append(
            {
                "query_id": entry.query_id,
                "question": entry.question,
                "correct_session_ids": list(entry.correct_session_ids),
                "embedding_key": embedding_key,
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "label": session.label,
                        "content": session.content,
                        "updated_at": session.updated_at.isoformat(),
                    }
                    for session in entry.sessions
                ],
                "chunks": [
                    {
                        "session_id": chunk.session_id,
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                    }
                    for chunk in entry.chunks
                ],
            }
        )
    return payload, arrays


def _deserialize_prepared_entries(payload: list[dict[str, Any]], arrays: Any) -> list[PreparedLongMemEvalEntry]:
    prepared_entries: list[PreparedLongMemEvalEntry] = []
    for entry in payload:
        sessions = [
            PreparedLongMemEvalSession(
                session_id=str(session["session_id"]),
                label=str(session["label"]),
                content=str(session["content"]),
                updated_at=datetime.fromisoformat(str(session["updated_at"])),
            )
            for session in entry.get("sessions", [])
        ]
        chunks = [
            PreparedLongMemEvalChunk(
                session_id=str(chunk["session_id"]),
                chunk_id=str(chunk["chunk_id"]),
                content=str(chunk["content"]),
            )
            for chunk in entry.get("chunks", [])
        ]
        prepared_entries.append(
            PreparedLongMemEvalEntry(
                query_id=str(entry["query_id"]),
                question=str(entry["question"]),
                correct_session_ids=[str(item) for item in entry.get("correct_session_ids", [])],
                sessions=sessions,
                embedding_matrix=np.asarray(arrays[str(entry["embedding_key"])], dtype=np.float32),
                chunks=chunks,
            )
        )
    return prepared_entries


def _load_prepared_cache(metadata_path: Path, arrays_path: Path) -> PreparedLongMemEvalCache | None:
    if not metadata_path.exists() or not arrays_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 1)) < 3:
        return None
    with np.load(arrays_path, allow_pickle=False) as arrays:
        prepared_entries = _deserialize_prepared_entries(payload.get("prepared_entries", []), arrays)
        question_embeddings = np.asarray(arrays[str(payload["question_embeddings_key"])], dtype=np.float32)
    return PreparedLongMemEvalCache(
        prepared_entries=prepared_entries,
        question_embeddings=question_embeddings,
    )


def _save_prepared_cache(
    metadata_path: Path,
    arrays_path: Path,
    *,
    prepared_entries: list[PreparedLongMemEvalEntry],
    question_embeddings: np.ndarray,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_payload, arrays = _serialize_prepared_entries(prepared_entries)
    question_embeddings_key = "question_embeddings"
    arrays[question_embeddings_key] = np.asarray(question_embeddings, dtype=np.float32)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "format": "waggle-longmemeval-cache",
                "prepared_entries": prepared_payload,
                "question_embeddings_key": question_embeddings_key,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(arrays_path, **arrays)


def _vector_similarity_matrix(question_embedding: np.ndarray, entry: PreparedLongMemEvalEntry, embedding_model: Any) -> np.ndarray:
    if not entry.sessions or entry.embedding_matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    question_vector = np.asarray(question_embedding, dtype=np.float32)
    if entry.embedding_matrix.ndim == 2 and entry.embedding_matrix.shape[1] == question_vector.shape[0]:
        return np.asarray(entry.embedding_matrix @ question_vector, dtype=np.float32)
    return np.asarray(
        [embedding_model.cosine_similarity(question_vector, session_embedding) for session_embedding in entry.embedding_matrix],
        dtype=np.float32,
    )


def _chunk_session_scores(question: str, entry: PreparedLongMemEvalEntry) -> dict[str, float]:
    if not entry.chunks:
        return {}
    scores: dict[str, float] = {}
    for chunk in entry.chunks:
        lexical_score = lexical_overlap(question, chunk.chunk_id, chunk.content)
        coverage_score = _term_coverage_score(question, chunk.content)
        quoted_score = _quoted_phrase_score(question, chunk.content)
        combined = (0.45 * lexical_score) + (0.45 * coverage_score) + (0.10 * quoted_score)
        previous = scores.get(chunk.session_id, 0.0)
        if combined > previous:
            scores[chunk.session_id] = combined
    return scores


def _by_gold_cardinality(results: list[LongMemEvalCaseResult]) -> dict[str, LongMemEvalCardinalityMetrics]:
    buckets: dict[int, list[LongMemEvalCaseResult]] = {}
    for result in results:
        buckets.setdefault(len(set(result.correct_session_ids)), []).append(result)
    summary: dict[str, LongMemEvalCardinalityMetrics] = {}
    for cardinality, bucket in sorted(buckets.items()):
        count = len(bucket)
        recall = sum(1 if item.hit_at_5 else 0 for item in bucket) / count if count else 0.0
        exact = sum(1 if item.exact_at_5 else 0 for item in bucket) / count if count else 0.0
        exact_10 = sum(1 if item.exact_at_10 else 0 for item in bucket) / count if count else 0.0
        exact_20 = sum(1 if item.exact_at_20 else 0 for item in bucket) / count if count else 0.0
        summary[str(cardinality)] = LongMemEvalCardinalityMetrics(
            count=count,
            recall_at_5=recall,
            exact_at_5=exact,
            exact_at_10=exact_10,
            exact_at_20=exact_20,
        )
    return summary


def _divergence_examples(results: list[LongMemEvalCaseResult], *, limit: int = 3) -> list[LongMemEvalDivergenceExample]:
    examples: list[LongMemEvalDivergenceExample] = []
    for result in results:
        if not result.hit_at_5 or result.exact_at_5:
            continue
        gold_set = set(result.correct_session_ids)
        retrieved_top5 = result.retrieved_session_ids[:5]
        retrieved_set = set(retrieved_top5)
        if not (retrieved_set & gold_set):
            continue
        missing = [session_id for session_id in result.correct_session_ids if session_id not in retrieved_set]
        examples.append(
            LongMemEvalDivergenceExample(
                case_id=result.query_id,
                gold_set=list(result.correct_session_ids),
                retrieved_top5=retrieved_top5,
                missing=missing,
            )
        )
        if len(examples) >= limit:
            break
    return examples


def format_summary_table(report: LongMemEvalReport) -> str:
    lines = [
        f"=== {report.mode} (n={report.case_count}) ===",
        (
            f"overall          R@5={report.r_at_5 * 100:.1f}%  "
            f"Exact@5={report.exact_at_5 * 100:.1f}%  "
            f"Exact@10={sum(1 if case.exact_at_10 else 0 for case in report.per_case) / report.case_count * 100:.1f}%  "
            f"Exact@20={sum(1 if case.exact_at_20 else 0 for case in report.per_case) / report.case_count * 100:.1f}%"
        ),
    ]
    for cardinality, metrics in sorted(report.by_gold_cardinality.items(), key=lambda item: int(item[0])):
        lines.append(
            f"cardinality={cardinality}    "
            f"R@5={metrics.recall_at_5 * 100:.1f}%  "
            f"Exact@5={metrics.exact_at_5 * 100:.1f}%  "
            f"Exact@10={metrics.exact_at_10 * 100:.1f}%  "
            f"Exact@20={metrics.exact_at_20 * 100:.1f}%   "
            f"(n={metrics.count})"
        )
    divergence_ids = ", ".join(example.case_id for example in report.divergence_examples) or "none"
    lines.append(f"divergence: {divergence_ids}")
    return "\n".join(lines)


def _raw_candidate_order(question: str, entry: PreparedLongMemEvalEntry, question_embedding: np.ndarray, embedding_model: Any) -> list[PreparedLongMemEvalSession]:
    session_scores = _combined_candidate_scores(
        question,
        entry,
        question_embedding,
        embedding_model,
        semantic_weight=0.60,
        lexical_weight=0.15,
        temporal_weight=0.10,
        coverage_weight=0.10,
        quoted_weight=0.05,
    )
    if session_scores.size == 0:
        return []
    chunk_scores = _chunk_session_scores(question, entry)
    fused_scores = np.asarray(
        [
            max(
                float(session_score),
                0.9 * chunk_scores.get(session.session_id, 0.0),
            )
            for session, session_score in zip(entry.sessions, session_scores, strict=True)
        ],
        dtype=np.float32,
    )
    ranked_indices = np.argsort(-fused_scores, kind="stable")
    return [entry.sessions[index] for index in ranked_indices]


def _hybrid_candidate_order(question: str, entry: PreparedLongMemEvalEntry, question_embedding: np.ndarray, embedding_model: Any) -> list[PreparedLongMemEvalSession]:
    first_pass = _raw_candidate_order(question, entry, question_embedding, embedding_model)
    if not first_pass:
        return []
    top_candidates = first_pass[:10]
    heuristic_reranked = _rank_candidates_heuristic(question, top_candidates, top_k=min(20, len(top_candidates)))
    chunk_reranked = _chunk_rerank_sessions(
        question,
        top_candidates,
        top_k=min(20, len(top_candidates)),
    )
    raw_rank = {session.session_id: index for index, session in enumerate(first_pass[:20], start=1)}
    heuristic_rank = {session.session_id: index for index, session in enumerate(heuristic_reranked, start=1)}
    chunk_rank = {session.session_id: index for index, session in enumerate(chunk_reranked, start=1)}
    fused_scores: list[tuple[float, int, PreparedLongMemEvalSession]] = []
    rrf_k = 20.0
    for index, session in enumerate(first_pass[:20], start=1):
        score = (
            (1.0 / (rrf_k + index))
            + (0.45 / (rrf_k + heuristic_rank.get(session.session_id, 1000)))
            + (0.75 / (rrf_k + chunk_rank.get(session.session_id, 1000)))
        )
        fused_scores.append((score, -raw_rank[session.session_id], session))
    ordered = [item[2] for item in sorted(fused_scores, key=lambda item: (-item[0], item[1]))]
    return ordered[:20]


def evaluate_longmemeval(
    dataset_path: str | Path,
    *,
    entries: list[dict[str, Any]] | None = None,
    embedding_model: Any | None = None,
    mode: Literal["graph_raw", "graph_hybrid"] = "graph_raw",
    limit: int | None = None,
    cache_dir: str | Path | None = None,
    split_type: str = "full",
    split_seed: int | None = None,
    profile: bool = False,
) -> LongMemEvalReport:
    profiler = LongMemEvalProfile() if profile else None
    if entries is None:
        entries = _load_entries(dataset_path)
    
    # We maintain the limit filter here for the 'full' run path
    if limit is not None:
        entries = entries[:limit]
    model_instance = embedding_model or EmbeddingModel()
    dataset_digest = _dataset_sha256(dataset_path)
    cache_key = _cache_key(
        dataset_path,
        mode=mode,
        embedding_model=model_instance,
        limit=limit,
        dataset_digest=dataset_digest,
    )
    # We always cache the full dataset preparation for efficiency, regardless of limit/split.
    # The limit/split only affects which prepared entries we actually evaluate.
    full_cache_key = _cache_key(
        dataset_path,
        mode=mode,
        embedding_model=model_instance,
        limit=None,
        dataset_digest=dataset_digest,
    )
    cache_metadata_path, cache_arrays_path = _cache_file_paths(
        dataset_path,
        cache_key=full_cache_key,
        cache_dir=cache_dir,
    )
    cache_lookup_started = time.perf_counter()
    cached = _load_prepared_cache(cache_metadata_path, cache_arrays_path)
    if profiler is not None:
        profiler.cache_lookup_seconds += time.perf_counter() - cache_lookup_started
    if cached is not None:
        cache_status = "warm"
        prepared_entries = cached.prepared_entries
        question_embeddings = cached.question_embeddings
    else:
        cache_status = "cold"
        prepare_started = time.perf_counter()
        prepared_entries = _prepare_entries(entries, mode=mode, embedding_model=model_instance)
        question_embeddings = _embed_texts_in_chunks(
            model_instance,
            [prepared_entry.question for prepared_entry in prepared_entries],
        )
        if profiler is not None:
            profiler.prepare_seconds += time.perf_counter() - prepare_started
        # Only save cache if we are evaluating the full dataset or at least a large chunk
        if limit is None and split_type == "full":
            _save_prepared_cache(
                cache_metadata_path,
                cache_arrays_path,
                prepared_entries=prepared_entries,
                question_embeddings=question_embeddings,
            )
    
    # Crucial: if we loaded from cache, we might have more entries than requested
    # We must match prepared_entries to entries by query_id
    if len(prepared_entries) != len(entries):
        entry_map = {e.question: (e, qe) for e, qe in zip(prepared_entries, question_embeddings)}
        prepared_entries_filtered = []
        question_embeddings_filtered = []
        for entry in entries:
            qtext = str(entry.get("question", ""))
            if qtext in entry_map:
                e, qe = entry_map[qtext]
                prepared_entries_filtered.append(e)
                question_embeddings_filtered.append(qe)
        prepared_entries = prepared_entries_filtered
        if question_embeddings_filtered:
            question_embeddings = np.asarray(question_embeddings_filtered)
        else:
            question_embeddings = np.empty((0, 0), dtype=np.float32)

    # Apply limit filter if requested
    if limit is not None:
        prepared_entries = prepared_entries[:limit]
        question_embeddings = question_embeddings[:limit]
        entries = entries[:limit]

    results: list[LongMemEvalCaseResult] = []
    for index, (entry, prepared_entry, question_embedding) in enumerate(
        zip(entries, prepared_entries, question_embeddings, strict=True),
        start=1,
    ):
        case_started = time.perf_counter()
        question = prepared_entry.question
        query_embed_started = time.perf_counter()
        question_vector = np.asarray(question_embedding, dtype=np.float32)
        if profiler is not None:
            profiler.query_embed_seconds += time.perf_counter() - query_embed_started
        if mode == "graph_raw":
            raw_rank_started = time.perf_counter()
            ranked_sessions = _raw_candidate_order(question, prepared_entry, question_vector, model_instance)[:5]
            if profiler is not None:
                profiler.raw_rank_seconds += time.perf_counter() - raw_rank_started
        else:
            hybrid_rank_started = time.perf_counter()
            ranked_sessions = _hybrid_candidate_order(question, prepared_entry, question_vector, model_instance)
            if profiler is not None:
                profiler.hybrid_rank_seconds += time.perf_counter() - hybrid_rank_started
        retrieved_session_ids = [session.session_id for session in ranked_sessions]
        gold_ids = prepared_entry.correct_session_ids
        retrieved_set = set(retrieved_session_ids[:5])
        retrieved_set_10 = set(retrieved_session_ids[:10])
        retrieved_set_20 = set(retrieved_session_ids[:20])
        gold_set = set(gold_ids)
        results.append(
            LongMemEvalCaseResult(
                query_id=_case_id(entry, index, question),
                question=question,
                correct_session_ids=gold_ids,
                retrieved_session_ids=retrieved_session_ids[:5],
                hit_at_5=bool(retrieved_set & gold_set),
                exact_at_5=gold_set.issubset(retrieved_set),
                exact_at_10=gold_set.issubset(retrieved_set_10),
                exact_at_20=gold_set.issubset(retrieved_set_20),
            )
        )
        if profiler is not None:
            profiler.case_total_seconds += time.perf_counter() - case_started
            profiler.cases_profiled += 1
    case_count = len(results)
    prepared_session_count = sum(len(entry.sessions) for entry in prepared_entries)
    hit_rate = sum(1 if result.hit_at_5 else 0 for result in results) / case_count if case_count else 0.0
    exact_rate = sum(1 if result.exact_at_5 else 0 for result in results) / case_count if case_count else 0.0
    return LongMemEvalReport(
        dataset_path=str(dataset_path),
        mode=mode,
        case_count=case_count,
        split_type=split_type,
        split_seed=split_seed,
        cache_status=cache_status,
        cache_path=str(cache_metadata_path),
        prepared_entry_count=len(prepared_entries),
        prepared_session_count=prepared_session_count,
        cache_key=full_cache_key,
        r_at_5=hit_rate,
        exact_at_5=exact_rate,
        by_gold_cardinality=_by_gold_cardinality(results),
        divergence_examples=_divergence_examples(results),
        per_case=results,
        profile=profiler.to_dict() if profiler is not None else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exploratory LongMemEval adapter for Waggle.")
    parser.add_argument("dataset_path", type=Path, help="Path to longmemeval_s_cleaned.json or equivalent cleaned dataset.")
    parser.add_argument("--mode", choices=["graph_raw", "graph_hybrid"], default="graph_raw")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of entries to evaluate.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory for prepared LongMemEval cache files (JSON metadata plus .npz embeddings).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--held-out", action="store_true", help="Split into 50 dev / 450 test based on fixed seed.")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for dev/test split.")
    parser.add_argument("--profile", action="store_true", help="Capture coarse timing for benchmark phases.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    
    if args.held_out:
        import random
        all_entries = _load_entries(args.dataset_path)
        if len(all_entries) < 500:
            print(f"Warning: held-out split requested but dataset only has {len(all_entries)} items. Proceeding with proportional split (10% dev).")
            dev_size = max(1, len(all_entries) // 10)
        else:
            dev_size = 50
        
        # Consistent shuffle based on seed
        indices = list(range(len(all_entries)))
        random.Random(args.split_seed).shuffle(indices)
        
        dev_indices = indices[:dev_size]
        test_indices = indices[dev_size:]
        
        dev_entries = [all_entries[i] for i in dev_indices]
        test_entries = [all_entries[i] for i in test_indices]
        
        print(f"Held-out split: {len(dev_entries)} dev / {len(test_entries)} test (seed {args.split_seed})")
        
        dev_report = evaluate_longmemeval(
            args.dataset_path,
            entries=dev_entries,
            embedding_model=EmbeddingModel(args.embedding_model),
            mode=args.mode,
            cache_dir=args.cache_dir,
            split_type="dev",
            split_seed=args.split_seed,
            profile=args.profile,
        )
        
        test_report = evaluate_longmemeval(
            args.dataset_path,
            entries=test_entries,
            embedding_model=EmbeddingModel(args.embedding_model),
            mode=args.mode,
            cache_dir=args.cache_dir,
            split_type="test",
            split_seed=args.split_seed,
            profile=args.profile,
        )
        
        print("=" * 72)
        print("waggle LongMemEval held-out benchmark")
        print("=" * 72)
        print(format_summary_table(dev_report))
        print()
        print(format_summary_table(test_report))
        
        if args.output is not None:
            output_dev = args.output.with_name(args.output.stem + "_dev" + args.output.suffix)
            output_test = args.output.with_name(args.output.stem + "_test" + args.output.suffix)
            
            output_dev.write_text(json.dumps(dev_report.to_dict(), indent=2), encoding="utf-8")
            output_test.write_text(json.dumps(test_report.to_dict(), indent=2), encoding="utf-8")
            print(f"Wrote held-out results to {output_dev} and {output_test}")
        return 0

    # Original full run path
    report = evaluate_longmemeval(
        args.dataset_path,
        embedding_model=EmbeddingModel(args.embedding_model),
        mode=args.mode,
        limit=args.limit,
        cache_dir=args.cache_dir,
        profile=args.profile,
    )
    print("=" * 72)
    print("waggle LongMemEval exploratory benchmark")
    print("=" * 72)
    print(f"dataset: {report.dataset_path}")
    print(f"mode: {report.mode}")
    print(f"cases: {report.case_count}")
    if report.cache_status == "warm":
        print(f"cache: warm ({report.cache_path})")
    else:
        print(f"cache: cold (wrote {report.cache_path})")
    print(format_summary_table(report))
    if report.profile:
        print(f"profile: {json.dumps(report.profile, sort_keys=True)}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote JSON report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
