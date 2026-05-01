from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from waggle.benchmark_cache import BenchmarkCache
from waggle.graph import MemoryGraph
from waggle.memory_benchmark import (
    index_case_into_waggle,
    load_locomo,
    load_longmemeval,
    run_case_cached,
    select_prompt,
)


class FakeEmbeddingModel:
    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            vector[sum(ord(character) for character in token) % len(vector)] += 1.0
        norm = np.linalg.norm(vector)
        return vector if norm == 0.0 else vector / norm

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


def _graph(tmp_path: Path) -> MemoryGraph:
    return MemoryGraph(
        tmp_path / "memory-benchmark.db",
        FakeEmbeddingModel(),
        dedup_similarity_threshold=1.01,
        dedup_same_label_threshold=1.01,
    )


def test_load_longmemeval_supports_compact_fixture_shape(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "What database is production using?",
            "answer": "PostgreSQL",
            "question_type": "knowledge-update",
            "haystack_sessions": [
                [{"role": "user", "content": "We use SQLite locally."}],
                [{"role": "user", "content": "Production uses PostgreSQL now."}],
            ],
            "haystack_session_ids": ["sess_local", "sess_prod"],
            "haystack_dates": ["2024/01/05", "2024/02/10"],
        }
    ]
    path = tmp_path / "longmemeval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    cases = load_longmemeval(path)

    assert len(cases) == 1
    assert cases[0].case_id == "entry_1"
    assert cases[0].question_type == "knowledge-update"
    assert cases[0].sessions[1]["session_id"] == "sess_prod"
    assert cases[0].metadata["gold_support_ids"] == []


def test_load_locomo_expands_multi_qa_rows(tmp_path: Path) -> None:
    dataset = [
        {
            "sample_id": "sample_1",
            "conversation": {
                "session_1": [{"speaker": "A", "text": "I like tea."}],
                "session_1_date_time": "2024-01-01T10:00:00Z",
                "session_2": [{"speaker": "B", "text": "Let's meet on Friday."}],
                "session_2_date_time": "2024-01-05T10:00:00Z",
            },
            "qa": [
                {"question": "Who likes tea?", "answer": "A", "category": 1},
                {"question": "When did they plan to meet?", "answer": "Friday", "category": 3},
            ],
        }
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    cases = load_locomo(path)

    assert len(cases) == 2
    assert cases[0].question_type == "single-hop"
    assert cases[1].question_type == "temporal"
    assert len(cases[0].sessions) == 2


def test_run_case_cached_reuses_disk_result(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "What database is production using?",
            "answer": "PostgreSQL",
            "question_type": "knowledge-update",
            "haystack_sessions": [
                [{"role": "user", "content": "We use SQLite locally."}],
                [{"role": "user", "content": "Production uses PostgreSQL now."}],
            ],
            "haystack_session_ids": ["sess_local", "sess_prod"],
            "haystack_dates": ["2024/01/05", "2024/02/10"],
        }
    ]
    path = tmp_path / "longmemeval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    case = load_longmemeval(path)[0]
    graph = _graph(tmp_path)
    indexed = index_case_into_waggle(case, graph)
    cache = BenchmarkCache(tmp_path / "cache")
    answer_calls = {"count": 0}
    judge_calls = {"count": 0}

    def answer(prompt: str) -> str:
        answer_calls["count"] += 1
        assert "RETRIEVED MEMORY" in prompt
        return "PostgreSQL"

    def judge(prompt: str) -> str:
        judge_calls["count"] += 1
        return "VERDICT: CORRECT\nREASON: Matches the gold answer."

    first, first_hit = run_case_cached(
        indexed,
        graph,
        benchmark="longmemeval",
        arm="waggle_graph",
        answer_model_call=answer,
        judge_model_call=judge,
        answer_model_name="fake-answer",
        judge_model_name="fake-judge",
        retrieval_limit=5,
        cache=cache,
        cache_extra={"dataset_sha256": "abc"},
    )
    second, second_hit = run_case_cached(
        indexed,
        graph,
        benchmark="longmemeval",
        arm="waggle_graph",
        answer_model_call=answer,
        judge_model_call=judge,
        answer_model_name="fake-answer",
        judge_model_name="fake-judge",
        retrieval_limit=5,
        cache=cache,
        cache_extra={"dataset_sha256": "abc"},
    )

    assert first.correct is True
    assert first_hit is False
    assert second_hit is True
    assert second.prediction == "PostgreSQL"
    assert answer_calls["count"] == 1
    assert judge_calls["count"] == 1


def test_index_case_into_waggle_adds_session_anchor_and_part_of_edges(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "What database is production using?",
            "answer": "PostgreSQL",
            "question_type": "knowledge-update",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "We use SQLite locally."},
                    {"role": "assistant", "content": "Local is still SQLite."},
                ],
            ],
            "haystack_session_ids": ["sess_local"],
            "haystack_dates": ["2024/01/05"],
            "correct_session_ids": ["sess_local"],
        }
    ]
    path = tmp_path / "longmemeval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    case = load_longmemeval(path)[0]
    graph = _graph(tmp_path)
    indexed = index_case_into_waggle(case, graph)

    with graph._lock, graph._connect() as connection:  # noqa: SLF001 - test-only inspection
        anchor_rows = connection.execute(
            "SELECT id FROM nodes WHERE project = ? AND session_id = ? AND label = ?",
            (indexed.project_scope, "sess_local", "Session sess_local"),
        ).fetchall()
        assert len(anchor_rows) == 1
        edge_rows = connection.execute(
            "SELECT relationship FROM edges WHERE source_id = ? OR target_id = ?",
            (anchor_rows[0]["id"], anchor_rows[0]["id"]),
        ).fetchall()
    relationships = {row["relationship"] for row in edge_rows}
    assert "part_of" in relationships
    assert "relates_to" in relationships


def test_select_prompt_uses_specialized_variants() -> None:
    assert "updated their information" in select_prompt("longmemeval", "knowledge-update")
    assert "multi-hop question" in select_prompt("locomo", "multi-hop")
