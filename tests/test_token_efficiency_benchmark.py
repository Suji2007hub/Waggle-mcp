from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from waggle.token_efficiency_benchmark import (
    BenchmarkReport,
    build_v2_comparison_report,
    build_markdown_report,
    chunk_text,
    count_tokens,
    generate_default_dataset,
    run_comparison_benchmark,
    run_benchmark,
)


class FakeEmbeddingModel:
    model_name = "fake"

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(16, dtype=np.float32)
        for token in text.lower().split():
            vector[sum(ord(character) for character in token) % len(vector)] += 1.0
        norm = np.linalg.norm(vector)
        return vector if norm == 0.0 else vector / norm

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.embed(text) for text in texts], dtype=np.float32)

    @staticmethod
    def to_bytes(embedding: np.ndarray) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def from_bytes(data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))


def _small_dataset(tmp_path: Path) -> Path:
    full_path = generate_default_dataset(tmp_path / "full_dataset.json")
    payload = json.loads(full_path.read_text(encoding="utf-8"))
    payload["conversations"] = payload["conversations"][:10]
    payload["queries"] = payload["queries"][:6]
    small_path = tmp_path / "small_dataset.json"
    small_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return small_path


def test_generate_default_dataset_matches_requested_shape(tmp_path: Path) -> None:
    path = generate_default_dataset(tmp_path / "dataset.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["conversations"]) == 50
    assert len(payload["queries"]) == 30
    assert sum(1 for query in payload["queries"] if query["kind"] == "single_fact") == 10
    assert sum(1 for query in payload["queries"] if query["kind"] == "multi_hop") == 10
    assert sum(1 for query in payload["queries"] if query["kind"] == "extraction_failure") == 10


def test_chunk_text_respects_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_size=128, overlap=32)

    assert len(chunks) > 1
    assert count_tokens(chunks[0]) <= 128


def test_run_benchmark_builds_report_with_local_fallback(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _small_dataset(tmp_path)

    monkeypatch.setattr(
        "waggle.token_efficiency_benchmark.EmbeddingModel",
        lambda model_name="all-MiniLM-L6-v2": FakeEmbeddingModel(),
    )

    report = run_benchmark(
        dataset_path=dataset_path,
        allow_local_baseline_fallback=True,
        local_fallback_embedding_model_name="fake",
    )

    assert isinstance(report, BenchmarkReport)
    assert report.corpus["conversation_count"] == 10
    assert report.corpus["query_count"] == 6
    assert "waggle_graph" in report.summaries
    assert "vanilla_rag" in report.summaries
    assert report.examples
    assert report.notes

    markdown = build_markdown_report(report)
    assert "| Avg input tokens per query |" in markdown
    assert "## Example Retrievals" in markdown


def test_run_comparison_benchmark_builds_three_way_report(tmp_path: Path, monkeypatch) -> None:
    dataset_path = _small_dataset(tmp_path)

    monkeypatch.setattr(
        "waggle.token_efficiency_benchmark.EmbeddingModel",
        lambda model_name="all-MiniLM-L6-v2": FakeEmbeddingModel(),
    )

    report = run_comparison_benchmark(
        dataset_path=dataset_path,
        local_fallback_embedding_model_name="fake",
    )

    assert set(report.summaries) == {"old", "new_no_rerank", "new_full"}
    markdown = build_v2_comparison_report(report)
    assert "| Metric | Old | New-no-rerank | New-full |" in markdown
    assert "## Breakdown by Subset" in markdown
