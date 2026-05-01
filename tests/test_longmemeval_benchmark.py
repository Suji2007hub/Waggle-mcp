from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from waggle.longmemeval_benchmark import evaluate_longmemeval


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


class CountingEmbeddingModel(FakeEmbeddingModel):
    def __init__(self) -> None:
        self.embedded_text_count = 0

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.embedded_text_count += len(texts)
        return np.asarray([self.embed(text) for text in texts], dtype=np.float32)


def test_evaluate_longmemeval_graph_modes(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "what database are we using in production",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "We are using SQLite locally."},
                    {"role": "assistant", "content": "SQLite sounds fine for local work."},
                ],
                [
                    {"role": "user", "content": "Production uses PostgreSQL for safer migrations."},
                    {"role": "assistant", "content": "PostgreSQL is the production choice."},
                ],
            ],
            "haystack_session_ids": ["sess_local", "sess_prod"],
            "haystack_dates": ["2024/01/05 (Fri) 09:00", "2024/02/10 (Sat) 09:00"],
            "correct_session_ids": ["sess_prod"],
        }
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    raw_report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_raw")
    hybrid_report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_hybrid")

    assert raw_report.case_count == 1
    assert hybrid_report.case_count == 1
    assert raw_report.r_at_5 == 1.0
    assert hybrid_report.r_at_5 == 1.0
    assert raw_report.per_case[0].retrieved_session_ids
    assert raw_report.by_gold_cardinality["1"].count == 1
    assert raw_report.by_gold_cardinality["1"].recall_at_5 == 1.0
    assert raw_report.by_gold_cardinality["1"].exact_at_5 == 1.0
    assert raw_report.by_gold_cardinality["1"].exact_at_10 == 1.0
    assert raw_report.by_gold_cardinality["1"].exact_at_20 == 1.0
    assert raw_report.per_case[0].query_id.startswith("entry_1")


def test_evaluate_longmemeval_caches_repeated_session_embeddings(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "what database are we using in production",
            "haystack_sessions": [
                [{"role": "user", "content": "Production uses PostgreSQL for safer migrations."}],
                [{"role": "user", "content": "Local development uses SQLite."}],
            ],
            "haystack_session_ids": ["sess_prod", "sess_local"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/01/05 (Fri) 09:00"],
            "correct_session_ids": ["sess_prod"],
        },
        {
            "id": "entry_2",
            "question": "what database do we use locally",
            "haystack_sessions": [
                [{"role": "user", "content": "Production uses PostgreSQL for safer migrations."}],
                [{"role": "user", "content": "Feature flags live in Redis."}],
            ],
            "haystack_session_ids": ["sess_prod_repeat", "sess_flags"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/03/01 (Fri) 09:00"],
            "correct_session_ids": ["sess_prod_repeat"],
        },
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    embedding_model = CountingEmbeddingModel()
    report = evaluate_longmemeval(dataset_path, embedding_model=embedding_model, mode="graph_raw")

    assert report.case_count == 2
    assert embedding_model.embedded_text_count == 5


def test_evaluate_longmemeval_reuses_disk_cache(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "what database are we using in production",
            "haystack_sessions": [
                [{"role": "user", "content": "Production uses PostgreSQL for safer migrations."}],
                [{"role": "user", "content": "Local development uses SQLite."}],
            ],
            "haystack_session_ids": ["sess_prod", "sess_local"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/01/05 (Fri) 09:00"],
            "correct_session_ids": ["sess_prod"],
        }
    ]
    dataset_path = tmp_path / "longmemeval.json"
    cache_dir = tmp_path / "cache"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    first_model = CountingEmbeddingModel()
    second_model = CountingEmbeddingModel()

    first_report = evaluate_longmemeval(
        dataset_path,
        embedding_model=first_model,
        mode="graph_raw",
        cache_dir=cache_dir,
    )
    second_report = evaluate_longmemeval(
        dataset_path,
        embedding_model=second_model,
        mode="graph_raw",
        cache_dir=cache_dir,
    )

    assert first_report.case_count == 1
    assert second_report.case_count == 1
    assert first_model.embedded_text_count == 3
    assert second_model.embedded_text_count == 0
    assert Path(first_report.cache_path).suffix == ".json"
    assert Path(first_report.cache_path).exists()
    assert Path(first_report.cache_path).with_suffix(".npz").exists()
    assert not Path(first_report.cache_path).with_suffix(".pkl").exists()


def test_graph_raw_prefers_verbatim_term_overlap_when_semantics_are_close(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "which session mentions PostgreSQL",
            "haystack_sessions": [
                [{"role": "user", "content": "We finalized the production database as PostgreSQL."}],
                [{"role": "user", "content": "We finalized the production database as the main datastore."}],
            ],
            "haystack_session_ids": ["sess_exact", "sess_generic"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/02/10 (Sat) 09:00"],
            "correct_session_ids": ["sess_exact"],
        }
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_raw")

    assert report.r_at_5 == 1.0
    assert report.per_case[0].retrieved_session_ids[0] == "sess_exact"


def test_graph_hybrid_keeps_raw_leader_when_rerank_is_only_secondary_signal(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "which session mentions PostgreSQL",
            "haystack_sessions": [
                [{"role": "user", "content": "We finalized the production database as PostgreSQL."}],
                [{"role": "user", "content": "We finalized the production database as the main datastore."}],
            ],
            "haystack_session_ids": ["sess_exact", "sess_generic"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/02/10 (Sat) 09:00"],
            "correct_session_ids": ["sess_exact"],
        }
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_hybrid")

    assert report.r_at_5 == 1.0
    assert report.per_case[0].retrieved_session_ids[0] == "sess_exact"


def test_graph_hybrid_chunk_rerank_finds_localized_answer_phrase(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "which session mentions ukulele lessons",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "We discussed travel plans and meal prep."},
                    {"role": "assistant", "content": "We also covered hobbies."},
                    {"role": "user", "content": "Rachel signed up for ukulele lessons last week."},
                ],
                [
                    {"role": "user", "content": "We discussed travel plans and meal prep."},
                    {"role": "assistant", "content": "We also covered hobbies in general terms."},
                    {"role": "user", "content": "Rachel signed up for music classes last week."},
                ],
            ],
            "haystack_session_ids": ["sess_uke", "sess_music"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/02/10 (Sat) 09:00"],
            "correct_session_ids": ["sess_uke"],
        }
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_hybrid")

    assert report.r_at_5 == 1.0
    assert report.per_case[0].retrieved_session_ids[0] == "sess_uke"


def test_graph_raw_uses_chunk_candidates_as_first_class_signal(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "entry_1",
            "question": "which session mentions ukulele lessons",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "We discussed travel plans and meal prep."},
                    {"role": "assistant", "content": "We also covered hobbies."},
                    {"role": "user", "content": "Rachel signed up for ukulele lessons last week."},
                ],
                [
                    {"role": "user", "content": "We discussed travel plans and meal prep."},
                    {"role": "assistant", "content": "We also covered hobbies in general terms."},
                    {"role": "user", "content": "Rachel signed up for music classes last week."},
                ],
            ],
            "haystack_session_ids": ["sess_uke", "sess_music"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/02/10 (Sat) 09:00"],
            "correct_session_ids": ["sess_uke"],
        }
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_raw")

    assert report.r_at_5 == 1.0
    assert report.per_case[0].retrieved_session_ids[0] == "sess_uke"


def test_report_surfaces_cardinality_breakdown_and_divergence_examples(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "case_single",
            "question": "which session mentions PostgreSQL",
            "haystack_sessions": [
                [{"role": "user", "content": "We finalized the production database as PostgreSQL."}],
                [{"role": "user", "content": "We finalized the production database as the main datastore."}],
            ],
            "haystack_session_ids": ["sess_exact", "sess_generic"],
            "haystack_dates": ["2024/02/10 (Sat) 09:00", "2024/02/10 (Sat) 09:00"],
            "correct_session_ids": ["sess_exact"],
        },
        {
            "id": "case_multi",
            "question": "which sessions mention the budget and launch timeline",
            "haystack_sessions": [
                [{"role": "user", "content": "The budget was approved for the product launch."}],
                [{"role": "user", "content": "The team reviewed office snack options."}],
                [{"role": "user", "content": "The launch timeline slipped by two weeks."}],
                [{"role": "user", "content": "We booked the company retreat venue."}],
                [{"role": "user", "content": "The budget forecast includes marketing."}],
                [{"role": "user", "content": "Launch comms will go out on Tuesday."}],
            ],
            "haystack_session_ids": [
                "sess_budget_primary",
                "sess_snacks",
                "sess_timeline",
                "sess_retreat",
                "sess_budget_secondary",
                "sess_comms",
            ],
            "haystack_dates": [
                "2024/02/10 (Sat) 09:00",
                "2024/02/11 (Sun) 09:00",
                "2024/02/12 (Mon) 09:00",
                "2024/02/13 (Tue) 09:00",
                "2024/02/14 (Wed) 09:00",
                "2024/02/15 (Thu) 09:00",
            ],
            "correct_session_ids": ["sess_budget_primary", "sess_timeline", "sess_budget_secondary"],
        },
    ]
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = evaluate_longmemeval(dataset_path, embedding_model=FakeEmbeddingModel(), mode="graph_raw")
    payload = report.to_dict()

    assert payload["summary"]["by_gold_cardinality"]["1"]["count"] == 1
    assert payload["summary"]["by_gold_cardinality"]["3"]["count"] == 1
    assert "exact_at_10" in payload["summary"]
    assert "exact_at_20" in payload["summary"]
    assert len(payload["divergence_examples"]) <= 3
    if payload["divergence_examples"]:
        example = payload["divergence_examples"][0]
        assert example["case_id"].startswith("case_")
        assert set(example["missing"]).issubset(set(example["gold_set"]))


def test_longmemeval_held_out_split(tmp_path):
    dataset_path = tmp_path / "longmemeval_split_test.json"
    # Create 100 items for easy splitting
    items = [
        {
            "id": f"q{i}",
            "question": f"Question {i}?",
            "haystack_sessions": [],
            "haystack_session_ids": [],
            "haystack_dates": [],
            "correct_session_ids": ["s1"]
        }
        for i in range(100)
    ]
    dataset_path.write_text(json.dumps(items))
    
    from waggle.longmemeval_benchmark import main
    output_path = tmp_path / "results.json"
    
    # Run with held-out
    main([str(dataset_path), "--held-out", "--output", str(output_path), "--limit", "100", "--embedding-model", "deterministic"])
    
    # Check for _dev and _test files
    assert (tmp_path / "results_dev.json").exists()
    assert (tmp_path / "results_test.json").exists()
    
    dev_data = json.loads((tmp_path / "results_dev.json").read_text())
    test_data = json.loads((tmp_path / "results_test.json").read_text())
    
    # In my logic, 100 items -> 10% dev = 10 items
    assert dev_data["case_count"] == 10
    assert test_data["case_count"] == 90
    assert dev_data["split_type"] == "dev"
    assert test_data["split_type"] == "test"
