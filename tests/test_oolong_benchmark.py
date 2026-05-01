from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from waggle.benchmark_harness import BenchmarkRuntimeError
from waggle.oolong_benchmark import (
    _validate_retrieved_user_scope,
    answers_match,
    evaluate_oolong,
    load_oolong_examples,
)
from waggle.rlm import _infer_semantic_label


class FakeEmbeddingModel:
    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(1024, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[sum(ord(character) for character in token) % len(vector)] += 1.0
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


def test_load_oolong_examples_normalizes_list_answers(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oolong.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "question": "Which topping was selected?",
                "answer": "['ham']",
                "context_window_text": "Order summary\nSelected topping: ham",
                "task_group": "single_hop",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_oolong_examples(dataset_path)

    assert len(examples) == 1
    assert examples[0].dataset_kind == "synth"
    assert examples[0].answer == "ham"


def test_load_oolong_examples_accepts_upstream_real_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oolong-real.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "3952f2d5-082f-14b2-5ec4-d9cbedd2f865",
                "context_window_id": "e4fb38b9-ffca-0729-d52a-02fffd17610a",
                "context_window_text": (
                    "The following lines contains a single episode transcript of a Dungeons and Dragons game. "
                    "[START OF EPISODE]\nBob rolled a 4.\n[END OF EPISODE]"
                ),
                "question": "Total number of rolls in this episode?",
                "answer": "1",
                "question_type": "singledoc_rolls",
                "episodes": [1],
                "campaign": "campaign2",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_oolong_examples(dataset_path)

    assert len(examples) == 1
    assert examples[0].dataset_kind == "real"
    assert examples[0].example_id == "3952f2d5-082f-14b2-5ec4-d9cbedd2f865"
    assert examples[0].context_window_id == "e4fb38b9-ffca-0729-d52a-02fffd17610a"
    assert examples[0].metadata["question_type"] == "singledoc_rolls"
    assert examples[0].metadata["campaign"] == "campaign2"


def test_load_oolong_examples_rejects_mixed_real_and_synth_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oolong-mixed.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "real-1",
                        "context_window_id": "ctx-real",
                        "context_window_text": "[START OF EPISODE]\nBob likes coffee.\n[END OF EPISODE]",
                        "question": "Which drink does Bob like?",
                        "answer": "coffee",
                        "question_type": "single_hop",
                        "episodes": [1],
                        "campaign": "campaign-a",
                    }
                ),
                json.dumps(
                    {
                        "id": "synth-1",
                        "context_window_id": "ctx-synth",
                        "context_window_text": "Example 1:\nText: The city of Paris is nice.\nUser: 101\nDate: 2026-04-03",
                        "question": "Which users have locations?",
                        "answer": "101",
                        "answer_type": "list",
                        "task_group": "oolong-pairs",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkRuntimeError, match="Mixed OOLONG dataset shapes are not allowed"):
        load_oolong_examples(dataset_path)


def test_load_oolong_examples_rejects_explicit_real_kind_for_synth_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oolong-synth.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "example_id": "clean-0",
                "context_window_id": "cw-clean-10000-user",
                "context_window_text": "Example 1:\nText: The city of Paris is nice.\nUser: 101\nDate: 2026-04-03",
                "question": "Which users have locations?",
                "answer": "101",
                "answer_type": "ANSWER_TYPE.USER",
                "task_group": "user",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkRuntimeError, match="not a valid upstream OOLONG-real record"):
        load_oolong_examples(dataset_path, dataset_kind="real")


def test_load_oolong_examples_rejects_unreachable_pair_gold(tmp_path: Path) -> None:
    dataset_path = tmp_path / "broken-oolong-pairs.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "question": (
                    "Each of the questions can be labelled as one of the labels: description and abstract concept, "
                    "entity, human being, numeric value, location, abbreviation.\n\n"
                    "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                    "where both users have at least one instance with a numeric value or location.\n\n"
                    "In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines."
                ),
                "answer": ["(110, 113)", "(110, 116)", "(113, 116)"],
                "answer_type": "list",
                "task_group": "oolong-pairs",
                "context_window_text": "\n".join(
                    [
                        "Example 1:",
                        "Text: This is an abstract concept or general entity.",
                        "User: 110",
                        "Date: 2026-04-01",
                        "",
                        "Example 2:",
                        "Text: The city of Paris is nice.",
                        "User: 113",
                        "Date: 2026-04-03",
                        "",
                        "Example 3:",
                        "Text: I have 105 dollars in my pocket.",
                        "User: 116",
                        "Date: 2026-04-27",
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkRuntimeError, match="Unreachable OOLONG gold answer"):
        load_oolong_examples(dataset_path)


def test_load_oolong_examples_accepts_reachable_pair_gold(tmp_path: Path) -> None:
    dataset_path = tmp_path / "reachable-oolong-pairs.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "question": (
                    "Each of the questions can be labelled as one of the labels: description and abstract concept, "
                    "entity, human being, numeric value, location, abbreviation.\n\n"
                    "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                    "where both users have at least one instance with a numeric value or location.\n\n"
                    "In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines."
                ),
                "answer": ["(101, 102)"],
                "answer_type": "list",
                "task_group": "oolong-pairs",
                "context_window_text": "\n".join(
                    [
                        "Example 1:",
                        "Text: The city of Paris is nice.",
                        "User: 101",
                        "Date: 2026-04-03",
                        "",
                        "Example 2:",
                        "Text: I have 105 dollars in my pocket.",
                        "User: 102",
                        "Date: 2026-04-27",
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_oolong_examples(dataset_path)

    assert len(examples) == 1
    assert examples[0].answer == "(101, 102)"


def test_load_oolong_examples_accepts_generator_location_templates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "reachable-location-oolong-pairs.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "question": (
                    "Each of the questions can be labelled as one of the labels: description and abstract concept, "
                    "entity, human being, numeric value, location, abbreviation.\n\n"
                    "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                    "where both users have at least one instance with a numeric value or location.\n\n"
                    "In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines."
                ),
                "answer": ["(100, 102)"],
                "answer_type": "list",
                "task_group": "oolong-pairs",
                "context_window_text": "\n".join(
                    [
                        "Example 1:",
                        "Text: The temperature is 75 degrees.",
                        "User: 100",
                        "Date: 2026-04-03",
                        "",
                        "Example 2:",
                        "Text: The park is downtown.",
                        "User: 102",
                        "Date: 2026-04-27",
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_oolong_examples(dataset_path)

    assert len(examples) == 1
    assert examples[0].answer == "(100, 102)"


def test_infer_semantic_label_matches_synthetic_location_templates() -> None:
    assert _infer_semantic_label("Paris is the capital of France.") == "location"
    assert _infer_semantic_label("The office is in New York.") == "location"
    assert _infer_semantic_label("Mount Everest is the highest mountain.") == "location"
    assert _infer_semantic_label("The park is downtown.") == "location"


def test_evaluate_oolong_retrieval_only_reports_context_usage(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "Which drink does Bob like?",
            "answer": "coffee",
            "question_type": "single_hop",
            "context_window_text": "\n".join(
                [
                    "[START OF EPISODE]",
                    "Alice likes tea.",
                    "[END OF EPISODE]",
                    "[START OF EPISODE]",
                    "Bob likes coffee.",
                    "[END OF EPISODE]",
                ]
            ),
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="retrieval_only",
    )

    assert report.case_count == 1
    assert report.scored_case_count == 0
    assert report.accuracy is None
    assert report.per_case[0].retrieved_node_count >= 1
    assert report.per_case[0].retrieved_tokens > 0


def test_evaluate_oolong_with_llm_answerer_scores_predictions(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "Which drink does Bob like? Return only the drink.",
            "answer": "coffee",
            "question_type": "single_hop",
            "context_window_text": "Bob likes coffee.",
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    def fake_llm(prompt: str) -> str:
        return "coffee" if "Bob likes coffee" in prompt else "tea"

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="waggle_llm",
        llm_answerer=fake_llm,
    )

    assert report.case_count == 1
    assert report.scored_case_count == 1
    assert report.accuracy == 1.0
    assert report.per_case[0].predicted_answer == "coffee"
    assert report.per_case[0].correct is True


def test_evaluate_oolong_with_waggle_rlm_runs_multi_step_loop(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "Which drink does Bob like? Return only the drink.",
            "answer": "coffee",
            "question_type": "single_hop",
            "context_window_text": "\n".join(
                [
                    "[START OF EPISODE]",
                    "Alice likes tea.",
                    "[END OF EPISODE]",
                    "[START OF EPISODE]",
                    "Bob likes coffee.",
                    "[END OF EPISODE]",
                ]
            ),
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    responses = iter(
        [
            """```repl
answer_with_waggle("Which drink does Bob like? Return only the drink.", top_k=2)
```""",
            'FINAL("coffee")',
        ]
    )
    call_count = 0

    def fake_response_fn(prompt: str | dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        assert prompt
        try:
            return next(responses)
        except StopIteration:
            return 'FINAL("coffee")'

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="waggle_rlm",
        rlm_mock_response_fn=fake_response_fn,
    )

    assert report.case_count == 1
    assert report.scored_case_count == 1
    assert report.accuracy == 1.0
    assert report.per_case[0].predicted_answer == "coffee"
    assert report.per_case[0].correct is True
    metadata = report.per_case[0].metadata["rlm_metadata"]
    assert metadata is not None
    assert metadata["iterations"]
    assert report.per_case[0].retrieved_node_count >= 1
    assert call_count >= 2


def test_evaluate_oolong_with_waggle_rlm_can_read_specific_node(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "Which drink does Bob like? Return only the drink.",
            "answer": "coffee",
            "question_type": "single_hop",
            "context_window_text": "\n".join(
                [
                    "[START OF EPISODE]",
                    "Alice likes tea.",
                    "[END OF EPISODE]",
                    "[START OF EPISODE]",
                    "Bob likes coffee.",
                    "[END OF EPISODE]",
                ]
            ),
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    call_count = 0

    def fake_response_fn(prompt: str | dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        assert prompt
        if call_count == 1:
            return """```repl
matches = search_waggle("Bob likes", top_k=2)
target_node_id = matches[0]["id"]
print(target_node_id)
```"""
        if call_count == 2:
            return """```repl
node_contents = [read_node(match["id"]) for match in matches]
evidence = "\n".join(node_contents)
print(evidence)
```"""
        return 'FINAL("coffee")'

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="waggle_rlm",
        rlm_mock_response_fn=fake_response_fn,
    )

    assert report.case_count == 1
    assert report.accuracy == 1.0
    assert report.per_case[0].predicted_answer == "coffee"
    assert report.per_case[0].retrieved_node_count >= 1
    assert call_count >= 3


def test_evaluate_oolong_with_waggle_rlm_can_extract_structured_records(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "Which users have numeric values? Return only the users.",
            "answer": "113 | 116",
            "question_type": "single_hop",
            "context_window_text": "\n".join(
                [
                    "Example 1:",
                    "Text: The city of Paris is nice.",
                    "User: 113",
                    "Date: 2026-04-03",
                    "",
                    "Example 2:",
                    "Text: I have 105 dollars in my pocket.",
                    "User: 116",
                    "Date: 2026-04-27",
                    "",
                    "Example 3:",
                    "Text: This is an abstract concept or general entity.",
                    "User: 101",
                    "Date: 2026-04-13",
                ]
            ),
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    responses = iter(
        [
            """```repl
records = extract_waggle_records("numeric value or location", top_k=2)
users = sorted({record["user"] for record in records if "dollars" in record["text"].lower() or "city of" in record["text"].lower()})
print(users)
```""",
            'FINAL("113 | 116")',
        ]
    )

    def fake_response_fn(prompt: str | dict[str, object]) -> str:
        assert prompt
        try:
            return next(responses)
        except StopIteration:
            return 'FINAL("113 | 116")'

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="waggle_rlm",
        rlm_mock_response_fn=fake_response_fn,
    )

    assert report.case_count == 1
    assert report.accuracy == 1.0
    assert report.per_case[0].predicted_answer == "113 | 116"
    assert report.per_case[0].retrieved_node_count >= 1


def test_evaluate_oolong_with_waggle_rlm_structured_records_include_label(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "Which users have numeric values or locations? Return only the users.",
            "answer": "113 | 116",
            "question_type": "single_hop",
            "context_window_text": "\n".join(
                [
                    "Example 1:",
                    "Text: The city of Paris is nice.",
                    "User: 113",
                    "Date: 2026-04-03",
                    "",
                    "Example 2:",
                    "Text: I have 105 dollars in my pocket.",
                    "User: 116",
                    "Date: 2026-04-27",
                ]
            ),
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    responses = iter(
        [
            """```repl
records = classify_waggle_records("numeric value or location", top_k=2)
users = sorted({record["user"] for record in records if record["label"] in {"numeric value", "location"}})
print(users)
```""",
            'FINAL("113 | 116")',
        ]
    )

    def fake_response_fn(prompt: str | dict[str, object]) -> str:
        assert prompt
        try:
            return next(responses)
        except StopIteration:
            return 'FINAL("113 | 116")'

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="waggle_rlm",
        rlm_mock_response_fn=fake_response_fn,
    )

    assert report.case_count == 1
    assert report.accuracy == 1.0
    assert report.per_case[0].predicted_answer == "113 | 116"


def test_evaluate_oolong_with_waggle_rlm_can_pair_users_by_label(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "List all user pairs where both users have a numeric value or location.",
            "answer": "(110, 113) | (110, 116) | (113, 116)",
            "question_type": "single_hop",
            "context_window_text": "\n".join(
                [
                    "Example 1:",
                    "Text: The city of Paris is nice.",
                    "User: 113",
                    "Date: 2026-04-03",
                    "",
                    "Example 2:",
                    "Text: I have 105 dollars in my pocket.",
                    "User: 116",
                    "Date: 2026-04-27",
                    "",
                    "Example 3:",
                    "Text: I have 385 dollars in my pocket.",
                    "User: 110",
                    "Date: 2026-04-20",
                ]
            ),
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    responses = iter(
        [
            """```repl
final_answer = answer_pair_users_by_label("numeric value or location", labels=["numeric value", "location"], top_k=3)
FINAL_VAR("final_answer")
```""",
        ]
    )

    def fake_response_fn(prompt: str | dict[str, object]) -> str:
        assert prompt
        try:
            return next(responses)
        except StopIteration:
            return 'FINAL("(110, 113) | (110, 116) | (113, 116)")'

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="waggle_rlm",
        rlm_mock_response_fn=fake_response_fn,
    )

    assert report.case_count == 1
    assert report.accuracy == 1.0
    assert report.per_case[0].predicted_answer == "(110, 113)\n(110, 116)\n(113, 116)"


def test_evaluate_oolong_with_waggle_rlm_pair_users_by_label_scans_full_session(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "real-1",
            "context_window_id": "ctx-1",
            "question": "List all user pairs where both users have a numeric value or location.",
            "answer": "(101, 102) | (101, 103) | (102, 103)",
            "question_type": "single_hop",
            "context_window_text": "\n".join(
                [
                    "Example 1:",
                    "Text: The company reported record profits.",
                    "User: 100",
                    "Date: 2026-04-01",
                    "",
                    "Example 2:",
                    "Text: It costs 10 dollars.",
                    "User: 101",
                    "Date: 2026-04-02",
                    "",
                    "Example 3:",
                    "Text: Freedom is a state of mind.",
                    "User: 100",
                    "Date: 2026-04-03",
                    "",
                    "Example 4:",
                    "Text: The office is in New York.",
                    "User: 102",
                    "Date: 2026-04-04",
                    "",
                    "Example 5:",
                    "Text: Justice is blind.",
                    "User: 100",
                    "Date: 2026-04-05",
                    "",
                    "Example 6:",
                    "Text: Mount Everest is the highest mountain.",
                    "User: 103",
                    "Date: 2026-04-06",
                ]
            ),
        }
    ]
    dataset_path = tmp_path / "oolong-real.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    responses = iter(
        [
            """```repl
final_answer = answer_pair_users_by_label("numeric value or location", labels=["numeric value", "location"], top_k=1)
FINAL_VAR("final_answer")
```""",
        ]
    )

    def fake_response_fn(prompt: str | dict[str, object]) -> str:
        assert prompt
        try:
            return next(responses)
        except StopIteration:
            return 'FINAL("(101, 102) | (101, 103) | (102, 103)")'

    report = evaluate_oolong(
        dataset_path,
        embedding_model=FakeEmbeddingModel(),
        eval_mode="waggle_rlm",
        rlm_mock_response_fn=fake_response_fn,
    )

    assert report.case_count == 1
    assert report.accuracy == 1.0
    assert report.per_case[0].predicted_answer == "(101, 102)\n(101, 103)\n(102, 103)"


def test_validate_retrieved_user_scope_rejects_cross_row_contamination(tmp_path: Path) -> None:
    dataset_path = tmp_path / "oolong-synth.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "synth-1",
                "question": (
                    "Each of the questions can be labelled as one of the labels: description and abstract concept, "
                    "entity, human being, numeric value, location, abbreviation.\n\n"
                    "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                    "where both users have at least one instance with a numeric value or location.\n\n"
                    "In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines."
                ),
                "answer": ["(100, 101)"],
                "answer_type": "list",
                "task_group": "oolong-pairs",
                "context_window_text": "\n".join(
                    [
                        "Example 1:",
                        "Text: The temperature is 75 degrees.",
                        "User: 100",
                        "Date: 2026-04-03",
                        "",
                        "Example 2:",
                        "Text: The office is in New York.",
                        "User: 101",
                        "Date: 2026-04-27",
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    example = load_oolong_examples(dataset_path)[0]
    leaked_bundle = "\n".join(
        [
            "Example 99:",
            "Text: The distance is 50 miles.",
            "User: 103",
            "Date: 2026-04-28",
        ]
    )

    with pytest.raises(BenchmarkRuntimeError, match="Cross-row retrieval contamination"):
        _validate_retrieved_user_scope(example, leaked_bundle)


def test_answers_match_handles_literal_lists() -> None:
    assert answers_match("ham", "['ham']")
    assert answers_match("42", "42")
    assert not answers_match("tea", "coffee")
