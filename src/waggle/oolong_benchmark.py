from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from waggle.benchmark_harness import BenchmarkRuntimeError
from waggle.embeddings import EmbeddingModel
from waggle.graph import MemoryGraph
from waggle.models import NodeType, SubgraphResult
from waggle.rlm import (
    DEFAULT_WAGGLE_RLM_SYSTEM_PROMPT,
    _infer_semantic_label,
    build_gemini_backend_kwargs,
    build_groq_openai_backend_kwargs,
    build_subprocess_response_fn,
    run_gemini_one_shot,
    run_groq_one_shot,
    run_waggle_rlm,
)


DEFAULT_PROJECT = "oolong-benchmark"


@dataclass(frozen=True)
class OolongExample:
    example_id: str
    dataset_kind: str
    context_window_id: str
    question: str
    answer: str
    raw_answer: Any
    context_text: str
    metadata: dict[str, Any]


@dataclass
class OolongCaseResult:
    example_id: str
    context_window_id: str
    dataset_kind: str
    question: str
    gold_answer: str
    predicted_answer: str
    correct: bool
    retrieved_node_ids: list[str]
    retrieved_node_labels: list[str]
    retrieved_node_count: int
    retrieved_tokens: int
    prompt_tokens: int
    metadata: dict[str, Any]


@dataclass
class OolongReport:
    dataset_path: str
    dataset_kind: str
    eval_mode: str
    retrieval_mode: str
    case_count: int
    scored_case_count: int
    accuracy: float | None
    max_nodes: int
    max_depth: int
    chunk_lines: int
    chunk_overlap_lines: int
    mean_retrieved_tokens: float
    mean_prompt_tokens: float
    per_case: list[OolongCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_kind": self.dataset_kind,
            "eval_mode": self.eval_mode,
            "retrieval_mode": self.retrieval_mode,
            "case_count": self.case_count,
            "scored_case_count": self.scored_case_count,
            "accuracy": self.accuracy,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "chunk_lines": self.chunk_lines,
            "chunk_overlap_lines": self.chunk_overlap_lines,
            "mean_retrieved_tokens": self.mean_retrieved_tokens,
            "mean_prompt_tokens": self.mean_prompt_tokens,
            "per_case": [asdict(case) for case in self.per_case],
        }


def _estimate_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


def _normalize_text_answer(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(_normalize_text_answer(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value).strip()


def _maybe_parse_answer_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    if text.startswith("[") or text.startswith("{") or text.startswith("("):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    return text


def _normalize_for_match(value: Any) -> str:
    parsed = _maybe_parse_answer_literal(value)
    if isinstance(parsed, list):
        pieces = [_normalize_for_match(item) for item in parsed]
        return " | ".join(piece for piece in pieces if piece)
    text = _normalize_text_answer(parsed).lower()
    filtered = "".join(character if character.isalnum() or character.isspace() else " " for character in text)
    return " ".join(filtered.split())


def answers_match(prediction: str, gold: str) -> bool:
    return _normalize_for_match(prediction) == _normalize_for_match(gold)


_PAIR_ANSWER_PATTERN = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
_RECORD_BLOCK_PATTERN = re.compile(
    r"Example\s+(?P<example_id>\d+):\s*"
    r"Text:\s*(?P<text>.*?)\s*"
    r"User:\s*(?P<user>[^\n]+)\s*"
    r"Date:\s*(?P<date>[^\n]+)",
    re.DOTALL,
)
_UPSTREAM_REAL_REQUIRED_FIELDS = {
    "id",
    "context_window_id",
    "context_window_text",
    "question",
    "answer",
    "question_type",
}
_SYNTH_INDICATOR_FIELDS = {
    "task_group",
    "answer_type",
    "context_window_text_with_labels",
}


def _parse_pair_answer_users(value: Any) -> set[str]:
    parsed = _maybe_parse_answer_literal(value)
    users: set[str] = set()

    def _collect(text: str) -> None:
        for left, right in _PAIR_ANSWER_PATTERN.findall(text):
            users.add(left)
            users.add(right)

    if isinstance(parsed, list):
        for item in parsed:
            _collect(str(item))
        return users
    if isinstance(parsed, str):
        _collect(parsed)
    return users


def _parse_structured_context_records(context_text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for match in _RECORD_BLOCK_PATTERN.finditer(context_text):
        records.append(
            {
                "example_id": match.group("example_id").strip(),
                "text": match.group("text").strip(),
                "user": match.group("user").strip(),
                "date": match.group("date").strip(),
            }
        )
    return records


def _validate_pair_answer_reachability(example: OolongExample) -> None:
    task_group = str(example.metadata.get("task_group", "")).strip().lower()
    answer_type = str(example.metadata.get("answer_type", "")).strip().lower()
    if task_group != "oolong-pairs" or answer_type != "list":
        return

    gold_users = _parse_pair_answer_users(example.raw_answer)
    if not gold_users:
        return

    visible_qualifying_users = {
        record["user"]
        for record in _parse_structured_context_records(example.context_text)
        if _infer_semantic_label(record["text"]) in {"numeric value", "location"}
    }
    missing_users = sorted(gold_users - visible_qualifying_users, key=int)
    if not missing_users:
        return

    raise BenchmarkRuntimeError(
        "Unreachable OOLONG gold answer: "
        f"example '{example.example_id}' requires users {', '.join(missing_users)} in the gold pair set, "
        "but those users have no visible numeric-value or location record in the provided context."
    )


def _validate_retrieved_user_scope(example: OolongExample, retrieved_text: str) -> None:
    task_group = str(example.metadata.get("task_group", "")).strip().lower()
    answer_type = str(example.metadata.get("answer_type", "")).strip().lower()
    if task_group != "oolong-pairs" or answer_type != "list":
        return

    visible_users = {
        record["user"]
        for record in _parse_structured_context_records(example.context_text)
        if record.get("user", "").strip()
    }
    if not visible_users:
        return

    retrieved_users = {
        record["user"]
        for record in _parse_structured_context_records(retrieved_text)
        if record.get("user", "").strip()
    }
    leaked_users = sorted(retrieved_users - visible_users, key=int)
    if not leaked_users:
        return

    raise BenchmarkRuntimeError(
        "Cross-row retrieval contamination: "
        f"example '{example.example_id}' retrieved users {', '.join(leaked_users)} that do not appear in its context_window_text."
    )


def _read_dataset_records(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            for key in ("data", "examples", "rows", "entries"):
                value = payload.get(key)
                if isinstance(value, list):
                    records = value
                    break
            else:
                raise BenchmarkRuntimeError("Unsupported OOLONG dataset shape. Expected a list or a dict containing rows.")
        else:
            raise BenchmarkRuntimeError("Unsupported OOLONG dataset payload.")
    if not all(isinstance(record, dict) for record in records):
        raise BenchmarkRuntimeError("OOLONG dataset records must be JSON objects.")
    return records


def _infer_record_dataset_kind(record: dict[str, Any]) -> str:
    if _UPSTREAM_REAL_REQUIRED_FIELDS.issubset(record.keys()) and not (_SYNTH_INDICATOR_FIELDS & record.keys()):
        return "real"
    if _SYNTH_INDICATOR_FIELDS & record.keys():
        return "synth"
    if "question_type" in record or "episodes" in record or "campaign" in record:
        return "real"
    return "custom"


def _context_field(record: dict[str, Any], context_field: str) -> str:
    if context_field != "auto":
        if context_field not in record:
            raise BenchmarkRuntimeError(f"Requested context field '{context_field}' is missing from a dataset row.")
        return context_field
    for candidate in ("context_window_text", "context_window_text_with_labels", "context", "document", "haystack"):
        value = record.get(candidate)
        if isinstance(value, str) and value.strip():
            return candidate
    raise BenchmarkRuntimeError("Could not locate a context field in the OOLONG dataset row.")


def _resolve_dataset_kind(records: list[dict[str, Any]], dataset_kind: str) -> str:
    if dataset_kind != "auto":
        return dataset_kind

    inferred_kinds = {_infer_record_dataset_kind(record) for record in records}
    concrete_kinds = {kind for kind in inferred_kinds if kind != "custom"}
    if len(concrete_kinds) > 1:
        raise BenchmarkRuntimeError(
            "Mixed OOLONG dataset shapes are not allowed in one run. "
            "Split real and synthetic rows into separate dataset files."
        )
    if concrete_kinds:
        return next(iter(concrete_kinds))
    return "custom"


def _build_real_example(record: dict[str, Any], index: int, context_field: str) -> OolongExample:
    missing_fields = sorted(field for field in _UPSTREAM_REAL_REQUIRED_FIELDS if field not in record)
    if missing_fields:
        raise BenchmarkRuntimeError(
            f"Dataset row {index} is not a valid upstream OOLONG-real record; missing fields: {', '.join(missing_fields)}."
        )

    field_name = "context_window_text" if context_field == "auto" else _context_field(record, context_field)
    raw_answer = record.get("answer", "")
    answer = _normalize_text_answer(_maybe_parse_answer_literal(raw_answer))
    question = str(record.get("question", "")).strip()
    if not question:
        raise BenchmarkRuntimeError(f"Dataset row {index} is missing a question.")
    context_text = str(record.get(field_name, "")).strip()
    if not context_text:
        raise BenchmarkRuntimeError(f"Dataset row {index} is missing context text in '{field_name}'.")

    metadata = {
        key: value
        for key, value in record.items()
        if key not in {field_name, "question", "answer", "gold_answer"}
    }
    return OolongExample(
        example_id=str(record.get("id") or f"example-{index}").strip(),
        dataset_kind="real",
        context_window_id=str(record.get("context_window_id") or f"context-{index}").strip(),
        question=question,
        answer=answer,
        raw_answer=raw_answer,
        context_text=context_text,
        metadata=metadata,
    )


def _build_nonreal_example(
    record: dict[str, Any],
    index: int,
    *,
    dataset_kind: str,
    context_field: str,
) -> OolongExample:
    field_name = _context_field(record, context_field)
    raw_answer = record.get("answer", record.get("gold_answer", ""))
    answer = _normalize_text_answer(_maybe_parse_answer_literal(raw_answer))
    question = str(record.get("question", "")).strip()
    if not question:
        raise BenchmarkRuntimeError(f"Dataset row {index} is missing a question.")
    context_text = str(record.get(field_name, "")).strip()
    if not context_text:
        raise BenchmarkRuntimeError(f"Dataset row {index} is missing context text in '{field_name}'.")
    context_window_id = str(
        record.get("context_window_id")
        or record.get("window_id")
        or record.get("conversation_id")
        or record.get("id")
        or f"context-{index}"
    ).strip()
    example_id = str(record.get("id") or record.get("example_id") or f"example-{index}").strip()
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {field_name, "question", "answer", "gold_answer"}
    }
    return OolongExample(
        example_id=example_id,
        dataset_kind=dataset_kind,
        context_window_id=context_window_id,
        question=question,
        answer=answer,
        raw_answer=raw_answer,
        context_text=context_text,
        metadata=metadata,
    )


def load_oolong_examples(
    dataset_path: str | Path,
    *,
    dataset_kind: str = "auto",
    context_field: str = "auto",
    limit: int | None = None,
) -> list[OolongExample]:
    records = _read_dataset_records(dataset_path)
    resolved_dataset_kind = _resolve_dataset_kind(records, dataset_kind)
    examples: list[OolongExample] = []
    for index, record in enumerate(records):
        if resolved_dataset_kind == "real":
            examples.append(_build_real_example(record, index, context_field))
        else:
            examples.append(
                _build_nonreal_example(
                    record,
                    index,
                    dataset_kind=resolved_dataset_kind,
                    context_field=context_field,
                )
            )
        _validate_pair_answer_reachability(examples[-1])
        if limit is not None and len(examples) >= limit:
            break
    return examples


def _episode_chunks(text: str) -> list[str]:
    marker = "[START OF EPISODE]"
    if marker not in text:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == marker and current:
            chunks.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _line_chunks(text: str, *, chunk_lines: int, overlap_lines: int) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) <= chunk_lines:
        return ["\n".join(lines)]
    chunks: list[str] = []
    step = max(1, chunk_lines - overlap_lines)
    for start in range(0, len(lines), step):
        chunk = lines[start : start + chunk_lines]
        if not chunk:
            continue
        chunks.append("\n".join(chunk))
        if start + chunk_lines >= len(lines):
            break
    return chunks


def _chunk_context(text: str, *, chunk_lines: int, overlap_lines: int) -> list[str]:
    episode_chunks = _episode_chunks(text)
    if episode_chunks:
        return episode_chunks
    return _line_chunks(text, chunk_lines=chunk_lines, overlap_lines=overlap_lines)


def _index_context_window(
    graph: MemoryGraph,
    example: OolongExample,
    *,
    project: str,
    chunk_lines: int,
    overlap_lines: int,
) -> None:
    chunks = _chunk_context(example.context_text, chunk_lines=chunk_lines, overlap_lines=overlap_lines)
    if not chunks:
        chunks = [example.context_text]

    previous_chunk_id = ""
    for index, chunk in enumerate(chunks):
        stored = graph.add_node(
            label=f"OOLONG Chunk {index + 1}",
            content=chunk,
            node_type=NodeType.NOTE,
            tags=["benchmark", "oolong", f"context:{example.context_window_id}", f"chunk:{index + 1}"],
            project=project,
            session_id=example.context_window_id,
        ).node
        if previous_chunk_id:
            graph.add_edge(source_id=stored.id, target_id=previous_chunk_id, relationship="relates_to")
            graph.add_edge(source_id=previous_chunk_id, target_id=stored.id, relationship="relates_to")
        previous_chunk_id = stored.id


def _bundle_text(result: SubgraphResult) -> str:
    lines: list[str] = []
    for index, node in enumerate(result.nodes, start=1):
        lines.append(f"[Node {index}] {node.label}")
        lines.append(node.content)
    return "\n\n".join(lines).strip()


def build_oolong_prompt(example: OolongExample, result: SubgraphResult) -> str:
    bundle = _bundle_text(result)
    return (
        "You are answering an OOLONG long-context benchmark question using only the retrieved Waggle context.\n"
        "Return just the final answer with no explanation. If the question requests a specific format, follow it exactly.\n"
        "If the answer is not supported by the retrieved context, return NOT_ENOUGH_CONTEXT.\n\n"
        f"Question:\n{example.question}\n\n"
        f"Retrieved Waggle context:\n{bundle}\n\n"
        "Final answer:"
    )


def run_subprocess_llm(prompt: str, *, command_template: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_path = Path(tmpdir) / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        command = command_template.format(prompt_file=str(prompt_path), prompt=prompt)
        completed = subprocess.run(
            shlex.split(command),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise BenchmarkRuntimeError(
                f"LLM command failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()


def evaluate_oolong(
    dataset_path: str | Path,
    *,
    dataset_kind: str = "auto",
    context_field: str = "auto",
    embedding_model: Any,
    eval_mode: str = "retrieval_only",
    retrieval_mode: str = "graph",
    max_nodes: int = 8,
    max_depth: int = 1,
    chunk_lines: int = 12,
    chunk_overlap_lines: int = 3,
    limit: int | None = None,
    llm_answerer: Callable[[str], str] | None = None,
    project: str = DEFAULT_PROJECT,
    rlm_system_prompt: str = DEFAULT_WAGGLE_RLM_SYSTEM_PROMPT,
    rlm_max_iterations: int = 6,
    rlm_backend: str = "openai",
    rlm_backend_kwargs: dict[str, Any] | None = None,
    rlm_mock_response_fn: Callable[[str | dict[str, Any]], str] | None = None,
) -> OolongReport:
    if eval_mode not in {"retrieval_only", "waggle_llm", "waggle_rlm"}:
        raise BenchmarkRuntimeError("eval_mode must be one of: retrieval_only, waggle_llm, waggle_rlm.")
    retrieval_mode = {"fusion": "hybrid", "replay": "verbatim"}.get(retrieval_mode, retrieval_mode)
    if retrieval_mode not in {"graph", "hybrid", "verbatim", "aggregate"}:
        raise BenchmarkRuntimeError("retrieval_mode must be one of: graph, hybrid, verbatim, aggregate.")
    if eval_mode == "waggle_llm" and llm_answerer is None:
        raise BenchmarkRuntimeError("waggle_llm mode requires an llm_answerer.")
    if eval_mode == "waggle_rlm" and rlm_backend_kwargs is None and rlm_mock_response_fn is None:
        raise BenchmarkRuntimeError("waggle_rlm mode requires backend kwargs or an rlm mock response function.")

    examples = load_oolong_examples(
        dataset_path,
        dataset_kind=dataset_kind,
        context_field=context_field,
        limit=limit,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = MemoryGraph(
            Path(tmpdir) / "oolong-benchmark.db",
            embedding_model,
            dedup_similarity_threshold=1.01,
            dedup_same_label_threshold=1.01,
        )
        indexed_windows: set[str] = set()
        per_case: list[OolongCaseResult] = []

        for example in examples:
            if example.context_window_id not in indexed_windows:
                _index_context_window(
                    graph,
                    example,
                    project=project,
                    chunk_lines=chunk_lines,
                    overlap_lines=chunk_overlap_lines,
                )
                indexed_windows.add(example.context_window_id)

            if retrieval_mode == "aggregate":
                result = graph.aggregate(
                    query=example.question,
                    max_nodes=max_nodes,
                    max_depth=max_depth,
                    project=project,
                    session_id=example.context_window_id,
                )
            else:
                result = graph.query(
                    query=example.question,
                    max_nodes=max_nodes,
                    max_depth=max_depth,
                    retrieval_mode=retrieval_mode,
                    project=project,
                    session_id=example.context_window_id,
                )
            prompt = build_oolong_prompt(example, result)
            prediction = ""
            correct = False
            retrieved_node_ids = [node.id for node in result.nodes]
            retrieved_node_labels = [node.label for node in result.nodes]
            bundle_text = _bundle_text(result)
            _validate_retrieved_user_scope(example, bundle_text)
            retrieved_tokens = _estimate_tokens(bundle_text)
            prompt_tokens = _estimate_tokens(prompt)
            if eval_mode == "waggle_llm" and llm_answerer is not None:
                prediction = llm_answerer(prompt).strip()
                correct = answers_match(prediction, example.answer)
            if eval_mode == "waggle_rlm":
                rlm_result = run_waggle_rlm(
                    graph,
                    question=example.question,
                    context=example.context_text,
                    project=project,
                    session_id=example.context_window_id,
                    retrieval_mode=retrieval_mode,
                    max_nodes=max_nodes,
                    max_depth=max_depth,
                    system_prompt=rlm_system_prompt,
                    max_iterations=rlm_max_iterations,
                    backend=rlm_backend,
                    backend_kwargs=rlm_backend_kwargs,
                    mock_response_fn=rlm_mock_response_fn,
                )
                prediction = rlm_result.answer.strip()
                correct = answers_match(prediction, example.answer)
                metadata = dict(example.metadata)
                metadata.update(rlm_result.to_metadata())
                visited_nodes = [graph.get_node(node_id) for node_id in rlm_result.visited_node_ids]
                retrieved_node_ids = [node.id for node in visited_nodes]
                retrieved_node_labels = [node.label for node in visited_nodes]
                bundle_text = "\n\n".join(node.content for node in visited_nodes).strip()
                _validate_retrieved_user_scope(example, bundle_text)
                retrieved_tokens = _estimate_tokens(bundle_text)
                prompt_tokens = _estimate_tokens(example.question)
            else:
                metadata = dict(example.metadata)

            per_case.append(
                OolongCaseResult(
                    example_id=example.example_id,
                    context_window_id=example.context_window_id,
                    dataset_kind=example.dataset_kind,
                    question=example.question,
                    gold_answer=example.answer,
                    predicted_answer=prediction,
                    correct=correct,
                    retrieved_node_ids=retrieved_node_ids,
                    retrieved_node_labels=retrieved_node_labels,
                    retrieved_node_count=len(retrieved_node_ids),
                    retrieved_tokens=retrieved_tokens,
                    prompt_tokens=prompt_tokens,
                    metadata=metadata,
                )
            )

    scored = [case for case in per_case if case.predicted_answer]
    accuracy = (sum(1 for case in scored if case.correct) / len(scored)) if scored else None
    return OolongReport(
        dataset_path=str(dataset_path),
        dataset_kind=examples[0].dataset_kind if examples else dataset_kind,
        eval_mode=eval_mode,
        retrieval_mode=retrieval_mode,
        case_count=len(per_case),
        scored_case_count=len(scored),
        accuracy=accuracy,
        max_nodes=max_nodes,
        max_depth=max_depth,
        chunk_lines=chunk_lines,
        chunk_overlap_lines=chunk_overlap_lines,
        mean_retrieved_tokens=(sum(case.retrieved_tokens for case in per_case) / len(per_case)) if per_case else 0.0,
        mean_prompt_tokens=(sum(case.prompt_tokens for case in per_case) / len(per_case)) if per_case else 0.0,
        per_case=per_case,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OOLONG long-context evaluation over Waggle retrieval.")
    parser.add_argument("dataset_path", help="Path to a local OOLONG JSON or JSONL export.")
    parser.add_argument("--dataset-kind", choices=["auto", "real", "synth", "custom"], default="auto")
    parser.add_argument("--context-field", default="auto")
    parser.add_argument("--eval-mode", choices=["retrieval_only", "waggle_llm", "waggle_rlm"], default="retrieval_only")
    parser.add_argument("--llm-command", default="", help="Command template that reads {prompt_file} and prints an answer.")
    parser.add_argument("--llm-backend", choices=["command", "groq", "gemini"], default="command")
    parser.add_argument("--llm-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--llm-api-key-env", default="GROQ_API_KEY")
    parser.add_argument("--llm-max-tokens", type=int, default=512)
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--retrieval-mode", choices=["graph", "fusion", "replay", "aggregate"], default="graph")
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--chunk-lines", type=int, default=12)
    parser.add_argument("--chunk-overlap-lines", type=int, default=3)
    parser.add_argument("--rlm-system-prompt-file", default="")
    parser.add_argument("--rlm-max-iterations", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    llm_answerer: Callable[[str], str] | None = None
    rlm_backend = "openai"
    rlm_backend_kwargs: dict[str, Any] | None = None
    rlm_mock_response_fn: Callable[[str | dict[str, Any]], str] | None = None

    if args.llm_backend == "command":
        if args.llm_command:
            llm_answerer = lambda prompt: run_subprocess_llm(prompt, command_template=args.llm_command)
            rlm_mock_response_fn = build_subprocess_response_fn(args.llm_command)
    elif args.llm_backend == "groq":
        api_key = os.environ.get(args.llm_api_key_env, "").strip()
        if not api_key:
            raise BenchmarkRuntimeError(f"{args.llm_api_key_env} is required for --llm-backend groq.")
        rlm_backend_kwargs = build_groq_openai_backend_kwargs(api_key=api_key, model_name=args.llm_model)
        llm_answerer = lambda prompt: run_groq_one_shot(
            prompt=prompt,
            api_key=api_key,
            model_name=args.llm_model,
            max_tokens=args.llm_max_tokens,
            timeout_seconds=args.llm_timeout_seconds,
        )
    elif args.llm_backend == "gemini":
        api_key = os.environ.get(args.llm_api_key_env, "").strip()
        if not api_key:
            raise BenchmarkRuntimeError(f"{args.llm_api_key_env} is required for --llm-backend gemini.")
        rlm_backend = "gemini"
        rlm_backend_kwargs = build_gemini_backend_kwargs(api_key=api_key, model_name=args.llm_model)
        llm_answerer = lambda prompt: run_gemini_one_shot(
            prompt=prompt,
            api_key=api_key,
            model_name=args.llm_model,
            timeout_seconds=args.llm_timeout_seconds,
        )

    rlm_system_prompt = DEFAULT_WAGGLE_RLM_SYSTEM_PROMPT
    if args.rlm_system_prompt_file:
        rlm_system_prompt = Path(args.rlm_system_prompt_file).read_text(encoding="utf-8")

    report = evaluate_oolong(
        args.dataset_path,
        dataset_kind=args.dataset_kind,
        context_field=args.context_field,
        embedding_model=EmbeddingModel(args.embedding_model),
        eval_mode=args.eval_mode,
        retrieval_mode=args.retrieval_mode,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        chunk_lines=args.chunk_lines,
        chunk_overlap_lines=args.chunk_overlap_lines,
        limit=args.limit,
        llm_answerer=llm_answerer,
        rlm_system_prompt=rlm_system_prompt,
        rlm_max_iterations=args.rlm_max_iterations,
        rlm_backend=rlm_backend,
        rlm_backend_kwargs=rlm_backend_kwargs,
        rlm_mock_response_fn=rlm_mock_response_fn,
    )

    payload = report.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
