from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_cache_key(
    *,
    benchmark: str,
    case_id: str,
    arm: str,
    answer_model: str,
    judge_model: str,
    prompt_template: str,
    retrieval_limit: int,
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        "benchmark": benchmark,
        "case_id": case_id,
        "arm": arm,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "prompt_template_sha": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "retrieval_limit": retrieval_limit,
        "extra": extra or {},
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:32]


class BenchmarkCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.payload_dir = self.cache_dir / "payloads"
        self.index_path = self.cache_dir / "index.jsonl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self._known_keys = self._load_known_keys()

    def _load_known_keys(self) -> set[str]:
        if not self.index_path.exists():
            return set()
        keys: set[str] = set()
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    keys.add(str(json.loads(raw)["key"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return keys

    def has(self, key: str) -> bool:
        return key in self._known_keys and (self.payload_dir / f"{key}.json").exists()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.payload_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: Any, *, meta: dict[str, Any] | None = None) -> None:
        payload = asdict(value) if is_dataclass(value) else value
        payload_path = self.payload_dir / f"{key}.json"
        _atomic_write_json(payload_path, payload)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, **(meta or {})}, ensure_ascii=False) + "\n")
        self._known_keys.add(key)

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._known_keys),
            "cache_dir": str(self.cache_dir),
            "index_path": str(self.index_path),
        }


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
