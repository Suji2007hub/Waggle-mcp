from __future__ import annotations

import argparse
import json
from pathlib import Path

from waggle.embeddings import EmbeddingModel
from waggle.graph import MemoryGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward-only migration for transcript-first storage metadata and embeddings.",
    )
    parser.add_argument("--db-path", required=True, help="SQLite database path.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name to stamp/backfill.")
    parser.add_argument("--tenant-id", default="local-default", help="Tenant to migrate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = Path(args.db_path).expanduser()
    graph = MemoryGraph(db_path, EmbeddingModel(args.model), tenant_id=args.tenant_id)
    payload = {
        "db_path": str(db_path),
        "tenant_id": args.tenant_id,
        "schema_version": 7,
        "store_health": graph.get_embedding_store_health(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
