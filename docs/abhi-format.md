# ABHI Format

`.abhi` is Waggle's portable graph snapshot format.

## Container choice

- Underlying format: JSON
- Reason: human-inspectable, easy to diff in git, stable enough for deterministic round-trip tests, and already compatible with the existing CLI and MCP export/import surface.

## Snapshot contents

- Nodes: `id`, `type`, `content`, and metadata including label, tenant, agent, project, session, timestamps, tags, evidence, source prompt, and `source_app`.
- Edges: `id`, `from`, `to`, `type`, and metadata including `label`, `prompt`, `weight`, and timestamp.
- Forward-compat versioning: `integrity.abhi_spec_version` plus `integrity.schema_version`.
- Optional embeddings blob: top-level `embeddings.vectors` as `float32` vectors encoded with base64 per node id.

## Encryption

- Optional at-rest encryption is supported with AES-256-GCM.
- Key derivation uses PBKDF2-HMAC-SHA256.
- Encrypted files require a passphrase at import/validate/inspect/query time.
- Unencrypted `.abhi` files should be treated as sensitive because they may contain full conversation history.

## Drive sync

- `waggle-mcp push` exports the current graph to `.abhi` and uploads it to a configured Google Drive folder.
- `waggle-mcp pull <file>` downloads a Drive `.abhi`, merges it locally, and imports the merged result.
- `waggle-mcp share <file>` creates an anyone-with-link Drive URL.

## Merge strategy

- Drive pull uses `last_write_wins`.
- Resolution is per stable object id.
- For conflicting node or edge payloads, the object with the newer timestamp wins.
- If timestamps tie, the remote side wins.
