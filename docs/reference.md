# Waggle Reference

This page keeps the lower-level operational and configuration material out of the top-level README.

## Installation variants

### Local / development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
waggle-mcp init
```

If `.venv` already exists from a different Python version, remove it and recreate it. Reusing a stale environment can leave wrapper scripts pointing at the wrong interpreter.

### Neo4j backend

```bash
pip install -e ".[dev,neo4j]"

WAGGLE_TRANSPORT=http \
WAGGLE_BACKEND=neo4j \
WAGGLE_DEFAULT_TENANT_ID=workspace-default \
WAGGLE_NEO4J_URI=bolt://localhost:7687 \
WAGGLE_NEO4J_USERNAME=neo4j \
WAGGLE_NEO4J_PASSWORD=change-me \
waggle-mcp
```

### Docker

```bash
docker build -t waggle-mcp:latest .

docker run --rm waggle-mcp:latest --help

docker run --rm -p 8080:8080 \
  -e WAGGLE_TRANSPORT=http \
  -e WAGGLE_BACKEND=neo4j \
  -e WAGGLE_DEFAULT_TENANT_ID=workspace-default \
  -e WAGGLE_NEO4J_URI=bolt://host.docker.internal:7687 \
  -e WAGGLE_NEO4J_USERNAME=neo4j \
  -e WAGGLE_NEO4J_PASSWORD=change-me \
  waggle-mcp:latest
```

## Manual client configuration

### Claude Desktop

```json
{
  "mcpServers": {
    "waggle": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "waggle.server"],
      "env": {
        "WAGGLE_TRANSPORT": "stdio",
        "WAGGLE_BACKEND": "sqlite",
        "WAGGLE_DB_PATH": "~/.waggle/memory.db",
        "WAGGLE_DEFAULT_TENANT_ID": "local-default",
        "WAGGLE_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

### Claude Code

Claude Code supports MCP servers directly. The two practical ways to add Waggle are:

```bash
# Project-local (default)
claude mcp add waggle --scope local --env WAGGLE_TRANSPORT=stdio --env WAGGLE_BACKEND=sqlite --env WAGGLE_DB_PATH=~/.waggle/memory.db --env WAGGLE_DEFAULT_TENANT_ID=local-default --env WAGGLE_MODEL=all-MiniLM-L6-v2 -- /path/to/.venv/bin/python -m waggle.server

# Shared project config in .mcp.json
claude mcp add waggle --scope project --env WAGGLE_TRANSPORT=stdio --env WAGGLE_BACKEND=sqlite --env WAGGLE_DB_PATH=~/.waggle/memory.db --env WAGGLE_DEFAULT_TENANT_ID=local-default --env WAGGLE_MODEL=all-MiniLM-L6-v2 -- /path/to/.venv/bin/python -m waggle.server
```

Equivalent `.mcp.json` entry:

```json
{
  "mcpServers": {
    "waggle": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "waggle.server"],
      "env": {
        "WAGGLE_TRANSPORT": "stdio",
        "WAGGLE_BACKEND": "sqlite",
        "WAGGLE_DB_PATH": "~/.waggle/memory.db",
        "WAGGLE_DEFAULT_TENANT_ID": "local-default",
        "WAGGLE_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

Useful Claude Code commands after setup:

```bash
claude mcp list
claude mcp get waggle
```

### Codex

```toml
[mcp_servers.waggle]
command = "/path/to/.venv/bin/python"
args    = ["-m", "waggle.server"]
env     = {
  WAGGLE_TRANSPORT         = "stdio",
  WAGGLE_BACKEND           = "sqlite",
  WAGGLE_DB_PATH           = "~/.waggle/memory.db",
  WAGGLE_DEFAULT_TENANT_ID = "local-default",
  WAGGLE_MODEL             = "all-MiniLM-L6-v2"
}
```

A pre-filled example is in [codex_config.example.toml](../codex_config.example.toml).

### Cursor

Cursor supports MCP in both the editor and the CLI. In the editor, open `Cursor Settings -> Features -> MCP Servers` and add a new stdio server with:

- Name: `waggle`
- Command: `/path/to/.venv/bin/python`
- Arguments: `-m`, `waggle.server`

Environment variables:

```text
WAGGLE_TRANSPORT=stdio
WAGGLE_BACKEND=sqlite
WAGGLE_DB_PATH=~/.waggle/memory.db
WAGGLE_DEFAULT_TENANT_ID=local-default
WAGGLE_MODEL=all-MiniLM-L6-v2
```

If you prefer JSON configuration, use the same `mcpServers` object shape shown for Claude Desktop above.

### Antigravity

Antigravity supports custom MCP servers through its MCP manager.

Steps:
- Open the agent panel
- Open the `...` menu
- Choose `Manage MCP Servers`
- Choose `View raw config`
- Add Waggle to the config file

Configuration:

```json
{
  "mcpServers": {
    "waggle": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "waggle.server"],
      "env": {
        "WAGGLE_TRANSPORT": "stdio",
        "WAGGLE_BACKEND": "sqlite",
        "WAGGLE_DB_PATH": "~/.waggle/memory.db",
        "WAGGLE_DEFAULT_TENANT_ID": "local-default",
        "WAGGLE_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

## Using Waggle In MCP Clients

After Waggle is installed as an MCP server, the normal workflow is conversational. Users usually do not run `waggle-mcp` commands during everyday work. They talk to the agent normally, and the agent decides when to call Waggle's MCP tools.

### Codex

- Work in a normal Codex thread.
- Codex can use `observe_conversation`, `store_node`, `store_edge`, `query_graph`, and `prime_context` to persist and retrieve memory.
- Later tasks can recover connected graph context even when the original thread is no longer in the current window.

### Claude Code

- Claude Code can use Waggle as a persistent MCP memory layer.
- It is useful for carrying decisions, constraints, and project state across sessions.
- `prime_context` and `export_context_bundle` are especially useful when starting a new task or handing work to another model.

### Cursor

- Cursor can use Waggle over MCP while you work in the editor.
- That lets the agent recover earlier facts and connected rationale instead of relying only on the current chat.

### Antigravity

- Antigravity can use Waggle as a persistent graph memory backend over MCP.
- Conversation memory can be extracted with `observe_conversation`, and linked context can be exported with `export_context_bundle`.

### Important behavior

- `store_node` saves one node directly, but does not create edges by itself.
- Edges come from:
  - explicit `store_edge` calls
  - `observe_conversation`
  - `decompose_and_store`
  - automatic contradiction/update detection in some cases
- The graph-aware retrieval tools are what return connected context to the model:
  - `query_graph`
  - `get_related`
  - `get_node_history`
  - `prime_context`
  - `export_context_bundle`

For a built-in CLI explainer, run:

```bash
waggle-mcp features
```

## Environment variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_BACKEND` | `sqlite` | `sqlite` or `neo4j` |
| `WAGGLE_TRANSPORT` | `stdio` | `stdio` or `http` |
| `WAGGLE_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `WAGGLE_DEFAULT_TENANT_ID` | `local-default` | default tenant |
| `WAGGLE_EXPORT_DIR` | — | optional export directory |

### SQLite

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_DB_PATH` | `memory.db` | path to the SQLite file |

### HTTP service

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_HTTP_HOST` | `0.0.0.0` | bind host |
| `WAGGLE_HTTP_PORT` | `8080` | bind port |
| `WAGGLE_LOG_LEVEL` | `INFO` | log level |
| `WAGGLE_RATE_LIMIT_RPM` | `120` | global rate limit |
| `WAGGLE_WRITE_RATE_LIMIT_RPM` | `60` | write-tool rate limit |
| `WAGGLE_MAX_CONCURRENT_REQUESTS` | `8` | concurrency cap |
| `WAGGLE_MAX_PAYLOAD_BYTES` | `1048576` | max request size |
| `WAGGLE_REQUEST_TIMEOUT_SECONDS` | `30` | per-request timeout |

### Neo4j

| Variable | Description |
|----------|-------------|
| `WAGGLE_NEO4J_URI` | Bolt URI, e.g. `bolt://localhost:7687` |
| `WAGGLE_NEO4J_USERNAME` | Neo4j username |
| `WAGGLE_NEO4J_PASSWORD` | Neo4j password |
| `WAGGLE_NEO4J_DATABASE` | Neo4j database name |

### Extraction

No extra extraction runtime is required. `observe_conversation` uses the built-in deterministic parser and stores only structured facts that map cleanly onto Waggle node types.

## Admin commands

```bash
# Create a tenant
waggle-mcp create-tenant --tenant-id workspace-a --name "Workspace A"

# Issue an API key (raw key returned once)
waggle-mcp create-api-key --tenant-id workspace-a --name "ci-agent"

# List keys for a tenant
waggle-mcp list-api-keys --tenant-id workspace-a

# Revoke a key
waggle-mcp revoke-api-key --api-key-id <id>

# Migrate SQLite data → Neo4j
WAGGLE_BACKEND=neo4j WAGGLE_NEO4J_URI=bolt://localhost:7687 \
WAGGLE_NEO4J_USERNAME=neo4j WAGGLE_NEO4J_PASSWORD=change-me \
  waggle-mcp migrate-sqlite --db-path ./memory.db --tenant-id workspace-a
```

## Full tool surface

| Tool | What it does |
|------|--------------|
| `observe_conversation` | Ingest a conversation turn into graph memory |
| `query_graph` | Semantic + temporal search across graph, replay, or fusion |
| `store_node` | Manually save a fact, preference, decision, or note |
| `store_edge` | Link two nodes with a typed relationship |
| `get_related` | Traverse edges from a specific node |
| `get_node_history` | Inspect evidence, validity window, and related context |
| `list_context_scopes` | Enumerate stored `agent_id`, `project`, and `session_id` scopes |
| `timeline` | Build a chronological memory view |
| `list_conflicts` | List unresolved contradiction and update edges |
| `resolve_conflict` | Mark a contradiction or update edge as resolved |
| `update_node` | Update content or tags on an existing node |
| `delete_node` | Remove a node and all its edges |
| `decompose_and_store` | Break long content into atomic nodes automatically |
| `graph_diff` | See what changed in the last N hours |
| `prime_context` | Generate a compact brief for a new conversation |
| `get_topics` | Detect topic clusters via community detection |
| `get_stats` | Node/edge counts and most-connected nodes |
| `export_graph_html` | Interactive browser visualization |
| `export_graph_backup` | Portable JSON backup |
| `import_graph_backup` | Restore from a JSON backup |
| `export_context_bundle` | Export Markdown/JSON context packs for another AI |
| `export_markdown_vault` | Export one-file-per-node Markdown vaults |
| `import_markdown_vault` | Re-import edited Markdown vault files |

## Architecture snapshot

```text
waggle-mcp
├── Core domain
│   ├── graph CRUD (nodes, edges, evidence)
│   ├── dedup (semantic + exact)
│   ├── conflict detection (auto-contradiction)
│   ├── context assembly (query, prime, timeline)
│   ├── export/import (JSON, Markdown, GraphML)
│   └── local embeddings (SentenceTransformers + SHA-256 fallback)
├── Transport
│   ├── stdio MCP (local clients)
│   └── HTTP MCP (server-to-server)
└── Platform
    ├── auth (API keys + tenant isolation)
    ├── storage (SQLite + Neo4j)
    └── operations (rate limiting, logging, metrics)
```

Backend defaults:
- local/dev → SQLite
- production → Neo4j

Repository layout:

```text
waggle-mcp/
├── assets/
├── benchmarks/fixtures/
├── benchmarks/longmemeval/
├── deploy/
├── docs/runbooks/
├── scripts/
├── src/waggle/
├── tests/artifacts/
├── Dockerfile
├── pyproject.toml
└── README.md
```
## Deduplication methodology

Deduplication in Waggle is intentionally conservative to prevent false merges of distinct but similar facts.

1. **Exact Content**: Case-insensitive, whitespace-normalized equality check.
2. **Same-Label High-Similarity**: If labels are identical or acronym matches, a lower similarity threshold (`0.90` default) is used.
3. **Semantic Similarity**: General node-to-node comparison using cosine similarity. Default threshold is `0.82` to ensure zero false positives across our 22-case fixture.

The system prefers creating "Derived From" or "Updates" edges over destructive merging when similarity is ambiguous.

## Context Assembly: Graph vs Flat

Naive RAG often stuffs irrelevant chunks into the prompt, wasting tokens and confusing reasoning. Waggle's graph retrieval builds a structured context subgraph focused on the query's dependency chain.

### Before: Naive Chunks (151 tokens)
> [Chunk 1] Keep SQLite for local development for now.
> [Chunk 2] Production is moving to PostgreSQL for parity.
> [Chunk 3] PostgreSQL is the production database.
> [Chunk 4] User: What is our database choice? Agent: You chose PostgreSQL.

### After: Waggle Graph (58 tokens)
> **Decisions**
> - [id:db_postgres] "PostgreSQL production" - PostgreSQL is the prod DB.
>   - *Updates*: "SQLite local only"
>   - *Contradicts*: "SQLite local only" (superseded-state)
