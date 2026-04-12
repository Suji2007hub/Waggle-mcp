<p align="center">
  <img src="https://raw.githubusercontent.com/Abhigyan-Shekhar/graph-memory-mcp/main/assets/banner.png" alt="waggle-mcp" width="720"/>
</p>

<p align="center">
  <strong>Persistent, structured memory for AI agents — backed by a real knowledge graph.</strong><br/>
  Your LLM remembers facts, decisions, and context <em>across every conversation</em>.
</p>

<p align="center">
  <a href="https://pypi.org/project/waggle-mcp"><img src="https://img.shields.io/pypi/v/waggle-mcp?color=39d5cf&label=pypi" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/MCP-compatible-brightgreen" alt="MCP compatible"/>
  <img src="https://img.shields.io/badge/embeddings-local%2C%20no%20API%20key-orange" alt="Local embeddings"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"/>
</p>

---

## Why waggle-mcp?

Most LLMs forget everything when the conversation ends.  
`waggle-mcp` fixes that by giving your AI a **persistent knowledge graph** it can read and write through any MCP-compatible client.

Waggle's key advantage is **token efficiency with structured context**:

| Without waggle-mcp | With waggle-mcp |
|--------------------------|----------------------|
| Context stuffed into a 200k-token prompt | **~4× fewer tokens** — compact subgraph, only relevant nodes retrieved |
| "What did we decide about the DB schema?" → ❌ no idea | ✅ Recalls the decision node, when it was made, and what it contradicts |
| Flat bullet-list memory | Typed edges: `relates_to`, `contradicts`, `depends_on`, `updates`… |
| One session, one agent | Multi-tenant, multi-session, multi-agent |

> **Note on retrieval:** Waggle trades some raw recall coverage for dramatically lower token cost and richer relational context. See the [benchmark section](#performance--benchmarking) for honest numbers.

---

## Quick start — 30 seconds

```bash
pip install waggle-mcp
waggle-mcp init
```

The `init` wizard detects your MCP client, writes its config file, and creates
the database directory — no JSON editing required. Supports **Claude Desktop**,
**Cursor**, **Codex**, and a generic JSON fallback.

After init, restart your MCP client and your AI has persistent memory.  
No cloud service. No API key. Semantic search runs fully locally.

---

## See it in action

Here's a concrete before/after for a developer using the AI daily:

**Session 1** — April 10
```
User:  Let's use PostgreSQL. MySQL replication has been painful.
Agent: [calls observe_conversation()]
       → stores decision node: "Chose PostgreSQL over MySQL"
       → stores reason node:   "MySQL replication painful"
       → links them with a depends_on edge
```

**Session 2** — April 12 (fresh context window, no history)
```
User:  What did we decide about the database?
Agent: [calls query_graph("database decision")]
       → retrieves the decision node + linked reason from April 10

       "You decided on PostgreSQL on April 10. The reason recorded was
        that MySQL replication had been painful."
```

**Session 3** — April 14
```
User:  Actually, let's reconsider — the team is more familiar with MySQL.
Agent: [calls store_node() + store_edge(new_node → old_node, "contradicts")]
       → conflict is flagged automatically; both positions are preserved in the graph
```

> The agent never needed explicit instructions to remember or retrieve — it called
> the right tools based on the conversation, and the graph gave it the right context.

---

## How it works

Memory doesn't just get stored — it flows through a lifecycle:

```
You talk to your AI
        │
        ▼
  observe_conversation()          ← AI drops the turn in; facts extracted via structured LLM (regex fallback)
        │
        ▼
  Graph nodes are created         ← "Chose PostgreSQL" becomes a decision node
  Edges are inferred              ← linked to the "database" entity node
        │
        ▼
  Future conversation starts
        │
        ▼
  query_graph("DB schema")        ← semantic search finds the node from 3 sessions ago
        │
        ▼
  AI answers with full context    ← "You decided on PostgreSQL on Apr 10, here's why…"
```

Every node carries semantic embeddings computed **locally** using
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) —
a fast, lightweight model that runs entirely on-device with no API key or network
call required. This means semantic search works offline, costs nothing per query,
and keeps your data private.

---

## The magic tool: `observe_conversation`

> **This is the tool you'll use most.** You don't have to manually store facts — just
> tell the agent to observe each conversation turn and it handles the rest.

```
observe_conversation(user_message, assistant_response)
```

Under the hood, it:
1. Extracts atomic facts from both sides of the conversation
2. Deduplicates against existing nodes using semantic similarity
3. Creates typed edges between related concepts
4. Flags contradictions with existing stored beliefs

No instructions needed. No schema to define. Just observe.

---

## Fact Extraction Schema

When the agent observes a conversation, the backend runs a Pydantic-validated LLM extraction pass (or falls back to a regex engine) to pull structured facts out of messy dialogue.

**Example Input:**
> "Let's use PostgreSQL for the generic event storage because MySQL replication is too painful to manage."

**Validated Output:**
```json
{
  "facts": [
    {
      "label": "PostgreSQL for generic events",
      "content": "Chose PostgreSQL over MySQL because MySQL replication is too painful.",
      "node_type": "decision",
      "confidence": 0.95,
      "tags": ["llm-extracted", "confidence:0.95"]
    }
  ]
}
```

*Any extraction with `confidence < 0.5` or an invalid schema is silently dropped to prevent hallucination noise.*

---

## Memory model

**Node types** — what gets stored:

| Type | Example |
|------|---------| 
| `fact` | "The API uses JWT tokens" |
| `preference` | "User prefers dark mode" |
| `decision` | "Chose PostgreSQL over MySQL" |
| `entity` | "Project: waggle-mcp" |
| `concept` | "Rate limiting" |
| `question` | "Should we add GraphQL?" |
| `note` | "TODO: add integration tests" |

**Edge types** — how nodes connect:

`relates_to` · `contradicts` · `depends_on` · `part_of` · `updates` · `derived_from` · `similar_to`

---

## MCP tools

> Your AI calls these directly — you don't need to use them manually.

| Tool | What it does |
|------|-------------|
| `observe_conversation` | **Drop a conversation turn in — facts extracted, stored, and linked** |
| `query_graph` | Semantic + temporal search across the graph |
| `store_node` | Manually save a fact, preference, decision, or note |
| `store_edge` | Link two nodes with a typed relationship |
| `get_related` | Traverse edges from a specific node |
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

---

## Performance & Benchmarking

All benchmark claims in this repository are reproducible from checked-in fixtures plus the local harness at [`scripts/benchmark_extraction.py`](./scripts/benchmark_extraction.py).

**Run the harness locally:**

```bash
# Regex extraction baseline
PYTHONPATH=src .venv/bin/python scripts/benchmark_extraction.py --extraction-backend regex

# LLM extraction (requires local Ollama)
PYTHONPATH=src .venv/bin/python scripts/benchmark_extraction.py --extraction-backend llm --ollama-model qwen2.5:7b --ollama-timeout-seconds 30

# Comparative pilot (waggle vs chunked-vector RAG)
PYTHONPATH=src .venv/bin/python scripts/benchmark_extraction.py --systems waggle rag_naive --output tests/artifacts/pilot_comparative.json
```

### Extraction accuracy

Fixture set: 12 dialogue pairs covering simple recall, interruptions, reversals, vague statements, and conflicting signals.

| Backend | Cases | Accuracy |
|---------|-------|----------|
| Regex (fallback) | 12 | 33% |
| LLM (`qwen2.5:7b`, 30s timeout) | 12 | 75% |

### Retrieval accuracy

Fixture set: 12 nodes, 12 queries — 6 easy (direct paraphrase) and 6 hard (adversarial: semantic generalization, temporal disambiguation, domain-level misdirection).

| Corpus | Easy queries | Hard queries | Overall Hit@k |
|--------|-------------|-------------|---------------|
| 12-node benchmark | 6/6 = 100% | 5/6 = 83% | 11/12 = **92%** |

### Token efficiency vs. chunked-vector RAG

This is where Waggle's graph model has a **clear, measurable advantage**. The graph extracts and stores discrete facts rather than raw text chunks, so the context injected into each prompt is far smaller:

| System | Mean tokens | Median tokens | p95 tokens | Hit@k |
|--------|-------------|---------------|------------|-------|
| **Waggle** | **37.2** | **38.0** | **42.0** | 92% |
| Naive chunked-vector RAG | 152.1 | 154.0 | 163.0 | 100% |

**Waggle uses ~4× fewer tokens per retrieval** than the naive chunked baseline.

The tradeoff is honest: the chunked baseline achieves 100% Hit@k on this corpus because the corpus is not yet hard enough to stress it. **The token efficiency advantage is real and large; the retrieval advantage is not yet demonstrated at this corpus scale.** Corpus hardening is the next evaluation step — see [`tests/artifacts/pilot_comparative.md`](./tests/artifacts/pilot_comparative.md) for details.

<details>
<summary>Deduplication results (22-pair fixture — click to expand)</summary>

Fixture set: 22 node pairs — 11 true duplicates (synonym, paraphrase, domain-level equivalence) and 11 false friends (same domain, distinct semantics). Balanced to test both false-positive and false-negative failure modes.

The dedup pipeline runs three layers in order:
1. **Exact string match** — normalized content or label equality (free, always run)
2. **Substring containment** — catches restatements where one phrase is a subset of the other
3. **Semantic similarity** — cosine via `all-MiniLM-L6-v2`:
   - Type-aware threshold: `decision`/`preference` nodes merge at 0.82; `fact` at 0.92; `entity` at 0.97
   - Jaccard-boosted path: if bag-of-words overlap ≥ 0.35 AND cosine ≥ (type threshold − 0.05), treat as duplicate
   - Conservative cosine fallback at global `dedup_similarity_threshold`

Best measured result on the expanded fixture set: **12/22 = 55%**

Threshold sweep (single uniform threshold, overrides type-aware logic in harness):

| Threshold | Result | Notes |
|-----------|--------|-------|
| 0.82 | 12/22 = 55% | Best on this fixture set |
| 0.85 | 11/22 = 50% | |
| 0.88 | 11/22 = 50% | |
| 0.90 | 10/22 = 45% | |
| 0.92 | 10/22 = 45% | |
| 0.95 | 11/22 = 50% | |
| 0.97 | 11/22 = 50% | Current product default |

**Root cause of the 55% ceiling:** `all-MiniLM-L6-v2` sentence embeddings cannot reliably separate near-synonym true duplicates from same-domain false friends at the cosine level. For example, "We decided to use PostgreSQL" and "We decided to use MySQL" score nearly as similar as two paraphrases of the PostgreSQL decision, because both sentences share the same structural template.

**Planned improvements:**
- Entity-type extraction: detect the named technology and add it to the merge key
- Bi-encoder fine-tuning on domain-specific duplicate pairs
- Production dedup logging to build a real-world fixture set from user data

</details>

### When extraction fails

*What happens when the user is too vague?*

> **User:** "Yeah, let's just do that thing we talked about."

Because the statement is entirely ambiguous, the LLM assigns low confidence (`confidence < 0.5`). Waggle **silently drops** the extraction rather than storing a guess. It is safer to omit a noisy extraction than to pollute the graph with unanchored nodes.

The LLM pipeline does **not** silently fall back to regex on timeout. The first saved benchmark attempt surfaced two explicit `backend-unavailable` errors at the original 15s timeout. The stable run on this branch uses a configurable 30s timeout, and backend failures are logged explicitly.

---

## Temporal queries — built-in, not bolted on

Most memory systems answer "what do you know about X?" — but can't answer
*when* you learned it or how knowledge changed over time.

`waggle-mcp` timestamps every node and understands temporal natural language:

| Query | What happens |
|-------|-------------|
| `query_graph("what did we decide recently")` | Filters nodes updated in the last 24–48h |
| `query_graph("what was the original plan")` | Retrieves the earliest version of relevant nodes |
| `query_graph("what changed last week")` | Returns a diff of nodes created/updated in that window |
| `graph_diff(since="48h")` | Explicit changelog: added nodes, updated nodes, new conflicts |

---

## Installation

<details>
<summary>Local / development (SQLite, no extra services)</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
waggle-mcp init        # ← writes your client config automatically
```

Key variables for local mode:

| Variable | What it does |
|----------|-------------|
| `WAGGLE_BACKEND=sqlite` | Local file DB, zero setup |
| `WAGGLE_TRANSPORT=stdio` | Connects to desktop MCP clients |
| `WAGGLE_DB_PATH` | Where the graph is stored (default: `memory.db`) |

</details>

<details>
<summary>Production (Neo4j backend)</summary>

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

</details>

<details>
<summary>Docker</summary>

```bash
docker build -t waggle-mcp:latest .

docker run --rm -p 8080:8080 \
  -e WAGGLE_TRANSPORT=http \
  -e WAGGLE_BACKEND=neo4j \
  -e WAGGLE_DEFAULT_TENANT_ID=workspace-default \
  -e WAGGLE_NEO4J_URI=bolt://host.docker.internal:7687 \
  -e WAGGLE_NEO4J_USERNAME=neo4j \
  -e WAGGLE_NEO4J_PASSWORD=change-me \
  waggle-mcp:latest
```

</details>

<details>
<summary>Manual client configuration</summary>

**Claude Desktop — `claude_desktop_config.json`**

```json
{
  "mcpServers": {
    "waggle": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "waggle.server"],
      "env": {
        "PYTHONPATH": "/path/to/waggle-mcp/src",
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

**Codex — `codex_config.toml`**

```toml
[mcp_servers.waggle]
command = "/path/to/.venv/bin/python"
args    = ["-m", "waggle.server"]
cwd     = "/path/to/waggle-mcp"
env     = {
  PYTHONPATH                     = "/path/to/waggle-mcp/src",
  WAGGLE_TRANSPORT         = "stdio",
  WAGGLE_BACKEND           = "sqlite",
  WAGGLE_DB_PATH           = "~/.waggle/memory.db",
  WAGGLE_DEFAULT_TENANT_ID = "local-default",
  WAGGLE_MODEL             = "all-MiniLM-L6-v2"
}
```

A pre-filled example is in [`codex_config.example.toml`](./codex_config.example.toml).

</details>

---

## Environment variables

<details>
<summary>Click to expand full reference</summary>

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_BACKEND` | `sqlite` | `sqlite` or `neo4j` |
| `WAGGLE_TRANSPORT` | `stdio` | `stdio` or `http` |
| `WAGGLE_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model (local inference) |
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
| `WAGGLE_RATE_LIMIT_RPM` | `120` | global rate limit (req/min) |
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

### LLM Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_EXTRACT_BACKEND` | `auto` | `auto` \| `llm` \| `regex` |
| `WAGGLE_EXTRACT_MODEL` | `mistral` | Ollama model name |
| `WAGGLE_EXTRACT_MIN_CONFIDENCE` | `0.5` | float 0–1, facts below this are dropped |
| `WAGGLE_OLLAMA_URL` | `http://localhost:11434` | Base URL for local Ollama |

</details>

---

<details>
<summary>Admin commands</summary>

```bash
# Create a tenant
waggle-mcp create-tenant --tenant-id workspace-a --name "Workspace A"

# Issue an API key (raw key returned once — store it securely)
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

</details>

<details>
<summary>Kubernetes & observability</summary>

Full production deployment assets are in [`deploy/`](./deploy/):

| Path | What's inside |
|------|--------------|
| `deploy/kubernetes/` | Deployment, Service, Ingress (TLS), NetworkPolicy, HPA, PDB, cert-manager, ExternalSecrets — see [`deploy/kubernetes/README.md`](./deploy/kubernetes/README.md) |
| `deploy/observability/` | Prometheus scrape config, Grafana dashboard, one-command Docker Compose observability stack |

Operational runbooks are in [`docs/runbooks/`](./docs/runbooks/):

- [API key rotation](./docs/runbooks/api-key-rotation.md) — zero-downtime create-then-revoke
- [Incident response](./docs/runbooks/incident-response.md) — Neo4j down, OOM, rate storm, auth failures
- [Backup & restore](./docs/runbooks/backup-restore.md) — manual and automated drill
- [Tenant onboarding](./docs/runbooks/onboarding.md) — new tenant checklist
- [Secret management](./docs/runbooks/secret-management.md) — External Secrets + cert-manager

</details>

<details>
<summary>Architecture & project layout</summary>

```
waggle-mcp
├── Core domain    graph CRUD · dedup · local embeddings · conflict detection · export/import
├── Transport      stdio MCP (Codex/Desktop) · streamable HTTP MCP (Kubernetes)
└── Platform       config · auth · tenant isolation · rate limiting · logging · metrics
```

**Backend:**
- Local/dev → SQLite (zero config, instant start)
- Production → Neo4j (`WAGGLE_TRANSPORT=http` requires `WAGGLE_BACKEND=neo4j`)

```
waggle-mcp/
├── assets/                   ← banner + demo SVG
├── benchmarks/fixtures/      ← checked-in eval datasets
├── deploy/
│   ├── kubernetes/           ← full K8s manifests + guide
│   └── observability/        ← Prometheus + Grafana stack
├── docs/runbooks/            ← operational runbooks
├── scripts/
│   ├── benchmark_extraction.py
│   ├── load_test.py / .sh
│   └── backup_restore_drill.py / .sh
├── src/waggle/         ← server, graph, neo4j_graph, auth, config …
├── tests/artifacts/    ← saved benchmark runs
├── Dockerfile
├── pyproject.toml
└── README.md
```

</details>

---

## Testing

```bash
.venv/bin/pytest -q
```

Coverage: graph CRUD, deduplication, conflict detection, tenant isolation,
backup/import, stdio MCP, HTTP auth/health/metrics, payload limits.

---

## License

MIT — see [LICENSE](./LICENSE).
