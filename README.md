<p align="center">
  <img src="https://raw.githubusercontent.com/Abhigyan-Shekhar/graph-memory-mcp/main/assets/banner.png" alt="waggle-mcp" width="720"/>
</p>

<p align="center">
  <strong>Persistent, structured memory for AI agents — 4× fewer tokens than chunk-based retrieval.</strong><br/>
  Your LLM remembers facts, decisions, and context <em>across every conversation</em>, backed by a real knowledge graph.
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

`waggle-mcp` is a local-first memory layer for MCP-compatible AI clients, built on a persistent knowledge graph.

MCP is the **Model Context Protocol**: the tool interface desktop AI clients like Claude Desktop, Cursor, and Codex use to talk to local servers.

Waggle gives your AI a persistent knowledge graph it can read and write through any MCP-compatible client.

| Stuffed context | Structured retrieval |
|-----------------|----------------------|
| Context stuffed into a huge prompt every session | Compact subgraph retrieved at query time |
| Session-local memory | Persistent multi-session memory |
| Flat notes and chunks | Typed nodes and edges: decisions, reasons, contradictions, updates |
| "What changed?" requires replaying logs | Temporal queries and diffs are first-class |

Waggle's core tradeoff is deliberate: it stores structured knowledge instead of replaying entire transcripts. On Waggle's checked-in 27-scenario multi-session corpus, that yields **~4× fewer tokens per retrieval** than naive chunked retrieval. The benchmark section below shows the actual numbers and limits.

---

## Quick start

```bash
pip install waggle-mcp
waggle-mcp init
# Restart your MCP client. Done.
```

`init` detects your MCP client, writes its config, and creates the local database directory. Default mode is local SQLite with on-device embeddings.

---

## See it in action

**Session 1** — April 10
```text
User:  Let's use PostgreSQL. MySQL replication has been painful.
Agent: [calls observe_conversation()]
       → stores decision node: "Chose PostgreSQL over MySQL"
       → stores reason node:   "MySQL replication painful"
       → links them with a depends_on edge
```

**Session 2** — April 12 (fresh context window, no history)
```text
User:  What did we decide about the database?
Agent: [calls query_graph("database decision")]
       → retrieves the decision node + linked reason from April 10

       "You decided on PostgreSQL on April 10. The reason recorded was
        that MySQL replication had been painful."
```

**Session 3** — April 14
```text
User:  Actually, let's reconsider — the team is more familiar with MySQL.
Agent: [calls store_node() + store_edge(new_node → old_node, "contradicts")]
       → both positions are preserved, and the contradiction is explicit
```

This is the main difference from chunk replay: the agent does not just recover a transcript snippet, it recovers the decision, the reason, and what changed.

---

## Portable context handoff

Hit a rate limit? Switching models mid-project? Handing context to another AI?

`export_context_bundle` generates a Markdown or JSON context pack that another AI can ingest directly.

Example MCP tool call:

```javascript
export_context_bundle({
  "mode": "query",
  "query": "database architecture decisions",
  "format": "both",
  "retrieval_mode": "fusion"
})
```

Supported export modes:
- `prime` — compact brief from `prime_context`
- `query` — answer a specific question with supporting graph context
- `graph` — export the whole tenant graph, chunked for large memory sets

Supported retrieval lanes for query-mode export:
- `graph` — graph-native retrieval
- `replay` — raw transcript/session replay
- `fusion` — graph + replay merged with reciprocal-rank fusion

Waggle also supports Obsidian-style round-trip editing:
- `export_markdown_vault`
- `import_markdown_vault`

That writes one Markdown file per node with YAML frontmatter and wikilinks, then re-imports user edits non-destructively.

---

## The core tool: `observe_conversation`

Once your client prompt or tool policy nudges the model to call `observe_conversation`, the memory workflow becomes automatic.

```text
observe_conversation(user_message, assistant_response)
```

Each call:
1. extracts atomic facts from the turn
2. deduplicates against existing nodes
3. links related concepts with typed edges
4. flags contradictions and updates
5. stores the raw turn for replay/fusion retrieval

No separate schema authoring is required. The deterministic parser turns conversation turns into typed graph memory directly.

---

## MCP tools

Core workflow:

| Tool | What it does |
|------|--------------|
| `observe_conversation` | Ingest a conversation turn into graph memory |
| `query_graph` | Retrieve memory with `graph`, `replay`, or `fusion` mode |
| `prime_context` | Build a compact brief for a fresh session |
| `export_context_bundle` | Hand memory to another AI as Markdown or JSON |
| `export_markdown_vault` | Export Obsidian-compatible Markdown files |
| `import_markdown_vault` | Re-import edited Markdown vault files |
| `timeline` | Build a chronological view of what changed |
| `list_conflicts` / `resolve_conflict` | Inspect and resolve contradictions without deleting history |

Additional graph/admin tools are documented in [docs/reference.md](./docs/reference.md).

---

## Installation

Local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
waggle-mcp init
```

Neo4j backend:

```bash
pip install -e ".[dev,neo4j]"
WAGGLE_BACKEND=neo4j WAGGLE_TRANSPORT=http waggle-mcp
```

Docker, manual client config, environment variables, and admin commands are in [docs/reference.md](./docs/reference.md).

---

## Benchmarks

Benchmark summary:

| Area | Corpus | Result |
|------|--------|--------|
| Extraction | 12-case deterministic fixture | `100%` |
| Retrieval | 18-query retrieval fixture | `83% Hit@k` |
| Comparative efficiency | 27-scenario / 66-query corpus | `88% Hit@k`, `73% exact support`, `37.7` mean tokens |
| Query stress | 40 adversarial retrieval-only cases | `98% Hit@k`, `98% exact support` |
| External baseline | LongMemEval `s` split, 500 questions | `graph_raw: 97.0% R@5 / 76.4% Exact@5`, `graph_hybrid: 95.8% R@5 / 82.0% Exact@5` |

What these numbers mean:
- Waggle is strongest when the query benefits from structured reasoning chains, temporal context, and contradiction tracking.
- The `~4× fewer tokens` claim comes from the comparative corpus: Waggle averages `37.7` tokens per retrieval vs `150.2` for naive chunked-vector RAG.
- The retrieval engine itself is strong in isolation (`98%` on the query-stress corpus). End-to-end misses still show up more in broader comparative evaluation than in retrieval-only tests.
- Deduplication is intentionally conservative: best measured `17/22 = 77%`, with **zero false merges** across the threshold sweep.

Deep dives and saved artifacts:
- Internal benchmark artifacts: [tests/artifacts/README.md](./tests/artifacts/README.md)
- LongMemEval artifacts: [benchmarks/longmemeval/README.md](./benchmarks/longmemeval/README.md)
- Evaluation roadmap: [docs/evaluation-plan.md](./docs/evaluation-plan.md)

---

## Docs and operations

Detailed reference material lives outside the landing flow:

- Install variants, client config, environment variables, admin commands, and architecture:
  [docs/reference.md](./docs/reference.md)
- Kubernetes deployment:
  [deploy/kubernetes/README.md](./deploy/kubernetes/README.md)
- Runbooks:
  [docs/runbooks/](./docs/runbooks/)
- Benchmark artifacts and methodology:
  [tests/artifacts/README.md](./tests/artifacts/README.md)
  and [benchmarks/longmemeval/README.md](./benchmarks/longmemeval/README.md)

---

## Next Steps

- Expand the extraction corpus beyond the current 12 cases so robustness claims are based on larger paraphrase- and temporality-heavy fixtures.
- Publish a short LongMemEval methodology note, including cold vs warm cache runs and the reranked comparison path.
- Tighten replay/fusion ranking for recall-heavy workloads and improve provenance summaries in exported bundles.
- Polish Neo4j query paths and large-vault import reporting.

---

## License

MIT — see [LICENSE](./LICENSE).
