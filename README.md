<p align="center">
  <strong>waggle-mcp</strong>
</p>

<!-- mcp-name: io.github.Abhigyan-Shekhar/Waggle-mcp -->

<p align="center">
  <strong>Your AI forgets everything between sessions. Waggle gives it a graph-backed brain.</strong><br/>
  Persistent, structured memory for AI agents — in the checked-in comparison snapshot, about 2.6× fewer tokens on factual lookups.
</p>

<p align="center">
  <em>Waggle is not a code indexer. It's a conversational memory engine — it remembers what you decided, why, and what changed, across every session.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/waggle-mcp"><img src="https://img.shields.io/pypi/v/waggle-mcp?color=39d5cf&label=pypi" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/MCP-compatible-brightgreen" alt="MCP compatible"/>
  <img src="https://img.shields.io/badge/embeddings-local%2C%20no%20API%20key-orange" alt="Local embeddings"/>
</p>

---

## 60-Second Demo

No MCP client needed. Run this from a fresh install:

```bash
waggle-mcp demo
```

Expected output:

```
waggle-mcp demo
──────────────────────────────────────────────────
  graph:   .../examples/demo.abhi
  db:      /tmp/waggle-demo-xxx/demo.db
  model:   deterministic

  Imported 15 nodes, 20 edges from demo.abhi

Query 1: What database did we choose?
  [decision] Database: PostgreSQL everywhere via Docker Compose
  [decision] Database: PostgreSQL
  [note]     Reason: PostgreSQL chosen for ACID and JSON

Query 2: What changed about the database decision?
  [contradicts] Database: SQLite for local dev
    → Database: PostgreSQL
  [updates] Database: PostgreSQL everywhere via Docker Compose
    → Database: SQLite for local dev

Query 3: What are our team's preferences?
  [preference] Preference: TypeScript for all frontend code
  [preference] Preference: dark mode as default UI theme
  [preference] Preference: small, focused pull requests

Query 4: Show decisions and their reasons
  [decision] Database: PostgreSQL everywhere via Docker Compose
    ↳ [reason] Reason: parity between dev and prod prevents migration drift
  [decision] Auth: use Auth0 for SSO
    ↳ [reason] Reason: Auth0 reduces security risk and time-to-market

──────────────────────────────────────────────────
  Graph Studio: http://127.0.0.1:8686/graph?mode=view
  (run 'WAGGLE_DB_PATH=... waggle-mcp ui' to open it)

  Cleanup: rm -rf /tmp/waggle-demo-xxx
```

<!-- TODO: add demo.gif -->

Add `--with-embeddings` to use the real sentence-transformers model for higher-fidelity retrieval (requires ~420 MB model download on first run).

## Demo First

The quickest way to understand Waggle is to run the feature demo and the smoke test:

- [`tests/artifacts/test-run/comprehensive_feature_demo.md`](./tests/artifacts/test-run/comprehensive_feature_demo.md)
- [`scripts/smoke_test_mcp.py`](./scripts/smoke_test_mcp.py)
- [`waggle-mcp features`](#cli-command-reference)

That demo exercises the full MCP surface: graph ingestion, retrieval, conflict handling, export/import, and graph inspection.

## Recent Additions

- **Auto-capture hooks for Claude Code:** `waggle-mcp setup --yes` now installs three Claude Code hooks (`UserPromptSubmit`, `Stop`, `PreCompact`) that capture memory deterministically without relying on prompt rules. See [docs/hooks.md](./docs/hooks.md).
- **60-second demo:** `waggle-mcp demo` imports a pre-loaded example graph and runs 4 scripted queries locally — no MCP client, no API key, no network.
- **`waggle-mcp uninstall-hooks`:** removes the waggle-managed hooks block from Claude Code settings cleanly.

- **Graph Studio refresh:** the local `/graph` editor now has dual-layer graph/conversation views, transcript provenance, retrieval inspection, collapsible side panels, focus mode, label toggling, connected/isolate/cluster stats, and a layout that handles sparse graphs better instead of dropping everything into a giant ring.
- **Visual review and handoff:** the Graph Studio refresh plus deterministic `.abhi` export makes memory graphs easier to inspect, discuss, and hand off across tools. For teams that already work visually in design tools such as Figma, this gives a concrete graph artifact and transcript-backed provenance instead of ad-hoc screenshots or lost chat context.
- **Broader retrieval:** the new `aggregate_graph` MCP tool returns a wide filtered subgraph for map-reduce style analysis, with optional `node_types`, `tags`, and scope filters.
- **Hybrid memory retrieval:** Waggle now supports graph, verbatim transcript, and hybrid retrieval modes. The no-rerank hybrid path is the current default because it fixes the original cross-session verbatim recall gap without shipping the rerank regression.
- **Deterministic `.abhi` v2 export:** unsigned, unencrypted `.abhi` exports are now byte-identical across repeated exports and export → import → export round-trips, with canonical hashing and signature support tied to `content_hash`.
- **Safer sharing workflow:** `waggle-mcp push` now encrypts `.abhi` exports by default, and export paths refuse to proceed when transcript text appears to contain likely secrets unless you explicitly pass `--force`.
- **Model migration repair:** `waggle-mcp doctor --fix` now acts as the supported re-embedding path when `WAGGLE_MODEL` changes on an existing DB.
- **OOLONG evaluation workflow:** `waggle-mcp benchmark-oolong ...` runs retrieval-only, one-shot retrieval+LLM, or Waggle-backed RLM evaluation against OOLONG datasets and can emit JSON reports for offline analysis.
- **Vendored upstream RLM package:** the repo now includes the upstream `alexzhang13/rlm` Python package under [`src/rlm`](./src/rlm/) with attribution material in [`third_party/rlm`](./third_party/rlm/).

## Who It's For

**→ Individual developer** extending Claude, Codex, Gemini CLI, Cursor, or Antigravity with persistent memory:
Use Python 3.11+ and install via `pipx` (no venv activation needed):
`brew install pipx && pipx ensurepath && pipx install waggle-mcp && waggle-mcp setup --yes`.
SQLite + local embeddings, zero infra.

**→ Team sharing a canonical project memory across multiple agents and developers:** Waggle ships with a Docker image, Kubernetes manifests, Prometheus metrics, and multi-tenant auth. See [deploy/kubernetes/](./deploy/kubernetes/) and [docs/runbooks/](./docs/runbooks/).

Both paths share the same MCP tool surface — the difference is only the backend and transport.

---

## Why Waggle

`waggle-mcp` is a local-first memory layer for MCP-compatible AI clients, built on a persistent knowledge graph.

| Stuffed context | Structured retrieval |
|-----------------|----------------------|
| Huge prompts every session | Compact subgraph retrieved at query time |
| Session-local memory | Persistent multi-session memory |
| Flat notes and chunks | Typed nodes and edges: decisions, reasons, contradictions |
| "What changed?" requires replaying logs | Temporal queries and diffs are first-class |

Waggle often uses materially fewer tokens than naive chunked retrieval on factual lookups, while graph-traversal queries intentionally spend more context to include reasoning chains such as updates, contradictions, and dependencies.

---

## Architecture

```mermaid
flowchart LR
  C["MCP Client\n(Claude/Gemini CLI/Codex/Cursor/Antigravity/ChatGPT)"] --> S["waggle.server\nMCP tool surface"]
  S --> G["Graph Engine\nMemoryGraph / Neo4jMemoryGraph"]
  G --> DB["SQLite (local default)\nor Neo4j (service mode)"]
  G --> E["Embeddings\n(sentence-transformers or deterministic fallback)"]
```

---

## Quick start (Recommended)

The simplest way to use Waggle is via `pipx`. This installs the package in an isolated environment and makes the `waggle-mcp` command available globally **without needing to manage a virtual environment (`.venv`) manually**.

```bash
# 1. Install waggle globally
pipx install waggle-mcp

# 2. Run the one-line non-interactive setup
waggle-mcp setup --yes

# 3. Verify everything looks healthy
waggle-mcp doctor
```
*(If you don't have `pipx`, install it via `brew install pipx && pipx ensurepath`.)*

Running `setup --yes` detects local MCP clients, writes the necessary configuration, and initializes your local database directory. Restart your client, and you're ready to go. Use `waggle-mcp init` if you prefer the older interactive wizard.

`waggle-mcp doctor` is your first stop if anything doesn't work — it checks config file locations, the embedding model cache, DB path, and surfaces the most common API mistakes.

If you change `WAGGLE_MODEL` for an existing shared DB, run `waggle-mcp doctor` immediately afterward. If it reports mixed embedding model IDs, run `waggle-mcp doctor --fix` to re-embed stale transcript and node rows to the current model before continuing.

> **Windows users:** Run all commands with `python -X utf8` or set `PYTHONUTF8=1` in your environment to avoid `UnicodeEncodeError` from emoji in log output.

For Codex, `waggle-mcp setup --yes` and `waggle-mcp init` also write a managed Waggle block into `AGENTS.md` in the current workspace so automatic memory is enabled by default for that repo.

Manual MCP setup examples for **Codex**, **Claude Code**, **Gemini CLI**, **Cursor**, and **Antigravity** are in [docs/reference.md](./docs/reference.md#manual-client-configuration).

## Prompt Instructions For Automatic Tool Calls

Registering Waggle as an MCP server only makes the tools available. If you want the agent to call them automatically during normal conversation, add an instruction block like this to your client prompt, rules, or project instructions:

```text
Use Waggle automatically for conversational memory.

At the start of a new session, if project, agent, or session scope is known, call prime_context.

Before answering questions that may depend on prior decisions, preferences, constraints, project state, or earlier conversation context, call query_graph with the narrowest relevant scope.

After completed turns that contain durable information such as decisions, preferences, constraints, requirements, user corrections, project facts, or meaningful task outcomes, call observe_conversation automatically.

Waggle should remember relevant context automatically. If memory appears empty, the session is likely missing the automatic memory policy or the runtime hooks that call build_context before answers and on_assistant_turn after answers.

Do not ask the user to trigger Waggle manually. Use it in the background when relevant.
```

Use the same stable `project` value for the same codebase across sessions, or recall will fragment. For Codex workspaces, `waggle-mcp setup --yes` already writes this managed block into `AGENTS.md`.

Comprehensive live feature run (full tool surface, multi-query graph tests, export/import validation):
[`tests/artifacts/test-run/comprehensive_feature_demo.md`](./tests/artifacts/test-run/comprehensive_feature_demo.md)

> **Best entry point:** if you do nothing else, run `waggle-mcp features` to get the tool map, workflows, and setup hints in one place.

> **⚠️ Edges are what make graph memory work.**
> `observe_conversation` and `decompose_and_store` create edges automatically.
> If you only call `store_node`, you get isolated facts — not a connected graph.
> Always prefer `observe_conversation` for conversational ingestion.

For broad summarization and offline synthesis tasks, prefer `aggregate_graph` over `query_graph` when you want a much larger scoped slice of memory instead of high-precision semantic ranking.

---

## Setting Up waggle as an MCP Server

> **One-time install:** `pipx install waggle-mcp` (requires Python 3.11+; recommended on macOS/Homebrew Python) — no API key, no cloud account, no Docker required for local use.

Use this shared JSON config shape for clients that accept `mcpServers` JSON (recommended when installed via `pipx`):

```json
{
  "mcpServers": {
    "waggle": {
      "command": "waggle-mcp",
      "args": ["serve"],
      "env": {
        "WAGGLE_TRANSPORT": "stdio",
        "WAGGLE_BACKEND": "sqlite",
        "WAGGLE_DB_PATH": "~/.waggle/memory.db",
        "WAGGLE_DEFAULT_TENANT_ID": "local-default",
        "WAGGLE_MODEL": "all-MiniLM-L6-v2",
        "WAGGLE_STARTUP_MODE": "normal"
      }
    }
  }
}
```

> **First run takes ~30 s** — `all-MiniLM-L6-v2` (~420 MB) downloads on first use.
> To skip the download entirely, set `"WAGGLE_MODEL": "deterministic"` (offline-safe, instant start, slightly lower retrieval quality).

<details>
<summary>Claude Desktop / Gemini CLI / Antigravity / Cursor / Claude Code setup details</summary>

**Claude Desktop config file location**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Gemini CLI**
```bash
gemini mcp add waggle \
  -e WAGGLE_TRANSPORT=stdio \
  -e WAGGLE_BACKEND=sqlite \
  -e WAGGLE_DB_PATH=~/.waggle/memory.db \
  -e WAGGLE_DEFAULT_TENANT_ID=local-default \
  -e WAGGLE_MODEL=all-MiniLM-L6-v2 \
  waggle-mcp serve
```

Equivalent `~/.gemini/settings.json` entry:

```json
{
  "mcpServers": {
    "waggle": {
      "command": "waggle-mcp",
      "args": ["serve"],
      "env": {
        "WAGGLE_TRANSPORT": "stdio",
        "WAGGLE_BACKEND": "sqlite",
        "WAGGLE_DB_PATH": "~/.waggle/memory.db",
        "WAGGLE_DEFAULT_TENANT_ID": "local-default",
        "WAGGLE_MODEL": "all-MiniLM-L6-v2"
      },
      "trust": false
    }
  }
}
```

After restarting Gemini CLI, run `/mcp` to confirm Waggle is connected.

**Antigravity**
- The **AI agent** reads: `~/.gemini/antigravity/mcp_config.json` (macOS/Linux) or `%USERPROFILE%\.gemini\antigravity\mcp_config.json` (Windows)
- The VS Code extension panel reads a **different** file (`%APPDATA%\Antigravity\User\mcp.json`) — adding waggle there will NOT make it available to the AI agent.
- Open the correct file and add the `waggle` block from above.

> Run `waggle-mcp doctor` to see exactly which config files exist and which ones have a waggle entry.

**Cursor**
- `Cursor Settings -> Features -> MCP Servers -> + Add`
- Command: `waggle-mcp`
- Args: `serve`
- Env vars: same keys as the JSON block above.

**Claude Code**
```bash
claude mcp add waggle \
  --env WAGGLE_TRANSPORT=stdio \
  --env WAGGLE_BACKEND=sqlite \
  --env WAGGLE_DB_PATH=~/.waggle/memory.db \
  --env WAGGLE_DEFAULT_TENANT_ID=local-default \
  --env WAGGLE_MODEL=all-MiniLM-L6-v2 \
  -- waggle-mcp serve
```

</details>

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.waggle]
command = "waggle-mcp"
args    = ["serve"]
env     = {
  WAGGLE_TRANSPORT         = "stdio",
  WAGGLE_BACKEND           = "sqlite",
  WAGGLE_DB_PATH           = "~/.waggle/memory.db",
  WAGGLE_DEFAULT_TENANT_ID = "local-default",
  WAGGLE_MODEL             = "all-MiniLM-L6-v2"
}
```

A live-source development example is included in [codex_config.example.toml](./codex_config.example.toml).

### ChatGPT

ChatGPT custom MCP connectors use a remote HTTPS MCP server, not a local `stdio` process. To connect Waggle to ChatGPT, deploy Waggle in HTTP mode, expose the `/mcp` endpoint over HTTPS, then add that URL as a custom connector in ChatGPT.

For a service deployment, use the Neo4j backend:

```bash
WAGGLE_TRANSPORT=http \
WAGGLE_BACKEND=neo4j \
WAGGLE_DEFAULT_TENANT_ID=workspace-default \
WAGGLE_NEO4J_URI=bolt://localhost:7687 \
WAGGLE_NEO4J_USERNAME=neo4j \
WAGGLE_NEO4J_PASSWORD=change-me \
waggle-mcp serve
```

Then configure ChatGPT with the HTTPS endpoint:

```text
https://waggle.example.com/mcp
```

In ChatGPT, enable developer mode and add the connector from:

```text
Settings -> Connectors -> Advanced -> Developer mode
```

Do not expose Waggle publicly without authentication. A remote memory server can read and mutate project memory, so it should sit behind your normal auth, network allowlisting, or gateway controls.

### `waggle-mcp` not on PATH?

If you installed with `pipx`, ensure its bin path is available:

```bash
pipx ensurepath
```

Then restart your terminal/client. If you're using a venv-based install, use the venv interpreter path instead of `waggle-mcp`:

```bash
which python3   # macOS / Linux
where python    # Windows
```

e.g. `/usr/local/bin/python3` or `C:\Python311\python.exe`.

### Verify it works

After restarting your client, ask the agent:

> *"Store a note: we're using PostgreSQL for this project."*

Then open a **fresh session** and ask:

> *"What database are we using?"*

Expected result (example):

```text
You're using PostgreSQL for this project.
```

If you see that kind of recall in a new session, you're live.

### What To Ask The Agent

| Ask the agent... | Tool called |
|---|---|
| "Remember that..." | `observe_conversation` |
| "What do you know about X?" | `query_graph` |
| "What changed recently?" | `graph_diff` |
| "Summarize context for a new session" | `prime_context` |
| "Show all stored topics" | `get_topics` |
| "Export my memory to a file" | `export_graph_backup` |

<details>
<summary>What's New — v0.1.9</summary>

- `waggle-mcp ingest-transcript-handoff` ingests ordered transcripts, deduplicates by `message_identity`, and exports a session-scoped handoff bundle.
- Append-only reruns avoid reprocessing completed turns while still finishing a trailing `user` block when the matching `assistant` arrives later.
- Batch ingestion keeps evidence turn indices aligned with stored transcript rows, including tool/system-interleaved sessions.
- Handoff export failures now surface as real CLI failures instead of being downgraded into `export_skipped`.
- The repo includes an end-to-end benchmark harness, LongMemEval artifacts, observability assets, and runbooks.

</details>

## Automatic Memory Setup For Codex And Antigravity

Registering Waggle as an MCP server is necessary, but it is not sufficient for automatic cross-session memory. The client still needs instructions telling the agent to use Waggle in the background.

`waggle-mcp setup --yes` and `waggle-mcp init` now do this automatically for Codex by writing a managed Waggle memory block to the workspace `AGENTS.md`. Other clients still need their equivalent instruction layer.

There is not a better generic repo-side mechanism for third-party MCP clients today. If the client does not provide a runtime hook that automatically calls memory tools, the practical setup is:
- register Waggle as an MCP server
- add an agent instruction / User Rule telling the model when to call Waggle

If a client later exposes a native pre-answer / post-turn orchestration hook, that is better than prompt rules. Until then, prompt-level rules are the portable solution across Codex and Antigravity.

For product integrations, the preferred path is the event-driven runtime in [docs/memory-orchestration.md](./docs/memory-orchestration.md): call `build_context(...)` before each answer and `on_assistant_turn(...)` after each completed turn. MCP tool exposure alone does not make memory automatic.

Use the same rule text in:
- **Codex**: your global/project instructions or equivalent agent rule layer
- **Antigravity**: **User Rules** / custom instructions for the agent

A copy-pasteable version also lives in [docs/automatic-memory-rules.md](./docs/automatic-memory-rules.md).

Recommended rule text:

```text
Use Waggle automatically for conversational memory.

At the start of a new session, if project, agent, or session scope is known, call prime_context.

Before answering questions that may depend on prior decisions, preferences, constraints, project state, or earlier conversation context, call query_graph with the narrowest relevant scope.

After completed turns that contain durable information such as decisions, preferences, constraints, requirements, user corrections, project facts, or meaningful task outcomes, call observe_conversation automatically.

Waggle should remember relevant context automatically. If memory appears empty, the session is likely missing the automatic memory policy or the runtime hooks that call build_context before answers and on_assistant_turn after answers.

Do not ask the user to trigger Waggle manually. Use it in the background when relevant.
```

### Important Findings

- **MCP registration alone does not create automatic memory.** If the client only exposes Waggle as a tool, cross-session recall can still fail.
- **Scope must match across sessions.** Store and recall need to use the same database, tenant, and relevant scope such as `project`.
- **Rollover handoff is separate from live-turn memory.** `ingest-transcript-handoff` fixes end-of-window/session import and export. Live conversational memory still depends on automatic `observe_conversation` and `query_graph` usage during normal chats.
- **For same-machine multi-client sharing, use the same `WAGGLE_DB_PATH`.** Codex and Antigravity can share one local brain if both point to the same SQLite file.

For the full tool surface and environment variable reference see [docs/reference.md](./docs/reference.md).

---

## CLI Command Reference

Waggle includes a built-in CLI for setup, maintenance, and learning the memory system.

| Command | Description |
|---|---|
| `waggle-mcp --help` | Show all available commands, options, and usage examples. |
| `waggle-mcp features` | **Best first command** — Explain the main tools, graph workflows, and how connected context reaches the model. |
| `waggle-mcp doctor` | **Run this if something isn't working** — checks config files, model cache, DB path, Windows encoding, and API gotchas. |
| `waggle-mcp setup --yes` | Non-interactive one-line setup that auto-patches detected supported clients. |
| `waggle-mcp init` | Interactive setup wizard to configure one MCP client. |
| `waggle-mcp serve` | Run the MCP server (usually started automatically by your client). |
| `waggle-mcp edit-graph` | Launch the local Graph Studio in the browser for direct graph editing and export/import workflows. |
| `waggle-mcp benchmark-oolong` | Run OOLONG retrieval-only, retrieval+LLM, or Waggle-backed RLM evaluation against local datasets and emit a JSON report. |
| `waggle-mcp ingest-transcript-handoff` | Ingest a rollover transcript and export a handoff bundle for the next window or IDE. |
| `waggle-mcp export-context-bundle` | Export a portable Markdown/JSON context pack for another AI. |
| `waggle-mcp export-markdown-vault` | Export your memory graph as an Obsidian-style vault. |

### `WAGGLE_STARTUP_MODE`

Controls how aggressively the embedding model is loaded at startup:

| Value | Behaviour | Best for |
|---|---|---|
| `normal` *(default)* | Model loads in background thread; server responds immediately | Daily use |
| `fast` | ML never loads; semantic tools return `unavailable` | Schema inspection, tool listing |
| `strict` | Server blocks until model is fully loaded before serving | Production deployments requiring guaranteed readiness |

Set in your client config: `"WAGGLE_STARTUP_MODE": "fast"`.

For advanced commands (tenant management, API keys, Neo4j migration), see the full help output:
```bash
waggle-mcp --help
```

### Graph Studio

Waggle includes a local browser-based graph editor for inspecting and editing memory directly.

```bash
waggle-mcp edit-graph
```

Graph Studio currently supports:

- Direct node and edge editing in the browser
- Conversation-layer and graph-layer views in the same UI
- Transcript provenance and retrieval-debug inspection for hybrid memory results
- Mouse-based node dragging and shift-drag edge creation
- Collapsible side panels, focus mode, and label toggling for large graphs
- Live graph stats including connected nodes, isolates, and cluster count
- Export/import of the current graph, including `.abhi` preview, diff, and sharing workflows
- `.abhi` is JSON underneath, supports optional embedded vectors, optional AES-256-GCM encryption, deterministic content hashing, and Google Drive sync via `waggle-mcp push|pull|share` (see [docs/abhi-format.md](/Users/abhigyanshekhar/Desktop/MCP/docs/abhi-format.md))

Security note: `waggle-mcp push` now encrypts exported `.abhi` files by default, and export paths refuse to proceed if transcript text appears to contain likely secrets unless you pass `--force`. The full threat model is in [SECURITY.md](/Users/abhigyanshekhar/Desktop/MCP/SECURITY.md).

Use this when you want to inspect the live memory graph visually, clean up relationships, or export a portable memory artifact for another Waggle instance. In practice this also makes design and product review easier: Graph Studio gives you a readable graph view for discussion, while `.abhi` gives you a deterministic, portable snapshot you can diff, validate, share, and re-import without losing provenance.

---

## Cross-Client Handoffs & Migration

Waggle is designed to be a "portable brain" for your AI sessions. Whether you are switching editors (e.g., Antigravity to Codex) or moving across machines, your memory can follow you.

### 1. Automatic Sharing (Same Machine)
If you run multiple MCP clients (like Codex and Antigravity) on the same machine, they can share a single "brain" automatically. 
*   **How:** Ensure both clients use the same `WAGGLE_DB_PATH` in their environment configuration (default is `~/.waggle/memory.db`).
*   **Result:** A decision made in one editor is immediately known by the agent in the other.

### 2. Session Handoffs (Context Bundles)
If you hit a session limit or want to jump to a fresh context while keeping important facts:
```bash
# Export a condensed, AI-ready summary of your current project context
waggle-mcp export-context-bundle --format markdown --output-path ./handoff.md
```
Paste the contents of `handoff.md` into your new session to "re-prime" the AI with your project's history.

### 3. Full Memory Migration (Backup/Import)
To move your entire memory history to a new machine:
*   **Export:** `waggle-mcp export-graph-backup --output-path my_memory.json`
*   **Import:** `waggle-mcp import-graph-backup --input-path my_memory.json`

---

## Using It In MCP Clients

Once installed, you usually do not run `waggle-mcp` commands by hand during daily work. Talk to the agent normally, and it calls Waggle MCP tools to store and retrieve memory.

- **Codex / Claude Code**: `observe_conversation`, `query_graph`, and `prime_context` are called automatically during normal threads.
- **Cursor**: decisions and facts can be persisted as graph memory instead of getting lost in old chat windows.
- **Antigravity**: conversation turns can be extracted via `observe_conversation`; context can be exported with `export_context_bundle`.

---

## See it in action

![Waggle init demo](./assets/demo.svg)

### How It Works (Interaction Flow)

```text
User  -> Agent -> observe_conversation(...) -> Graph stores typed nodes + edges
User  -> Agent -> query_graph("database")    -> Subgraph returned -> Agent answers with linked rationale
```

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

### Knowledge graph visual (example)

```mermaid
graph TD
  D1["Decision: Use PostgreSQL"]
  R1["Reason: MySQL replication pain"]
  D2["Decision update: reconsider MySQL"]
  P1["Preference: dark mode UI"]
  N1["Note: add integration tests"]

  D1 -- "depends_on" --> R1
  D2 -- "contradicts" --> D1
  N1 -- "relates_to" --> D2
  P1 -- "part_of project context" --> D1
```

---

## Key Features

- **Automatic Extraction**: `observe_conversation` ingests facts into the graph without manual schema work.
- **Portable Context**: `export_context_bundle` generates Markdown/JSON context packs for another AI.
- **Vault Round-trip**: `export_markdown_vault` / `import_markdown_vault` for Obsidian-style node editing.
- **Conflict Resolution**: `list_conflicts` / `resolve_conflict` to manage contradictions without losing history.
- **Deterministic Fallback**: Stable SHA-256 hashing for reliable, reproducible offline operation when transformer models are unavailable.

For a concise "what to say about Waggle" version, see [docs/briefing.md](./docs/briefing.md). That doc is intentionally a media-kit style summary; this README remains the canonical product and setup reference.

---

## Security & Privacy

By default, data stays local on your machine (`sqlite` backend, local database path such as `~/.waggle/memory.db`).  
Waggle does not require telemetry or cloud calls for core local operation.  
Your conversation memory only leaves your machine if you explicitly configure a remote backend or remote infrastructure.

Waggle currently stores local memory as a normal SQLite file and does not add application-level encryption at rest. Use standard filesystem permissions and OS disk encryption if the stored conversation history is sensitive.

Before `.abhi` export or sharing workflows, Waggle now scans transcript text for likely secrets such as API keys, passwords, and JWTs. If it finds probable secrets, export is refused unless you pass `--force`. For remote sharing, `waggle-mcp push` defaults to encrypted export.

The current threat model and sharing guidance are documented in [SECURITY.md](./SECURITY.md).

---

## Graph Data Model

### Node Types

`fact`, `entity`, `concept`, `preference`, `decision`, `question`, `note`

### Edge Types

`relates_to`, `contradicts`, `depends_on`, `part_of`, `updates`, `derived_from`, `similar_to`

---

## Model Support

Waggle currently uses a local `sentence-transformers` embedding model selected by `WAGGLE_MODEL`.

- Default: `all-MiniLM-L6-v2`
- Any locally available `sentence-transformers` model name can be used.
- If the selected model is unavailable locally, Waggle falls back to deterministic embeddings for portability.
- For existing DBs, model changes are a migration event: run `waggle-mcp doctor`, then `waggle-mcp doctor --fix` if mixed `embedding_model_id` values are reported.

Set model in env:

```bash
WAGGLE_MODEL=all-mpnet-base-v2 waggle-mcp serve
```

Set model in MCP client config (example):

```json
{
  "mcpServers": {
    "waggle": {
      "command": "python",
      "args": ["-m", "waggle.server"],
      "env": {
        "WAGGLE_MODEL": "all-mpnet-base-v2"
      }
    }
  }
}
```

Notes:
- Waggle does not currently route to hosted embedding providers directly; embedding inference is local to the runtime.
- Deterministic mode is useful for offline/testing portability, but semantic retrieval quality is lower than transformer mode.
- For a shared DB migration after changing `WAGGLE_MODEL`, run `waggle-mcp doctor`; if stale rows are reported, run `waggle-mcp doctor --fix` and let the re-embed pass complete before resuming normal use.

---

## Benchmarks & Verification

### LongMemEval Retrieval Results

Measured on the full `500`-question LongMemEval-S split (`all-MiniLM-L6-v2`, warm cache, 2026-05-03):

| Mode | R@5 | Exact@5 | Exact@10 | Exact@20 |
|------|-----|---------|---------|---------|
| `graph_raw` | `97.4%` | **`89.0%`** | `89.0%` | `89.0%` |
| `graph_hybrid` | `97.0%` | `87.2%` | `94.8%` | `98.0%` |

**The headline number is `graph_raw` at `89.0% Exact@5`.** It has no tunable reranking heuristics and is the fairest apples-to-apples comparison against other retrieval systems.

`graph_hybrid` scores slightly lower on Exact@5 but recovers strongly at higher cutoffs (Exact@20 = 98.0%), meaning it finds all the right sessions — it just doesn't always pack them all into the top 5 slots for high-cardinality queries.

#### Overfitting check — held-out dev/test split

To verify the scores are not artefacts of the specific 500-question distribution, the benchmark was also run with `--held-out` (50 dev / 450 test, seed 42):

| Mode | Dev Exact@5 (n=50) | Test Exact@5 (n=450) | Gap |
|------|-------------------|---------------------|-----|
| `graph_raw` | `88.0%` | `89.1%` | `+1.1pp` — no overfitting |
| `graph_hybrid` | `92.0%` | `86.7%` | `−5.3pp` — reranking weights have dev-set sensitivity |

`graph_raw` is stable across the split. `graph_hybrid`'s 5.3pp dev/test gap indicates the heuristic reranking weights are partially tuned to this distribution and should not be treated as a robust headline number.

#### How to reproduce

```bash
# Full run — raw (headline)
.venv/bin/python scripts/benchmark_longmemeval.py \
  benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --mode graph_raw \
  --output benchmarks/longmemeval/results_graph_raw_$(date +%F).json

# Full run — hybrid
.venv/bin/python scripts/benchmark_longmemeval.py \
  benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --mode graph_hybrid \
  --output benchmarks/longmemeval/results_graph_hybrid_$(date +%F).json

# Held-out overfitting check (50 dev / 450 test)
.venv/bin/python scripts/benchmark_longmemeval.py \
  benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --mode graph_raw --held-out \
  --output benchmarks/longmemeval/results_graph_raw_heldout_$(date +%F).json
```

Full methodology, per-cardinality breakdown, and all checked-in artifacts are in [`benchmarks/longmemeval/README.md`](./benchmarks/longmemeval/README.md).

Checked-in artifacts back the headline claims:

- Token use: `63.0` comparative mean context tokens for Waggle vs `161.8` for the naive RAG baseline, about `2.6×` fewer tokens on the saved comparison snapshot.
- LongMemEval: `graph_raw` reaches `97.4% R@5` and `89.0% Exact@5` on the 500-question split with no dev/test gap.
- Local operation latency snapshot: `observe_conversation` mean `1.54 ms`, `query_graph` mean `1.60 ms`, `graph_diff` mean `0.80 ms` using local SQLite plus deterministic embeddings.
- Automated verification: MCP integration, transcript handoff, and benchmark harness tests are checked into the repo, and the local smoke path exercises live `store_node`, `query_graph`, and `graph://stats`.

README benchmark claims in this repo are limited to Waggle runs with checked-in artifacts and reproducible commands. Cross-project comparisons are intentionally excluded unless they are apples-to-apples on split, protocol, and scoring.

> **Note:** artifacts dated April 25 – May 2, 2026 are from a regression window (embeddings cache bug + ranking formula change) and should not be used as baselines. See the [postmortem](./docs/postmortems/2026-05-02-embeddings-cache-and-ranking-regression.md).

Detailed artifacts and methodology live in:

- [docs/benchmark-methodology.md](./docs/benchmark-methodology.md)
- [docs/longmemeval-methodology.md](./docs/longmemeval-methodology.md)
- [tests/artifacts/README.md](./tests/artifacts/README.md)
- [tests/artifacts/benchmark_current.md](./tests/artifacts/benchmark_current.md)
- [tests/artifacts/verification/2026-04-20-performance-snapshot/performance_snapshot.md](./tests/artifacts/verification/2026-04-20-performance-snapshot/performance_snapshot.md)

> **Note on dates:** some checked-in benchmark and demo artifacts use `2026-04-2x` timestamps as fixture metadata for reproducibility. Treat them as artifact dates, not a product release calendar.

---

## Known Limitations

- **Best on structured recall, weaker on answer synthesis**: Waggle is strongest at "retrieve the right facts and relationships" — not at emitting a single benchmark-formatted final answer from memory.
- **Edges are load-bearing**: `observe_conversation` and `decompose_and_store` create them automatically. Raw `store_node` calls without follow-up edges produce disconnected nodes with no traversal value.
- **Graph retrieval trades tokens for reasoning context**: factual lookups are often cheaper than chunked RAG; graph-expansion queries intentionally spend more tokens to carry update chains and contradictions.
- **Graph-only recall still depends on extraction**: the cross-session codeword guarantee is fixed by the verbatim and hybrid layers; graph-only mode can still miss facts if extraction produced zero nodes for the original turn.
- **Hybrid rerank is not the default**: the shipped no-rerank hybrid path is stronger right now. The rerank path remains available for follow-up tuning, but it is intentionally not the launch default.
- **Deduplication is fixture-backed, not universal semantic equivalence**: the current 32-case fixture covers common memory-node paraphrases and false friends, but broader production text can still require additional aliases or stricter domain guards.

For operational details, scaling considerations, tool-level behavior, and the full MCP feature surface, see [docs/reference.md](./docs/reference.md).

---

## Troubleshooting

Run `waggle-mcp doctor` first — it catches the most common issues automatically.

### Install / dependency failures (and fixes)

These are the most common install-time failures and what to do next.

| Failure (example) | Likely cause | Fix |
|---|---|---|
| `waggle-mcp: command not found` right after install | `pipx` bin dir not on `PATH` | Run `pipx ensurepath` and restart your terminal |
| `zsh: command not found: python` | System has only `python3` (or `python` is not on `PATH`) | Use `python3 ...` for ad-hoc commands; install Python 3.11+ (macOS: Homebrew Python recommended) |
| `ModuleNotFoundError: No module named 'mcp'` or `... 'networkx'` | Wrong environment (e.g. running from global Python instead of the `pipx` venv) or incomplete install | Prefer `pipx install waggle-mcp` and run via `waggle-mcp ...`. If you need to repair an existing pipx env: `pipx runpip waggle-mcp install -U mcp networkx` |
| `pipx install waggle-mcp` fails while building / installing `sentence-transformers` / `torch` | Python/OS wheel mismatch or outdated `pip` | Use Python 3.11+; upgrade tooling: `python3 -m pip install -U pip setuptools wheel`. If you only need an offline-safe setup, you can still run Waggle in deterministic mode once installed: `WAGGLE_MODEL=deterministic waggle-mcp serve` |
| `pytest` collection errors about missing packages | Dev deps not installed | Install dev extras in the same env: `pip install -e '.[dev]'` (repo) or `pipx inject waggle-mcp 'waggle-mcp[dev]'` (pipx) |

| Symptom | Likely cause | Fix |
|---|---|---|
| `store_node` / `query_graph` hangs forever | Embedding model downloading on first run (~420 MB) | Set `WAGGLE_MODEL=deterministic` for instant offline mode, or wait for download to finish |
| `UnicodeEncodeError: 'charmap' codec...` | Windows stdout not UTF-8 | Run with `python -X utf8` or set `PYTHONUTF8=1` |
| `Additional properties are not allowed ('user_text', 'assistant_text' were unexpected)` | Using old pre-v0.2 field names | Use `user_message` + `assistant_response` (not `user_text`/`assistant_text`) |
| `Additional properties are not allowed ('project' was unexpected)` in `get_topics` | Fixed in current version | Update: `pipx upgrade waggle-mcp` |
| Waggle registered but AI agent doesn't see it (Antigravity) | Added to wrong config file | Agent reads `~/.gemini/antigravity/mcp_config.json`, **not** the VS Code extension file |
| Stdio framing garbled / dropped messages (Windows) | Hand-rolled JSON-RPC | Use the official `mcp` Python client: `pip install mcp` |
| `waggle-mcp: command not found` | pipx bin dir not on PATH | Run `pipx ensurepath` then restart terminal |

### Checking which config file the agent reads

```bash
waggle-mcp doctor
```

Section `[1] MCP client config files` lists every known config path, whether it exists, and whether it contains a waggle entry. The Antigravity AI agent path (`~/.gemini/antigravity/mcp_config.json`) is listed separately from the VS Code extension path.

---

## Contributing

This repository is maintained privately. Internal contributors can use the docs in this repo as the source of truth.

---

## Reference & Docs

Detailed reference material lives in external documentation:

- **[docs/reference.md](./docs/reference.md)**: Environment variables, admin commands, Docker setup, and full tool surface.
- **[deploy/kubernetes/README.md](./deploy/kubernetes/README.md)**: Production deployment.
- **[docs/runbooks/](./docs/runbooks/)**: Operations and troubleshooting.
- **[tests/artifacts/README.md](./tests/artifacts/README.md)**: Benchmark artifacts and traceability.

---
