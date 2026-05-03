# Waggle Hook Integration

Waggle supports automatic memory capture via client hooks, eliminating the need for prompt rules in most cases.

## Auto-capture matrix

| Client | Method | Auto-capture quality |
|--------|--------|----------------------|
| Claude Code | Hooks | Strong (deterministic) |
| Codex | AGENTS.md prompt rule | Moderate (prompt-driven) |
| Cursor | User Rules | Moderate |
| Antigravity | User Rules | Moderate |

Hooks are preferred where supported. They fire deterministically on IDE events, independent of whether the model follows prompt instructions.

## Claude Code hooks

Three hook scripts are installed under `src/waggle/hooks/claude_code/`:

| Script | Claude Code event | What it does |
|--------|-------------------|--------------|
| `pre_response.py` | `UserPromptSubmit` | Calls `prime_context` or `query_graph` and injects relevant memory as a system reminder before Claude responds |
| `post_response.py` | `Stop` | Calls `observe_conversation` with the last user/assistant turn after Claude finishes |
| `pre_compact.py` | `PreCompact` | Calls `ingest_transcript_handoff` to preserve durable info before context compression |

### Installation

Hooks are installed automatically when you run:

```bash
waggle-mcp setup --yes
```

To skip hook installation:

```bash
waggle-mcp setup --yes --no-hooks
```

To remove hooks:

```bash
waggle-mcp uninstall-hooks
```

### How it works

Each hook script:
- Reads JSON from stdin per the Claude Code hook protocol
- Calls the local Waggle in-process API (no network required)
- Writes JSON to stdout per the protocol
- **Always exits 0** — a Waggle bug never blocks your session
- Has a **5-second timeout** — if exceeded, exits silently

### Security

`post_response.py` scans turn text for likely secrets (API keys, tokens, passwords) before calling `observe_conversation`. If secrets are detected, the turn is skipped silently.

### Manual verification

After running `waggle-mcp setup --yes`, check `~/.claude/settings.json` for a `hooks` block containing entries with `waggle` in the command path.

Have a 2-turn conversation in Claude Code, then in a fresh session ask about the previous turn — it should recall it without any prompt rule.
