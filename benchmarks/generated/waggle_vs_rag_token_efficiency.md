# Waggle Graph vs Vanilla RAG Token Efficiency

- Corpus: 50 conversations, 30 queries, ~132485 transcript tokens
- Waggle embedding: `all-MiniLM-L6-v2`
- Vanilla RAG embedding: `text-embedding-3-small` (actual: `all-MiniLM-L6-v2`)

## Headline Table

| Metric | Waggle Graph | Vanilla RAG |
|--------|--------------|-------------|
| Avg input tokens per query | 175.3 | 1308.9 |
| Avg retrieved-context tokens | 158.3 | 1291.9 |
| Recall@k (did it find the fact) | 83.3% | 83.3% |
| Multi-hop accuracy | 0.0% | 10.0% |
| Latency p50 / p95 | 53.5 ms / 86.4 ms | 7448.5 ms / 10983.5 ms |

## Waggle Retrieval Policy

- `retrieval_mode=graph`
- `max_nodes=5`
- `max_depth=2`
- `expand_depth=0`
- `tiered_retrieval=False`
- `tiered_top_k_windows=3`
- Avg returned subgraph: 3.9 nodes, 4.8 edges

## Vanilla RAG Baseline

- `chunk_size=512` tokens
- `overlap=64` tokens
- `top_k=5`
- Requested embedding model: `text-embedding-3-small`

## Example Retrievals

### database_factual (factual)

**Query:** What is the current production choice for database?

**Gold facts:** `database_new`

**Waggle (`graph`):**
- Database current state: The current production choice for database is PostgreSQL production.
- Database previous state: The previous baseline for database was SQLite local only.
- Database change reason: The reason the team changed database is safer migrations and parity.
- Database dependency: The analytics joins workflow depends on the current database choice.
Edges:
- c6648dc4-9116-45b6-a90c-0e89dcde4baa->updates->60dd5d04-8f39-4d0e-bce6-f21d1abf57aa
- c6648dc4-9116-45b6-a90c-0e89dcde4baa->contradicts->60dd5d04-8f39-4d0e-bce6-f21d1abf57aa
- 8b632c77-29d4-4853-b8af-e5a3f15ce7ed->depends_on->c6648dc4-9116-45b6-a90c-0e89dcde4baa
- 61dd3d9d-45a3-4279-8aad-21523ff8a2b2->depends_on->c6648dc4-9116-45b6-a90c-0e89dcde4baa
- 61dd3d9d-45a3-4279-8aad-21523ff8a2b2->relates_to->8b632c77-29d4-4853-b8af-e5a3f15ce7ed

**Vanilla RAG top chunks:**
- choice is now PostgreSQL production.. User: People also discussed naming consistency, reporting requirements, support load, sprint scheduling, and documentation debt tied to For database, the current production choice is now PostgreSQL prod
- incident history, rollback options, and coordination overhead around database thread 2. Agent: People also discussed naming consistency, reporting requirements, support load, sprint scheduling, and documentation debt tied to database thread
- also discussed naming consistency, reporting requirements, support load, sprint scheduling, and documentation debt tied to For database, the current baseline is SQLite local only..

### database_relational (relational)

**Query:** What replaced the older database baseline and why was it changed?

**Gold facts:** `database_new, database_reason`

**Waggle (`graph`):**
- Database previous state: The previous baseline for database was SQLite local only.
- Database current state: The current production choice for database is PostgreSQL production.
- Database change reason: The reason the team changed database is safer migrations and parity.
- Database owner: The owner for the database stack is the Database platform team.
- Database dependency: The analytics joins workflow depends on the current database choice.
Edges:
- c6648dc4-9116-45b6-a90c-0e89dcde4baa->updates->60dd5d04-8f39-4d0e-bce6-f21d1abf57aa
- c6648dc4-9116-45b6-a90c-0e89dcde4baa->contradicts->60dd5d04-8f39-4d0e-bce6-f21d1abf57aa
- 8b632c77-29d4-4853-b8af-e5a3f15ce7ed->depends_on->c6648dc4-9116-45b6-a90c-0e89dcde4baa
- 61dd3d9d-45a3-4279-8aad-21523ff8a2b2->depends_on->c6648dc4-9116-45b6-a90c-0e89dcde4baa
- 61dd3d9d-45a3-4279-8aad-21523ff8a2b2->relates_to->8b632c77-29d4-4853-b8af-e5a3f15ce7ed

**Vanilla RAG top chunks:**
- also discussed naming consistency, reporting requirements, support load, sprint scheduling, and documentation debt tied to For database, the current baseline is SQLite local only..
- incident history, rollback options, and coordination overhead around database thread 1. Agent: People also discussed naming consistency, reporting requirements, support load, sprint scheduling, and documentation debt tied to database thread
- tradeoffs, onboarding pain points, incident history, rollback options, and coordination overhead around For analytics, the current baseline is daily batch exports.. User: People also discussed naming consistency, reporting requirements, sup

### database_multi_hop (multi_hop)

**Query:** For database, which current choice supports analytics joins, and who owns that stack?

**Gold facts:** `database_new, database_dependency, database_owner`

**Waggle (`graph`):**
- Database dependency: The analytics joins workflow depends on the current database choice.
- Database current state: The current production choice for database is PostgreSQL production.
- Database previous state: The previous baseline for database was SQLite local only.
- Database change reason: The reason the team changed database is safer migrations and parity.
Edges:
- c6648dc4-9116-45b6-a90c-0e89dcde4baa->updates->60dd5d04-8f39-4d0e-bce6-f21d1abf57aa
- c6648dc4-9116-45b6-a90c-0e89dcde4baa->contradicts->60dd5d04-8f39-4d0e-bce6-f21d1abf57aa
- 8b632c77-29d4-4853-b8af-e5a3f15ce7ed->depends_on->c6648dc4-9116-45b6-a90c-0e89dcde4baa
- 61dd3d9d-45a3-4279-8aad-21523ff8a2b2->depends_on->c6648dc4-9116-45b6-a90c-0e89dcde4baa
- 61dd3d9d-45a3-4279-8aad-21523ff8a2b2->relates_to->8b632c77-29d4-4853-b8af-e5a3f15ce7ed

**Vanilla RAG top chunks:**
- The team compared implementation tradeoffs, onboarding pain points, incident history, rollback options, and coordination overhead around The owner for the analytics stack is the Analytics platform team.. User: People also discussed naming c
- checklists, ownership gaps, and release sequencing for analytics thread 4. User: The team compared implementation tradeoffs, onboarding pain points, incident history, rollback options, and coordination overhead around analytics thread 4. Ag
- incident history, rollback options, and coordination overhead around database thread 3. Agent: People also discussed naming consistency, reporting requirements, support load, sprint scheduling, and documentation debt tied to database thread

## Verdict

Waggle wins on token efficiency in this benchmark: it cut average retrieved context from 1291.9 to 158.3 tokens while also raising multi-hop accuracy from 10.0% to 0.0%. The tradeoff is that graph retrieval only helps when the memory graph already captured the right facts and links; plain chunked RAG remains competitive on direct factual recall and has the simpler ingestion story because it indexes raw transcripts without structure.

## Notes

- Vanilla RAG requested `text-embedding-3-small` but ran with local fallback `all-MiniLM-L6-v2` because OpenAI embeddings were unavailable: OPENAI_API_KEY is required for OpenAI embedding benchmarks.
