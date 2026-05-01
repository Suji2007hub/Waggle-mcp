# Token Efficiency Benchmark V2 Comparison

- Corpus: 50 conversations, 30 queries, ~194481 transcript tokens

## Config Table

| Metric | Old | New-no-rerank | New-full |
|--------|-----|---------------|----------|
| Recall@k | 50.0% | 100.0% | 100.0% |
| Multi-hop accuracy | 0.0% | 30.0% | 0.0% |
| Extraction-failure recall | 0.0% | 100.0% | 100.0% |
| Avg input tokens | 178.0 | 1391.5 | 1477.3 |
| Avg retrieved context tokens | 156.6 | 1370.1 | 1455.8 |
| Avg rerank tokens | 0.0 | 0.0 | 3079.2 |
| Avg total token cost | 178.0 | 1391.5 | 4556.4 |
| Latency p50 / p95 | 5.9 / 36.0 ms | 14.8 / 53.3 ms | 15.4 / 55.7 ms |

## Breakdown by Subset

| Subset | Metric | Old | New-no-rerank | New-full |
|--------|--------|-----|---------------|----------|
| single_fact | recall_at_k | 50.0% | 100.0% | 100.0% |
| multi_hop | exact_support | 0.0% | 30.0% | 0.0% |
| extraction_failure | recall_at_k | 0.0% | 100.0% | 100.0% |

## Verdict

`Old` stays cheapest and simplest, but its recall (50.0%) collapses on extraction-failure and multi-hop questions. `New-no-rerank` is the best performer in this run: it reaches 100.0% overall recall and 30.0% multi-hop accuracy without rerank overhead. `New-full` currently regresses relative to `New-no-rerank`; reranking adds 3079.2 tokens per query on average and should be treated as a bug to debug rather than a win.

## Failing Multi-hop Traces

### old
- `database_multi_hop`: {"graph_nodes": ["Database dependency", "Database current state"], "graph_edges": ["depends_on"], "chunk_trace": []}
- `auth_multi_hop`: {"graph_nodes": ["Auth dependency", "Auth current state"], "graph_edges": ["depends_on"], "chunk_trace": []}
- `cache_multi_hop`: {"graph_nodes": ["Cache dependency", "Cache current state"], "graph_edges": ["depends_on"], "chunk_trace": []}

### new_full
- `database_multi_hop`: {"graph_nodes": ["Database dependency", "Database current state"], "graph_edges": ["depends_on"], "chunk_trace": [{"chunk_id": "database_c5::chunk13", "conversation_id": "database_c5", "first_pass_rank": 1, "preview": "migrations and parity. User: We reviewed database thread 5 rollout notes, risk registers, test plans, migration checklists, ownership gaps, and release sequencing for Keep the prev", "rerank_rank": 2}, {"chunk_id": "database_c3::chunk13", "conversation_id": "database_c3", "first_p
- `auth_multi_hop`: {"graph_nodes": ["Auth dependency", "Auth current state"], "graph_edges": ["depends_on"], "chunk_trace": [{"chunk_id": "auth_c3::chunk12", "conversation_id": "auth_c3", "first_pass_rank": 1, "preview": "incident history, rollback options, and coordination overhead around auth thread 3. Agent: People also discussed naming consistency, reporting requirements, support load, sprint sc", "rerank_rank": 1}, {"chunk_id": "mobile_c3::chunk13", "conversation_id": "mobile_c3", "first_pass_rank": 2, "previ
- `cache_multi_hop`: {"graph_nodes": ["Cache dependency", "Cache current state"], "graph_edges": ["depends_on"], "chunk_trace": [{"chunk_id": "cache_c2::chunk13", "conversation_id": "cache_c2", "first_pass_rank": 1, "preview": "migration checklists, ownership gaps, and release sequencing for For cache, the current production choice is now Redis cache enabled in prod.. Agent: The team compared implementati", "rerank_rank": 1}, {"chunk_id": "cache_c5::chunk13", "conversation_id": "cache_c5", "first_pass_rank": 2, "pre

## Notes

- Hybrid modes use graph retrieval plus transcript chunk retrieval. `New-full` adds heuristic reranking over the first-pass chunk pool.
- Benchmark used local embeddings `all-MiniLM-L6-v2` for graph and transcript chunk ranking.
- Sanity-check mismatch: Single-fact recall did not meet the expected target in this run.
- Sanity-check mismatch: Multi-hop accuracy did not meet the expected target in this run.
- Sanity-check mismatch: Latency target did not meet the expected target in this run.
