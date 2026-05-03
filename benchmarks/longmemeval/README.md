# LongMemEval Baseline

This directory contains the downloaded `longmemeval_s_cleaned.json` split and Waggle's exploratory benchmark outputs.

## Dataset

- Source: `xiaowu0162/longmemeval-cleaned`
- File: `longmemeval_s_cleaned.json`
- Cases: `500`


## Current measured result

Measured locally on the full `500`-question `s` split (`all-MiniLM-L6-v2`, warm cache, 2026-05-03):

| Mode | R@5 | Exact@5 | Exact@10 | Exact@20 |
|------|-----|---------|---------|---------|
| `graph_raw` | `97.4%` | `89.0%` | `89.0%` | `89.0%` |
| `graph_hybrid` | `97.0%` | `87.2%` | `94.8%` | `98.0%` |

**The headline number is `graph_raw` at `89.0% Exact@5`.** It has no tunable reranking heuristics and is the fairest apples-to-apples comparison against other retrieval systems.

`graph_hybrid` scores slightly lower on Exact@5 but recovers strongly at higher cutoffs (Exact@20 = 98.0%), meaning it finds all the right sessions — it just doesn't always pack them all into the top 5 slots for high-cardinality queries.

### Overfitting check — held-out dev/test split

To verify the scores are not artefacts of the specific 500-question distribution, the benchmark was also run with `--held-out` (50 dev / 450 test, seed 42):

| Mode | Dev Exact@5 (n=50) | Test Exact@5 (n=450) | Gap |
|------|-------------------|---------------------|-----|
| `graph_raw` | `88.0%` | `89.1%` | `+1.1pp` — no overfitting |
| `graph_hybrid` | `92.0%` | `86.7%` | `−5.3pp` — reranking weights have dev-set sensitivity |

`graph_raw` is stable across the split. `graph_hybrid`'s 5.3pp dev/test gap indicates the heuristic reranking weights (RRF fusion, lexical/coverage scoring) are partially tuned to this distribution and should not be treated as a robust headline number.

### High-cardinality behaviour

All gold sets in this dataset are chunks of the same base session (e.g. `answer_abc_1`, `_2`, `_3`, `_4`). This means:

- **R@5 = 100%** at cardinality ≥ 4 is expected — the embedding finds the base session easily.
- **Exact@5 drops** at high cardinality because fitting 4–6 chunks into 5 slots while excluding noise is a packing problem, not a retrieval quality problem.
- **Exact@10/20 recovering to ~90–100%** confirms the model is finding all the chunks; it just can't rank them all in the top 5.

Raw output artifacts:

- [`results_graph_raw_2026-05-03.json`](./results_graph_raw_2026-05-03.json)
- [`results_graph_hybrid_2026-05-03.json`](./results_graph_hybrid_2026-05-03.json)
- [`results_graph_raw_heldout_2026-05-03_dev.json`](./results_graph_raw_heldout_2026-05-03_dev.json)
- [`results_graph_raw_heldout_2026-05-03_test.json`](./results_graph_raw_heldout_2026-05-03_test.json)
- [`results_graph_hybrid_heldout_2026-05-03_dev.json`](./results_graph_hybrid_heldout_2026-05-03_dev.json)
- [`results_graph_hybrid_heldout_2026-05-03_test.json`](./results_graph_hybrid_heldout_2026-05-03_test.json)
- Methodology note: [docs/longmemeval-methodology.md](../../docs/longmemeval-methodology.md)

## How we ran this benchmark

### Full 500-question run

```bash
# Raw retrieval (no reranking) — the headline number
.venv/bin/python scripts/benchmark_longmemeval.py \
  benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --mode graph_raw \
  --output benchmarks/longmemeval/results_graph_raw_2026-05-03.json

# Hybrid retrieval with heuristic reranking
.venv/bin/python scripts/benchmark_longmemeval.py \
  benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --mode graph_hybrid \
  --output benchmarks/longmemeval/results_graph_hybrid_2026-05-03.json
```

### Held-out dev/test split (overfitting check)

```bash
# Raw — held-out split (50 dev / 450 test, seed 42)
.venv/bin/python scripts/benchmark_longmemeval.py \
  benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --mode graph_raw \
  --held-out \
  --output benchmarks/longmemeval/results_graph_raw_heldout_2026-05-03.json

# Hybrid — held-out split
.venv/bin/python scripts/benchmark_longmemeval.py \
  benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --mode graph_hybrid \
  --held-out \
  --output benchmarks/longmemeval/results_graph_hybrid_heldout_2026-05-03.json
```

To control where prepared-entry cache files are written:

```bash
.venv/bin/python scripts/benchmark_longmemeval.py \
  benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --mode graph_raw \
  --cache-dir /tmp/longmemeval-cache
```

Each run prints whether the prepared-session cache was `cold` or `warm`, and the saved JSON artifact records the cache status, cache key, cache path, and prepared-entry counts.

## Notes

- This adapter prepares each LongMemEval entry in memory, batches unique session embeddings, and caches them per run instead of rebuilding a fresh graph per case.
- By default cache files are written to `benchmarks/longmemeval/.cache/`; warm reruns reuse the prepared-session cache and skip re-embedding the same split.
- `graph_raw` is the fairest current comparison to raw retrieval systems because it does not add reranking.
- The 2026-05-03 `graph_raw` run used `all-MiniLM-L6-v2`, evaluated all `500` cases, prepared `23,796` sessions, and reported `13/500` misses plus `55/500` non-exact top-5 support sets.
- `graph_hybrid` uses the same prepared-session cache. Its Exact@5 is slightly lower than `graph_raw` on the full split, and the held-out test confirms a 5.3pp dev/test gap in the reranking heuristics — treat it as exploratory rather than a headline number.
- Artifacts dated April 25 – May 2, 2026 are from a regression window (embeddings cache bug + ranking formula change) and should not be used as baselines. See the [postmortem](../../docs/postmortems/2026-05-02-embeddings-cache-and-ranking-regression.md).
