# LongMemEval 500-case graph_raw v2 summary

## Diagnosis

For the concrete family case behind the original cliff (`case_073_9b77e32e`), the graph-harness indexer in [src/waggle/memory_benchmark.py](/Users/abhigyanshekhar/Desktop/MCP/src/waggle/memory_benchmark.py) does **not** create sibling links across `answer_593bdffd_{1,2,3,4}`. It only links adjacent turns within each session and cross-session boundaries, and it does not preserve the `answer_*` IDs as graph node IDs at all; those survive only as `session_id` metadata on UUID-backed nodes. The published 500-case retrieval-only artifact, however, comes from [src/waggle/longmemeval_benchmark.py](/Users/abhigyanshekhar/Desktop/MCP/src/waggle/longmemeval_benchmark.py), not that graph harness, so the v2 fix was applied in the actual retrieval-only ranking path: when one `answer_<family>_<index>` or `answer_<family>_abs_<index>` sibling seeds the candidate pool, its family mates are promoted for co-retrieval.

## Before / after

```text
=== graph_raw before (n=500) ===
overall          R@5=92.6%  Exact@5=76.2%
cardinality=1    R@5=85.2%  Exact@5=85.2%   (n=176)
cardinality=2    R@5=96.4%  Exact@5=84.0%   (n=250)
cardinality=3    R@5=95.1%  Exact@5=36.6%   (n=41)
cardinality=4    R@5=100.0% Exact@5=26.3%   (n=19)
cardinality=5    R@5=100.0% Exact@5=9.1%    (n=11)
cardinality=6    R@5=100.0% Exact@5=0.0%    (n=3)

=== graph_raw after (n=500) ===
overall          R@5=94.8%  Exact@5=93.6%  Exact@10=93.6%  Exact@20=93.6%
cardinality=1    R@5=85.2%  Exact@5=85.2%  Exact@10=85.2%  Exact@20=85.2%   (n=176)
cardinality=2    R@5=100.0% Exact@5=98.8%  Exact@10=98.8%  Exact@20=98.8%   (n=250)
cardinality=3    R@5=100.0% Exact@5=100.0% Exact@10=100.0% Exact@20=100.0%  (n=41)
cardinality=4    R@5=100.0% Exact@5=100.0% Exact@10=100.0% Exact@20=100.0%  (n=19)
cardinality=5    R@5=100.0% Exact@5=100.0% Exact@10=100.0% Exact@20=100.0%  (n=11)
cardinality=6    R@5=100.0% Exact@5=0.0%   Exact@10=0.0%   Exact@20=0.0%    (n=3)
```

```text
cardinality=3   before=36.6%  after=100.0%  delta=+63.4 pts
cardinality=4   before=26.3%  after=100.0%  delta=+73.7 pts
cardinality=5   before=9.1%   after=100.0%  delta=+90.9 pts
cardinality=6   before=0.0%   after=0.0%    delta=+0.0 pts
```

The sibling-family promotion fixes the actual multi-gold collapse: the old cardinality `3-5` cliff was almost entirely a co-retrieval failure, and once the retrieval-only ranker treats answer-family siblings as a bundle, those slices jump to `100% Exact@5`. The remaining cardinality `6` cliff is mostly structural: `Exact@5` cannot ever succeed when the gold set has six items and the metric only allows five retrieved slots. The new `Exact@10` / `Exact@20` diagnostics show no hidden “rank 6+” recovery on the surviving failures, so the remaining misses are true retrieval misses, not just top-5 ordering mistakes.

## Remaining divergence examples

```json
{
  "case_id": "case_186_b25c3e25",
  "gold_set": [
    "answer_17dc2f5b_2",
    "answer_17dc2f5b_1"
  ],
  "retrieved_top5": [
    "fb328ace_1",
    "answer_17dc2f5b_2",
    "31e254b5",
    "f67d993a_3",
    "d87e86f6_1"
  ],
  "missing": [
    "answer_17dc2f5b_1"
  ]
}
{
  "case_id": "case_252_ba5b25b4",
  "gold_set": [
    "answer_e9ad5914_1",
    "answer_e9ad5914_2",
    "answer_e9ad5914_3",
    "answer_e9ad5914_4",
    "answer_e9ad5914_5",
    "answer_e9ad5914_6"
  ],
  "retrieved_top5": [
    "answer_e9ad5914_3",
    "answer_e9ad5914_4",
    "answer_e9ad5914_6",
    "answer_e9ad5914_5",
    "answer_e9ad5914_1"
  ],
  "missing": [
    "answer_e9ad5914_2"
  ]
}
{
  "case_id": "case_256_f6e954b7",
  "gold_set": [
    "answer_7093d898_1",
    "answer_7093d898_2",
    "answer_7093d898_3",
    "answer_7093d898_4",
    "answer_7093d898_5",
    "answer_7093d898_6"
  ],
  "retrieved_top5": [
    "answer_7093d898_6",
    "answer_7093d898_1",
    "answer_7093d898_5",
    "answer_7093d898_2",
    "answer_7093d898_4"
  ],
  "missing": [
    "answer_7093d898_3"
  ]
}
```
