# LongMemEval 500-case retrieval-only summary

## graph_raw

```text
=== graph_raw (n=500) ===
overall          R@5=92.6%  Exact@5=76.2%
cardinality=1    R@5=85.2%  Exact@5=85.2%   (n=176)
cardinality=2    R@5=96.4%  Exact@5=84.0%   (n=250)
cardinality=3    R@5=95.1%  Exact@5=36.6%   (n=41)
cardinality=4    R@5=100.0%  Exact@5=26.3%   (n=19)
cardinality=5    R@5=100.0%  Exact@5=9.1%   (n=11)
cardinality=6    R@5=100.0%  Exact@5=0.0%   (n=3)
divergence: case_072_41e549b7, case_073_9b77e32e, case_074_80ca7a28
```

## graph_hybrid

```text
=== graph_hybrid (n=500) ===
overall          R@5=92.4%  Exact@5=75.2%
cardinality=1    R@5=86.4%  Exact@5=86.4%   (n=176)
cardinality=2    R@5=95.6%  Exact@5=81.2%   (n=250)
cardinality=3    R@5=92.7%  Exact@5=39.0%   (n=41)
cardinality=4    R@5=100.0%  Exact@5=21.1%   (n=19)
cardinality=5    R@5=100.0%  Exact@5=9.1%   (n=11)
cardinality=6    R@5=100.0%  Exact@5=0.0%   (n=3)
divergence: case_072_41e549b7, case_073_9b77e32e, case_074_80ca7a28
```

On single-gold questions, `R@5` and `Exact@5` are identical by definition, and the reports show that directly. The divergence appears on multi-gold cases: both modes usually retrieve at least one supporting gold chunk, but they often fail to recover the full gold support set inside the top 5. That gap is modest for cardinality `2` and then becomes large for cardinality `3+`, with `Exact@5` collapsing hardest on cardinalities `4-6` even when `R@5` is near or at `100%`. In practice, this says the graph retriever is good at finding some relevant support for multi-hop or multi-evidence questions, but much less reliable at assembling all required supporting chunks within a 5-item budget. The hardest slice is therefore the higher-cardinality multi-gold questions, not the single-gold retrieval cases.

## Divergence examples: graph_raw

```json
{
  "case_id": "case_072_41e549b7",
  "gold_set": [
    "answer_ec904b3c_1",
    "answer_ec904b3c_4",
    "answer_ec904b3c_3",
    "answer_ec904b3c_2"
  ],
  "retrieved_top5": [
    "2e4430d8_2",
    "sharegpt_zciCXP1_12",
    "ultrachat_271963",
    "answer_ec904b3c_1",
    "5558a42e_2"
  ],
  "missing": [
    "answer_ec904b3c_4",
    "answer_ec904b3c_3",
    "answer_ec904b3c_2"
  ]
}
{
  "case_id": "case_073_9b77e32e",
  "gold_set": [
    "answer_593bdffd_4",
    "answer_593bdffd_1",
    "answer_593bdffd_3",
    "answer_593bdffd_2"
  ],
  "retrieved_top5": [
    "answer_593bdffd_4",
    "eb47739f_2",
    "c263f1c0",
    "answer_593bdffd_1",
    "2f09d4c8"
  ],
  "missing": [
    "answer_593bdffd_3",
    "answer_593bdffd_2"
  ]
}
{
  "case_id": "case_074_80ca7a28",
  "gold_set": [
    "answer_a8b4290f_3",
    "answer_a8b4290f_1",
    "answer_a8b4290f_2"
  ],
  "retrieved_top5": [
    "sharegpt_XWqXdom_0",
    "answer_a8b4290f_2",
    "answer_a8b4290f_3",
    "35dcacdc_2",
    "ultrachat_565056"
  ],
  "missing": [
    "answer_a8b4290f_1"
  ]
}
```

## Divergence examples: graph_hybrid

```json
{
  "case_id": "case_072_41e549b7",
  "gold_set": [
    "answer_ec904b3c_1",
    "answer_ec904b3c_4",
    "answer_ec904b3c_3",
    "answer_ec904b3c_2"
  ],
  "retrieved_top5": [
    "2e4430d8_2",
    "answer_ec904b3c_4",
    "answer_ec904b3c_1",
    "f6246b5f",
    "0bab76de"
  ],
  "missing": [
    "answer_ec904b3c_3",
    "answer_ec904b3c_2"
  ]
}
{
  "case_id": "case_073_9b77e32e",
  "gold_set": [
    "answer_593bdffd_4",
    "answer_593bdffd_1",
    "answer_593bdffd_3",
    "answer_593bdffd_2"
  ],
  "retrieved_top5": [
    "a7d014e4_1",
    "2f09d4c8",
    "answer_593bdffd_4",
    "e60a93ff_2",
    "5b9c49f7"
  ],
  "missing": [
    "answer_593bdffd_1",
    "answer_593bdffd_3",
    "answer_593bdffd_2"
  ]
}
{
  "case_id": "case_074_80ca7a28",
  "gold_set": [
    "answer_a8b4290f_3",
    "answer_a8b4290f_1",
    "answer_a8b4290f_2"
  ],
  "retrieved_top5": [
    "answer_a8b4290f_3",
    "ultrachat_565056",
    "answer_a8b4290f_2",
    "ultrachat_53385",
    "sharegpt_XWqXdom_0"
  ],
  "missing": [
    "answer_a8b4290f_1"
  ]
}
```
