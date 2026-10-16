# CAMC-JL metadata

759 learner-writing examples (200 development and 559 test) from the Natane and ICJS/TUFS learner corpora,
each with one of ten error categories: aspect, tense, style, passive, transitivity, honorific, benefactive,
demonstrative, sentence connection, and causative.

| File | Contents |
|---|---|
| `metadata.csv` | `release_id`, `split`, `source`, `error_type` for all 759 examples |
| `development_folds.csv` | `release_id`, `fold`, `component`, `source`, `error_type`; three grouped folds with 67, 67, and 66 development examples |
| `grouping_keys.csv` | Opaque `document_key`, `available_unit_key`, `window_key`, and `pair_key` labels for overlap and cluster analysis |
| `historical_subsets.csv` | `pair_seen` and `no_shared_available_unit` flags for the 559 test examples |
| `test_cluster_order.csv` | Integer cluster ranks that fix the bootstrap sampling order |

The grouping labels are de-linked: they are not source IDs or text hashes. No learner writing, reference spans,
corrected strings, or predictions are included; the texts are obtained from the original corpora under their
providers' conditions.
