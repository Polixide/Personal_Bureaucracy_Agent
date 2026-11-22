# Retrieval evaluation (4 packs, 567 passages, top-3)

| model | queries | hit@1 | hit@3 | MRR | index s | s/query |
|---|---|---|---|---|---|---|
| bge-m3 | 66 | 0.561 | 0.758 | 0.644 | 392.2 | 0.671 |
| qwen3-embedding:0.6b | 66 | 0.561 | 0.667 | 0.606 | 174.6 | 0.589 |

hit@3 by procedure:

| model | ikamet_extension | ikamet_first | tc_kimlik_first | tc_kimlik_renewal |
|---|---|---|---|---|
| bge-m3 | 0.625 | 0.931 | 0.6 | 0.643 |
| qwen3-embedding:0.6b | 1.0 | 0.897 | 0.333 | 0.357 |
