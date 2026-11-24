# Speed benchmark

Hardware: Apple M5 Pro, Darwin 27.2.0, Python 3.11.18. Date: 2026-11-24.

## Offline parts (no model)

| part | samples | median ms | p95 ms | note |
|---|---|---|---|---|
| load all packs | 20 | 0.96 | 17.3 | 4 packs |
| route one goal | 760 | 0.17 | 0.2 | 38 goals |
| open a Twin with a user's facts | 240 | 0.56 | 0.65 | 12 Kimlik scenarios, in memory |
| evaluate all rules | 240 | 0.03 | 0.03 | 14-15 rules per Kimlik pack |
| build keyword index | 40 | 2.03 | 2.2 | one Kimlik pack |
| keyword search (BM25) | 40 | 0.01 | 0.03 | one question |
| whole agent run, scripted policy | 24 | 6.59 | 7.93 | Kimlik scenarios; orchestration + tools + Gate 1, no model |

## Model parts (local Ollama)

| part | samples | median ms | p95 ms | note |
|---|---|---|---|---|
| first chat call (loads the model) | 1 | 783.48 | 783.48 | qwen3:4b, unloaded first |
| chat call, warm | 5 | 162.0 | 163.76 | qwen3:4b, tiny JSON reply |
| index one Kimlik pack (embeddings) | 1 | 986.82 | 986.82 | bge-m3, 50 passages |
| embedding search, one question | 5 | 20.31 | 23.46 | bge-m3 |
| read document D06-kimlik-card-text | 1 | 1342.1 | 1342.1 | 1/1 documents found (by title), 6 values read |
| read document D07-kimlik-card-photo | 1 | 12639.84 | 12639.84 | 1/1 documents found (by title), 4 values read |
| read document D08-kimlik-card-and-fee-in-one-pdf | 1 | 2293.79 | 2293.79 | 2/2 documents found (by title), 8 values read |
| read document D09-nufus-and-fee-scanned-together | 1 | 7389.78 | 7389.78 | 2/2 documents found (by title), 6 values read |
| whole Kimlik agent run, live | 2 | 23956.8 | 33581.09 | qwen3:4b; scenarios F01-adult-replaces-nufus-cuzdani, F02-newborn-free-card |
