# Speed benchmark

Hardware: Apple M5 Pro, Darwin 27.2.0, Python 3.11.18. Date: 2026-11-24.

## Offline parts (no model)

| part | samples | median ms | p95 ms | note |
|---|---|---|---|---|
| load all packs | 20 | 1.0 | 16.84 | 4 packs |
| route one goal | 760 | 0.17 | 0.2 | 38 goals |
| open a Twin with a user's facts | 240 | 0.53 | 0.63 | 12 Kimlik scenarios, in memory |
| evaluate all rules | 240 | 0.03 | 0.03 | 14-15 rules per Kimlik pack |
| build keyword index | 40 | 1.98 | 2.2 | one Kimlik pack |
| keyword search (BM25) | 40 | 0.01 | 0.04 | one question |
| whole agent run, scripted policy | 24 | 6.19 | 8.01 | Kimlik scenarios; orchestration + tools + Gate 1, no model |

## Model parts (local Ollama)

Not run (use --live with Ollama running).
