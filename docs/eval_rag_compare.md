# Retrieval: keyword vs embedding vs hybrid (embedding model bge-m3)

Top-5; hybrid = reciprocal rank fusion of the top 20 of each. Index time is for all packs in the run, from an empty cache.

## rules (66 queries)

| retriever | hit@1 | hit@3 | hit@5 | MRR@5 | query p50 ms | query p95 ms | index s |
|---|---|---|---|---|---|---|---|
| bm25 | 0.364 | 0.576 | 0.636 | 0.475 | 0.23 | 0.52 | 0.02 |
| embedding | 0.561 | 0.758 | 0.803 | 0.655 | 32.56 | 46.18 | 12.09 |
| hybrid | 0.47 | 0.667 | 0.773 | 0.586 | 32.82 | 46.5 | 12.11 |

hit@3 by procedure_id:

| retriever | ikamet_extension (n=8) | ikamet_first (n=29) | tc_kimlik_first (n=15) | tc_kimlik_renewal (n=14) |
|---|---|---|---|---|
| bm25 | 0.875 | 0.931 | 0.133 | 0.143 |
| embedding | 0.625 | 0.931 | 0.6 | 0.643 |
| hybrid | 0.875 | 0.862 | 0.4 | 0.429 |

## questions (31 queries)

| retriever | hit@1 | hit@3 | hit@5 | MRR@5 | query p50 ms | query p95 ms | index s |
|---|---|---|---|---|---|---|---|
| bm25 | 0.097 | 0.29 | 0.29 | 0.183 | 0.06 | 0.08 | 0.02 |
| embedding | 0.742 | 0.903 | 0.968 | 0.832 | 23.11 | 26.99 | 12.09 |
| hybrid | 0.548 | 0.774 | 0.903 | 0.681 | 23.19 | 27.06 | 12.11 |

hit@3 by procedure_id:

| retriever | tc_kimlik_first (n=16) | tc_kimlik_renewal (n=15) |
|---|---|---|
| bm25 | 0.312 | 0.267 |
| embedding | 0.875 | 0.933 |
| hybrid | 0.688 | 0.867 |

hit@3 by lang:

| retriever | en (n=29) | tr (n=2) |
|---|---|---|
| bm25 | 0.276 | 0.5 |
| embedding | 0.897 | 1.0 |
| hybrid | 0.793 | 0.5 |

## Questions missed in the top 3

| id | question | bm25 | embedding | hybrid |
|---|---|---|---|---|
| KF01 | How many days do I have to register my baby's birth? | - | 1 | 1 |
| KF02 | Do I have to pay the card fee for my newborn's first ID card? | - | 1 | 1 |
| KF05 | Who can apply for the ID card of a child under 15? | - | 1 | 4 |
| KF08 | What documents does a new Turkish citizen need for the first ID card? | - | 2 | - |
| KF09 | As a new citizen, must my address be registered before I apply? | - | - | - |
| KF11 | Can somebody else collect my new ID card for me? | - | 5 | 5 |
| KF12 | I paid too much for the ID card fee. How do I get the money back? | - | 3 | 3 |
| KF13 | Can I pay last year's card fee, or which year's fee applies? | - | 1 | 1 |
| KF14 | My grandmother is bedridden. Can the office come to her home for the ID card application? | - | 1 | 2 |
| KF15 | Yeni doğan bebek için kimlik kartı başvurusu nasıl yapılır? | - | 1 | - |
| KF16 | Can I use the Turkish ID card to travel abroad? | - | 1 | 3 |
| KR01 | What should I do if I lost my Turkish ID card? | - | 1 | 4 |
| KR03 | Is there a fine for losing my identity card? | - | 2 | 2 |
| KR04 | Which documents do I need to apply for a replacement of a lost ID card? | - | 2 | 2 |
| KR05 | The chip fell off my ID card. Do I have to pay for the new card? | - | 1 | 1 |
| KR06 | My ID card is cracked but still readable. Is the new application a replacement or a loss? | - | 1 | 1 |
| KR09 | How can I check the status of my ID card application? | - | 4 | 4 |
| KR10 | Where can I see when my ID card expires? | - | 1 | 1 |
| KR11 | I booked an appointment but cannot go. What happens if I do not cancel it? | - | 1 | 1 |
| KR12 | Can I go to the population office without an appointment? | - | 1 | 1 |
| KR14 | How can I pay the ID card fee online with a credit card? | - | 1 | 1 |
| KR15 | How long is a new card kept at the office if I do not collect it? | - | 1 | 1 |
