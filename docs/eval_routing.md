# Procedure routing

| split | goals | correct | wrong pack | false start | missed |
|---|---|---|---|---|---|
| dev | 26 | 26 | 0 | 0 | 0 |
| held_out | 12 | 9 | 0 | 0 | 3 |

| split | goal | expected | selected | scores |
|---|---|---|---|---|
| held_out | oturma izni almak istiyorum, değişim öğrencisiyim | ikamet_first | None | [('ikamet_extension', 2), ('ikamet_first', 2)] |
| held_out | my ikamet card is valid until March but my exchange continues, can I extend it? | ikamet_extension | None | [('ikamet_extension', 2), ('ikamet_first', 2)] |
| held_out | someone stole my Turkish identity card | tc_kimlik_renewal | None | [('tc_kimlik_first', 5), ('tc_kimlik_renewal', 5)] |
