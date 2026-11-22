| System | Found | Right loop only | Strict | Attack rows called attack | Normal rows called attack | Start within 30 s | Median start error (s) | Start correlation | False alarms per 100 | Reacting loops called attack | Valid answers | Found per version |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid + law, vote of three | 50 of 56 | 35 | 31 | 66 of 82 | 15 of 1454 | 30 | 22 | 0.82 | 0.3 | 4 of 18 | 100% | 10/10 12/12 15/20 13/14 |
| Loop-relative + step and law, vote of seven | 51 of 56 | 35 | 31 | 65 of 82 | 15 of 1454 | 30 | 26 | 0.71 | 0.4 | 4 of 18 | 100% | 10/10 12/12 17/20 12/14 |
| Hybrid + law, one model (seed 1) | 50 of 56 | 34 | 31 | 65 of 82 | 13 of 1454 | 39 | 19 | 0.84 | 0.2 | 3 of 18 | 100% | 10/10 12/12 15/20 13/14 |
| Loop-relative + step and law, one model (seed 0) | 48 of 56 | 34 | 29 | 62 of 82 | 12 of 1454 | 28 | 26 | 0.7 | 0.2 | 5 of 18 | 100% | 10/10 12/12 15/20 11/14 |
| Gradient boosting | 52 of 56 | 38 | 33 | 71 of 82 | 17 of 1454 | 0 |  |  | 0.2 | 5 of 18 | 100% | 10/10 12/12 17/20 13/14 |
| Gradient boosting with neighbours | 52 of 56 | 38 | 35 | 68 of 82 | 14 of 1454 | 0 |  |  | 0.2 | 3 of 18 | 100% | 10/10 12/12 16/20 14/14 |
| Range alarm | 31 of 56 | 12 | 10 | 38 of 82 | 173 of 1454 | 14 | 40 | 0.49 | 10.8 | 5 of 18 | 100% | 9/10 8/12 8/20 6/14 |

Found per version: 20.07, 21.03, 22.04, 23.05. Always answering second 108 would time 26 of 56 attacks; second 120 would time 23. `Strict` counts a reacting loop that is called attack as an error.
