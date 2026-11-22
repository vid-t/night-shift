# Final results on the test set

Test set: 1536 windows: 82 attack, 254 neighbour, 1200 quiet. Score = attack windows called attack minus normal windows called attack. Perfect is 82.

## The whole test set

| System | Attack windows caught (of 82) | False alarms (of 1454) | Score | Attacks found |
|---|---|---|---|---|
| r4-zsize-law | 65 | 23 | 42 | 50 of 56 |
| r5-zsize-law-s1 | 65 | 13 | 52 | 50 of 56 |
| r7-zsize-law-s2 | 68 | 16 | 52 | 52 of 56 |
| r8-zsize-law-s3 | 61 | 18 | 43 | 48 of 56 |
| r5-looprange-steplaw | 62 | 12 | 50 | 48 of 56 |
| r5-looprange-steplaw-s1 | 63 | 20 | 43 | 48 of 56 |
| r6-lr-steplaw-s2 | 63 | 14 | 49 | 49 of 56 |
| r6-lr-steplaw-s3 | 59 | 20 | 39 | 45 of 56 |
| r7-lr-steplaw-s4 | 64 | 17 | 47 | 49 of 56 |
| r7-lr-steplaw-s5 | 62 | 16 | 46 | 51 of 56 |
| r7-lr-steplaw-s6 | 63 | 19 | 44 | 49 of 56 |
| r7-lr-steplaw-s7 | 63 | 21 | 42 | 50 of 56 |
| A vote (seeds 0 to 2, 2 of 3) | 66 | 15 | 51 |  |
| B vote (seeds 0 to 6, 4 of 7) | 65 | 15 | 50 |  |
| Gradient boosting | 71 | 17 | 54 | 52 of 56 |
| Gradient boosting with neighbours | 68 | 14 | 54 | 52 of 56 |
| Range alarm | 38 | 173 | -135 | 31 of 56 |
| A mean of the four single runs (hybrid + law) [42 to 52] | 64.75 | 17.5 | 47.25 |  |
| B mean of the eight single runs (loop-relative + step and law) [39 to 50] | 62.38 | 17.38 | 45 |  |

## The same 811 windows for every system (82 attack, 129 neighbour, 600 quiet)

| System | Caught (of 82) | False alarms, neighbour (of 129) | False alarms, quiet (of 600) | Score on these windows | Estimated score on the whole set | Answers with a verdict |
|---|---|---|---|---|---|---|
| r4-zsize-law | 65 | 5 | 6 | 54 | 43.2 | 811 of 811 |
| r5-zsize-law-s1 | 65 | 4 | 2 | 59 | 53.1 | 811 of 811 |
| r7-zsize-law-s2 | 68 | 5 | 2 | 61 | 54.2 | 811 of 811 |
| r8-zsize-law-s3 | 61 | 6 | 3 | 52 | 43.2 | 811 of 811 |
| r5-looprange-steplaw | 62 | 6 | 1 | 55 | 48.2 | 811 of 811 |
| r5-looprange-steplaw-s1 | 63 | 6 | 3 | 54 | 45.2 | 811 of 811 |
| r6-lr-steplaw-s2 | 63 | 5 | 1 | 57 | 51.2 | 811 of 811 |
| r6-lr-steplaw-s3 | 59 | 5 | 5 | 49 | 39.2 | 811 of 811 |
| r7-lr-steplaw-s4 | 64 | 5 | 3 | 56 | 48.2 | 811 of 811 |
| r7-lr-steplaw-s5 | 62 | 5 | 3 | 54 | 46.2 | 811 of 811 |
| r7-lr-steplaw-s6 | 63 | 5 | 4 | 54 | 45.2 | 811 of 811 |
| r7-lr-steplaw-s7 | 63 | 6 | 6 | 51 | 39.2 | 811 of 811 |
| A vote (seeds 0 to 2, 2 of 3) | 66 | 5 | 2 | 59 | 52.2 | 811 of 811 |
| B vote (seeds 0 to 6, 4 of 7) | 65 | 5 | 3 | 57 | 49.2 | 811 of 811 |
| Gradient boosting | 71 | 7 | 1 | 63 | 55.2 | 811 of 811 |
| Gradient boosting with neighbours | 68 | 4 | 2 | 62 | 56.1 | 811 of 811 |
| Range alarm | 38 | 20 | 69 | -51 | -139.4 | 811 of 811 |
| Kimi K3.1 Pro, cold | 78 | 71 | 291 | -284 | -643.8 | 807 of 811 (788 before the cut-off answers were asked again) |
| Kimi K3.1 Pro, hints and solved examples | 78 | 52 | 207 | -181 | -438.4 | 811 of 811 (810 before the cut-off answers were asked again) |

## Per loop on those 811 windows (caught / false alarms)

| System | P1-FC | P1-LC | P1-PC | P1-TC | P2-SC | P3-LC |
|---|---|---|---|---|---|---|
| A vote (seeds 0 to 2, 2 of 3) | 18 / 0 | 12 / 2 | 16 / 1 | 0 / 1 | 15 / 2 | 5 / 1 |
| B vote (seeds 0 to 6, 4 of 7) | 15 / 2 | 9 / 4 | 17 / 0 | 2 / 1 | 15 / 1 | 7 / 0 |
| Gradient boosting | 18 / 0 | 13 / 5 | 16 / 0 | 0 / 1 | 15 / 1 | 9 / 1 |
| Gradient boosting with neighbours | 18 / 0 | 11 / 2 | 16 / 0 | 0 / 1 | 14 / 1 | 9 / 2 |
| Range alarm | 13 / 2 | 4 / 2 | 3 / 4 | 2 / 45 | 13 / 30 | 3 / 6 |
| Kimi K3.1 Pro, cold | 19 / 80 | 14 / 35 | 17 / 64 | 5 / 63 | 13 / 22 | 10 / 98 |
| Kimi K3.1 Pro, hints and solved examples | 19 / 72 | 14 / 26 | 18 / 41 | 4 / 50 | 15 / 7 | 8 / 63 |

## The 58 windows of the side run with thinking on (28 attack, 30 normal)

| System | Caught (of 28) | False alarms (of 30) | Score | Answers with a verdict |
|---|---|---|---|---|
| Kimi K3.1 Pro, cold, thinking off | 27 | 20 | 7 | 57 of 58 |
| Kimi K3.1 Pro, cold, thinking on (low) | 25 | 15 | 10 | 58 of 58 |
| Kimi K3.1 Pro, hints and solved examples, thinking off | 25 | 18 | 7 | 58 of 58 |
| Gradient boosting | 25 | 0 | 25 | 58 of 58 |
| A vote (seeds 0 to 2, 2 of 3) | 20 | 0 | 20 | 58 of 58 |
| B vote (seeds 0 to 6, 4 of 7) | 21 | 0 | 21 | 58 of 58 |

## Our runs with the seconds shuffled (test set)

| Run | Attack windows called attack | Normal windows called attack |
|---|---|---|
| r4-zsize-law | 25 | 9 |
| r5-zsize-law-s1 | 33 | 12 |
| r7-zsize-law-s2 | 32 | 5 |
| r8-zsize-law-s3 | 25 | 4 |
| r5-looprange-steplaw | 38 | 624 |
| r5-looprange-steplaw-s1 | 44 | 677 |
| r6-lr-steplaw-s2 | 38 | 648 |
| r6-lr-steplaw-s3 | 39 | 672 |
| r7-lr-steplaw-s4 | 37 | 502 |
| r7-lr-steplaw-s5 | 33 | 524 |
| r7-lr-steplaw-s6 | 29 | 395 |
| r7-lr-steplaw-s7 | 30 | 497 |

## The same systems on the check set (1,092 windows, 144 attack), for the drop from check to test

| System | Caught (of 144) | False alarms (of 948) | Score |
|---|---|---|---|
| r4-zsize-law | 97 | 26 | 71 |
| r5-zsize-law-s1 | 99 | 20 | 79 |
| r7-zsize-law-s2 | 95 | 17 | 78 |
| r8-zsize-law-s3 | 96 | 18 | 78 |
| r5-looprange-steplaw | 95 | 11 | 84 |
| r5-looprange-steplaw-s1 | 98 | 12 | 86 |
| r6-lr-steplaw-s2 | 90 | 27 | 63 |
| r6-lr-steplaw-s3 | 97 | 23 | 74 |
| r7-lr-steplaw-s4 | 99 | 22 | 77 |
| r7-lr-steplaw-s5 | 97 | 18 | 79 |
| r7-lr-steplaw-s6 | 100 | 17 | 83 |
| r7-lr-steplaw-s7 | 93 | 23 | 70 |
| A vote (seeds 0 to 2, 2 of 3) | 98 | 18 | 80 |
| B vote (seeds 0 to 6, 4 of 7) | 101 | 11 | 90 |
| Gradient boosting | 108 | 26 | 82 |
| Gradient boosting with neighbours | 108 | 21 | 87 |
| Range alarm | 49 | 113 | -64 |

## What the big model cost

| File | Questions | Spent (USD) | USD per 1,000 questions | Seconds per answer | Tokens read | Tokens written |
|---|---|---|---|---|---|---|
| final_k31_cold.jsonl | 811 | 1.2457 | 1.54 | 1.4 | 4421 | 55 |
| final_k31_cold_more_room.jsonl | 23 | 0.0632 | 2.75 | 2.6 | 5226 | 183 |
| final_k31_cold_thinking.jsonl | 58 | 0.3578 | 6.17 | 6.1 | 4587 | 580 |
| final_k31_hints_examples.jsonl | 811 | 1.3274 | 1.64 | 1.4 | 5440 | 43 |
| final_k31_hints_examples_more_room.jsonl | 1 | 0.0019 | 1.92 | 1.4 | 6628 | 52 |
| pilot20_k31_cold.jsonl | 20 | 0.0324 | 1.62 | 1.6 | 4573 | 58 |
| pilot20_k31_hints_examples.jsonl | 20 | 0.0354 | 1.77 | 1.5 | 5600 | 46 |
| pilot_k31_cold.jsonl | 12 | 0.0182 | 1.52 | 1.4 | 4558 | 50 |

Total spent on the big model: 3.082 USD.

