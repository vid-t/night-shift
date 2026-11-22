# Night summary: HAI training search, 21 November 2026

> **Read this first.** This is the report of the training search, written on 21 November 2026, before the one test run. Every score in it is a check-set (validation) score. On the test set, run once afterwards, gradient boosting scores 54 of 82 and our votes score 51 and 50, so the check-set lead of the seven-model vote (90 against 82) did not hold. The final facts are in `runs/final/FACTS.md` and the main README.

**On the check set (validation), a majority vote of seven small models beat gradient boosting: 90 against 82, and against 87 for the stronger gradient boosting variant that also reads the other loops. This lead did not hold on the test set (see the note above). No design beat gradient boosting as a single model on average: the best designs average 79.6 or less, and one design's runs range from 63 to 86. Four of the 64 single runs score above 82.**

Everything below is counted on the 1,092 validation rows (144 attack rows from 15 attacks, 948 normal rows). Score = attack rows called "attack" minus normal rows called "attack". Gradient boosting, the baseline fixed before the night, scores 82 (108 and 26). **At the time of writing, no run of the night had answered, scored or opened the test rows.** Earlier models and both baselines had been scored on them before the night (see the main README, Limits).

The runs trained on a rented graphics card (Nebius) that costs 5.40 US dollars an hour. A run took about an hour on one eighth of the card, under a dollar; the whole search of 64 runs cost about 57 dollars of card time. The answers, logs and scores of all 64 runs are in release v1.0 of this repository (`answers.tar.gz`). The release also holds twelve saved models, the runs of the two final designs (see `MODELS.md`); the other 52 saved models are not published.

## The short version

- **64 full runs in 8 rounds, 10 hours.** The models went from 17 to 37 (the design the night started with) to 71.25 to 79.6 on average for the 6 leading designs.
- **What made the difference:** short text clauses. Code computes one or two numbers per series and writes them into the text the model reads next to the curve. Runs with clauses average 73.1 (56 to 86); runs of the same views without clauses average 56.5 (50 to 63).
- **The leading designs tie as single models.** One design's runs differ by up to 23 points from seed to seed, so their means (79.6, 77, 76.5, 73, 72.25, 71.25) cannot be told apart.
- **A vote over seeds is the best result:** seven seeds of loop-relative + step and law call 101 attack rows and 11 normal rows "attack", score 90. Gradient boosting: 108 and 26, score 82. A stronger gradient boosting variant (it also reads the other loops) scores 87.
- **The price of that result:** it leans on the computed numbers. With every curve blanked, one model of that design (seed 0) still makes 84 attack calls, against 95 with the curve, on the rows we checked.
- **There was a choice to make,** between the design that scores best as a vote (90, seven models, leans on computed numbers) and a one-model design that leant less on them in the one check we ran (76.5 on average; vote 80). It is laid out below as the pick and two alternates.
- **The pick itself is a close call.** It turns on how we counted clauses for the tie-break. Under the other fair count the pick is alternate 1.
- **Every number here is on validation, which chose the designs.** Expect less on the test rows (it was less: see the note at the top).

## The pick

**Hybrid view + law clause** (`--view zsize --clauses law`), by the rule we fixed at 16:37, before rounds 5 to 8 were in: the highest mean among designs with at least three seeds; a lead under the seed gap is a tie; the design with the fewest clauses wins a tie. 6 designs tie; this one has one clause. **How we counted, and the other fair reading:** we counted clauses as the options added on top of a view. But the hybrid view itself writes one number into every text line, the size of the largest change; in the code it is the size clause built in. Counting it, hybrid + law and loop-relative + step and law have two clauses each. The rule's next criterion, the higher mean, then picks loop-relative + step and law (77 against 76.5), which is alternate 1. Both readings are fair. Which one to apply was left open at the time of writing.

| Run | Seed | Attack rows called attack (of 144) | Normal rows called attack (of 948) | Score |
|---|---|---|---|---|
| r4-zsize-law | 0 | 97 | 26 | 71 |
| r5-zsize-law-s1 | 1 | 99 | 20 | 79 |
| r7-zsize-law-s2 | 2 | 95 | 17 | 78 |
| r8-zsize-law-s3 | 3 | 96 | 18 | 78 |
| Mean of the single runs | | 96.75 | 20.25 | 76.5 |
| Majority vote (2 of 3: r4-zsize-law, r5-zsize-law-s1, r7-zsize-law-s2) | | 98 | 18 | 80 |
| Gradient boosting | | 108 | 26 | 82 |

What it is:

- **The curve:** each series scaled to its own mean and spread (the z-score view).
- **The text line of each series:** its largest change against its usual quiet change in that loop.
- **One clause, in the boiler-pressure loop only:** for how many seconds the valve moves the opposite way to what the controller would do.
- **No step clause.** So it passes the shuffled check: 17 to 30 of 948 shuffled normal rows are called "attack", against 242 to 414 for the loop-relative runs with the step clause.

Why it is a defensible pick:

- **It is level with the leaders as a single model** (71, 79, 78, 78; mean 76.5), and its four seeds lie closer together (71 to 79) than the first four loop-relative seeds (63 to 86). Four seeds are too few to call it steadier with confidence.
- **It leant less on the stated numbers in the one check we ran.** In the one model of each design we checked (r5-zsize-law-s1 and r5-looprange-steplaw), blank curves take hybrid + law from 99 attack calls to 72 and add 7 false alarms on neighbour rows; they take the loop-relative design from 95 to 84. That is one model each, on inputs neither saw in training: a sign, not a measurement. And it still makes 72 attack calls from its text alone, so it is no pure curve reader.
- **It finds the water-level jumps from the curve,** not from a stated step number: 13 to 16 of 42 attack rows there (the step clause finds 18; gradient boosting 27).

What it does not do: beat gradient boosting. By the rule its vote is the majority of seeds 0 to 2: 80 (98 and 18), against 82 for the named baseline and 87 for the gradient boosting variant that also reads the other loops ("with neighbours"). With all four seeds, "at least three say attack" scores 83 (96 and 13) and "at least two" scores 80 (101 and 21). We found the 83 after the fact, so it does not count.

Per loop (attack rows called attack / normal rows called attack):

| Loop (attack rows / normal rows) | Hybrid + law: mean of single runs ; vote | Loop-relative + step and law: mean of single runs ; vote | Gradient boosting |
|---|---|---|---|
| P1-FC (18 / 164) | 18 / 8.5 ; 18 / 7 | 13.88 / 5.38 ; 15 / 3 | 11 / 9 |
| P1-LC (42 / 140) | 39 / 11.25 ; 40 / 11 | 37.75 / 8.38 ; 40 / 8 | 40 / 9 |
| P1-PC (18 / 164) | 12.5 / 0.5 ; 14 / 0 | 15 / 5.38 ; 16 / 0 | 18 / 3 |
| P1-TC (12 / 170) | 1 / 0 ; 0 / 0 | 0 / 0 ; 0 / 0 | 0 / 0 |
| P2-SC (12 / 170) | 12 / 0 ; 12 / 0 | 11.62 / 0 ; 12 / 0 | 12 / 1 |
| P3-LC (42 / 140) | 14.25 / 0 ; 14 / 0 | 17.88 / 0 ; 18 / 0 | 27 / 4 |

To train one more: `python3 training/cloud/run_variant.py NAME --start sleep --view zsize --order label_last --epochs 4 --clauses law --choose-pass last --seed N`. Each run kept its saved model as `runs/nebius/2026-11-21/night/<run>/kept/best_model.pt`. The four runs in the table are among the twelve models published in release v1.0 (see `MODELS.md`).

## Two alternates

**These are not the alternates our rule of 16:37 named.** The rule asked for the best design without the step clause and the best design with no numbers in the text. By that letter they are the pick itself (after it: hybrid + law and plant, 72.25; vote 74) and loop-relative with no clause (60, 50, 55, 54; mean 54.75; vote 60). We chose the two below after the results were in, because they show the two ends of the choice more clearly.

### Alternate 1: the seven-model vote, for the highest number

**Loop-relative view + step and law clauses** (`--view looprange --clauses step,law`), eight seeds, answered by majority vote.

| Run | Seed | Attack rows called attack (of 144) | Normal rows called attack (of 948) | Score |
|---|---|---|---|---|
| r5-looprange-steplaw | 0 | 95 | 11 | 84 |
| r5-looprange-steplaw-s1 | 1 | 98 | 12 | 86 |
| r6-lr-steplaw-s2 | 2 | 90 | 27 | 63 |
| r6-lr-steplaw-s3 | 3 | 97 | 23 | 74 |
| r7-lr-steplaw-s4 | 4 | 99 | 22 | 77 |
| r7-lr-steplaw-s5 | 5 | 97 | 18 | 79 |
| r7-lr-steplaw-s6 | 6 | 100 | 17 | 83 |
| r7-lr-steplaw-s7 | 7 | 93 | 23 | 70 |
| Mean of the single runs | | 96.12 | 19.12 | 77 |
| Majority vote (4 of 7: r5-looprange-steplaw, r5-looprange-steplaw-s1, r6-lr-steplaw-s2, r6-lr-steplaw-s3, r7-lr-steplaw-s4, r7-lr-steplaw-s5, r7-lr-steplaw-s6) | | 101 | 11 | 90 |
| Gradient boosting | | 108 | 26 | 82 |

- **It is the only vote of one design, fixed by the rule, that is above gradient boosting:** 90 with seven seeds (101 and 11). The rule leaves out seed 7, and that happens to be the best of the eight ways to leave one seed out: they score 87 to 90. With all eight seeds: 89 for "at least four", 88 for "at least five". The stronger gradient boosting file scores 87.
- **It needs several models.** How the vote depends on their number, over every possible choice among the eight seeds:

| Models in the vote | Mean score over every choice of seeds | Lowest | Highest | Choices |
|---|---|---|---|---|
| 1 | 77 | 63 | 86 | 8 |
| 3 | 83.61 | 74 | 90 | 56 |
| 5 | 86.71 | 80 | 91 | 56 |
| 7 | 88.5 | 87 | 90 | 8 |

- **A single model of this design is a gamble:** 63 to 86.
- **It leans on the stated numbers.** In the one model we checked (seed 0), blank curves leave 84 attack calls against 95 with the curve, and every run of this design fails the shuffled check. In a write-up it is "a language model that reads computed features, with the curve as a second witness".
- **How to use it:** answer with each seed, then `uv run python training/vote.py --out OUT.jsonl PRED1 PRED2 ...`. Seven models take seven times the answering time.

With the plant clause added (`--clauses step,law,plant`) single runs are steadier (75 to 86, mean 79.6, 11 false alarms on average instead of 19), but its vote scores only 81. For one model of this family rather than seven: the plant runs so far lie closer together and raise fewer false alarms, but their mean is inside the tie.

One more vote, fixed in advance at 17:33: the majority of the seed-0 runs of the three designs with the best means (r6-lr-slp, r5-looprange-steplaw, r4-zsize-law) scores 89 (100 and 11). Mixed votes that we tried after the fact reached 93; we do not report them as results.

### Alternate 2: the curve reader, if the claim "it reads the curve" matters most

**Loop-relative + law** (`--view looprange --clauses law`).

| Run | Seed | Attack rows called attack (of 144) | Normal rows called attack (of 948) | Score |
|---|---|---|---|---|
| r6-lr-law | 0 | 83 | 19 | 64 |
| r8-lr-law-s1 | 1 | 89 | 27 | 62 |
| r8-lr-law-s2 | 2 | 81 | 25 | 56 |
| r8-lr-law-s3 | 3 | 82 | 20 | 62 |
| Mean of the single runs | | 83.75 | 22.75 | 61 |
| Majority vote (2 of 3: r6-lr-law, r8-lr-law-s1, r8-lr-law-s2) | | 85 | 20 | 65 |
| Gradient boosting | | 108 | 26 | 82 |

- **Its answers come from the curve.** In four of the six loops every one of the 182 rows shows the model one identical text. The four runs still give 11 to 47 different answers in each of those loops. Only the boiler-pressure loop carries a computed number (the law clause).
- **It passes the shuffled check in three of four runs:** 18, 35, 93 and 25 of 948 shuffled normal rows are called "attack".
- **It scores less:** 64, 62, 56, 62 (mean 61; vote 65). It finds only 1 to 5 of the 42 water-level attack rows.
- The loop-relative view with no clause at all has no computed number anywhere: 60, 50, 55, 54 (mean 54.75; vote 60).

## Where every design stands

Clause designs and their parents (last pass kept, start `sleep` unless said):

| Design | Seeds | Single runs | Mean | Lowest to highest | Majority vote over its seeds |
|---|---|---|---|---|---|
| Gradient boosting (baseline) | | 82 | 82 | | 108 attack rows, 26 normal rows |
| Loop-relative + step, law and plant | 5 | 86, 75, 82, 79, 76 | 79.6 | 75 to 86 | 81 (91 and 10; 5 seeds) |
| Loop-relative + step and law | 8 | 84, 86, 63, 74, 77, 79, 83, 70 | 77 | 63 to 86 | 90 (101 and 11; 7 seeds) |
| Hybrid + law | 4 | 71, 79, 78, 78 | 76.5 | 71 to 79 | 80 (98 and 18; 3 seeds) |
| Hybrid + step and law, start `har` | 3 | 81, 71, 67 | 73 | 67 to 81 | 75 (95 and 20; 3 seeds) |
| Hybrid + law and plant | 4 | 72, 71, 72, 74 | 72.25 | 71 to 74 | 74 (94 and 20; 3 seeds) |
| Loop-relative + step and law, start `har` | 1 | 72 | 72 |  |  |
| Hybrid + step and law | 4 | 71, 70, 73, 71 | 71.25 | 70 to 73 | 73 (97 and 24; 3 seeds) |
| Hybrid + step | 2 | 72, 67 | 69.5 | 67 to 72 |  |
| Loop-relative + size, step and law | 1 | 65 | 65 |  |  |
| Loop-relative + law | 4 | 64, 62, 56, 62 | 61 | 56 to 64 | 65 (85 and 20; 3 seeds) |
| Hybrid, no clause | 4 | 62, 51, 57, 63 | 58.25 | 51 to 63 | 59 (83 and 24; 3 seeds) |
| Loop-relative, no clause | 4 | 60, 50, 55, 54 | 54.75 | 50 to 60 | 60 (81 and 21; 3 seeds) |
| Z-score view | 2 | 46, 61 | 53.5 | 46 to 61 |  |

The earlier designs, all under 50:

| Design | Seeds | Single runs | Mean | Lowest to highest | Majority vote over its seeds |
|---|---|---|---|---|---|
| Range view, mirrored rows 0.5 | 1 | 44 | 44 |  |  |
| Z-score view, pass chosen by loss | 3 | 39, 43, 42 | 41.33 | 39 to 43 | 46 (70 and 24; 3 seeds) |
| Range view | 2 | 37, 32 | 34.5 | 32 to 37 |  |
| Range view, starts from another run, pass chosen by loss | 1 | 31 | 31 |  |  |
| Range view, 6 passes | 1 | 29 | 29 |  |  |
| Range view, pass chosen by loss | 3 | 33, 33, 17 | 27.67 | 17 to 33 | 29 (68 and 39; 3 seeds) |
| Range view, verdict only, pass chosen by loss | 2 | 20, 31 | 25.5 | 20 to 31 |  |
| Range view, start `tsqa`, pass chosen by loss | 1 | 21 | 21 |  |  |
| Range view, verdict only, encoder step 5e-05, pass chosen by loss | 1 | 20 | 20 |  |  |
| Range view, attack share 0.35, pass chosen by loss | 1 | 13 | 13 |  |  |
| Range view, verdict first, pass chosen by loss | 1 | 9 | 9 |  |  |
| Range view, attack share 0.5, pass chosen by loss | 1 | -3 | -3 |  |  |

How to read these tables:

- **Single runs of one design differ by up to 23 points** (63 to 86 for eight seeds of one design). Two seeds can agree by luck: after two seeds that design showed a gap of 2.
- So a design beats another only on its mean over at least four seeds, and only if the means differ by more than about 10 points. By that rule the 6 designs at the top are a tie as single models: Loop-relative + step, law and plant (79.6); Loop-relative + step and law (77); Hybrid + law (76.5); Hybrid + step and law, start `har` (73); Hybrid + law and plant (72.25); Hybrid + step and law (71.25).
- A "vote" is a majority vote over the answers of several seeds of the same design. With an even number of seeds it leaves out the highest-numbered one, so no threshold was chosen on validation.

## Does the model read the curve?

In the clause designs, code computes numbers and writes them into the text line of each series. The usual shuffle check cannot tell what the model then reads, because shuffling the seconds also changes the stated numbers. With the step clause on the loop-relative view, all 15 runs fail that check: 242 to 414 of 948 shuffled normal rows are called "attack". On the hybrid view the step-clause runs call 5 to 368.

So we ran a direct check on a desktop computer (processor only, no graphics card). The saved model answers the 144 attack rows and the 396 neighbour rows of validation again. The text is byte-identical; only the curve changes: blank (all zeros), or its seconds shuffled.

**Loop-relative + step and law** (run r5-looprange-steplaw, score 84). The desktop computer's own answers equal those of the rented machine on 539 of the 540 rows.

| Loop | Attack rows | Called attack: normal curves | blank curves | shuffled curves |
|---|---|---|---|---|
| P1-FC | 18 | 13 | 12 | 8 |
| P1-LC | 42 | 36 | 33 | 9 |
| P1-PC | 18 | 16 | 16 | 13 |
| P1-TC | 12 | 0 | 0 | 0 |
| P2-SC | 12 | 12 | 5 | 0 |
| P3-LC | 42 | 18 | 18 | 18 |
| **All attack rows** | 144 | 95 | 84 | 48 |
| **Neighbour rows called attack** | 396 | 10 | 15 | 4 |

**Hybrid + law** (run r5-zsize-law-s1, score 79).

| Loop | Attack rows | Called attack: normal curves | blank curves | shuffled curves |
|---|---|---|---|---|
| P1-FC | 18 | 18 | 5 | 3 |
| P1-LC | 42 | 39 | 39 | 24 |
| P1-PC | 18 | 14 | 15 | 12 |
| P1-TC | 12 | 3 | 0 | 0 |
| P2-SC | 12 | 12 | 12 | 11 |
| P3-LC | 42 | 13 | 1 | 0 |
| **All attack rows** | 144 | 99 | 72 | 50 |
| **Neighbour rows called attack** | 396 | 15 | 22 | 2 |

What this shows:

- **Both models can make most of their attack calls from the text alone.** With every curve blank, the loop-relative + step and law model makes 84 attack calls against 95 with the curve (15 lost, 4 new), and the hybrid + law model 72 against 99 (30 lost, 3 new).
- **The hybrid + law model depends on the curve about twice as much.** On these 540 rows, blank curves cost it 34 points (from 84 to 50), against 16 for the other design (from 85 to 69). It reads two loops from the curve: P1-FC (18 attack calls fall to 5) and the water level (13 fall to 1). In P1-LC, P1-PC and P2-SC its text numbers are enough. With the curves shuffled it makes 50 attack calls against 99 and calls 2 neighbour rows "attack".
- **Loop-relative + step and law reads the water level from the step number alone:** 18 attack calls under every variant. The curve gives it 7 of its 12 P2-SC calls and holds back 7 false alarms in the boiler-pressure loop.
- **Both models do read the curve.** With the curves shuffled and the same text, the loop-relative + step and law model makes only 48 attack calls against 95: when the curve does not look like an attack, it overrules the numbers.
- **A third piece of evidence, without changed inputs:** the four loop-relative runs with the law clause only show the model one identical text for all 182 rows of four loops. They still give 11 to 47 different answers in each of those loops. In the hybrid + law runs, 69 to 74 of the 165 texts that several rows share get more than one answer.
- **A limit:** this is one model per design (r5-looprange-steplaw and r5-zsize-law-s1), on inputs neither saw in training. It is a sign, not a measurement of the designs. Single models of one design differ by up to 23 points in score.

## What held up

- **Text clauses beat no clauses.** The 36 runs with a clause on the hybrid or loop-relative view score 56 to 86 and average 73.1. The 8 runs of these two views without a clause score 50 to 63 and average 56.5. Only the weakest clause design, loop-relative + law (56 to 64), overlaps with them.
- **The loop-relative view beats the plain range view.** With the last pass kept it scores 60 and 55 against 37 and 32 for the same seeds. Its four seeds score 50 to 60; all five range-view runs score 17 to 37.
- **Keep the last pass, not the lowest-loss pass.** In all 8 runs that scored every pass, the last pass was the best and the loss-chosen pass was never the kept one. Since round 4 every run keeps the last pass.
- **A vote over seeds reduces the seed noise, for one design clearly.** Seven seeds of loop-relative + step and law score 90 together; alone they score 63 to 86. A vote of three of them still ranges from 74 to 90, depending on which three. Votes of the other designs gain 1 to 5 points over their mean.
- **Most of the remaining misses were out of reach for every design of tonight.** Of the 44 attack rows a three-run vote missed in round 4, 36 fall in two groups. 24 are water-level rows without a valve jump: none of the 116 statistics tried in the error analyses separates them in training, though gradient boosting still finds 9 of them. 12 are P1-TC rows: their attacks are version 22.04 in validation and version 23.05 in training. No run finds more than 18 of the 42 water-level attack rows; gradient boosting finds 27.

## What failed, and why

Tested with several runs, or with an effect far outside the noise:

- **More attack rows in training** (35% and 50% instead of 20%), one run each. At 50% the score is -3, 20 points under any comparable run (17 to 37). At 35% it is 13, by itself inside the noise. Both call far more normal rows "attack" (68 and 85 against 32 to 43) for a few more attack rows (81 and 82). Dropped.
- **Choosing the pass by validation loss.** Costs points in every run that scored all passes (for example 39 against 61 in one run). The loss rises at the last pass while the score still improves. Replaced by "keep the last pass".
- **The `har` start instead of `sleep`.** Three seeds score 81, 71, 67 (mean 73) against 70 to 73 (mean 71.25). The 81 was a lucky seed. No gain.
- **The plant clause on the loop-relative view.** Five seeds average 79.6 against 77.0 for eight seeds without it: 2.6 points, a third of one standard deviation between seeds and far inside the 23-point spread. In P1-FC it does what it was built for (3.4 fewer false alarms on average). It also removes 3.6 false alarms in P1-PC, where it writes nothing, so that part is unexplained. In P1-LC it costs 7 attack rows. A vote over its seeds gains little (81 against a mean of 79.6).
- **The plant clause on the hybrid view.** Four seeds score 72, 71, 72, 74 (mean 72.25) against 71, 79, 78, 78 (mean 76.5) without it: seed by seed +1, -8, -6, -4. It removes 2.5 false alarms in P1-FC, where it was aimed, and loses 4.75 water-level attack rows, where it writes nothing. We have no explanation for that side effect.
- **The plain range view, the design the night started from:** 17 to 37 over five runs.

Tested with one run only, so unproven either way (single runs differ by up to 23 points):

- **Six passes instead of four:** 29 against 37 and 32.
- **Mirrored copies of training rows:** 44 against 37 and 32, at 1.5 times the training rows (about 1.4 times the training time).
- **Start `tsqa`:** 21 against 17 to 37.
- **Verdict first, then the sentence:** 9. **Verdict only:** 20 and 31; one of these runs let the encoder response fall to 0.09. A smaller encoder step (5e-5) kept the encoder alive (0.54) and scored 20.
- **Verdict only first, then the full answer from that model:** 31, a cautious model (38 attack rows, 7 normal rows).
- **The size number added to the loop-relative text:** 65 against 63 to 86 without it. Not needed; not shown to hurt.
- **The `har` start on the loop-relative view:** 72 against 63 to 86.

Out of reach for every design of tonight:

- **24 water-level attack rows without a valve jump.** Their stated numbers sit at or below those of normal rows, and none of the 116 statistics tried in the error analyses separates them in training. Gradient boosting still finds 9 of them, so some signal exists that tonight's designs do not use.
- **The 12 P1-TC attack rows.** No vote and no leading design finds them. A few single runs call 1 to 3 of them, probably by chance. Training holds P1-TC attacks of version 23.05 only; validation holds version 22.04 only. Gradient boosting finds none.

## Limits to state in the write-up

- **Validation is small, and every choice of the night was made on it.** It holds 15 attacks. On it we scored 64 runs, the passes of 8 runs, many votes, three error analyses and the curve checks. The three clauses and the loop-relative view were designed from what the models missed there. The plant clause was even chosen by comparing about 15 statistics against the validation false alarms. Their fits use training rows only; the choices do not. So every score above is optimistic. The test rows are the honest check.
- **The test rows are not pristine.** Several models and both baselines were scored on them before tonight, in the Kaggle runs, and the redesign that tonight builds on was chosen after seeing those results. Nothing in the night touched them.
- **There is a stronger gradient boosting baseline.** `runs/baselines/repaired_validation/predictions_gradient_boosting_with_neighbours.jsonl` scores 87 on validation (108 and 21). The 82 is the variant fixed as the baseline before the night. Only the seven-seed vote (90) is above 87.
- **The same seed does not reproduce the same run.** Seeds are repeats, not fixed points.
- **The clause designs lean on computed numbers.** See the curve check. A gradient boosting classifier on exactly the numbers in the hybrid view's text (size, step, law) scores 62 (99 and 37). The hybrid runs with those clauses average 71; single clause runs range from 56 to 86.
- **One-run results are not evidence.** Wherever the tables show one seed, read the number as "somewhere within about 23 points of this".
- **The gradient boosting figure (82) was computed before the night.** We recounted it from its answer file that night (108 and 26); we did not retrain it.

## The rounds

| Round | Started | Ended | Code on the machine | Notes |
|---|---|---|---|---|
| 1 | 11:17 | 12:08 | c8b01fe | 8 of 8 finished, 51 minutes; MPS on; card at 100% |
| 2 | 12:20 | 13:29 | 32dcaa8 | 8 of 8 finished, 69 minutes (six by 13:12, the two pass-choice runs at 13:29) |
| 3 | 13:34 | 14:50 | 77fda42 | 8 of 8 finished, 76 minutes (six runs scored every pass) |
| 4 | 14:54 | 16:03 | aa94b19 | 8 of 8 finished, 69 minutes (seven by 15:57) |
| 5 | 16:13 | 17:21 | 1f2a677 | 8 of 8 finished, 68 minutes |
| 6 | 17:27 | 18:35 | 3714564 | 8 of 8 finished, 68 minutes |
| 7 | 18:36 | 19:43 | 3714564 | 8 of 8 finished, 67 minutes |
| 8 | 19:45 | 20:48 | 3714564 | 8 of 8 finished, 63 minutes |

Round 9 was not run: it could not have ended before the 10-hour limit at 21:16. Round 1 took 51 minutes. The later rounds took 63 to 76: from round 2 on, runs that keep a pass other than the loss-chosen one answer the 1,092 validation rows a second time with the kept pass (about 7 minutes), round 3 scored every pass, and the clause texts made each pass about a minute longer.

## Full table of the 64 runs

"Scrambled: a and b" = attack rows and normal rows called attack when the seconds of every series are shuffled. Encoder response = how differently the curve encoder answers six fixed test curves; under 0.01 means dead. No run was dead, none stopped, and no plain answer was invalid.

| Name | Round | Parent | The one change | Passes | Kept pass | Encoder start / lowest / kept | Attacks of 15 | Attack rows of 144 | Normal rows of 948 | Score | Differ within a loop | State |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1-base | 1 | none | base | 4 | 3 | 0.38 / 0.50 / 0.57 | 11 | 71 | 38 | 33 | yes | done |
| r1-tsqa | 1 | r1-base | --start tsqa | 4 | 4 | 0.28 / 0.46 / 0.54 | 6 | 34 | 13 | 21 | yes | done |
| r1-zscore | 1 | r1-base | --view zscore | 4 | 3 | 0.38 / 0.50 / 0.54 | 12 | 78 | 39 | 39 | n/a (z-score view; scrambled: 4 and 1) | done |
| r1-first | 1 | r1-base | --order label_first | 4 | 3 | 0.38 / 0.58 / 0.62 | 10 | 61 | 52 | 9 | yes | done |
| r1-only | 1 | r1-base | --order label_only | 4 | 1 | 0.38 / 0.09 / 0.09 | 5 | 27 | 7 | 20 | yes (at most 2 answers) | done |
| r1-base-s1 | 1 | r1-base | --seed 1 | 4 | 4 | 0.38 / 0.47 / 0.54 | 11 | 75 | 42 | 33 | yes | done |
| r1-base-s2 | 1 | r1-base | --seed 2 | 4 | 2 | 0.38 / 0.52 / 0.54 | 8 | 49 | 32 | 17 | yes | done |
| r1-only-s1 | 1 | r1-only | --seed 1 | 4 | 3 | 0.38 / 0.27 / 0.56 | 6 | 43 | 12 | 31 | yes (at most 2 answers) | done |
| r2-base-pv | 2 | r1-base | --choose-pass validation | 4 | 4 (loss pass 3) | 0.38 / 0.54 / 0.56 | 12 | 73 | 36 | 37 | yes | done; pass scores -10, 16, 34, 37 |
| r2-base-s2-pv | 2 | r1-base-s2 | --choose-pass validation | 4 | 4 (loss pass 3) | 0.38 / 0.56 / 0.64 | 12 | 75 | 43 | 32 | yes | done; pass scores 9, 10, 20, 32 |
| r2-zscore-s1 | 2 | r1-zscore | --seed 1 | 4 | 2 | 0.38 / 0.46 / 0.50 | 10 | 54 | 11 | 43 | n/a (z-score view; scrambled: 0 and 0) | done |
| r2-zscore-s2 | 2 | r1-zscore | --seed 2 | 4 | 3 | 0.38 / 0.56 / 0.56 | 11 | 72 | 30 | 42 | n/a (z-score view; scrambled: 1 and 0) | done |
| r2-share35 | 2 | r1-base | --attack-share 0.35 | 4 | 2 | 0.38 / 0.51 / 0.54 | 12 | 81 | 68 | 13 | yes | done |
| r2-share50 | 2 | r1-base | --attack-share 0.5 | 4 | 2 | 0.38 / 0.53 / 0.53 | 12 | 82 | 85 | -3 | yes | done |
| r2-only-lr | 2 | r1-only | --lr-encoder 5e-5 | 4 | 1 | 0.38 / 0.54 / 0.54 | 8 | 51 | 31 | 20 | yes (at most 2 answers) | done |
| r2-from-only | 2 | r1-base | --start run:r1-only-s1 (verdict only first, then the full answer) | 4 | 2 | 0.56 / 0.72 / 0.73 | 6 | 38 | 7 | 31 | yes | done |
| r3-zscore-pv | 3 | r1-zscore | --choose-pass validation | 4 | 4 (loss pass 3) | 0.38 / 0.61 / 0.65 | 10 | 76 | 30 | 46 | n/a (numbers in the text; scrambled: 4 and 0) | done; pass scores 39, 35, 39, 46 |
| r3-zscore-s1-pv | 3 | r2-zscore-s1 | --choose-pass validation | 4 | 4 (loss pass 3) | 0.38 / 0.49 / 0.53 | 11 | 85 | 24 | 61 | n/a (numbers in the text; scrambled: 2 and 8) | done; pass scores 12, 36, 39, 61 |
| r3-looprange-pv | 3 | r2-base-pv | --view looprange | 4 | 4 (loss pass 3) | 0.38 / 0.49 / 0.67 | 12 | 81 | 21 | 60 | yes | done; pass scores 42, 41, 53, 60 |
| r3-looprange-s2-pv | 3 | r2-base-s2-pv | --view looprange | 4 | 4 (loss pass 3) | 0.38 / 0.46 / 0.60 | 12 | 82 | 27 | 55 | yes | done; pass scores 29, 30, 43, 55 |
| r3-zsize-pv | 3 | r3-zscore-pv | --view zsize | 4 | 4 (loss pass 3) | 0.38 / 0.47 / 0.54 | 10 | 82 | 20 | 62 | n/a (numbers in the text) | done; pass scores 44, 52, 60, 62 |
| r3-zsize-s1-pv | 3 | r3-zscore-s1-pv | --view zsize | 4 | 4 (loss pass 2) | 0.38 / 0.47 / 0.60 | 11 | 85 | 34 | 51 | n/a (numbers in the text) | done; pass scores 24, 49, 49, 51 |
| r3-mirror-last | 3 | r2-base-pv | --mirror-share 0.5 (keeps the last pass, as its parent did) | 4 | 4 (loss pass 3) | 0.38 / 0.57 / 0.61 | 12 | 81 | 37 | 44 | yes | done |
| r3-e6-last | 3 | r2-base-pv | --epochs 6 (keeps the last pass) | 6 | 6 (loss pass 2) | 0.38 / 0.47 / 0.58 | 12 | 69 | 40 | 29 | yes | done |
| r4-looprange-s1 | 4 | r3-looprange-pv | --seed 1 (keeps the last pass) | 4 | 4 (loss pass 3) | 0.38 / 0.64 / 0.71 | 12 | 77 | 27 | 50 | yes | done |
| r4-looprange-s3 | 4 | r3-looprange-pv | --seed 3 (keeps the last pass) | 4 | 4 (loss pass 2) | 0.38 / 0.55 / 0.57 | 12 | 77 | 23 | 54 | yes | done |
| r4-zsize-s2 | 4 | r3-zsize-pv | --seed 2 (keeps the last pass) | 4 | 4 (loss pass 3) | 0.38 / 0.47 / 0.60 | 11 | 87 | 30 | 57 | n/a (numbers in the text) | done |
| r4-zsize-s3 | 4 | r3-zsize-pv | --seed 3 (keeps the last pass) | 4 | 4 (loss pass 3) | 0.38 / 0.46 / 0.53 | 11 | 91 | 28 | 63 | n/a (numbers in the text) | done |
| r4-zsize-step | 4 | r3-zsize-pv | --clauses step | 4 | 4 (loss pass 3) | 0.38 / 0.50 / 0.53 | 10 | 96 | 24 | 72 | n/a (numbers in the text; scrambled: 78 and 368) | done |
| r4-zsize-law | 4 | r3-zsize-pv | --clauses law | 4 | 4 (loss pass 3) | 0.38 / 0.47 / 0.61 | 12 | 97 | 26 | 71 | n/a (numbers in the text; scrambled: 29 and 27) | done |
| r4-zsize-steplaw | 4 | r4-zsize-step | --clauses step,law (adds the law clause) | 4 | 4 (loss pass 3) | 0.38 / 0.48 / 0.56 | 12 | 95 | 24 | 71 | n/a (numbers in the text; scrambled: 9 and 49) | done |
| r4-zsize-steplaw-s1 | 4 | r4-zsize-steplaw | --seed 1 | 4 | 4 (loss pass 2) | 0.38 / 0.46 / 0.53 | 12 | 95 | 25 | 70 | n/a (numbers in the text; scrambled: 13 and 49) | done |
| r5-zsize-steplaw-s2 | 5 | r4-zsize-steplaw | --seed 2 | 4 | 4 (loss pass 2) | 0.38 / 0.51 / 0.57 | 12 | 97 | 24 | 73 | n/a (numbers in the text; scrambled: 14 and 14) | done |
| r5-zsize-steplaw-s3 | 5 | r4-zsize-steplaw | --seed 3 | 4 | 4 (loss pass 1) | 0.38 / 0.48 / 0.48 | 11 | 98 | 27 | 71 | n/a (numbers in the text; scrambled: 19 and 42) | done |
| r5-looprange-steplaw | 5 | r4-zsize-steplaw | --view looprange (clauses step,law, seed 0) | 4 | 4 (loss pass 3) | 0.38 / 0.51 / 0.67 | 11 | 95 | 11 | 84 | n/a (numbers in the text; fewest distinct answers in a loop: 11; scrambled: 47 and 382) | done |
| r5-looprange-steplaw-s1 | 5 | r5-looprange-steplaw | --seed 1 | 4 | 4 (loss pass 2) | 0.38 / 0.65 / 0.70 | 11 | 98 | 12 | 86 | n/a (numbers in the text; fewest distinct answers in a loop: 14; scrambled: 61 and 414) | done |
| r5-looprange-ssl | 5 | r5-looprange-steplaw | --clauses size,step,law (adds the size clause) | 4 | 4 (loss pass 3) | 0.38 / 0.47 / 0.62 | 11 | 91 | 26 | 65 | n/a (numbers in the text; scrambled: 85 and 372) | done |
| r5-zsize-steplaw-har | 5 | r4-zsize-steplaw | --start har | 4 | 4 (loss pass 3) | 0.37 / 0.50 / 0.55 | 12 | 97 | 16 | 81 | n/a (numbers in the text; scrambled: 12 and 112) | done |
| r5-zsize-step-s1 | 5 | r4-zsize-step | --seed 1 | 4 | 4 (loss pass 3) | 0.38 / 0.49 / 0.56 | 11 | 88 | 21 | 67 | n/a (numbers in the text; scrambled: 2 and 5) | done |
| r5-zsize-law-s1 | 5 | r4-zsize-law | --seed 1 | 4 | 4 (loss pass 3) | 0.38 / 0.47 / 0.53 | 12 | 99 | 20 | 79 | n/a (numbers in the text; scrambled: 43 and 30) | done |
| r6-lr-steplaw-s2 | 6 | r5-looprange-steplaw | --seed 2 | 4 | 4 (loss pass 2) | 0.38 / 0.52 / 0.59 | 10 | 90 | 27 | 63 | n/a (numbers in the text; fewest distinct answers in a loop: 12; scrambled: 59 and 403) | done |
| r6-lr-steplaw-s3 | 6 | r5-looprange-steplaw | --seed 3 | 4 | 4 (loss pass 3) | 0.38 / 0.65 / 0.70 | 11 | 97 | 23 | 74 | n/a (numbers in the text; fewest: 14; scrambled: 77 and 403) | done |
| r6-zsize-steplaw-har-s1 | 6 | r5-zsize-steplaw-har | --seed 1 | 4 | 4 (loss pass 3) | 0.37 / 0.59 / 0.61 | 11 | 97 | 26 | 71 | n/a (numbers in the text; scrambled: 14 and 80) | done |
| r6-zsize-steplaw-har-s2 | 6 | r5-zsize-steplaw-har | --seed 2 | 4 | 4 (loss pass 2) | 0.37 / 0.55 / 0.61 | 10 | 88 | 21 | 67 | n/a (numbers in the text; scrambled: 22 and 155) | done |
| r6-lr-slp | 6 | r5-looprange-steplaw | --clauses step,law,plant (adds the plant clause) | 4 | 4 (loss pass 3) | 0.38 / 0.67 / 0.70 | 11 | 94 | 8 | 86 | n/a (numbers in the text; fewest: 12; scrambled: 62 and 282) | done |
| r6-lr-slp-s1 | 6 | r6-lr-slp | --seed 1 | 4 | 4 (loss pass 3) | 0.38 / 0.49 / 0.61 | 10 | 87 | 12 | 75 | n/a (numbers in the text; fewest: 13; scrambled: 100 and 346) | done |
| r6-lr-steplaw-har | 6 | r5-looprange-steplaw | --start har | 4 | 4 (loss pass 3) | 0.37 / 0.49 / 0.65 | 11 | 94 | 22 | 72 | n/a (numbers in the text; fewest: 13; scrambled: 94 and 367) | done |
| r6-lr-law | 6 | r5-looprange-steplaw | --clauses law (drops the step clause) | 4 | 4 (loss pass 3) | 0.38 / 0.49 / 0.62 | 12 | 83 | 19 | 64 | yes: in the four loops whose rows all share one text it gives 13 to 37 distinct answers (scrambled: 26 and 18) | done |
| r7-lr-steplaw-s4 | 7 | r5-looprange-steplaw | --seed 4 | 4 | 4 (loss pass 3) | 0.38 / 0.63 / 0.71 | 11 | 99 | 22 | 77 | n/a (numbers in the text; fewest distinct answers in a loop: 10; scrambled: 59 and 362) | done |
| r7-lr-steplaw-s5 | 7 | r5-looprange-steplaw | --seed 5 | 4 | 4 (loss pass 3) | 0.38 / 0.59 / 0.67 | 10 | 97 | 18 | 79 | n/a (numbers in the text; fewest: 13; scrambled: 36 and 313) | done |
| r7-lr-steplaw-s6 | 7 | r5-looprange-steplaw | --seed 6 | 4 | 4 (loss pass 2) | 0.38 / 0.55 / 0.63 | 11 | 100 | 17 | 83 | n/a (numbers in the text; fewest: 11; scrambled: 30 and 268) | done |
| r7-lr-steplaw-s7 | 7 | r5-looprange-steplaw | --seed 7 | 4 | 4 (loss pass 3) | 0.38 / 0.65 / 0.75 | 10 | 93 | 23 | 70 | n/a (numbers in the text; fewest: 14; scrambled: 41 and 333) | done |
| r7-lr-slp-s2 | 7 | r6-lr-slp | --seed 2 | 4 | 4 (loss pass 2) | 0.38 / 0.45 / 0.66 | 10 | 92 | 10 | 82 | n/a (numbers in the text; fewest: 14; scrambled: 22 and 242) | done |
| r7-lr-slp-s3 | 7 | r6-lr-slp | --seed 3 | 4 | 4 (loss pass 2) | 0.38 / 0.60 / 0.63 | 10 | 91 | 12 | 79 | n/a (numbers in the text; fewest: 12; scrambled: 95 and 343) | done |
| r7-lr-slp-s4 | 7 | r6-lr-slp | --seed 4 | 4 | 4 (loss pass 3) | 0.38 / 0.49 / 0.61 | 11 | 90 | 14 | 76 | n/a (numbers in the text; fewest: 12; scrambled: 71 and 276) | done |
| r7-zsize-law-s2 | 7 | r4-zsize-law | --seed 2 | 4 | 4 (loss pass 3) | 0.38 / 0.47 / 0.53 | 12 | 95 | 17 | 78 | yes: 70 of the 165 texts shared by several rows get more than one answer (scrambled: 34 and 23) | done |
| r8-zsize-lawplant | 8 | r4-zsize-law | --clauses law,plant (adds the plant clause) | 4 | 4 (loss pass 3) | 0.38 / 0.48 / 0.52 | 12 | 91 | 19 | 72 | yes: 52 of the 147 texts shared by several rows get more than one answer (scrambled: 41 and 43) | done |
| r8-zsize-lawplant-s1 | 8 | r8-zsize-lawplant | --seed 1 | 4 | 4 (loss pass 3) | 0.38 / 0.46 / 0.47 | 12 | 94 | 23 | 71 | yes: 50 of 147 shared texts (scrambled: 32 and 28) | done |
| r8-zsize-lawplant-s2 | 8 | r8-zsize-lawplant | --seed 2 | 4 | 4 (loss pass 3) | 0.38 / 0.46 / 0.52 | 12 | 92 | 20 | 72 | yes: 56 of 147 shared texts (scrambled: 30 and 12) | done |
| r8-zsize-lawplant-s3 | 8 | r8-zsize-lawplant | --seed 3 | 4 | 4 (loss pass 3) | 0.38 / 0.38 / 0.47 | 12 | 94 | 20 | 74 | yes: 54 of 147 shared texts (scrambled: 18 and 13) | done |
| r8-zsize-law-s3 | 8 | r4-zsize-law | --seed 3 | 4 | 4 (loss pass 2) | 0.38 / 0.45 / 0.47 | 12 | 96 | 18 | 78 | yes: 74 of 165 shared texts (scrambled: 33 and 17) | done |
| r8-lr-law-s1 | 8 | r6-lr-law | --seed 1 | 4 | 4 (loss pass 4) | 0.38 / 0.50 / 0.57 | 12 | 89 | 27 | 62 | yes: 11 to 47 distinct answers in the four loops whose rows share one text (scrambled: 53 and 35) | done |
| r8-lr-law-s2 | 8 | r6-lr-law | --seed 2 | 4 | 4 (loss pass 3) | 0.38 / 0.55 / 0.61 | 12 | 81 | 25 | 56 | yes: 11 to 44 distinct answers in those loops (scrambled: 23 and 93) | done |
| r8-lr-law-s3 | 8 | r6-lr-law | --seed 3 | 4 | 4 (loss pass 3) | 0.38 / 0.63 / 0.67 | 10 | 82 | 20 | 62 | yes: 11 to 42 distinct answers in those loops (scrambled: 15 and 25) | done |

## How the night was run

The analysis of each round and the code changes were made with AI coding agents (Claude) under the owner's direction. Every table here was recounted from the raw answer files by a separate check. No change of the night touched the data or the scoring script (`training/score.py`).

## Code changes (commits on the branch `feature/evaluation`)

Each change is a new option whose default keeps the earlier behaviour, with unit tests (254 pass, 1 skipped) and, for options used on the machine, a plumbing test and a learning test there. The times are when the change was logged in the working log of the night (not published), about a minute after the commit.

| Time | Commit | What | Tests on the machine | Accepted |
|---|---|---|---|---|
| 11:16 | c8b01fe | state before round 1 | plumbing and learning tests passed earlier that day on this code path | yes |
| 11:31 | 0dba878 | `--lr-encoder` option, default unchanged | plumbing clean; learning 11 of 11 and 50 of 50 | yes: both machine tests passed at 12:19 |
| 11:35 | 01844e1 | `--attack-share` option, default unchanged | plumbing clean; learning 11 of 11 and 48 of 50 | yes: both machine tests passed at 12:19 |
| 11:39 | 112b3c5 | `--start run:NAME`, default unchanged | started from r1-only: plumbing clean, adapters loaded; learning 11 of 11 and 50 of 50 | yes: both machine tests passed at 12:19 |
| 11:45 | 32dcaa8 | `--choose-pass`, default unchanged | plumbing with `validation`: three passes scored, kept pass 3; learning with `last`: kept pass 24 (loss pass 19), 11 of 11 and 50 of 50 | yes: both machine tests passed at 12:19 |
| 12:40 | 3826abe | `--mirror-share`, default unchanged | plumbing clean (75 training rows from 50); learning 11 of 11 and 50 of 50 | yes: both machine tests passed at 13:33 |
| 12:46 | 99ce67d | `--view looprange`; zscore and range unchanged | plumbing clean (87 units from 1,552 quiet rows); learning 11 of 11 and 50 of 50 | yes: both machine tests passed at 13:33 |
| 13:20 | 77fda42 | `--view zsize`; the other views unchanged | plumbing clean (the size sentence is in the prompt); learning 11 of 11 and 50 of 50 | yes: both machine tests passed at 13:33 |
| 14:13 | 2d57244 | `--clauses step` (zsize only), default unchanged | plumbing clean (a step clause on all 198 series texts); learning 11 of 11 and 50 of 50 | yes: both machine tests passed at 14:53 |
| 14:19 | f7b3841 | `--clauses law` (zsize only), default unchanged | plumbing clean (a law clause on the 8 boiler-pressure command lines); learning 11 of 11 and 50 of 50; both clauses together: 11 of 11 and 50 of 50 | yes: both machine tests passed at 14:53 |
| 14:23 | aa94b19 | law clause gated by fit quality | covered by the law tests above | yes: both machine tests passed at 14:53 |
| 15:00 | 95db437 | clauses with `--view looprange`; clause `size`; existing texts unchanged | with `--view looprange --clauses size,step,law`: plumbing clean (all three clause logs present, exit 0); learning 11 of 11 and 50 of 50 | yes: both machine tests passed at 16:13 |
| 15:08 | 1f2a677 | `training/vote.py`, a new offline tool | not applicable: it does not run on the machine | yes |
| 16:36 | 3714564 | `--clauses plant`, default unchanged | with `--view looprange --clauses step,law,plant`: plumbing clean (plant models for 2 of 4 loops, the phrase on 15 of 50 rows, exit 0); learning 11 of 11 and 50 of 50 | yes: both machine tests passed at 17:26 |

## Our errors that night

- **We called a leader on two seeds.** After round 5 we reported loop-relative + step and law as clearly ahead (84 and 86, "gap 2"). Seeds 2 and 3 scored 63 and 74. We corrected the working log, and two other single-run readings with it (the size clause, six passes).
- **We stalled the curve check for 55 minutes** by starting three copies at once on the desktop computer without testing that they run side by side. Restarted one at a time.
- **A quoting slip killed the first start of two machine tests** before round 5 (the run name swallowed its options). Found after 5 minutes; nothing trained, nothing else affected.
- **A wait command hung** because its process check matched itself. Stopped; no harm.
- **We logged an early look at round 8 as its 10-minute check** after misreading the clock. Corrected, and the real check was done at 19:56.
- **We planned on 52-minute rounds; from round 2 on they took 63 to 76 minutes.** So eight rounds fit into the 10 hours, not nine.
- **We read too much into a vote after round 4.** We reported that voting adds 8 to 9 points over the best single run. That 80 came from mixing three different clause designs; three seeds of one design scored 73. Corrected at 17:32.
- **We bent the "two more seeds first" rule in rounds 4, 5 and 6.** Each of those rounds tested additions to a design in the same round as its extra seeds, to save clock time. We logged it each time and counted the additions only once the seeds were in. For round 5's leader the seeds then did not confirm the lead.
- **From round 5 on we started each round on our own count**, before the separate round analysis was back. Every round analysis then matched that count.
- **Before round 1, the command to set old test folders aside failed silently.** We moved them a minute later, while round 1 was already running. They are separate folders; no run was touched.
- **A wording slip in the specification of the plant clause:** it said "strictly above" for a threshold table that counts "at or above". Both reviews of the code caught it; the code has no threshold, so nothing changed.

## State at the end of the night

- **The search was over.** Eight rounds, 64 runs, none lost.
- **Card time:** the whole search of 64 runs cost about 57 dollars of card time, at 5.40 US dollars an hour.
- **Where the runs are:** every run's answers, logs and scores are in release v1.0 of this repository; `answers.tar.gz`, unpacked at the repository root, puts them in `runs/nebius/2026-11-21/night/<run>/kept/`. Each run also kept a saved model (64 models, 55 to 164 MB each, about 5 GB in all). Twelve of them, the runs of the two final designs, are in the same release (see `MODELS.md`); the other 52 are not published. The round analyses and the working log of the night are not published. The numbers of this report are recounted by `runs/nebius/summary_source/night_numbers.py`; its output is `runs/nebius/summary_source/night_numbers.json`.
- **Code:** 14 commits on `feature/evaluation`, from `c8b01fe` to `3714564`.
- **Test rows:** at the time of writing, the test rows had not been answered, scored or opened.

## What came after

The test rows were answered once, with both final designs fixed in advance: hybrid + law (the pick) as the vote of seeds 0 to 2, and loop-relative + step and law (alternate 1) as the vote of seeds 0 to 6 (r5-looprange-steplaw to r7-lr-steplaw-s6). Each run answered with `python3 training/cloud/run_variant.py NAME --answer-only test test_shuffled`, and `training/vote.py` combined exactly those files. What was fixed before the run is in `runs/final/DECLARED.md`; the result is in `runs/final/FACTS.md` and the main README.
