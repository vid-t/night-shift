# Final facts (written 21 November 2026, 23:45, after the one test run)

Source files, all in runs/final/: final_results.json (every count), FINAL-RESULTS.md (the tables), results_table_test.md (attacks found and start times), final_score.py (how it was counted). Background: runs/nebius/NIGHT-SUMMARY.md (the night on the check set) and runs/final/DECLARED.md (what was fixed before the test run).

## The test set
1,536 windows: 82 attack windows from 56 attacks, 1,454 normal windows (254 neighbour, 1,200 quiet). Score = attack windows called attack minus normal windows called attack. Perfect is 82. Both designs, which models vote, and the scoring were written down before the run. It was run once.

## Results on the whole test set
| System | Attack windows caught (of 82) | False alarms (of 1,454) | Score | Attacks found (of 56) |
|---|---|---|---|---|
| Gradient boosting (classic statistical method) | 71 | 17 | 54 | 52 |
| Gradient boosting that also reads the other loops | 68 | 14 | 54 | 52 |
| Our simpler design (hybrid view + law hint), three models voting | 66 | 15 | 51 | 50 |
| Our hint-heavy design (loop-relative view + step and law hints), seven models voting | 65 | 15 | 50 | 51 |
| Our simpler design, one model: four repeats | 61 to 68 | 13 to 23 | 42, 52, 52, 43 (mean 47.25) | 48 to 52 |
| Our hint-heavy design, one model: eight repeats | 59 to 64 | 12 to 21 | 50, 43, 49, 39, 47, 46, 44, 42 (mean 45) | 45 to 51 |
| Range alarm (the floor) | 38 | 173 | -135 | 31 |

Reading: on fresh data gradient boosting is slightly ahead of our best (54 against 51 and 50). One model of ours averages 45 to 47. The lead our seven-model vote had on the check set (90 against 82, of 144) did not hold on the test set.

Check set against test set, as a share of the perfect score: seven-model vote 90 of 144 (63%) then 50 of 82 (61%); gradient boosting 82 of 144 (57%) then 54 of 82 (66%).

## What our model gives that gradient boosting does not
A start second and a sentence. The three-model vote names the start within 30 seconds for 30 of the 56 attacks (median error 22 seconds); the seven-model vote for 30 (median error 26 seconds); one model of the simpler design (seed 1) for 39 (median error 19 seconds); one model of the hint-heavy design (seed 0) for 28 (median error 26 seconds). Only these two single models were timed. Always answering "second 108" would be within 30 seconds for 26 of the 56. Gradient boosting gives no start time.

## The huge general model: Kimi K3.1 Pro, asked directly
Asked through an API on 811 of the test windows: all 82 attack windows plus half of each loop's normal windows (129 neighbour, 600 quiet). Two ways: cold (the task and the raw numbers), and with our computed hints plus six solved examples. Thinking off, short answers. The wording was tried on 20 check-set windows per way and frozen before any test result was read.

On those same 811 windows:
| System | Attack windows caught (of 82) | False alarms (of 729 normal) | Score on these windows |
|---|---|---|---|
| Gradient boosting | 71 | 8 | 63 |
| Our simpler design, three models voting | 66 | 7 | 59 |
| Our hint-heavy design, seven models voting | 65 | 8 | 57 |
| Kimi K3.1 Pro with our hints and six solved examples | 78 | 259 | -181 |
| Kimi K3.1 Pro, cold | 78 | 362 | -284 |

Reading: Kimi calls almost every attack window attack because it calls a third to a half of all normal windows attack too (362 of 729 cold, 259 with our hints and examples). It does not know what is normal for this plant. Our hints and examples help it (362 false alarms down to 259) but not enough.

Thinking: on 58 of those windows (28 attack, 30 normal) Kimi cold with thinking off caught 27 with 20 false alarms; with thinking on (low) 25 with 15 false alarms, at four times the price. On the same 58 windows gradient boosting caught 25 with 0 false alarms, our votes 20 and 21 with 0.

24 of Kimi's 1,622 answers were cut off before a verdict; asked again with more room, 20 gave one. Counts above use the second answer for those; 4 cold answers have no verdict and count as "not attack".

## Cost and speed
- Kimi K3.1 Pro: 1.54 US dollars per 1,000 questions cold, 1.64 with hints and examples, 6.17 with thinking on; 1.4 seconds an answer (6.1 with thinking). Total spent on the comparison: 3.08 US dollars.
- Our model on the rented card (5.40 dollars an hour): 12 models answered 1,536 windows each in 34 minutes, so about 0.17 dollars per 1,000 windows for one model, 0.50 for three voting, 1.16 for seven voting. On a plain desktop processor with no graphics card: about 3 seconds a window (2.7 to 4 measured).
- Training one of our models: about 67 minutes on one eighth of the card, under 1 dollar. The night's search of 64 runs: about 57 dollars of card time.
- Gradient boosting: trains in 5 seconds on a laptop.
- The plant data can stay on site with our model and with gradient boosting. With Kimi it leaves the site.

## The test rows with their seconds shuffled (added 23:48, when the card's last pass was in)
The card answered the test set a second time with the seconds of every series in random order. The hints are computed again from the shuffled seconds, so this check cannot tell curve reading from hint reading; the direct curve check (same text, curve blank or shuffled) was made on check-set windows only.
- Simpler design (hybrid view + law hint), four models: 25, 33, 32 and 25 attack windows called attack (61 to 68 in true order) and 9, 12, 5 and 4 false alarms of 1,454 (13 to 23 in true order). It stays calm, and it loses about half of its catches: the order of the seconds matters to it. What it keeps is mostly the boiler-pressure loop (13 to 17 catches).
- Hint-heavy design (loop-relative view + step and law hints), eight models: 29 to 44 attack windows called attack (59 to 64 in true order) and 395 to 677 false alarms of 1,454 (12 to 21 in true order). It fails this check, as it did on the check set: the step hint is computed again from the shuffled seconds and then states a large jump.

## How much chance could move the gap (added 23:59, after the test result was read; it changes no result)
runs/final/gap_check.py draws the test set again 20,000 times: 56 attack time windows and 200 quiet ones, with replacement, the same draw for both systems (results in runs/final/gap_check.json).
- Gradient boosting minus our three-model vote: 3 points as scored. In 90% of draws the gap lies between -6 and +12. Our vote is level or ahead in 32% of draws (ahead in 25%).
- Gradient boosting minus our seven-model vote: 4 points as scored. In 90% of draws between -6 and +14. Our vote is level or ahead in 29% of draws.
- Three-model vote minus seven-model vote: 1 point; between -8 and +11.
- The plain count behind it: the test set holds 256 time windows (56 with an attack, 200 quiet), six loop rows each. Gradient boosting and the three-model vote score differently on 29 of them: 16 go to gradient boosting, 13 to the vote (two-sided sign test p = 0.71). Against the seven-model vote: 19 and 16 (p = 0.74).
- How much the choice of repeats moves a vote: the declared vote of three used seeds 0 to 2 and scores 51. The three other votes of three from the same four models score 52, 48 and 49.
Reading: the test set cannot tell gradient boosting and our votes apart. Redrawn, gradient boosting stays ahead in about two draws of three, which restates the gap we saw and is no clear lead. The redraw understates the spread: the test rows come from four recording files, and it does not capture the noise between training repeats.

## Limits
- The test set is not fully fresh: earlier models and both baselines were scored on it before this night, and the starting design was chosen after seeing those scores. Tonight's 64 runs never touched it.
- 56 attacks is still few. A difference of 3 points between 54 and 51 is small.
- The designs lean on hints computed by code (see the curve check in NIGHT-SUMMARY.md).
- Kimi got one wording, no long thinking, and numbers as text. That is "what you get if you just ask a huge model", not a contest of equals.
- HAI is a lab-scale plant. This is a research demo, not a deployed detector.
