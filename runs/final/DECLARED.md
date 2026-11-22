# What was fixed before the test run

Copied from the working log of the night of 21 November 2026. The first part was written at 22:32, before any test answer existed. The second was approved at 23:03 and written before 23:15, before any test result was read. The working log itself is not published. The two entries are given as they were written, except that one operational line is left out and two headings no longer name who approved.

## The test run (declared 22:32 on 21 November, before any test answer exists)

Decided at 22:31: answer the test rows once, with both designs. Fixed before the run, so nothing is chosen after seeing a test number:

- **Design A, hybrid + law** (`--view zsize --clauses law`): r4-zsize-law, r5-zsize-law-s1, r7-zsize-law-s2, r8-zsize-law-s3. Reported: each of the four single runs, their mean, and the majority vote of seeds 0 to 2 (2 of 3).
- **Design B, loop-relative + step and law** (`--view looprange --clauses step,law`): the eight runs r5-looprange-steplaw, r5-looprange-steplaw-s1, r6-lr-steplaw-s2, r6-lr-steplaw-s3, r7-lr-steplaw-s4 to -s7. Reported: each of the eight single runs, their mean, and the majority vote of seeds 0 to 6 (4 of 7).
- Score as on validation: attack rows called attack minus normal rows called attack. Also attacks found, and the same with the seconds shuffled. Next to them: gradient boosting, gradient boosting with neighbours and the range alarm on the same test rows.
- No other model answers the test rows tonight. No design, vote size or threshold changes after this point.
- Command per run: `python3 training/cloud/run_variant.py NAME --answer-only test test_shuffled --threads 2`.

## The big-model comparison (approved at 23:03 on 21 November; declared here before any test result is read)

- Model: `moonshotai/kimi-k3.1-pro` through OpenRouter, thinking off, answers of at most 160 tokens.
- Rows: the TEST set. All 82 attack windows, and of each loop every second neighbour window (129 of 254) and every second quiet window (600 of 1,200): 811 windows, 131 to 138 per loop, listed in runs/llm_compare/ids_final_sample.txt. False alarms on the sampled normal windows count double when a full-set score is estimated; every system is also compared on exactly these 811 windows.
- Two ways of asking: cold (the task and the raw numbers); and with our hints plus six solved examples in hint form from the training rows. Both prompts tell the model that about one window in five in the training material shows an attack.
- Order: (1) the wording is tried on 20 CHECK-set windows per way and may be corrected there; (2) the wording is frozen (commit and prompt file); (3) one run on the 811 test windows per way; (4) a side run on 60 of those windows with thinking on, if money is left. Our own test answers are fetched but NOT scored until step 2 is done, so nothing about the wording can follow a test result.
- This changes one line of "The test run" above: one more system now answers test rows, the big model, as a baseline named here in advance. No design, vote size or threshold of ours changes.

## What happened next

- 22:33: the twelve answer-only runs started.
- 23:14: the wording was tried on 20 check-set windows per way. One correction came from that trial, as step (1) allows: answers may be 240 tokens long, not 160, so that none is cut off.
- 23:15: the wording for the big model was frozen in commit `cbc0fba` (`training/llm_compare.py`). No test result of any system had been read up to this point.
- 23:20: the big model answered the 811 test windows in both ways. 24 answers were cut off at 240 tokens with no verdict and were asked again with room for 800 tokens; both counts are reported.
- 23:22: the side run with thinking on used 58 of the test windows (28 attack, 30 normal), not 60.
- 23:26: `runs/final/final_score.py` was written and first run on what was in: the baselines and the big model. Our own test answers were not yet fetched.
- 23:40: our models' test answers were in, and the test result was scored once with `runs/final/final_score.py`. The numbers are in `FACTS.md`.
- The checks in `gap_check.py` were made after the result was read. They change no score.
