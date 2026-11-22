# hai-attack-reader

A small time-series language model reads 300 seconds of one control loop of a lab-scale power-plant testbed (HAI). It says whether the loop is under attack, names the second the attack began, and describes what it saw in a sentence. It is OpenTSLM-SP: the small language model Llama 3.2 1B plus a part that turns curves into input it can read. We trained it on all four releases of the HAI dataset for the TimeNet hackathon. We compare it with a classic statistical method and with a huge general model asked directly.

**On the test set, run once, gradient boosting scores 54 of a perfect 82. Three of our models voting score 51, seven score 50. Chance alone could erase that gap. A huge general model asked directly is far behind both. Our model adds what gradient boosting does not give: the attack's start second and a sentence.**

Built with Llama. The language model is Llama 3.2 1B, under Meta's Llama 3.2 Community License.

An interactive demo page with real windows is in [`demo/`](demo/README.md). The twelve trained models are in the release; [MODELS.md](MODELS.md) says how to use one.

## Conclusion

1. **Training on the plant's own data beats size.** On the same 811 test windows our votes score 59 and 57. Kimi K3.1 Pro, asked directly, scores -181 with our hints and -284 cold: it calls a third to a half of all normal windows an attack. Each of our models is small, about 1 billion parameters, and trains in about an hour for under a dollar.
2. **On detection, gradient boosting is 3 to 4 points ahead, and that is inside chance.** Of the test set's 256 stretches of time, gradient boosting and our three-model vote differ on only 29: gradient boosting does better on 16, the vote on 13. On the check set we used to choose designs, seven of our models voting led gradient boosting 90 to 82 (87 when gradient boosting also reads the other loops). That lead did not carry over. Choosing on the check set likely flattered our designs there. Gradient boosting ran with default settings and still came out ahead.
3. **What it adds: when the attack began, and a sentence a person can check.** An alarm alone does not tell an operator when or what. Either vote names the start within 30 seconds for 30 of the 56 attacks, where a fixed guess gets 26. That beats a fixed guess, but not by much. We did not measure how often the sentence is right. Gradient boosting gives no start and no sentence.
4. **Two weaknesses set the next step.** The model relies more on the numbers our code writes next to each curve than on the curve itself. And one model alone is a gamble: retrained from a new random start, one design scores 42 to 52 of 82, the other 39 to 50. What we would try next, not tested here: let gradient boosting raise the alarm and let the language model explain it.

## Results

The test set has 1,536 windows: 82 attack windows from 56 attacks and 1,454 normal windows (254 neighbour windows, where another loop of the same window is attacked, and 1,200 quiet windows). A window is 300 seconds seen through one control loop. **Score = attack windows called attack minus normal windows called attack. Perfect is 82.** We fixed both designs, the voters and the scoring before the run, and ran it once.

| System | Attack windows caught (of 82) | False alarms (of 1,454) | Score | Attacks found (of 56) |
|---|---|---|---|---|
| Gradient boosting (classic statistical method) | 71 | 17 | 54 | 52 |
| Gradient boosting that also reads the other loops | 68 | 14 | 54 | 52 |
| Our simpler design (hybrid view + law hint), three models voting | 66 | 15 | 51 | 50 |
| Our hint-heavy design (loop-relative view + step and law hints), seven models voting | 65 | 15 | 50 | 51 |
| Our simpler design, one model: four repeats | 61 to 68 | 13 to 23 | 42, 52, 52, 43 (mean 47.25) | 48 to 52 |
| Our hint-heavy design, one model: eight repeats | 59 to 64 | 12 to 21 | 50, 43, 49, 39, 47, 46, 44, 42 (mean 45) | 45 to 51 |
| Range alarm (the floor) | 38 | 173 | -135 | 31 |

- Gradient boosting is slightly ahead, but the test set cannot separate it from our votes. It and our three-model vote score differently on only 29 of the 256 stretches of 300 seconds: 16 go to gradient boosting, 13 to our vote (two-sided sign test, p = 0.71).
- We rebuilt the test set at random 20,000 times. In nine draws of ten, gradient boosting's lead was between -6 and +12 points, and our vote tied or led in 32%. Both checks came after we read the result and change no number (step 11).
- One model alone scores 45 to 47 on average, so several models vote. As a share of the perfect score, the seven-model vote held steady from check set to test set (90 of 144 is 63%; 50 of 82 is 61%); gradient boosting went up (82 of 144 is 57%; 54 of 82 is 66%).

**What our model gives that gradient boosting does not: a start second and a sentence.**

| System | Attacks timed within 30 seconds (of 56) | Median start error |
|---|---|---|
| Our simpler design, three models voting | 30 | 22 s |
| Our hint-heavy design, seven models voting | 30 | 26 s |
| Our simpler design, one model (seed 1) | 39 | 19 s |
| Our hint-heavy design, one model (seed 0) | 28 | 26 s |
| Always answering "second 108" (the best constant guess) | 26 | |
| Gradient boosting | no start time | |

The best constant guess already times 26 of the 56 attacks. Our timing is only useful where it beats those 26. We timed only these two single models.

**The huge general model, Kimi K3.1 Pro, asked directly** on 811 of the test windows (setup in step 9):

| System | Attack windows caught (of 82) | False alarms (of 729 normal) | Score on these windows |
|---|---|---|---|
| Gradient boosting | 71 | 8 | 63 |
| Our simpler design, three models voting | 66 | 7 | 59 |
| Our hint-heavy design, seven models voting | 65 | 8 | 57 |
| Kimi K3.1 Pro with our hints and six solved examples | 78 | 259 | -181 |
| Kimi K3.1 Pro, cold | 78 | 362 | -284 |

Kimi catches almost every attack window because it calls many normal windows an attack too: cold, about half of them; with our hints and examples, about a third. It does not know what is normal for this plant. Our hints and examples help it, but not enough.

**Cost and speed.**

| System | Cost | Speed | Where the plant data goes |
|---|---|---|---|
| Our model, one model | about 0.17 US dollars per 1,000 windows on a rented card at 5.40 dollars an hour | 12 models answered 1,536 windows each in 34 minutes on the card; about 3 seconds a window on a plain desktop processor with no graphics card (2.7 to 4 measured) | stays on site |
| Our model, three voting / seven voting | about 0.50 / 1.16 dollars per 1,000 windows | three / seven times one model | stays on site |
| Kimi K3.1 Pro | 1.54 dollars per 1,000 questions cold, 1.64 with hints and examples, 6.17 with thinking on | 1.4 seconds an answer (6.1 with thinking) | leaves the site |
| Gradient boosting | trains in 5 seconds on a laptop | | stays on site |

Training one of our models took about 67 minutes on one eighth of the card, under 1 dollar. The whole training search of 64 runs cost about 57 dollars of card time. The Kimi comparison cost 3.08 dollars in all.

Full tables per loop and per version: `runs/final/FINAL-RESULTS.md` and `runs/final/results_table_test.md`. Every count: `runs/final/final_results.json`. The sheet of final facts: `runs/final/FACTS.md`. Step 11 recounts them.

## Step by step

These are the steps we took, in order. Each step says what it does, how to run it, what comes out and where to read more.

| # | Step | Run | Needs |
|---|---|---|---|
| 1 | Get the data | `uv run hai-build` | any machine, 650 MB download |
| 2 | Cut windows, label them per loop, split by file | `uv run hai-export` | any machine |
| 3 | The baselines | `training/baselines_on_test.py` | any machine |
| 4 | The model and what it is shown | nothing to run | |
| 5 | Train one model | `training/cloud/run_variant.py` | Linux, one NVIDIA card, about 67 minutes |
| 6 | The training search: 64 runs in one night | step 5, repeated | the same card |
| 7 | Answer the test set, once | `run_variant.py --answer-only` | the same card |
| 8 | Vote and score | `training/vote.py`, `training/score.py` | any machine |
| 9 | Ask a huge general model | `training/llm_compare.py` | any machine, an OpenRouter key |
| 10 | Check what the model reads | the answers of steps 5 and 7 | |
| 11 | Recount every number yourself | `runs/final/final_score.py` | any machine |
| 12 | The demo page | `demo/build_data.py`, `demo/build_results.py` | any machine |
| 13 | The TimeNet connector | a separate pull request | |

**No graphics card?** Do step 1, skip steps 5 to 7 and download our answer files with the [GitHub CLI](https://cli.github.com/). Steps 8, 11 and 12 then run on a plain desktop:

```bash
gh release download v1.0 --repo vid-t/hai-attack-reader --pattern answers.tar.gz
tar -xzf answers.tar.gz            # at the repository root; it fills runs/
```

`answers.tar.gz` (7 MB) holds the answers, logs and scores of all 64 training runs, the baselines, the curve check and the Kimi comparison. Data under `data/raw` and `data/processed` is never committed; step 1 rebuilds it from the pinned source.

### 1. Get the data

- **What.** [HAI](https://github.com/icsdataset/hai) records a hardware-in-the-loop testbed of a boiler, a turbine and a water treatment process, one row per second, with staged attacks. The build fetches all four releases (20.07, 21.03, 22.04 and 23.05) at a pinned commit and checks every file by SHA-256. It reads the attack tables from the HAI manual, checks each attack against the attack flags in the data, and fixes the manual where the data disagrees. It maps every release's column names to the 22.04 names.
- **Run it** (any machine; needs [uv](https://docs.astral.sh/uv/) and Python 3.12):
  ```bash
  uv sync
  uv run hai-build      # download about 650 MB, verify hashes, check the attack table, build windows and examples, write REPORT.md
  uv run hai-export     # write the training, check-set and test files for OpenTSLM, and the TimeNet table
  uv run pytest         # tests run on what the build produced
  ```
  `hai-build --versions 21.03,22.04` builds only some releases; the default is all four. A fresh build takes about two and a half minutes, most of it the download. To rerun the table check against the flags: `uv run python -m hai.check_table 21.03` (any releases; default all). To regenerate the attack table from the PDF: `uv run python -m hai.attacks_pdf`; to check the committed table against it: `uv run python -m hai.attacks_pdf --check`.
- **What comes out.** `data/raw/` with `SHA256SUMS`; the committed attack table `data/frozen/attacks.csv` and the hand-made name map `data/frozen/tag_map.csv` (checked by tests); in `data/processed/`, the table check (`table_check.csv`), `windows.parquet`, `examples.parquet` with its `answer` column, the check-set baselines (`baselines.json`), the OpenTSLM files (`opentslm/*.jsonl`) and the TimeNet table (`timenet/examples_table.csv.gz`); and `REPORT.md`. Each part has its own module in `src/hai/`.
- **More.** Every fix to the manual's table is in the `note` column of `data/frozen/attacks.csv`, in `CORRECTIONS`, `ATTACK_FIXES` and `POINT_FIXES` in `src/hai/attacks_pdf.py`, and in [REPORT.md](REPORT.md), with the build's mismatches and open questions.

### 2. Cut windows, label them per loop, split by file

- **What.** We look at six control loops, each with a target, a measurement and a command.

| Loop | Target | Measurement | Command |
|---|---|---|---|
| Boiler pressure (P1-PC) | `P1_B2016` | `P1_PIT01` | `P1_PCV01D`, `P1_PCV02D` |
| Boiler level (P1-LC) | `P1_B3004` | `P1_LIT01` | `P1_LCV01D` |
| Boiler flow (P1-FC) | `P1_B3005` | `P1_FT03` | `P1_FCV03D` |
| Boiler temperature (P1-TC) | `P1_B4022`, `P1_B4002` | `P1_TIT01` | `P1_FCV01D`, `P1_FCV02D` |
| Turbine speed (P2-SC) | `P2_AutoSD` | `P2_SIT01` | `P2_SCO` |
| Water level (P3-LC) | `P3_LH01`, `P3_LL01` | `P3_LIT01` | `P3_LCP01D`, `P3_LCV01D` |

- **Windows.** Every attack gets six attack windows; in each, the attack starts at a random second between 30 and 270 (one draw from each of six equal bands, fixed seed). One of the six, picked at random, is the scored window of that attack. Normal windows lie at least 600 seconds from any attack, half from the attack files and half from the normal file of the release.
- **Labels.** One window gives up to six examples, one per loop. An example is `attack` when an attack the HAI manual assigns to that loop is flagged in at least one second of the window. Otherwise it is `normal`: "neighbour" when another loop of the window is attacked, "quiet" in a normal window. Code writes the reference sentence from the series: `rises` or `falls` for a level shift, `starts swinging` or `stops swinging` for a change in swinging. Two kinds of rows are kept out of training and counted apart: attack rows whose series show no visible change, and normal boiler-loop rows that start or stop swinging when a neighbouring loop is attacked (`reacting`).
- **Splits.** By whole recording file inside each release, so no attack crosses splits. Normal-file windows go to training.

| Split | 20.07 | 21.03 | 22.04 | 23.05 |
|---|---|---|---|---|
| Train | test1 | test1, test2, test3 | test1, test2 | hai-test2 |
| Check set (validation) | | test4 | test3 | |
| Test | test2 | test5 | test4 | hai-test1 |

- **Run it.** Nothing new: `uv run hai-build` and `uv run hai-export` in step 1 do this.
- **What comes out.** Training uses a balanced sample of 3,880 rows: every training attack row with a visible change, plus two neighbour rows and two quiet rows for each. The check set holds 1,092 rows (144 attack rows from 15 attacks); every design choice was made on it. The test set holds 1,536 rows: the six loop rows of each of the 56 scored test attack windows and of 50 normal test windows per release.
- **More.** `src/hai/dataset.py` and `src/hai/answers.py`; the rows per file in [training/README.md](training/README.md), step 1.

### 3. The baselines

- **What.** Two yardsticks: a range alarm (the floor) and gradient boosting (a classic statistical method), with and without the other loops' measurements.
- **Run it** (any machine; gradient boosting trains in 5 seconds on a laptop):
  ```bash
  uv run python training/baselines_on_test.py --out runs/baselines
  ```
  It trains gradient boosting on the training sample, answers every test row in the model's answer format, and scores the range alarm, both gradient boosting variants and "always normal" with the same scorer as our model.
- **What comes out.** The baselines' answers on the test rows in `runs/baselines`, the same files as `runs/baselines/repaired/` in `answers.tar.gz`. Step 11 counts their scores: gradient boosting 54 of 82 both ways, the range alarm -135. On the check set gradient boosting scored 82 of 144 (87 when it also reads the other loops).
- **More.** `training/baselines_on_test.py`; the check-set baselines of step 1 in `data/processed/baselines.json`.

### 4. The model and what it is shown

- **What.** [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM)-SP: a curve encoder turns each series into soft tokens, and the language model, Llama 3.2 1B, reads them between the text lines. We add a training stage, `stage6_hai`, and fine-tune with LoRA adapters (small trainable add-on weights) on the language model. We start from OpenTSLM's stage 4 checkpoint (`OpenTSLM/llama-3.2-1b-sleep-sp`), not the end of its curriculum: in the stage 5 checkpoint the encoder gives the same output for every curve, so a model that starts there never sees a curve. Our launcher checks the encoder at the start, after every pass and before the answers, and refuses a dead one. Answers are greedy (no sampling). The answer is one or two sentences, then `Change starts near second N.` for an attack, then `Answer: attack` or `Answer: normal`.
  - **Two views.** How a curve is scaled decides what the model can see. The **hybrid view** (`--view zsize`) scales each curve to its own average and spread. Its text line says how big the largest change is, compared with a normal quiet change in that loop. The **loop-relative view** (`--view looprange`) shows each curve as its change from its first 30 seconds, measured against how much that series normally moves in a quiet training window of its loop.
  - **Text hints** (we call them clauses). Each is one or two numbers computed from the series, fitted on quiet training rows only, added to its text line. `step`: the largest one-second step against the usual one, and its second. `law`: how many seconds the command moves against what the controller should do. Only the boiler pressure loop gets it, the one loop where we could model the controller well. `size` and `plant` were tried too; they did not help.
  - **Our two final designs.** Simpler design: hybrid view + law hint (`--view zsize --clauses law`). Hint-heavy design: loop-relative view + step and law hints (`--view looprange --clauses step,law`).
- **Run it.** Nothing to run; steps 5 and 6 use these flags.
- **What comes out: one worked example.** This window is from the **check set** (validation), not the test set. It is the example the demo page opens with: 300 seconds of the boiler level loop (P1-LC) in HAI 22.04, file `test3`. An attack on this loop starts at second 112. The model gets one instruction, then three series. Each series is a short text line, our computed hint, followed by its curve as soft tokens. This is the hint-heavy design (the law hint only appears in the boiler pressure loop).

```text
You are given 300 seconds of the boiler level control loop of a power plant testbed, one reading
per second. The series are the target, the measurement and the command. Each series is shown as
its change from its first 30 seconds, measured against how much it normally moves in this loop.
Say whether the loop behaves normally or is under attack. If it is under attack, say the second
where the change starts. End with 'Answer: attack' or 'Answer: normal'.

This is the target, the value the controller aims for; it never changes from one second to the next:
  [curve: 300 seconds]
This is the measurement, the measured value; its largest one-second step is 5.8 times its usual one, at second 243:
  [curve: 300 seconds]
This is the command, what the controller sends to the actuator; its largest one-second step is 2.5 times its usual one, at second 179:
  [curve: 300 seconds]
```

The reference answer, written by code from the labels and the series:

```text
The measurement starts swinging from second 120 while the target and command stay flat.
Change starts near second 112. Answer: attack
```

All seven models of the vote said attack. One of their answers, the one the demo shows next to the vote:

```text
The command falls from second 128 while the target stays flat. The measurement rises after it.
Change starts near second 127. Answer: attack
```

The verdict is right. The start is 15 seconds late, and the sentence describes the movement differently from the reference. The seven models' start seconds spread widely: 65, 93, 93, 131, 127, 175 and 139. Gradient boosting also called this window an attack, with no start time. One of the seven models (seed 0) answered the same window again with the curve blanked and with its seconds shuffled, text unchanged: both times it said "normal". On this window the curve decides.
- **More.** [training/README.md](training/README.md), sections "The starting checkpoint" and "The tokenizer of the open copy pads on the left".

### 5. Train one model

- **What.** One run trains one design with one seed, then answers the check set plain and with the seconds shuffled, and scores both.
- **Run it** (Linux, one NVIDIA card):
  ```bash
  bash training/setup.sh unsloth/Llama-3.2-1B     # OpenTSLM at the pinned commit, our stage, the stage 4 starting checkpoint
  python3 training/cloud/run_variant.py r4-zsize-law --start sleep --view zsize --order label_last \
      --epochs 4 --clauses law --choose-pass last --seed 0
  ```
  That is the simpler design, seed 0. For the hint-heavy design use `--view looprange --clauses step,law`; for another repeat, another `--seed`. `--rows 50` is a rehearsal: train, check set and test are the same 50 training rows. `python3 training/cloud/run_variant.py --help` lists every option.
- **What comes out.** Everything of the run in `work/NAME/`, the saved model and answers in `work/NAME/kept/`. One run takes about 67 minutes on one eighth of a rented card, under 1 dollar.
- **More.** [training/README.md](training/README.md): the original single-card path, and why the T4 needs float16.

### 6. The training search: 64 runs in one night

- **What.** In one night we ran 64 full training runs in 8 rounds over 10 hours, each on one eighth of a rented card. We judged every run on the check set only.
- **Run it.** Step 5, repeated with other flags and seeds.
- **What comes out.**
  - **Start and end.** The starting design (plain range view, no hints) scored 17 to 37 of 144. The leading designs ended at 71.25 to 79.6 on average.
  - **What worked.** Text hints: runs with hints averaged 73.1 (56 to 86), runs of the same views without them 56.5 (50 to 63). The loop-relative view beat the plain range view. Keeping the last training pass beat keeping the pass with the lowest validation loss, in every run that scored all passes.
  - **What failed or gave no clear gain.** More attack rows in training failed: at a 50% share the score fell to -3. A different starting checkpoint (`har`) and the `plant` hint gave no gain beyond seed noise.
  - **The catch.** Single runs of one design differ by up to 23 points from seed to seed, so the six leading designs could not be told apart as single models.
- **More.** `runs/nebius/NIGHT-SUMMARY.md`: the report of the night, with every run's score.

### 7. Answer the test set, once

- **What.** Because one model is a gamble, several models of the same design vote (step 8). Before this run we fixed both designs, which models vote, and the scoring. Then each model answered the test rows once, plain and with the seconds shuffled.
- **Run it** (Linux, one NVIDIA card; once per model, with a finished run of step 5):
  ```bash
  python3 training/cloud/run_variant.py r4-zsize-law --answer-only test test_shuffled
  ```
- **What comes out.** `work/NAME/kept/test_predictions_plain.jsonl` and `test_predictions_shuffled.jsonl`. 12 models answered 1,536 windows each in 34 minutes on the card.
- **More.** `runs/final/DECLARED.md`: what we fixed before the run, with the times.

### 8. Vote and score

- **What.** A window is called attack when a majority says attack, and the sentence and start second come from a model that agrees. The simpler design votes with three models (seeds 0 to 2, 2 of 3), the hint-heavy design with seven (seeds 0 to 6, 4 of 7).
- **Run it** (any machine, with the answer files of the release or of step 7):
  ```bash
  N=runs/nebius/2026-11-21/night    # where answers.tar.gz puts the answers; for your own runs: N=work
  uv run python training/vote.py --out runs/vote_A.jsonl \
      $N/{r4-zsize-law,r5-zsize-law-s1,r7-zsize-law-s2}/kept/test_predictions_plain.jsonl
  uv run python training/vote.py --out runs/vote_B.jsonl \
      $N/{r5-looprange-steplaw,r5-looprange-steplaw-s1,r6-lr-steplaw-s2,r6-lr-steplaw-s3,r7-lr-steplaw-s4,r7-lr-steplaw-s5,r7-lr-steplaw-s6}/kept/test_predictions_plain.jsonl
  python3 training/score.py --run A runs/vote_A.jsonl data/processed/opentslm/test.jsonl
  uv run python training/results_table.py --truth data/processed/opentslm/test.jsonl \
      --run A runs/vote_A.jsonl --run B runs/vote_B.jsonl
  ```
  `--min K` in `vote.py` sets the number of attack votes needed; the default is a strict majority. `score.py` prints, per release and in all, the attacks found, the right loop only, the start within 30 seconds, false alarms and the constant-guess line.
- **What comes out.** The two votes. From the release's answer files they come out byte for byte identical to the committed `runs/final/vote_A.jsonl` and `runs/final/vote_B.jsonl`. They score 51 and 50 of 82 (see Results).
- **More.** `training/vote.py`, `training/score.py`.

### 9. Ask a huge general model: Kimi K3.1 Pro

- **What.** We asked Kimi K3.1 Pro (`moonshotai/kimi-k3.1-pro` through OpenRouter) on 811 of the test windows: all 82 attack windows plus half of each loop's normal windows (129 neighbour, 600 quiet). Two ways: cold (the task and the raw numbers), and with our computed hints plus six solved examples. Thinking off, short answers. The wording was tried on 20 check-set windows per way and frozen before any test result was read.
- **Run it** (any machine; needs an OpenRouter key, `--key-file` or the file named in `OPENROUTER_API_KEY_FILE`). With `--dry-run` the tool reads the public price list, writes three sample prompts, prints the worst-case cost, and makes no paid call:
  ```bash
  uv run python training/llm_compare.py --model moonshotai/kimi-k3.1-pro --mode cold \
      --rows test --allow-test --ids runs/llm_compare/ids_final_sample.txt \
      --out runs/llm_compare/NAME.jsonl --max-usd 2.00 --dry-run
  ```
  `--mode hints_examples` adds our hints and six solved training examples. `--reasoning low` turns thinking on. The tool refuses the test rows unless both `--rows test` and `--allow-test` are given, and it stops at the spending cap. The top of `training/llm_compare.py` explains every mode and the cost cap. The tool also stops at 5 dollars for all runs under `runs/llm_compare/` together, and the unpacked archive already holds our 3.08 dollars of answers: move that folder aside before a real run of your own.
- **What comes out.** Scores -181 with hints and -284 cold (see Results), for 3.08 dollars in all. Thinking on does not get close either. On 58 of those windows (28 attack, 30 normal), Kimi cold caught 27 with 20 false alarms with thinking off, and 25 with 15 with thinking on (low), at four times the price. Gradient boosting caught 25 with 0 false alarms, our votes 20 and 21 with 0. 24 of Kimi's 1,622 answers were cut off before a verdict; asked again with more room, 20 gave one. The counts use the second answer for those; 4 cold answers have no verdict and count as "not attack".
- **More.** `runs/final/FACTS.md`, section "The huge general model".

### 10. Check what the model reads: the curve check

- **What.** Does the model read the curve or only the hints? Models answered the same windows again with the curve blanked or its seconds shuffled.
- **Run it.** The shuffled answers come with every run of step 5 (check set) and step 7 (`test_shuffled`). The blank-curve check ran on a desktop processor with `curve_check.py`; the script and its answers are in `answers.tar.gz`, under `runs/nebius/2026-11-21/night/analysis/curvecheck/`.
- **What comes out.**
  - **Check set, text unchanged.** The hints carry much of the signal: with every curve blanked, one hint-heavy model still made 84 attack calls (95 with the curve), one simpler model 72 (99). The curve still matters: with the seconds shuffled, they made only 48 and 50, so when the curve does not look like an attack the model overrules the numbers. One model per design, on inputs neither saw in training: a sign, not a measurement.
  - **Test set, a weaker check.** Our code recomputes the hints from the shuffled seconds, so this check cannot show whether the model reads the curve or the hints. Simpler design, four models: 25 to 33 attack windows called attack (61 to 68 in true order), with 4 to 12 false alarms of 1,454; it stays calm, loses about half of its catches, and keeps mostly the boiler-pressure loop. Hint-heavy design, eight models: 29 to 44 attack windows called attack, and 395 to 677 of 1,454 normal windows (12 to 21 in true order); it fails, because the recomputed step hint then states a large jump.
- **More.** `runs/nebius/NIGHT-SUMMARY.md`, section "Does the model read the curve?"; `runs/final/FACTS.md`, section "The test rows with their seconds shuffled".

### 11. Recount every number yourself, no graphics card needed

- **What.** The scores on this page come from the answer files in the release. You can recount them on a plain desktop.
- **Run it** (any machine; after step 1, with the answers unpacked as in "No graphics card?" above):
  ```bash
  uv run python runs/final/final_score.py     # rewrites runs/final/final_results.json and FINAL-RESULTS.md
  uv run python runs/final/gap_check.py       # rewrites runs/final/gap_check.json
  ```
- **What comes out.** `runs/final/final_results.json`, `runs/final/FINAL-RESULTS.md` and `runs/final/gap_check.json`. On a plain desktop all three came out byte for byte identical to the committed ones: `git status` stays clean.
- **More.** `runs/final/final_score.py` counts with `training/score.py`. `runs/final/gap_check.py` draws the 56 attack stretches and 200 quiet ones again, with replacement, the same draw for both systems.

### 12. The demo page

- **What.** A static page with real check-set windows and the results.
- **Run it** (any machine; after step 1, with the answers unpacked):
  ```bash
  uv run python demo/build_data.py
  uv run python demo/build_results.py
  cd demo && python3 -m http.server 8765 --bind 127.0.0.1
  ```
  Then open http://127.0.0.1:8765/. Opening `index.html` as a file does not work: the page fetches its JSON.
- **What comes out.** `demo/data/demo.json`, built from check-set files only, and `demo/data/results_extra.json`, built from `runs/final/`. Each script stops if its recount disagrees with the committed numbers.
- **More.** [demo/README.md](demo/README.md): the files, where each number comes from, and what to publish.

### 13. The TimeNet connector

- **What.** A connector that brings these examples to TimeNet (dataset id `icsdataset/hai-loop-attacks`). It goes to OpenTSLM/TimeNet as a separate pull request.
- **Run it.** `uv run hai-export` in step 1 writes the examples table.
- **What comes out.** `data/processed/timenet/examples_table.csv.gz` (24,132 rows). The connector adds 24,132 loop examples as classification tasks and 1,458 localization tasks for the attack interval, over 17 recordings. The examples table will be published in its own repository under CC BY-SA 4.0.
- **More.** `src/hai/export.py`.

## Use a trained model

The twelve trained models are attached to the GitHub release v1.0 of `vid-t/hai-attack-reader`: `Llama-3.2-1B-hai-attack-reader-simpler-seed0..3.pt` and `Llama-3.2-1B-hai-attack-reader-hint-heavy-seed0..7.pt`, 55 MB each, with `SHA256SUMS`, `NOTICE` and the Llama 3.2 Community License text.
Each file holds only the trained parts, not the Llama base weights.
[MODELS.md](MODELS.md) explains how to use them.

## What is where

| Path | What |
|---|---|
| `src/hai/` | the data build (steps 1 and 2) |
| `training/` | setup, training, answers, voting, scoring, baselines, the Kimi tool (steps 3 to 9) |
| `tests/` | tests of the build, the training tools, the vote and the Kimi tool |
| `demo/` | the demo page and the scripts that build its data (step 12) |
| `data/frozen/` | the committed attack table (`attacks.csv`) and name map (`tag_map.csv`) |
| `runs/final/` | final facts, tables, counts, votes and the scripts that count them (step 11) |
| `runs/nebius/` | the report of the training night and the script that recounts its numbers (step 6) |
| `REPORT.md` | what the data build found: results, mismatches, open questions |
| `MODELS.md` | how to use the trained models of the release |
| `DATA_LICENSE.md` | the licence of the files derived from HAI, and what was changed |
| `NOTICE`, `LICENSES/` | third-party notices and licence texts (Llama 3.2, OpenTSLM) |
| `.github/workflows/pages.yml` | publishes the demo page to GitHub Pages; started by hand |

The answer files and logs under `runs/` are in the release's `answers.tar.gz`. The twelve released models are separate files of the release. The other 52 saved models and the working log of the night are not published.

## Limits

- **Gradient boosting is slightly ahead on the test set.** Our model does not beat the classic method on detection. Its check-set lead (90 against 82, or 87 when gradient boosting also reads the other loops) did not hold.
- **The test set is not fully fresh.** Earlier models and both baselines were scored on it before the night of the training search, and the starting design was chosen after seeing those scores. The night's 64 runs never touched it.
- **56 test attacks are few.** The test set cannot separate 54 from 51 (gap between -6 and +12 in nine draws of ten). That redraw understates the spread: the windows come from four recording files, one per HAI release. The three other votes of three we could have formed from the same four models score 48, 49 and 52.
- **The check set is small, and every choice was made on it.** It holds 15 attacks. The hints and the loop-relative view were designed from what the models missed there, so every check-set score is optimistic.
- **The designs lean on hints computed by code** (step 10). With the step hint, every loop-relative run fails the shuffle check: it calls 242 to 414 of 948 shuffled normal check-set rows an attack, because shuffling also changes the numbers in its text. The same holds on the test set. Gradient boosting on exactly the numbers of the hybrid view's text scores 62 on the check set.
- **One model is a gamble.** Single runs of one design differ by up to 23 points on the check set, and the same seed does not reproduce the same run.
- **Two loops are out of reach.** No vote finds the boiler temperature (P1-TC) attacks of the check set, and no single run calls more than 3 of its 12 attack rows an attack: training holds P1-TC attacks of 23.05 only, the check set of 22.04 only. In the water level loop (P3-LC), no run finds more than 18 of the 42 check-set attack rows; gradient boosting finds 27.
- **Kimi got one wording, no long thinking, and numbers as text.** That is "what you get if you just ask a huge model", not a contest of equals.
- **HAI is a lab-scale plant.** This is a research demo, not a deployed detector.

## Data and licences

- **HAI.** The data comes from the HAI security datasets, <https://github.com/icsdataset/hai>, at commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737`, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Raw and processed data are never committed here. The two committed tables derived from HAI, `data/frozen/attacks.csv` and `data/frozen/tag_map.csv`, are shared under the same licence. So are the demo page's data files, which hold 16 check-set windows and counts, `REPORT.md`, the result files under `runs/` and the release's answer archive. [DATA_LICENSE.md](DATA_LICENSE.md) says what was changed. The examples table the TimeNet connector reads is derived from HAI and carries CC BY-SA 4.0 as well; it will be published in its own repository.
- **OpenTSLM.** [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM) is MIT-licensed (Stanford University, ETH Zurich and the project authors). `training/setup.sh` clones it at a pinned commit into `external/`, which is not in git. Its licence text is in `LICENSES/`. The starting checkpoint is OpenTSLM's published `OpenTSLM/llama-3.2-1b-sleep-sp`.
- **Llama 3.2 1B.** The language model is Meta's Llama 3.2 1B, under Meta's Llama 3.2 Community License. On the training machine we load the open copy `unsloth/Llama-3.2-1B`, whose weights match Meta's file by SHA-256. The trained models of the release are built with Llama and fall under the same licence; see `NOTICE` and `LICENSES/`.
- **The TimeNet hackathon.** This is an entry to the [TimeNet Online Hackathon](https://challenge.timenet.ai/). We also built a TimeNet connector for these examples (dataset id `icsdataset/hai-loop-attacks`).
- **This repository's code.** The repository has no licence file for the code. `DATA_LICENSE.md` covers only the data files derived from HAI and states that no licence is granted for the code.

### Citation

The HAI README asks users to cite the following.

HAI 22.04, HAI 23.05, HAIEnd 23.05:

```bibtex
@misc{github,
  author={Shin, Hyeok-Ki; Lee, Woomyo; Choi, Seungoh; Yun, Jeong-Han; and Min, Byung-Gi},
  title={HAI security datasets},
  year={2023},
  url={https://github.com/icsdataset/hai},
}
```

HAI 21.03, HAICon 2020, HAICon 2021:

```bibtex
@inproceedings{10.1145/3474718.3474719,
  author = {Shin, Hyeok-Ki and Lee, Woomyo and Yun, Jeong-Han and Min, Byung-Gi},
  title = {Two ICS Security Datasets and Anomaly Detection Contest on the HIL-Based Augmented ICS Testbed},
  year = {2021},
  isbn = {9781450390651},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3474718.3474719},
  doi = {10.1145/3474718.3474719},
  booktitle = {Cyber Security Experimentation and Test Workshop},
  pages = {36--40},
  numpages = {5},
  location = {Virtual, CA, USA},
  series = {CSET '21}
}
```

HAI 20.07:

```bibtex
@inbook{10.5555/3485754.3485755,
  author = {Shin, Hyeok-Ki and Lee, Woomyo and Yun, Jeong-Han and Kim, HyoungChun},
  title = {HAI 1.0: HIL-Based Augmented ICS Security Dataset},
  year = {2020},
  publisher = {USENIX Association},
  address = {USA},
  booktitle = {Proceedings of the 13th USENIX Conference on Cyber Security Experimentation and Test},
  articleno = {1},
  numpages = {1}
}
```
