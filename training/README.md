# Fine-tune OpenTSLM on the HAI loop examples

Seven steps on a Linux machine with one NVIDIA GPU. Run them in order from the repo root unless a step says otherwise.

Needs: `git`, [uv](https://docs.astral.sh/uv/) 0.9 or newer, Python 3.12, CUDA, and a Hugging Face login with access to `meta-llama/Llama-3.2-1B` (export `HF_TOKEN` with a read token before step 3). The starting checkpoint `OpenTSLM/llama-3.2-1b-tsqa-sp` is open.

**Another copy of the model.** With any id other than `meta-llama/Llama-3.2-1B`, replace `curriculum_learning.py` by `../../training/launch.py` in the commands below. The open copy `unsloth/Llama-3.2-1B` pads text on the left, and OpenTSLM then loses every series label (see "Run on Kaggle"). `launch.py` corrects that.

**Which GPU.** OpenTSLM loads the language model in bfloat16 (`torch_dtype=torch.bfloat16` in `OpenTSLMSP.py`). An L4, an A10G or an RTX 30- or 40-series card computes in it directly: run the steps below as written. A T4 (Kaggle, Colab) does not: it emulates bfloat16 slowly and is fast in float16. For a T4 use `training/launch.py` with `HAI_LLM_DTYPE=float16`, as the Kaggle section below does.

| File | What it does |
|---|---|
| `HAIAttackQADataset.py` | OpenTSLM dataset class for the JSONL files of `hai-export` |
| `add_stage.py` | adds stage `stage6_hai` to `curriculum_learning.py` (safe to rerun; stops if an anchor is missing) |
| `setup.sh` | clones OpenTSLM at the pinned commit, installs it, copies the class, runs the patcher, seeds the starting checkpoint |
| `score.py` | scores `test_predictions.jsonl` per attack; standard library only |
| `smoke_cpu.py` | one or more training steps and one generation on 4 examples, on CPU in float32 (wiring check only) |
| `launch.py` | runs `curriculum_learning.py` with optional settings: number format, loss scaling, answer length, epochs, seed |
| `kaggle/push.sh` | uploads the run files as a private Kaggle dataset and starts a script on a Kaggle GPU |
| `kaggle/gpu_probe.py` | Kaggle run 0: which number format and batch size fit the GPU, and how fast each is |
| `kaggle/first_run.py` | Kaggle run 1: the overfit check of step 3, scored, with a time estimate for the full run |
| `kaggle/kaggle_common.py`, `kaggle/gpu_check.py` | shared helpers of the Kaggle scripts |

## 1. Get the data

```bash
git clone https://github.com/vid-t/hai-attack-reader.git && cd hai-attack-reader
uv sync
uv run hai-build && uv run hai-export
```

`hai-build` downloads about 650 MB from the pinned HAI commit, checks every hash and writes `data/processed/examples.parquet`. `hai-export` then prints the rows per file. On the current build:

```
  3770 rows  data/processed/opentslm/train.jsonl
  1092 rows  data/processed/opentslm/validation.jsonl
  1536 rows  data/processed/opentslm/test.jsonl
  4404 rows  data/processed/opentslm/test_full.jsonl
  1536 rows  data/processed/opentslm/test_shuffled.jsonl
 24132 rows  data/processed/timenet/examples_table.csv.gz
```

`test.jsonl` holds the six loop rows of each of the 56 scored test attack windows and of 50 normal test windows per version.

## 2. Set up OpenTSLM

```bash
bash training/setup.sh                    # LLM id, default meta-llama/Llama-3.2-1B
```

It ends with:

```
seeded results/Llama_3_2_1B/OpenTSLMSP/stage5_ecg_cot/checkpoints/best_model.pt (28.6 MB) and results/Llama_3_2_1B/OpenTSLMSP/stage5_ecg_cot/results/metrics.json
```

A stage loads the stage before it, so the pretrained checkpoint goes where `stage5_ecg_cot` would be. The folder name comes from the code (`Llama_3_2_1B`), not from OpenTSLM's README (`Llama3_2_1B`). If you pass another LLM id, pass the same id to every command below.

All later commands run inside `external/OpenTSLM` with these variables:

```bash
cd external/OpenTSLM
export HAI_JSONL_DIR="$(cd ../.. && pwd)/data/processed/opentslm"
STAGE=results/Llama_3_2_1B/OpenTSLMSP/stage6_hai
mkdir -p ../../runs                       # predictions we keep (git-ignored)
```

## 3. Overfit 50 examples

```bash
time HAI_OVERFIT_ROWS=50 uv run python curriculum_learning.py --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B --stages stage6_hai --batch_size 4 --device cuda
python3 ../../training/score.py --only-predicted \
    --run overfit $STAGE/results/test_predictions.jsonl "$HAI_JSONL_DIR/train.jsonl"
```

It should print `HAI overfit check: train, validation and test are the same 50 training rows`, `Loading checkpoint from previous stage: results/Llama_3_2_1B/OpenTSLMSP/stage5_ecg_cot/checkpoints/best_model.pt`, `Enabling LoRA for stage6_hai`, then a train and val loss per epoch for up to 10 epochs. Both losses should fall well below their first value. It writes `$STAGE/checkpoints/best_model.pt`, `$STAGE/checkpoints/loss_history.txt`, `$STAGE/results/test_predictions.jsonl` (50 lines) and `$STAGE/results/metrics.json`.

The scorer's `label_right_share` on the `all` line should be close to 1.0. If it is not, the model does not learn the format: stop and look at `generated` in the predictions. Note the wall time: it sizes the full run (3,770 training rows and 1,092 validation rows per epoch, against 50 and 50 here).

## 4. Delete the overfit result

```bash
rm -rf "$STAGE"
```

Without this, the full run resumes from the overfit checkpoint, or skips the stage as already completed.

## 5. Full run

```bash
time uv run python curriculum_learning.py --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B --stages stage6_hai --batch_size 8 --device cuda
cp $STAGE/results/test_predictions.jsonl ../../runs/test_predictions_plain.jsonl
```

It prints `HAI rows: train 3770, validation 1092, test 1536`, a loss line per epoch and `New best model saved` when the val loss improves. Early stopping waits 5 epochs; the stage runs at most 10. At the end it loads the best checkpoint, generates an answer for each of the 1,536 rows of `test.jsonl`, and writes `$STAGE/results/test_predictions.jsonl` and `metrics.json` (`test_loss` there is NaN by design of OpenTSLM's evaluation; ignore it). `--gradient_checkpointing` only affects OpenTSLMFlamingo, so it is left out.

## 6. Test predictions on the shuffled file

The full run already wrote the predictions for `test.jsonl`. The shuffled run uses the same checkpoint on `test_shuffled.jsonl`, the same rows with the seconds permuted inside each window. OpenTSLM skips a stage whose `metrics.json` says completed, so move the first results aside:

```bash
mv $STAGE/results/metrics.json $STAGE/results/metrics_plain.json
mv $STAGE/results/test_predictions.jsonl $STAGE/results/test_predictions_plain.jsonl
HAI_TEST_FILE=test_shuffled.jsonl uv run python curriculum_learning.py --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B --stages stage6_hai --eval_only --device cuda
cp $STAGE/results/test_predictions.jsonl ../../runs/test_predictions_shuffled.jsonl
```

It prints `EVAL-ONLY MODE` and `HAI rows: train 3770, validation 1092, test 1536`, then writes a new `test_predictions.jsonl` (1,536 lines). Use `HAI_TEST_FILE=test_full.jsonl` the same way to predict every test row (4,404).

## 7. Score

```bash
cd ../..
python3 training/score.py \
    --run plain    runs/test_predictions_plain.jsonl    data/processed/opentslm/test.jsonl \
    --run shuffled runs/test_predictions_shuffled.jsonl data/processed/opentslm/test_shuffled.jsonl \
    --json runs/scores.json
```

For each run it prints one line per version and one `all` line: `attacks` (56 in all), `found` (an attacked loop answers attack), `right_loop_only` (attacked loops say attack, the other loops say normal), `timed` (start second within 30 s of the truth), `false_alarms_per_100` on normal-window loop examples, `valid_share` and `label_right_share`. A second table splits the attacked loop examples into `visible` and `no_visible_change`. An empty or unparseable answer is always wrong. The last line gives the constant guess: on the current `test.jsonl`, always answering second 120 times 23 of the 56 attacks, and the best constant (second 108) times 26, because the true start seconds of the 56 scored windows happen to bunch between 95 and 145. A model's `timed` count means something only above that. If the shuffled run scores as well as the plain one, the model is not reading the shape of the curves.

Each prediction carries `example_id` (added by `add_stage.py`), so the scorer joins on it. Predictions without it are joined by order, and the scorer stops if a `gold` answer does not match the test file.

## Run on Kaggle (free T4)

Kaggle gives two T4 GPUs with 15 GB each, 30 GPU hours a week and 12 hours per run. The scripts in `training/kaggle/` use one of them. The account must be phone-verified, or the run gets no GPU and no internet.

**One-time:** create an API token at <https://www.kaggle.com/settings/api> ("Generate New Token"). Put it in `KAGGLE_API_TOKEN` (the token, or the path of a file that holds it) or save it as `~/.kaggle/access_token`. Never commit it.

```bash
uv run hai-build && uv run hai-export                       # the data, once
bash training/kaggle/push.sh training/kaggle/gpu_probe.py   # run 0: settings for this GPU, about 10 minutes
bash training/kaggle/push.sh                                # run 1: first_run.py, the overfit check of step 3
```

`push.sh` uploads the training scripts and the JSONL files as the private dataset `<you>/hai-attack-reader-run-files`, then starts the script as a private Kaggle script on a T4 with internet on. It prints the page of the run and the two commands that give its status and download its results. `KAGGLE_SKIP_UPLOAD=1` starts a script on the files already there. `KAGGLE_TIME_LIMIT` sets the seconds after which Kaggle stops the run (default 5400).

On Kaggle the script copies the run files into the layout of this repo under `/tmp/hai`, runs `training/setup.sh unsloth/Llama-3.2-1B`, names the GPU, trains through `training/launch.py`, and keeps only `/kaggle/working/results`. It uses the open copy `unsloth/Llama-3.2-1B`, so it needs no Hugging Face token: its `model.safetensors` has the same SHA-256 as the one of `meta-llama/Llama-3.2-1B` (`68a2e4be...6809a`, 2,471,645,608 bytes).

### Why float16 on a T4

Measured on a Kaggle T4 on 2026-11-20 with `gpu_probe.py`: 16 training rows, batch 2, seed 0, three epochs. One 4096 x 4096 matrix product took 35 ms in float32, 57 ms in bfloat16 and 3 to 5 ms in float16; the card reports no native bfloat16.

| Number format, attention | Peak memory | Seconds per training row | Train loss, epochs 1 to 3 | Val loss, epochs 1 to 3 |
|---|---|---|---|---|
| float32, eager | 12.1 GB | 0.81 | 7.345 3.073 2.149 | 3.153 1.271 0.880 |
| float16, eager | 9.9 GB | 0.31 | 7.342 3.110 2.216 | 3.134 1.312 0.903 |
| float16, eager, no loss scaling | 9.9 GB | 0.28 | 7.359 3.130 2.186 | 3.191 1.315 0.901 |
| float16, sdpa | 7.6 GB | 0.25 | 7.354 3.124 2.185 | 3.170 1.318 0.900 |
| bfloat16, eager (what OpenTSLM asks for) | 9.9 GB | 1.50 | one epoch only | |

The probe times whole seconds on short epochs, so read the speeds as rough. The 50-row runs below are more exact.

- **float16 follows float32** in its losses and trains about 2.5 times faster. bfloat16 is about 5 times slower than float16.
- **Batch size:** float32 at batch 4 runs out of the T4's 15 GB. float16 with sdpa fits batch 4 (12.8 GB) at the same speed per row, and not batch 8.
- **Loss scaling** made no visible difference over 24 steps. It stays on with float16 because it costs nothing and protects small gradients late in training. One step in 250 to 1,000 overflowed at scale 4096; the scale then halved to 2048 and stayed.
- **sdpa attention** gives the same losses with a quarter less memory.

### The tokenizer of the open copy pads on the left

`unsloth/Llama-3.2-1B` has the weights of Meta's copy and a different tokenizer setting: it pads on the left with `<|finetune_right_pad_id|>`. OpenTSLM's `pad_and_apply_batch` keeps the first n tokens of each padded text piece. With left padding those are all padding: the model got the opening instruction and the curves, but no series label ("This is the target ...") and no closing "Answer:". `launch.py` and `smoke_cpu.py` now make every tokenizer pad on the right with the end-of-text token. `tests/test_launch.py` checks it against the real tokenizer.

### First run, 2026-11-20: the overfit check on 50 training rows

float16, sdpa, seed 0, up to 40 epochs, on one T4. The same 50 rows serve as train, validation and test.

| Run | Epochs | Labels right | Answers word for word right | Start second exact (15 attack rows) |
|---|---|---|---|---|
| before the padding fix, batch 2 | 21 (stopped early) | 42 of 50 | 33 of 50 | 4 |
| padding fixed, batch 2 | 40 | 50 of 50 | 50 of 50 | 15 |
| padding fixed, batch 1 | 27 (stopped early) | 50 of 50 | 50 of 50 | 15 |

With the fix both losses reach 0.000. Before the fix the train loss stayed near 0.8 while the val loss on the same rows fell to 0.1, which is how the fault showed.

| | Batch 2 | Batch 1 |
|---|---|---|
| Peak GPU memory | 9.3 GB | 6.2 GB |
| Seconds per training row | 0.28 | 0.36 |
| Seconds per validation row | 0.14 | 0.14 |
| Seconds per answer | 1.1 | 1.0 |
| Estimate for the full run, 10 epochs and the test answers | 3.8 h | 4.6 h |

Setup took 63 to 66 seconds in four runs and 427 seconds in one.

### Full run

```bash
KAGGLE_TIME_LIMIT=36000 bash training/kaggle/push.sh training/kaggle/full_run.py
```

`full_run.py` trains on the 3,770 training rows at batch 1 (a row then looks the same in training as in testing: no padding between its prompt and its answer), answers the 1,536 test rows, answers them again with the seconds shuffled, and scores both. Every five minutes it copies the best checkpoint (164 MB) and the loss history to `/kaggle/working/results`. Set `REHEARSAL_ROWS = 50` at the top to walk through every step in about ten minutes first; that rehearsal passed on 2026-11-20.

## CPU smoke test (no GPU)

Checks that the dataset class, the new stage and the seeded checkpoint load and run together. It proves wiring, not quality. Uses the ungated copy `unsloth/Llama-3.2-1B` because it needs no Hugging Face login. Never use `--device mps`: OpenTSLM's README says MPS does not work with the pretrained checkpoints.

```bash
bash training/setup.sh unsloth/Llama-3.2-1B
mkdir -p /tmp/hai-smoke && cd /tmp/hai-smoke
HAI_JSONL_DIR=<repo>/data/processed/opentslm uv run --project <repo>/external/OpenTSLM \
    python <repo>/training/smoke_cpu.py <repo>/external/OpenTSLM unsloth/Llama-3.2-1B
```

The script loads the language model in float32 and makes the tokenizer pad on the right (see "Run on Kaggle"). OpenTSLM asks for bfloat16, which a CPU runs hundreds of times slower: on the Apple M4 used here a 1024x1024 matrix product took 1.4 ms in float32 and 603 ms in bfloat16. The first attempt on 2026-11-20 ran in bfloat16, and its single training step had not finished after 40 minutes.

### Outcome on 2026-11-20, in float32, tokenizer padding fixed: passed

Mac mini (Apple M4, 16 GB RAM, 10 cores, CPU only), weights already in the cache.

```
[launch] tokenizer padded on the left with '<|finetune_right_pad_id|>': now on the right with '<|end_of_text|>'
[smoke] model loaded in 5 s
Epoch 1 — train loss: 9.1496
Epoch 1 — val   loss: 7.7182
[smoke] 1 training step(s) + evaluation in 91 s; metrics {'test_loss': nan, 'epoch': 1}
[smoke] predictions: results/Llama_3_2_1B/OpenTSLMSP/stage6_hai/results/test_predictions.jsonl
```

It wrote `checkpoints/best_model.pt`, `checkpoints/loss_history.txt`, `results/metrics.json` and `results/test_predictions.jsonl` (4 lines, each with `example_id`, `generated` and `gold`). `score.py --only-predicted` reads that file. After one step none of the four generated answers is valid, as expected.

What this shows: the forward and backward pass, checkpoint saving, generation and the predictions file all work with the dataset class, the new stage and the seeded checkpoint. What it does not show: a GPU, speed, memory, or any quality. For those see "Run on Kaggle" above.

An earlier run the same day, before the padding fix, started at a train loss of 15.4809: the model was reading padding in place of the series labels.

### Twelve steps on the same four rows: the loss falls on every step

```bash
HAI_SMOKE_EPOCHS=12 HAI_JSONL_DIR=<repo>/data/processed/opentslm uv run --project <repo>/external/OpenTSLM \
    python <repo>/training/smoke_cpu.py <repo>/external/OpenTSLM unsloth/Llama-3.2-1B
```

It took 861 s on the same machine.

| step | train loss | val loss |
|---|---|---|
| 1 | 9.1496 | 7.7243 |
| 2 | 4.7884 | 6.0691 |
| 3 | 4.4257 | 5.4603 |
| 4 | 3.4387 | 4.9419 |
| 5 | 3.0601 | 4.5141 |
| 6 | 2.7554 | 4.0722 |
| 7 | 2.4662 | 3.6381 |
| 8 | 2.2047 | 3.2700 |
| 9 | 1.9935 | 2.9889 |
| 10 | 1.8290 | 2.7864 |
| 11 | 1.7045 | 2.6532 |
| 12 | 1.6197 | 2.5851 |

Train and val are the same four rows; the val loss is measured after the step, in evaluation mode, one row at a time. After twelve steps the generated answers echo words of the prompt, and none is a valid answer yet. Twelve steps on four rows show that training moves in the right direction, nothing more. The 50-row run on the GPU is the test of whether the model reaches the right answers.

## Tests

`uv run --group training pytest tests/test_launch.py` tests the launcher: its settings, the answer cap, loss scaling (gradients unchanged, small ones saved from float16, the scale halving after an overflow), the tokenizer padding, and the log readers of the Kaggle scripts. The padding test against the real tokenizer needs `transformers` and the cached `unsloth/Llama-3.2-1B`; run it with `PYTHONPATH=src uv run --project external/OpenTSLM --with pytest python -m pytest tests/test_launch.py`.

`uv run --group training pytest tests/test_training.py` tests the patcher on the pinned commit (needs `external/OpenTSLM`), the scorer on made-up predictions and the dataset class on a 20-row JSONL (needs torch, hence the `training` group). Without the checkout or torch those tests skip.
