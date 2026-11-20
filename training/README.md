# Fine-tune OpenTSLM on the HAI loop examples

Seven steps on a Linux machine with one NVIDIA GPU. Run them in order from the repo root unless a step says otherwise.

Needs: `git`, [uv](https://docs.astral.sh/uv/) 0.9 or newer, Python 3.12, CUDA, and a Hugging Face login with access to `meta-llama/Llama-3.2-1B` (export `HF_TOKEN` with a read token before step 3). The starting checkpoint `OpenTSLM/llama-3.2-1b-tsqa-sp` is open.

**Which GPU.** OpenTSLM loads the language model in bfloat16 (`torch_dtype=torch.bfloat16` in `OpenTSLMSP.py`). Pick a GPU that runs bfloat16 natively: an L4, an A10G, an RTX 30- or 40-series card, or newer. A T4 runs it in a slow fallback mode (one write-up measured about four times slower than float16), and a P100 is reported not to support it. None of this was tested here: the first minutes of step 3 show the real speed.

| File | What it does |
|---|---|
| `HAIAttackQADataset.py` | OpenTSLM dataset class for the JSONL files of `hai-export` |
| `add_stage.py` | adds stage `stage6_hai` to `curriculum_learning.py` (safe to rerun; stops if an anchor is missing) |
| `setup.sh` | clones OpenTSLM at the pinned commit, installs it, copies the class, runs the patcher, seeds the starting checkpoint |
| `score.py` | scores `test_predictions.jsonl` per attack; standard library only |
| `smoke_cpu.py` | one or more training steps and one generation on 4 examples, on CPU in float32 (wiring check only) |

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

## CPU smoke test (no GPU)

Checks that the dataset class, the new stage and the seeded checkpoint load and run together. It proves wiring, not quality. Uses the ungated copy `unsloth/Llama-3.2-1B` because it needs no Hugging Face login. Never use `--device mps`: OpenTSLM's README says MPS does not work with the pretrained checkpoints.

```bash
bash training/setup.sh unsloth/Llama-3.2-1B
mkdir -p /tmp/hai-smoke && cd /tmp/hai-smoke
HAI_JSONL_DIR=<repo>/data/processed/opentslm uv run --project <repo>/external/OpenTSLM \
    python <repo>/training/smoke_cpu.py <repo>/external/OpenTSLM unsloth/Llama-3.2-1B
```

The script loads the language model in float32. OpenTSLM asks for bfloat16, which a CPU runs hundreds of times slower: on the Apple M4 used here a 1024x1024 matrix product took 1.4 ms in float32 and 603 ms in bfloat16. The first attempt on 2026-11-20 ran in bfloat16, and its single training step had not finished after 40 minutes.

### Outcome on 2026-11-20, in float32: passed

Mac mini (Apple M4, 16 GB RAM, 10 cores, CPU only), weights already in the cache.

```
[smoke] model loaded in 4 s
Epoch 1 — train loss: 15.4809
Epoch 1 — val   loss: 8.1421
[smoke] 1 training step(s) + evaluation in 103 s; metrics {'test_loss': nan, 'epoch': 1}
[smoke] predictions: results/Llama_3_2_1B/OpenTSLMSP/stage6_hai/results/test_predictions.jsonl
```

It wrote `checkpoints/best_model.pt`, `checkpoints/loss_history.txt`, `results/metrics.json` and `results/test_predictions.jsonl` (4 lines, each with `example_id`, `generated` and `gold`). `score.py --only-predicted` reads that file. After one step none of the four generated answers is valid, as expected.

What this shows: the forward and backward pass, checkpoint saving, generation and the predictions file all work with the dataset class, the new stage and the seeded checkpoint. What it does not show: bfloat16 on a GPU, speed, memory, or any quality.

### Twelve steps on the same four rows: the loss falls on every step

```bash
HAI_SMOKE_EPOCHS=12 HAI_JSONL_DIR=<repo>/data/processed/opentslm uv run --project <repo>/external/OpenTSLM \
    python <repo>/training/smoke_cpu.py <repo>/external/OpenTSLM unsloth/Llama-3.2-1B
```

It took 712 s on the same machine.

| step | train loss | val loss |
|---|---|---|
| 1 | 15.4809 | 8.1209 |
| 2 | 11.6538 | 7.2132 |
| 3 | 10.6716 | 6.5420 |
| 4 | 9.5514 | 6.3706 |
| 5 | 8.8738 | 6.0114 |
| 6 | 8.2542 | 5.6493 |
| 7 | 7.6361 | 5.2491 |
| 8 | 7.1135 | 4.8427 |
| 9 | 6.6678 | 4.4874 |
| 10 | 6.3264 | 4.2135 |
| 11 | 6.0772 | 4.0310 |
| 12 | 5.9049 | 3.9393 |

Train and val are the same four rows; the val loss is measured after the step, in evaluation mode. After twelve steps the generated answers use the words of the task (`under attack`, `The change starts`, `Answer:`) but none is a valid answer yet. The loss starts high because the starting checkpoint was trained to give very short answers. Twelve steps on four rows show that training moves in the right direction, nothing more. Step 3 on the GPU is still the first test of whether the model reaches the right answers.

## Tests

`uv run --group training pytest tests/test_training.py` tests the patcher on the pinned commit (needs `external/OpenTSLM`), the scorer on made-up predictions and the dataset class on a 20-row JSONL (needs torch, hence the `training` group). Without the checkout or torch those tests skip.
