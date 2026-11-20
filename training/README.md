# Fine-tune OpenTSLM on the HAI loop examples

Seven steps on a Linux machine with one NVIDIA GPU. Run them in order from the repo root unless a step says otherwise.

Needs: `git`, [uv](https://docs.astral.sh/uv/) 0.9 or newer, Python 3.12, CUDA, and a Hugging Face login with access to `meta-llama/Llama-3.2-1B` (export `HF_TOKEN` with a read token before step 3). The starting checkpoint `OpenTSLM/llama-3.2-1b-tsqa-sp` is open.

| File | What it does |
|---|---|
| `HAIAttackQADataset.py` | OpenTSLM dataset class for the JSONL files of `hai-export` |
| `add_stage.py` | adds stage `stage6_hai` to `curriculum_learning.py` (safe to rerun; stops if an anchor is missing) |
| `setup.sh` | clones OpenTSLM at the pinned commit, installs it, copies the class, runs the patcher, seeds the starting checkpoint |
| `score.py` | scores `test_predictions.jsonl` per attack; standard library only |
| `smoke_cpu.py` | one training step and one generation on 4 examples, on CPU (wiring check only) |

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

### Outcome on 2026-11-20: did not finish

Run on a Mac mini (Apple silicon, 16 GB RAM, 10 cores, CPU only), with a 45-minute time box (11:00 to 11:45). The first 4 minutes went to downloading the `unsloth/Llama-3.2-1B` weights (2.47 GB). The run then got this far:

```
[smoke] model loaded in 288 s
Loading checkpoint from previous stage:  results/Llama_3_2_1B/OpenTSLMSP/stage5_ecg_cot/checkpoints/best_model.pt and model type:  OpenTSLMSP and llm_id:  unsloth/Llama-3.2-1B
📂 Loading best model from stage5_ecg_cot:
   Achieved at epoch: 20
   Validation loss: 0.0509
   seeded_from: OpenTSLM/llama-3.2-1b-tsqa-sp/model_checkpoint.pt
🔧 Enabling LoRA for stage6_hai
✅ LoRA enabled:
   LoRA parameters: 11,272,192
   Total parameters: 1,247,086,592
HAI overfit check: train, validation and test are the same 4 training rows
📈 Total training steps: 1
🆕 Starting fresh training for stage6_hai
Epoch 1/1:   0%|          | 0/1 [00:00<?, ?it/s]
```

The single training step had not finished after about 40 minutes, and the time box stopped the run there. No generation ran. The cause is unknown: the run left no error, and nobody measured CPU or memory during the step.

What this shows: the patched `curriculum_learning.py` imports, `stage6_hai` is a known stage, the seeded checkpoint loads into OpenTSLMSP with the `unsloth/Llama-3.2-1B` llm id, LoRA turns on for `stage6_hai`, and the dataset class reads and formats the JSONL rows. What it does not show: a forward and backward pass, a generation, or the predictions file. Step 3 (overfit 50 examples) on the GPU is the first real test of those.

## Tests

`uv run --group training pytest tests/test_training.py` tests the patcher on the pinned commit (needs `external/OpenTSLM`), the scorer on made-up predictions and the dataset class on a 20-row JSONL (needs torch, hence the `training` group). Without the checkout or torch those tests skip.
