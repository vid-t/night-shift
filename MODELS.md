# Released models

Twelve fine-tuned add-on files for Llama 3.2 1B. Each one reads 300 seconds of one control loop of the HAI plant testbed and says whether the loop is under attack, when the change starts, and how the series move.

The dated result records under `runs/final/` say "hint-heavy" for the design this page calls feature-heavy. "Hint" was the project's earlier word for a computed feature, the numbers that code writes beside each curve; `runs/nebius/NIGHT-SUMMARY.md` calls them "clauses". They all mean the same thing.

**Built with Llama.** The files work only with Meta's Llama 3.2 1B and fall under the Llama 3.2 Community License and Meta's [Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy). The licence text and the `NOTICE` file are attached to the release and are in this repository (`LICENSES/`, `NOTICE`). The model code is OpenTSLM (MIT licence). The models were trained on the HAI dataset (CC BY-SA 4.0).

## What one file holds

One file is 54,695,785 bytes: 13,647,616 numbers, all float32, in a PyTorch pickle.

| Key | What it is | Numbers |
|---|---|---|
| `encoder_state` | the curve encoder (76 tensors) | 2,110,976 |
| `projector_state` | the projector from curve to soft tokens (4 tensors) | 264,448 |
| `lora_state` | LoRA adapters on all 16 layers: q, k, v, o, gate, up and down projections, rank 16, alpha 32 (224 tensors) | 11,272,192 |
| `lora_config` | the `peft` LoRA settings (no numbers) | |
| `epoch`, `val_loss`, `lora_enabled` | the kept pass (4) and its check-set loss | |

No Llama base weight is inside: no embedding, no `lm_head`, no full layer weight. Every `lora_state` key is a `lora_A` or `lora_B` matrix. No optimizer state is inside. Three files were opened (simpler seed 0 and 3, feature-heavy seed 7): they have the same keys and the same tensor counts. You need the base model `unsloth/Llama-3.2-1B` or `meta-llama/Llama-3.2-1B` (same weights file) to use them.

The file needs `torch.load(path, map_location="cpu", weights_only=False)` and `peft`, because `lora_config` is a pickled `peft` object. A pickle can run code: check the SHA-256 before you load a file.

## The twelve models

Both designs start from `OpenTSLM/llama-3.2-1b-sleep-sp`, train 4 passes and keep the last one. **Simpler:** `--view zsize --clauses law`. **Feature-heavy:** `--view looprange --clauses step,law`. Check-set score: attack rows called attack minus normal rows called attack, of 144 attack rows (1,092 rows). Test score: attack windows caught (of 82) minus false alarms (of 1,454). Numbers come from `runs/final/final_results.json`.

| Release file | Run | Design | Seed | Kept pass | Check set | Test caught | Test false alarms | Test score | SHA-256 |
|---|---|---|---|---|---|---|---|---|---|
| `Llama-3.2-1B-night-shift-simpler-seed0.pt` | `r4-zsize-law` | simpler | 0 | 4 | 71 | 65 | 23 | 42 | `4a16421e9b2d5fad6da7cba0faaf9eb91585cb6f12ff2c366ea15a6112e0094f` |
| `Llama-3.2-1B-night-shift-simpler-seed1.pt` | `r5-zsize-law-s1` | simpler | 1 | 4 | 79 | 65 | 13 | 52 | `6ec335ad38b80e74577cdd8ffcf85a00fd7f745f192b6d2e3f8f6629bf860267` |
| `Llama-3.2-1B-night-shift-simpler-seed2.pt` | `r7-zsize-law-s2` | simpler | 2 | 4 | 78 | 68 | 16 | 52 | `9e4b54abaa507cba3545288d151acd14b18443048e4abaa607657d4c707e0b16` |
| `Llama-3.2-1B-night-shift-simpler-seed3.pt` | `r8-zsize-law-s3` | simpler | 3 | 4 | 78 | 61 | 18 | 43 | `c63efe16583c6436e79cf368e9a54cd5e37f774d7bdf650aea611745d470da98` |
| `Llama-3.2-1B-night-shift-feature-heavy-seed0.pt` | `r5-looprange-steplaw` | feature-heavy | 0 | 4 | 84 | 62 | 12 | 50 | `7f9dc2205fbe27114461fc70527d181756d14ca1b1e9a9e826a1e44dd6542227` |
| `Llama-3.2-1B-night-shift-feature-heavy-seed1.pt` | `r5-looprange-steplaw-s1` | feature-heavy | 1 | 4 | 86 | 63 | 20 | 43 | `fdd52e13ae0304945d5395716d503b87d6c756bd80d9bb8fdff2ebc948fb343d` |
| `Llama-3.2-1B-night-shift-feature-heavy-seed2.pt` | `r6-lr-steplaw-s2` | feature-heavy | 2 | 4 | 63 | 63 | 14 | 49 | `da51f1184dd41c54ef891d87e0860e7fbf164a9135cd6014338d5a1792a35f39` |
| `Llama-3.2-1B-night-shift-feature-heavy-seed3.pt` | `r6-lr-steplaw-s3` | feature-heavy | 3 | 4 | 74 | 59 | 20 | 39 | `75cfd9c6edd6e6e601882f795dac0c713cffa6d6bd220639f3b154b20ae1da35` |
| `Llama-3.2-1B-night-shift-feature-heavy-seed4.pt` | `r7-lr-steplaw-s4` | feature-heavy | 4 | 4 | 77 | 64 | 17 | 47 | `a11003ab5b0fabcb21ea8d683efe309b356c0e74bf06863c84706a363829496d` |
| `Llama-3.2-1B-night-shift-feature-heavy-seed5.pt` | `r7-lr-steplaw-s5` | feature-heavy | 5 | 4 | 79 | 62 | 16 | 46 | `cff6534734c0a3afa102278c34694266fde54a2aff2fd4fba356cf38f31ff928` |
| `Llama-3.2-1B-night-shift-feature-heavy-seed6.pt` | `r7-lr-steplaw-s6` | feature-heavy | 6 | 4 | 83 | 63 | 19 | 44 | `31a56029b40221f833b247a56ceb5fe6c518683e19d7fc64b6aa43002f6bc3df` |
| `Llama-3.2-1B-night-shift-feature-heavy-seed7.pt` | `r7-lr-steplaw-s7` | feature-heavy | 7 | 4 | 70 | 63 | 21 | 42 | `64911ba07a36c03f30e6aa7a82bc688982887dc166278c29d8b2ba83dc497d40` |

## The two declared votes

Both votes were written down before the one test run. A window counts as attack when enough models say attack.

| System | Check set | Test caught | Test false alarms | Test score |
|---|---|---|---|---|
| Simpler design, seeds 0 to 2, attack when 2 of 3 say so | 80 | 66 | 15 | 51 |
| Feature-heavy design, seeds 0 to 6, attack when 4 of 7 say so | 90 | 65 | 15 | 50 |
| Gradient boosting (for comparison) | 82 | 71 | 17 | 54 |

Gradient boosting scores 54 on the test set; our votes score 51 and 50. The test set is too small to tell them apart (`runs/final/FACTS.md`). What our models add is a start second and a sentence. `training/vote.py` builds a vote from the answer files in `answers.tar.gz`.

## Use a model

These steps answer rows on a CPU with OpenTSLM's own eval-only path (`training/launch.py --eval_only`), so the prompt is the one the model learned. Steps 4 to 6 ran on a desktop computer (Apple M4, CPU only) on 22 November 2026, with simpler seed 0 and feature-heavy seed 0, on 6 check-set rows (the first 3 attack rows and the first 3 normal rows of `validation.jsonl`), with symlinks for both checkpoint files and a copy of the seed's `metrics.json`. Steps 1 and 2 ran on earlier days in this checkout. Step 3 ran the same day against the release: the downloaded model file and `answers.tar.gz` passed the checksum test.

1. Get the repo and the data. `hai-build` downloads about 650 MB. You need `train.jsonl` even to answer: the computed features are measured on the quiet training rows.
   ```bash
   git clone https://github.com/vid-t/night-shift.git && cd night-shift
   uv sync && uv run hai-build && uv run hai-export
   REPO=$(pwd); C=$REPO/external/OpenTSLM; S=results/Llama_3_2_1B/OpenTSLMSP
   ```
2. Set up OpenTSLM: `bash training/setup.sh unsloth/Llama-3.2-1B`. It also copies the current `training/HAIAttackQADataset.py` into the checkout. Rerun it after every pull: an old copy silently ignores the series view and builds the wrong prompt (this happened in the check: 4 of 6 labels agreed until the copy was refreshed). The starting checkpoint it seeds only fills a slot that OpenTSLM requires; the released file replaces all its weights. `HAI_START_CHECKPOINT=OpenTSLM/llama-3.2-1b-tsqa-sp` makes that download 29 MB in place of 164 MB; the check used it.
3. Download a model and check it:
   ```bash
   gh release download v1.0 -R vid-t/night-shift -D ~/hai-models \
       -p 'Llama-3.2-1B-night-shift-simpler-seed0.pt' -p SHA256SUMS
   (cd ~/hai-models && shasum -a 256 -c --ignore-missing SHA256SUMS)
   ```
4. Make a work folder. The file goes where OpenTSLM looks for the stage's best model; the rows to answer go in a folder next to `train.jsonl` and `validation.jsonl`.
   ```bash
   WORK=~/hai-work; mkdir -p $WORK/jsonl $WORK/$S/stage5_ecg_cot $WORK/$S/stage6_hai/checkpoints
   cp -R $C/$S/stage5_ecg_cot/. $WORK/$S/stage5_ecg_cot/
   ln -s ~/hai-models/Llama-3.2-1B-night-shift-simpler-seed0.pt $WORK/$S/stage6_hai/checkpoints/best_model.pt
   ln -s $REPO/data/processed/opentslm/train.jsonl $REPO/data/processed/opentslm/validation.jsonl $WORK/jsonl/
   head -n 6 $REPO/data/processed/opentslm/validation.jsonl > $WORK/jsonl/rows.jsonl   # or your own rows
   ```
5. Answer. For a feature-heavy file, use `HAI_SERIES_VIEW=looprange HAI_TEXT_CLAUSES=step,law`.
   ```bash
   cd $WORK && env UV_PROJECT=$C UV_NO_SYNC=1 OPENTSLM_DIR=$C CUDA_VISIBLE_DEVICES= \
     HAI_JSONL_DIR=$WORK/jsonl HAI_TEST_FILE=rows.jsonl HAI_LLM_DTYPE=float32 HAI_ATTENTION=sdpa \
     HAI_MAX_NEW_TOKENS=128 HAI_SERIES_VIEW=zsize HAI_TEXT_CLAUSES=law HAI_ANSWER_ORDER=label_last \
     uv run python $REPO/training/launch.py --model OpenTSLMSP --llm_id unsloth/Llama-3.2-1B \
       --stages stage6_hai --eval_only --device cpu
   ```
   Do not set `HF_HUB_OFFLINE=1`: transformers 4.57.3 then fails while loading the tokenizer. Never use `--device mps`.
6. Read the answers. The run prints `HAI rows: train 3880, validation 1092, test 6`, `Loaded LoRA adapters: 224 parameters`, `encoder response of the model that answers: 0.6055` (simpler seed 0) and `stage6_hai evaluation complete`. Each answer is a line of `$WORK/results/Llama_3_2_1B/OpenTSLMSP/stage6_hai/results/test_predictions.jsonl` with `example_id`, `generated` and `gold`, for example `The first valve command rises from second 157 while the target and second valve command stay flat. The measurement falls after it. Change starts near second 155. Answer: attack`. `training/score.py` scores the file. To answer again, move that `results` folder aside first: OpenTSLM skips a stage that it has marked complete.

**Speed on the CPU:** 3 to 4 seconds a row in float32 on the Apple M4 (6 rows in 18 to 24 seconds), and 28 seconds for the whole command with the model already cached.

**What the check showed:** both models gave the same label as the training card on 6 of 6 check-set rows (3 attack, 3 normal), from the same prompt text. The feature-heavy model gave the same text word for word on all 6. The simpler model did on 3; on the others only the seconds moved, as expected between float32 on a CPU and bfloat16 on the card. Example, CPU: `The command rises from second 67 while the target stays flat. The measurement starts swinging after it. Change starts near second 54. Answer: attack`; card: `... Change starts near second 67. Answer: attack`.

## Limits

- Single models of one design differ a lot between seeds: on the test set by 10 points in the simpler design (42 to 52) and 11 in the feature-heavy design (39 to 50). Use a vote, or judge a design on four seeds or more.
- The answers lean on computed features that code writes into the text (the law and step clauses). With the seconds shuffled, the feature-heavy design raises 395 to 677 false alarms of 1,454.
- HAI is a lab-scale plant testbed. This is a research demo, not a deployed detector.
