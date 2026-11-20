# hai-attack-reader

Labelled control-loop windows from the HAI industrial control system security dataset, for a small time-series language model.

One example is 300 seconds of one control loop (target, measurement, command), labelled `attack` or `normal`, with the second the attack starts and one line of text written by code.

## Rebuild everything

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12. From a fresh clone (`data/raw` and `data/processed` empty):

```bash
uv sync
uv run hai-build      # download, verify, check, build, score, write REPORT.md
uv run pytest         # tests run on what the build produced
```

`hai-build` downloads about 650 MB from `icsdataset/hai` at a pinned commit and checks every file against `data/raw/SHA256SUMS`. `--versions 21.03,22.04` picks versions; the default is all four (`ENABLED` in `src/hai/config.py`). A fresh build takes about two and a half minutes, most of it the download.

## What the build does

| Step | Module | Output |
|---|---|---|
| Download and hash | `hai.download` | `data/raw/`, `data/raw/SHA256SUMS` |
| Attack table from the manual (pypdf) | `hai.attacks_pdf` | `data/frozen/attacks.csv` (committed) |
| Table against the attack flags | `hai.check_table` | `data/processed/table_check.csv` |
| Name map to the 22.04 names | hand-made, checked by tests | `data/frozen/tag_map.csv` (committed) |
| One loader per version | `hai.loaders` | tidy table: `time`, standard tags, `attack` |
| Windows, loop examples, splits | `hai.dataset` | `data/processed/windows.parquet`, `examples.parquet` |
| Answer text | `hai.answers` | `answer` column in `examples.parquet` |
| Baselines on validation | `hai.baselines` | `data/processed/baselines.json` |
| Report | `hai.report` | `REPORT.md` |

To recount the rows and rerun the table check against the flags: `uv run python -m hai.check_table 21.03` (any versions; default all). To regenerate the attack table from the PDF: `uv run python -m hai.attacks_pdf`. To check the committed table against the PDF: `uv run python -m hai.attacks_pdf --check`. Every fix to the manual's table is in `CORRECTIONS`, `ATTACK_FIXES` and `POINT_FIXES` in `src/hai/attacks_pdf.py`, with its reason in the `note` column.

## Loops

| Loop | Target | Measurement | Command |
|---|---|---|---|
| Boiler pressure (P1-PC) | `P1_B2016` | `P1_PIT01` | `P1_PCV01D`, `P1_PCV02D` |
| Boiler level (P1-LC) | `P1_B3004` | `P1_LIT01` | `P1_LCV01D` |
| Boiler flow (P1-FC) | `P1_B3005` | `P1_FT03` | `P1_FCV03D` |
| Boiler temperature (P1-TC) | `P1_B4022`, `P1_B4002` | `P1_TIT01` | `P1_FCV01D`, `P1_FCV02D` |
| Turbine speed (P2-SC) | `P2_AutoSD` | `P2_SIT01` | `P2_SCO` |
| Water level (P3-LC) | `P3_LH01`, `P3_LL01` | `P3_LIT01` | `P3_LCP01D`, `P3_LCV01D` |

Names are the 22.04 names. `data/frozen/tag_map.csv` gives each version's column and the evidence for it.

## Splits

By whole file, so no attack crosses splits. Normal-file windows go to train.

| Split | 20.07 | 21.03 | 22.04 | 23.05 |
|---|---|---|---|---|
| Train | test1 | test1, test2, test3 | test1, test2 | hai-test2 |
| Validation | | test4 | test3 | |
| Test | test2 | test5 | test4 | hai-test1 |

## Using the examples

`data/processed/examples.parquet` has one row per loop example. Four columns decide how to use a row:

| Column | Meaning | Use |
|---|---|---|
| `attack_start_s` | Second of the window where the attack starts | Random between 30 and 270, so it cannot be guessed |
| `scored` | True for one of the six windows of each attack, picked at random | Score per attack on these windows only |
| `no_visible_change` | True for an attack example whose series show no clear step | Leave out of training; count on its own line in scoring |
| `train_use` | True for the training sample | Train on these rows; the full train split is 96% normal |

The training sample holds every train attack example with a visible change, plus two normal examples from attack windows and two from normal windows for each. Validation and test are never sampled.

To add the neighbouring loops' measurements to an example, join the other rows with the same `window_id`. `hai.baselines.neighbour_features` does this for the boosting baseline.

## Data

Raw and processed data are never committed. The two frozen CSVs derive from HAI and are under CC BY-SA 4.0: see [DATA_LICENSE.md](DATA_LICENSE.md). Results, mismatches and open questions are in [REPORT.md](REPORT.md).
