# HAI attack reader: demo page

A static page (no build step, no framework, no backend). Every path is relative.

## Rebuild the data

From the repo root:

    uv run python demo/build_data.py
    uv run python demo/build_results.py

Both scripts read files under `runs/`. The result files are in git; the answer files are in the release:
unpack its `answers.tar.gz` at the repository root first (see the main README, "No graphics card?").

`build_data.py` writes `demo/data/demo.json` (about 200 KB) from check-set (validation) files only.
It never opens a test file. It stops with an error if its recount of the vote (101 caught, 11 false
alarms, 90) or of the seven voting runs disagrees with `runs/nebius/summary_source/night_numbers.json`.

`build_results.py` writes `demo/data/results_extra.json` (the test set and the huge-model comparison).
It reads only `runs/final/final_results.json` and `runs/final/results_table_test.md`, never an answer
file. Cost and speed figures for our own model are not in those files; they are quoted in the script
from `runs/final/FACTS.md`. It stops if a count disagrees with FACTS.md.

## View it

    cd demo && python3 -m http.server 8765 --bind 127.0.0.1

then open http://127.0.0.1:8765/. Opening `index.html` as a file does not work: the page fetches its JSON.

## Files

| Path | What |
|---|---|
| `index.html` | the page |
| `css/style.css` | all styles |
| `js/app.js` | loads the data, draws the charts (SVG), runs the window explorer |
| `fonts/archivo-latin.woff2` | Archivo, SIL Open Font License, self-hosted |
| `fonts/OFL.txt` | the font's licence text; it goes out with the font |
| `data/demo.json` | built by `build_data.py` |
| `data/results_extra.json` | test set and huge-model results, built by `build_results.py` |
| `build_data.py` | builds `demo.json` |
| `build_results.py` | builds `results_extra.json` |

## Publish list

Publish exactly these five entries and nothing else:

    index.html
    css/
    js/
    fonts/
    data/

Everything else in this folder is a working file (`README.md`, `build_data.py`, `build_results.py`). To make a clean copy to upload, run from this folder:

    rm -rf ../demo-publish && mkdir ../demo-publish && cp -R index.html css js fonts data ../demo-publish/

and upload the contents of `../demo-publish/`.

`.github/workflows/pages.yml` publishes exactly this list to GitHub Pages. It starts by hand only: turn Pages on
for the repository with GitHub Actions as the source, then run `gh workflow run pages.yml`.

## `data/results_extra.json`

Built by `build_results.py`; do not edit by hand. Two keys: `test` (whole test set: the systems,
the single-model repeats, the range alarm, attacks found, start times and the constant guess, the test rows answered again with shuffled seconds, plus
the check-set scores for the drop from check to test) and `big_model` (the 811 shared windows, the
thinking side run, cost rows). Static text binds to it with `data-x="path.to.value"`, as it binds
to `demo.json` with `data-k`. If the file is missing or empty the page shows a short notice and the
static text keeps its numbers.

## Where the numbers come from

Every check-set number the page computes comes from `demo.json`; every test-set and huge-model number from `results_extra.json`. Numbers in the static text either match
`demo.json` (the page overwrites them with the live value) or are quoted from
`runs/nebius/NIGHT-SUMMARY.md`: runs with hints average 73.1 and without 56.5; blank curves take the simpler design from 99 to 72 attack calls; 23 points between
repeats; 10 hours; the last pass was best in all 8 runs that scored every pass; -3 at half attacks;
about 36 windows out of reach (24 water level, 12 boiler temperature).

Some figures are quoted from `runs/final/FACTS.md` and `runs/final/gap_check.json` (checks made after the
test result was read; they change no score): the test set drawn again 20,000 times, the gap between gradient
boosting and the three-model vote between −6 and +12 in nine draws of ten, the vote level or ahead in about a
third of the draws (32%); the two scoring differently on 29 of 256 stretches of time (16 to gradient boosting,
13 to the vote); and the three other votes of three scoring 48, 49 and 52. The conclusion's "about an hour for
under a dollar" is the 67 minutes of FACTS.md; "about 1 billion parameters" is Llama 3.2 1B.

The curve check's "normal curve" column uses the training machine's answers (95 attack calls), as the
night summary does; the re-run on another computer differs on 1 of 540 windows (96). The shuffled curves
in the explorer use the exact seconds order of the curve check (`permutation` in `curve_check.py`; the script is in the
release's `answers.tar.gz`, under `runs/nebius/2026-11-21/night/analysis/curvecheck/`).

## Credits

HAI dataset, https://github.com/icsdataset/hai, CC BY-SA 4.0 (attribution sits under every figure
that shows HAI data). OpenTSLM. TimeNet hackathon. Archivo font, SIL Open Font License.
