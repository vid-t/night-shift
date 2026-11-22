# HAI attack reader: demo page

A static page (no build step, no framework, no backend). Every path is relative.

## Rebuild the data

From the repo root:

    uv run python demo/build_data.py
    uv run python demo/build_results.py
    uv run python demo/build_scenes.py

Both scripts read files under `runs/`. The result files are in git; the answer files are in the release:
unpack its `answers.tar.gz` at the repository root first (see the main README, "No graphics card?").

`build_data.py` writes `demo/data/demo.json` (about 200 KB) from check-set (validation) files only.
It never opens a test file. It stops with an error if its recount of the vote (101 caught, 11 false
alarms, 90) or of the seven voting runs disagrees with `runs/nebius/summary_source/night_numbers.json`,
or if its recount of the range alarm (`runs/baselines/repaired_validation/predictions_range_alarm.jsonl`
against the check-set labels: 49 caught, 113 false alarms, score -64, in `score.range_alarm`) is off.

`build_results.py` writes `demo/data/results_extra.json` (the test set and the huge-model comparison).
It reads only `runs/final/final_results.json` and `runs/final/results_table_test.md`, never an answer
file. Cost and speed figures for our own model are not in those files; they are quoted in the script
from `runs/final/FACTS.md`. It stops if a count disagrees with FACTS.md.

`build_scenes.py` writes `demo/data/scenes.json` (about 365 KB): six moments of the check set, each one
300-second window with all six loops, their curves, hint lines, the seven recorded answers, the vote and
gradient boosting's call, plus each curve's usual movement over the check set. It reuses `build_data.py`'s
readers and stops if a scene disagrees with the facts it asserts, or if its recount of the vote (101 caught,
11 false alarms) or of gradient boosting (108, 26) is off.

## View it

    cd demo && python3 -m http.server 8765 --bind 127.0.0.1

then open http://127.0.0.1:8765/. Opening `index.html` as a file does not work: the page fetches its JSON.

## Files

| Path | What |
|---|---|
| `index.html` | the page: two views, the control room (default) and the write-up |
| `css/style.css` | fonts, tokens (light and dark), header, the write-up |
| `css/console.css` | the control room |
| `js/views.js` | the Demo / Write-up toggle and the routing by hash, including the old anchors |
| `js/console.js` | the control room: loads `scenes.json` and `results_extra.json`, draws and replays the scenes |
| `js/app.js` | the write-up: loads `demo.json` and `results_extra.json`, binds the numbers in the text, draws the charts (SVG) |
| `fonts/geist-latin.woff2`, `fonts/geist-mono-latin.woff2` | Geist and Geist Mono, SIL Open Font License, self-hosted |
| `fonts/OFL.txt` | the fonts' licence text; it goes out with the fonts |
| `data/demo.json` | built by `build_data.py` |
| `data/results_extra.json` | test set and huge-model results, built by `build_results.py` |
| `data/scenes.json` | the control room's six scenes, built by `build_scenes.py` |
| `build_data.py` | builds `demo.json` |
| `build_results.py` | builds `results_extra.json` |
| `build_scenes.py` | builds `scenes.json` |

## Publish list

Publish exactly these five entries and nothing else:

    index.html
    css/
    js/
    fonts/
    data/

Everything else in this folder is a working file (`README.md`, `build_data.py`, `build_results.py`, `build_scenes.py`). To make a clean copy to upload, run from this folder:

    rm -rf ../demo-publish && mkdir ../demo-publish && cp -R index.html css js fonts data ../demo-publish/

and upload the contents of `../demo-publish/`.

`.github/workflows/pages.yml` publishes exactly this list to GitHub Pages. It starts by hand only: turn Pages on
for the repository with GitHub Actions as the source, then run `gh workflow run pages.yml`.

## `data/results_extra.json`

Built by `build_results.py`; do not edit by hand. Two keys: `test` (whole test set: the systems,
the single-model repeats, the range alarm, attacks found, start times and the constant guess, the test rows answered again with shuffled seconds, plus
the check-set scores for the drop from check to test) and `big_model` (the 811 shared windows, the
thinking side run, cost rows; the cost chart leaves out the thinking row, which the page text no longer covers). Static text binds to it with `data-x="path.to.value"`, as it binds
to `demo.json` with `data-k`. If the file is missing or empty the page shows a short notice and the
static text keeps its numbers.

## The write-up

The second view. Its text is the approved copy, word for word; `js/app.js` only binds numbers and draws
charts. Structure (each part is a `<section class="part">`, numbered by a CSS counter):

| Anchor | Part | Charts and tables |
|---|---|---|
| `#top` | title, summary, links to the five parts | |
| `#data` | 1 Data and attacks | the example window (`#window-chart`); the three sets |
| `#baseline` | 2 Baseline | check-set score of the range alarm and the two gradient boosting versions (`#baseline-chart`) |
| `#training` | 3 Training | curves and hints in, answer out (`#flow-chart`); the 64 runs (`#night-chart`, with its table); the two designs; check-set scores of the vote, its single repeats and gradient boosting (`#score-chart`) |
| `#results` | 4 Results | test-set scores (`#test-chart`) and their table; attacks found (`#found-chart`); start within 30 seconds (`#start-chart`); `#kimi`: caught and false alarms on the 811 windows (`#big-chart`) and cost (`#cost-chart`) |
| `#conclusions` | 5 Conclusions and limits | seven models voting against gradient boosting (table, differences computed on the page); `#limits`: the curve check (`#reads-chart`); then the credits |

Charts draw only while the write-up is visible, and redraw when it is shown and on resize. The control
room does not wait for `demo.json`: `js/console.js` loads its own data.

Old anchors still work (`js/views.js`, `ALIAS`): `#test` opens `#results`, `#big` opens `#kimi`,
`#reads` and `#limits` open `#limits`, `#score` and `#night` open `#training`, `#conclusion` opens
`#conclusions`, and `#try` opens the demo. The hash stays as typed.

The window explorer ("Pick a window"), with its `#w=` deep link and curve-check switch, is gone: the control
room replaces it. `demo.json` still carries its 16 windows; the write-up uses one of them (`hero`).

Bindings: `data-k="path"` reads `demo.json`, `data-x="path"` reads `results_extra.json`; a `data-k` path
that starts with `hero.` reads the example window. On the same element, `data-minus="path"` shows the
difference (part 5's "6 fewer", "2 fewer", "4 behind", "3 points"), `data-of="path"` a whole percentage
(part 4's 63%, 61%, 57%, 66%) and `data-round` a rounded value (the means 47 and 45). Every static fallback
equals its bound value; `demo/.review/wu_check.cjs` checks this.

## The control room

The default view replays six recorded moments of the check set. Nothing runs in the browser. Each scene
has four steps: watch (the clock runs 0 to 300 seconds; no verdict is shown), read (the six windows travel
into the central control), answer (the seven models' votes, the start second, one model's sentence and the
hint lines) and check (the plant log and gradient boosting's call). Space plays and pauses; the arrow keys
scrub 10 seconds (30 with Shift) during the watch and step through the later steps.

Deep links, for review. Each opens the scene paused at the end of that step:

    ?scene=1&step=check
    ?scene=4&step=verdict
    ?scene=2&step=watch&t=180             (t: the second, watch only)
    ?autoplay=0                           (no autoplay; the page starts paused at second 0)

Scenes are numbered 1 to 6, so an address does not give the outcome away. The older keys still work: `caught`, `two_loops`, `one_missed`, `missed`, `false_alarm`, `quiet`. Steps: `watch`,
`read`, `verdict`, `check`. Without a step the first scene plays on load, unless the reader prefers
reduced motion; then it shows the whole window and two buttons, "Show the model's answer" and "Check
against the plant log". Choosing a scene or a step writes `scene` and `step` into the address with
`replaceState`; autoplay never changes the address.

Where each control-room number comes from (all in `data/scenes.json` unless named):

| On screen | Field |
|---|---|
| curves, readouts, small drawings | `scenes[].loops[].series[].values` (the measurement drives the readout and the drawing) |
| lane scale | `loops[].series[].usual_span_p90`, see below |
| hint chips and "Why" lines | `scenes[].loops[].hints` (the chip is the "N times" figure, or "flat") |
| seven marks, "k of 7" | `scenes[].loops[].voters[].label`, `votes_attack` |
| attack / normal per loop, "4 of 7" | `vote`, `votes_needed`, `voters` |
| "began near second N", model marker | `vote_start` |
| "One copy wrote" | `vote_example.text` |
| plant log, log marker | `kind`, `attack_start` |
| "N seconds late / early" | `vote_start` − `attack_start`, computed on the page |
| gradient boosting's call | `gb` |
| scene date and time | `window_start` (plant local time) |
| test-set line | `data/results_extra.json`: `test.systems` (gb, vote3, vote7) and `test.perfect` |

Scale rule, the same for every loop and scene: each curve is centred on the mean of its first 30 seconds,
and its lane spans the larger of the window's own extent and 3 times `usual_span_p90` (the span 9 in 10
check-set windows of that curve stay within). The page uses the 90th percentile, not the median
`usual_span`: in version 21.03 the turbine command moves about 80 times its pooled median in normal
windows, so the median would make the quiet scene look like a swing. A curve that never moves is drawn
flat in the middle of its lane.

Dataset version 21.03 records no target for the turbine-speed loop, so in the three 21.03 scenes
(`two_loops`, `false_alarm`, `quiet`) that instrument has two lanes, not three. The page matches each
curve to its scale by role and name.

## Where the numbers come from

Every number in the write-up that exists in `demo.json` or `results_extra.json` is bound to it (see "The
write-up" above): every result, every chart, every table cell. The numbers below exist in neither file, so
they stay static text. Each was checked against the file named.

The data (part 1 and the baseline):

| On the page | Source |
|---|---|
| 198 staged attacks | `REPORT.md`, section 2: 38 + 50 + 58 + 52 attacks in the four releases |
| about 620 hours | `REPORT.md`, section 1: 209.5 + 171.7 + 126.3 + 114.0 hours |
| 196 change a target or command; 72 also freeze the measurement | `data/frozen/attacks.csv`, column `scenario`: an `SP`, `CO` or `CV` part in 196 attacks; a `PV` part as well in 72 of them (the other two, 23.05 A235 and A236, are `P1-TC-PV_FUNCTION` only) |
| half last under three minutes | `data/frozen/attacks.csv`, column `duration_s`: median 163.5 s, 108 of 198 under 180 s |
| 2% of the recording | `REPORT.md`, section 1: 49,888 attack seconds of 2,237,407 |
| agreed for only 49 of the 198 | `REPORT.md`, section 2, column `ok_as_printed`: 1 + 48 + 0 + 0 |
| 3,880, a sample of 18,636; 776 attack windows | `REPORT.md`, section 4 (training sample) |
| 300 seconds | `REPORT.md`, section 4 ("Windows are 300 s") |
| seven numbers per curve | `src/hai/baselines.py`, `N_STATS = 7` |
| gradient boosting trains in 5 seconds | `runs/final/FACTS.md`, "Cost and speed" |
| always answering "normal" would score 0 | `runs/baselines/repaired_validation/scores.json`, `always_normal`: nothing found, no false alarms |

Training (part 3), all from `runs/nebius/NIGHT-SUMMARY.md` unless named:

| On the page | Source |
|---|---|
| "its largest one-second step is 5.8 times its usual one, at second 243" | the example window's hint line (`demo.json`, `windows[hero].hints`), quoted as text |
| every training window four times | the run table, passes column 4 (`--epochs 4`) |
| about 67 minutes, one eighth of a rented card, under 1 dollar | `runs/final/FACTS.md`, "Cost and speed" |
| without added hints average 56.5 | "Text clauses beat no clauses" |
| over five runs; over four runs (50 to 60) | the run counts behind `score.start_of_night` and `night.designs[11]` |
| after every pass in 8 runs; in all 8 the last pass scored best | "Keep the last pass, not the lowest-loss pass" |
| up to 23 points | "Single runs of one design differ by up to 23 points" |
| eight repeats | the count behind `score.voting_design_single` |

Results and limits (parts 4 and 5):

| On the page | Source |
|---|---|
| 256 stretches; 29 differ, 16 and 13 | `runs/final/gap_check.json`, `disagree` (gradient boosting against vote of three) |
| 20,000 rebuilds; between −6 and +12; level or ahead in about a third | `runs/final/gap_check.json`: `draws`, `pairs` (`low_90`, `high_90`, `share_b_at_least_a` 0.32) |
| four recordings | `runs/final/gap_check.json`, `recording_files` |
| the other three-model votes score 52, 48 and 49 | `runs/final/FACTS.md`, "How much chance could move the gap" |
| one simpler model goes from 99 to 72 | the curve check ("blank curves take hybrid + law from 99 attack calls to 72") |
| missed 44; 36 in two groups; 24 water level; 12 boiler temperature; gradient boosting finds 9 | "Most of the remaining misses were out of reach" |
| a third to a half of the normal windows | `runs/final/FACTS.md`: 259 and 362 of 729 |
| within 30 seconds; per 1,000 windows | definitions (`runs/final/final_score.py`; the cost unit) |
| trains in 5 seconds (conclusion 1) | `runs/final/FACTS.md` |
| 1 billion parameters (credits) | Llama 3.2 1B |

The curve check (part 5) counts the training machine's answers (95 attack calls), as the night summary
does; the re-run on another computer differs on 1 of 540 windows (96).

## Credits

HAI dataset, https://github.com/icsdataset/hai, CC BY-SA 4.0 (attribution sits under every figure
that shows HAI data, and in the control room's footer). OpenTSLM. Built with Llama. TimeNet hackathon.
Geist and Geist Mono fonts (github.com/vercel/geist-font), SIL Open Font License.
