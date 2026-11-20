"""Kaggle run 0: which number format and batch size fit this GPU, and how fast is each?

Short training runs on the same 16 training rows with the same seed. The float32 run is the
reference: a setting is trusted when its losses follow the float32 losses. A run that fails
(out of memory) is recorded and the next one starts.

Saves /kaggle/working/results/probe.json and prints a table.
"""
import glob
import json
import sys

sys.path.insert(0, glob.glob("/kaggle/input/**/kaggle_common.py", recursive=True)[0].rsplit("/", 1)[0])
import kaggle_common as k  # noqa: E402

ROWS = 16
PROBES = [  # number format, attention, batch size, epochs, loss scale (None = the default)
    ("float32", "eager", 2, 3, None),
    ("float16", "eager", 2, 3, None),
    ("float16", "eager", 2, 3, 1),      # loss scaling off: is it needed?
    ("float16", "sdpa", 2, 3, None),
    ("float16", "sdpa", 4, 2, None),
    ("float16", "sdpa", 8, 2, None),
    ("bfloat16", "eager", 2, 1, None),  # what OpenTSLM asks for
]


def main() -> None:
    paths = k.prepare()
    results = []
    for dtype, attention, batch_size, epochs, loss_scale in PROBES:
        k.say(f"\n===== {dtype}, {attention}, batch {batch_size}, {epochs} epoch(s), loss scale {loss_scale or 'default'} =====")
        results.append(k.train(paths, rows=ROWS, batch_size=batch_size, dtype=dtype, epochs=epochs, attention=attention,
                               loss_scale=loss_scale, max_new_tokens=16, check=False))
        results[-1]["loss_scale"] = loss_scale
        (k.RESULTS / "probe.json").write_text(json.dumps(
            {"version": paths.version, "gpu": paths.gpu, "rows": ROWS, "probes": results}, indent=2))

    k.say("\nformat    attention  batch  scale  exit  peak GB  s/row train  s/row val  overflow steps  train loss by epoch    val loss by epoch")
    for r in results:
        rate = r["seconds_per_row"]
        k.say(f"{r['dtype']:9s} {r['attention']:9s} {r['batch_size']:5d}  {str(r['loss_scale'] or '-'):>5}  {r['exit']:4d}  "
              f"{r['peak_gpu_memory_gb'] if r['peak_gpu_memory_gb'] is not None else '-':>7}  "
              f"{(rate['train'] or 0):11.3f}  {(rate['validation'] or 0):9.3f}  "
              f"{str(r['overflow_steps']) + ' of ' + str(r['train_steps']) if r['train_steps'] else '-':>14}  "
              f"{' '.join(f'{x:.3f}' for x in r['losses']['train']):20s}  "
              f"{' '.join(f'{x:.3f}' for x in r['losses']['validation'])}"
              f"{'  OUT OF MEMORY' if r['out_of_memory'] else ''}")


if __name__ == "__main__":
    main()
