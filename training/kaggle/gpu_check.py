"""Name the GPU and time one matrix product in each number format.

OpenTSLM loads the language model in bfloat16. Newer GPUs (L4, A10G, RTX 30 and 40 series)
compute in it directly. Older ones (the T4) emulate it, slowly, and are fast in float16.

Prints a report, then one line of JSON. Run it with the Python that will train:

    uv run --project external/OpenTSLM python training/kaggle/gpu_check.py
"""
import json
import time

import torch

SIZE = 4096
REPEATS = 20


def milliseconds(dtype: torch.dtype) -> float:
    a = torch.randn(SIZE, SIZE, device="cuda").to(dtype)
    b = torch.randn(SIZE, SIZE, device="cuda").to(dtype)
    (a @ b).sum().item()  # warm up
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(REPEATS):
        c = a @ b
    torch.cuda.synchronize()
    del c
    return (time.time() - start) / REPEATS * 1000


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. Turn on the accelerator for this session.")
    name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    try:
        native = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # older torch: no such argument
        native = None
    times = {}
    for label in ("float32", "bfloat16", "float16"):
        try:
            times[label] = round(milliseconds(getattr(torch, label)), 2)
        except Exception as error:  # a format the GPU cannot run at all
            times[label] = None
            print(f"{label}: failed ({type(error).__name__}: {error})")
    print(f"GPU: {name}, {memory_gb:.1f} GB, {torch.cuda.device_count()} device(s), torch {torch.__version__}")
    print(f"bfloat16 native: {native}")
    for label, value in times.items():
        print(f"{label:9s} {value} ms per {SIZE} x {SIZE} matrix product")
    print(json.dumps({"gpu": name, "memory_gb": round(memory_gb, 1), "devices": torch.cuda.device_count(),
                      "torch": torch.__version__, "bfloat16_native": native, "matmul_ms": times}))


if __name__ == "__main__":
    main()
