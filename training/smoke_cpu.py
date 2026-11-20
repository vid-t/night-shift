"""CPU smoke test: the dataset class, stage6_hai and the seeded checkpoint load and run together.

One epoch of one training step on 4 examples, then one generation per example (capped at
MAX_NEW_TOKENS), through the patched CurriculumTrainer. Results go to ./results under the
current folder, so run it from a scratch folder, not from the OpenTSLM checkout:

    mkdir -p /tmp/hai-smoke && cd /tmp/hai-smoke
    HAI_JSONL_DIR=<repo>/data/processed/opentslm \
      uv run --project <repo>/external/OpenTSLM python <repo>/training/smoke_cpu.py <repo>/external/OpenTSLM unsloth/Llama-3.2-1B

It seeds ./results/<llm>/OpenTSLMSP/stage5_ecg_cot from the one setup.sh seeded in the checkout.
Never uses MPS. Proves wiring only, not quality.

The language model loads in float32 here. OpenTSLM asks for bfloat16, which a CPU runs hundreds
of times slower (436 times for a 1024x1024 matmul on the Apple M4 this was measured on), so one
training step does not finish. Only the number format changes; the wiring under test is the same.

    HAI_SMOKE_EPOCHS=12   run 12 one-step epochs on the same 4 rows instead of 1, to watch the loss fall
"""
import os
import shutil
import sys
import time

ROWS = 4
MAX_NEW_TOKENS = 64


def force_float32() -> None:
    """Make OpenTSLM load the language model in float32 instead of bfloat16. For CPU runs only."""
    import torch
    from transformers import AutoModelForCausalLM

    original = AutoModelForCausalLM.from_pretrained.__func__

    def float32(cls, *args, **kwargs):
        kwargs["torch_dtype"] = torch.float32
        return original(cls, *args, **kwargs)

    AutoModelForCausalLM.from_pretrained = classmethod(float32)


def main() -> None:
    opentslm_dir, llm_id = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "unsloth/Llama-3.2-1B"
    epochs = int(os.environ.get("HAI_SMOKE_EPOCHS", "1"))
    os.environ["HAI_OVERFIT_ROWS"] = str(ROWS)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.path.insert(0, os.path.abspath(opentslm_dir))
    force_float32()
    import curriculum_learning as cl
    from opentslm.time_series_datasets.hai.HAIAttackQADataset import HAIAttackQADataset

    trainer_cls = cl.CurriculumTrainer
    safe = trainer_cls._sanitize_llm_id(None, llm_id)
    seeded = os.path.join(opentslm_dir, "results", safe, "OpenTSLMSP", "stage5_ecg_cot")
    here = os.path.join("results", safe, "OpenTSLMSP", "stage5_ecg_cot")
    if not os.path.exists(os.path.join(here, "checkpoints", "best_model.pt")):
        shutil.copytree(seeded, here, dirs_exist_ok=True)

    t0 = time.time()
    trainer = trainer_cls("OpenTSLMSP", device="cpu", llm_id=llm_id)
    print(f"[smoke] model loaded in {time.time() - t0:.0f} s")
    model = trainer._get_model()
    generate = model.generate
    model.generate = lambda batch, max_new_tokens=MAX_NEW_TOKENS, **kw: generate(batch, max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS), **kw)

    t1 = time.time()
    metrics = trainer._train_stage(
        stage_name="stage6_hai", dataset_class=HAIAttackQADataset, num_epochs=epochs,
        lr_encoder=2e-4, lr_projector=1e-4, lr_base=2e-4, metric_func=None, batch_size=ROWS,
    )
    print(f"[smoke] {epochs} training step(s) + evaluation in {time.time() - t1:.0f} s; metrics {metrics}")
    print(f"[smoke] predictions: {os.path.join(trainer.results_dir, 'stage6_hai', 'results', 'test_predictions.jsonl')}")


if __name__ == "__main__":
    main()
