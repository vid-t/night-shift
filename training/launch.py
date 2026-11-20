"""Run OpenTSLM's curriculum_learning.py with optional settings from the environment.

Use it in place of curriculum_learning.py, with the same arguments, from the OpenTSLM checkout:

    uv run python ../../training/launch.py --model OpenTSLMSP --llm_id meta-llama/Llama-3.2-1B \
        --stages stage6_hai --batch_size 4 --device cuda

    HAI_LLM_DTYPE=float16     number format of the language model: bfloat16 (what OpenTSLM asks
                              for), float32 or float16. A T4 has no native bfloat16: float16 is
                              the fast format there. float16 turns loss scaling on.
    HAI_LOSS_SCALE=4096       loss scaling (default 4096 with float16, off otherwise): gradients
                              travel through the network this many times larger, so small ones do
                              not round to zero in float16, and are divided back at each trained
                              weight. Trained weights stay in float32. When a step overflows, the
                              scale halves for the next one.
    HAI_ATTENTION=sdpa        attention code of the language model: eager (what OpenTSLM asks for)
                              or sdpa (the same result with less memory)
    HAI_MAX_NEW_TOKENS=128    cap the length of an answer in evaluation. OpenTSLM allows 2000
                              tokens per answer; the longest HAI answer has about 60.
    HAI_EPOCHS=3              epochs of stage6_hai (default: the 10 of the stage)
    HAI_SEED=0                seed for the weights that start at random and the row order
    OPENTSLM_DIR=<path>       where curriculum_learning.py is (default: the current folder)

One thing it always does: it makes the tokenizer pad on the right with the end-of-text token.
OpenTSLM keeps the first n tokens of each padded text piece, so a tokenizer that pads on the
left (the open copy unsloth/Llama-3.2-1B does) hands it padding in place of every series label
and of the closing "Answer:". With meta-llama/Llama-3.2-1B this changes nothing.

Otherwise, with no setting, it behaves like curriculum_learning.py.
"""
import os
import sys

DTYPES = ("bfloat16", "float32", "float16")
ATTENTIONS = ("eager", "sdpa")
DEFAULT_LOSS_SCALE = 4096.0
# loss scaling: the scale for the next step, the one in use, and what went wrong so far
SCALE = {"value": 1.0, "in_use": 1.0, "steps": 0, "bad_steps": 0, "bad_tensors": 0, "pending": 0}


def force_load(dtype: str | None = None, attention: str | None = None) -> None:
    """Make every AutoModelForCausalLM.from_pretrained call use this number format and attention code."""
    import torch
    from transformers import AutoModelForCausalLM

    original = AutoModelForCausalLM.from_pretrained.__func__

    def forced(cls, *args, **kwargs):
        if dtype:
            kwargs["torch_dtype"] = getattr(torch, dtype)
        if attention:
            kwargs["attn_implementation"] = attention
        return original(cls, *args, **kwargs)

    AutoModelForCausalLM.from_pretrained = classmethod(forced)


def pad_like_opentslm_expects(tokenizer_class=None) -> None:
    """Make every tokenizer pad on the right with the end-of-text token, as OpenTSLM assumes."""
    if tokenizer_class is None:
        from transformers import AutoTokenizer as tokenizer_class

    original = tokenizer_class.from_pretrained.__func__

    def loaded(cls, *args, **kwargs):
        tokenizer = original(cls, *args, **kwargs)
        if tokenizer.padding_side != "right" or tokenizer.pad_token != tokenizer.eos_token:
            print(f"[launch] tokenizer padded on the {tokenizer.padding_side} with {tokenizer.pad_token!r}: "
                  f"now on the right with {tokenizer.eos_token!r}")
            tokenizer.padding_side = "right"
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    tokenizer_class.from_pretrained = classmethod(loaded)


def _model_class():
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

    return OpenTSLMSP


def cap_answers(limit: int, model_class=None) -> None:
    """Stop the model's generate after `limit` new tokens, whatever the caller asks for."""
    model_class = model_class or _model_class()
    original = model_class.generate

    def capped(self, batch, max_new_tokens=limit, **kwargs):
        return original(self, batch, max_new_tokens=min(max_new_tokens, limit), **kwargs)

    model_class.generate = capped


def settle_step() -> None:
    """Close the last training step: if a gradient overflowed in it, halve the scale for the next one."""
    if SCALE["pending"]:
        new = max(SCALE["value"] / 2, 1.0)
        print(f"[launch] step {SCALE['steps']}: {SCALE['pending']} gradients were not finite and were set to zero; "
              f"loss scale {SCALE['value']:g} -> {new:g}")
        SCALE["bad_steps"] += 1
        SCALE["bad_tensors"] += SCALE["pending"]
        SCALE["value"], SCALE["pending"] = new, 0


def scale_loss(scale: float, model_class=None) -> None:
    """Loss scaling without touching the training loop.

    The loss keeps its value. On the way back its gradient is multiplied by the scale, and every
    trained weight divides its own gradient by the scale again. So the loop, the gradient clipping
    and the optimizer see the true numbers, while the float16 part of the network works with
    numbers large enough to survive. A gradient that overflows anyway is set to zero for that
    step, and the scale halves for the next step.
    """
    import torch

    model_class = model_class or _model_class()
    SCALE.update(value=float(scale), in_use=float(scale), steps=0, bad_steps=0, bad_tensors=0, pending=0)

    class Scaled(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss):
            return loss.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad * SCALE["in_use"]

    def unscale(grad):
        grad = grad / SCALE["in_use"]
        if not torch.isfinite(grad).all():
            SCALE["pending"] += 1
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return grad

    hooked: set[int] = set()
    original = model_class.compute_loss

    def compute_loss(self, batch, *args, **kwargs):
        for p in self.parameters():
            if p.requires_grad and id(p) not in hooked:
                if p.dtype != torch.float32:
                    p.data = p.data.float()
                p.register_hook(unscale)
                hooked.add(id(p))
        loss = original(self, batch, *args, **kwargs)
        if not loss.requires_grad:  # validation
            return loss
        settle_step()
        SCALE["steps"] += 1
        SCALE["in_use"] = SCALE["value"]
        return Scaled.apply(loss)

    model_class.compute_loss = compute_loss


def set_epochs(cl, epochs: int) -> None:
    original = cl.CurriculumTrainer._train_stage

    def train_stage(self, *args, **kwargs):
        if kwargs.get("stage_name") == "stage6_hai" and "num_epochs" in kwargs:
            kwargs["num_epochs"] = epochs
        return original(self, *args, **kwargs)

    cl.CurriculumTrainer._train_stage = train_stage


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setting(name: str, kind, valid) -> object | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = kind(raw)
    except ValueError:
        sys.exit(f"{name}: cannot read {raw!r}")
    if not valid(value):
        sys.exit(f"{name}: {raw!r} is not allowed")
    return value


def main() -> None:
    checkout = os.path.abspath(os.environ.get("OPENTSLM_DIR", "."))
    script = os.path.join(checkout, "curriculum_learning.py")
    if not os.path.exists(script):
        sys.exit(f"no curriculum_learning.py in {checkout}: run from the OpenTSLM checkout or set OPENTSLM_DIR")

    dtype = setting("HAI_LLM_DTYPE", str, lambda v: v in DTYPES)
    attention = setting("HAI_ATTENTION", str, lambda v: v in ATTENTIONS)
    scale = setting("HAI_LOSS_SCALE", float, lambda v: v >= 1)
    limit = setting("HAI_MAX_NEW_TOKENS", int, lambda v: v >= 1)
    epochs = setting("HAI_EPOCHS", int, lambda v: v >= 1)
    seed = setting("HAI_SEED", int, lambda v: v >= 0)
    if scale is None and dtype == "float16":
        scale = DEFAULT_LOSS_SCALE

    sys.path.insert(0, checkout)
    sys.argv = [script, *sys.argv[1:]]
    import torch

    import curriculum_learning as cl

    pad_like_opentslm_expects()
    if dtype or attention:
        force_load(dtype, attention)
        print(f"[launch] language model loads in {dtype or 'bfloat16'} with {attention or 'eager'} attention")
    if scale and scale > 1:
        scale_loss(scale)
        print(f"[launch] loss scaling by {scale:g}")
    if limit:
        cap_answers(limit)
        print(f"[launch] answers stop after {limit} new tokens")
    if epochs:
        set_epochs(cl, epochs)
        print(f"[launch] stage6_hai runs {epochs} epoch(s)")
    if seed is not None:
        seed_everything(seed)
        print(f"[launch] seed {seed}")
    try:
        cl.main()
    finally:
        if torch.cuda.is_available():
            print(f"[launch] peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        if scale and scale > 1:
            settle_step()
            print(f"[launch] loss scale ended at {SCALE['value']:g}; steps with a gradient that was not finite: "
                  f"{SCALE['bad_steps']} of {SCALE['steps']}")


if __name__ == "__main__":
    main()
