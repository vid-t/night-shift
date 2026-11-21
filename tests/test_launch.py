"""Tests for training/launch.py and the log readers of training/kaggle. No GPU, no network."""
import importlib.util

import pytest

from tests.training_helpers import TRAINING, load_script

launch = load_script("launch")


def load_kaggle_common():
    spec = importlib.util.spec_from_file_location("kaggle_common", TRAINING / "kaggle" / "kaggle_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- settings ---------------------------------------------------------------

def test_a_missing_setting_is_none(monkeypatch):
    monkeypatch.delenv("HAI_EPOCHS", raising=False)
    assert launch.setting("HAI_EPOCHS", int, lambda v: v >= 1) is None


def test_a_setting_is_read_and_checked(monkeypatch):
    monkeypatch.setenv("HAI_EPOCHS", "3")
    assert launch.setting("HAI_EPOCHS", int, lambda v: v >= 1) == 3
    for bad in ("0", "three"):
        monkeypatch.setenv("HAI_EPOCHS", bad)
        with pytest.raises(SystemExit):
            launch.setting("HAI_EPOCHS", int, lambda v: v >= 1)
    monkeypatch.setenv("HAI_LLM_DTYPE", "float8")
    with pytest.raises(SystemExit):
        launch.setting("HAI_LLM_DTYPE", str, lambda v: v in launch.DTYPES)


# --- tokenizer padding -------------------------------------------------------

def test_a_left_padding_tokenizer_is_made_to_pad_on_the_right():
    class Tokenizer:
        def __init__(self, side, pad):
            self.padding_side, self.pad_token, self.eos_token = side, pad, "<eos>"

    class Loader:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return Tokenizer("left", "<pad>") if name == "open copy" else Tokenizer("right", None)

    launch.pad_like_opentslm_expects(Loader)
    for name in ("open copy", "meta copy"):
        tokenizer = Loader.from_pretrained(name, use_fast=True)
        assert tokenizer.padding_side == "right" and tokenizer.pad_token == "<eos>"


def test_left_padding_would_hand_opentslm_padding_for_the_series_labels():
    """Why the fix exists: OpenTSLM keeps the first n positions of each padded text piece."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B", local_files_only=True)
    except Exception:
        pytest.skip("unsloth/Llama-3.2-1B is not in the local Hugging Face cache")
    texts = ["You are given 300 seconds of a control loop, one reading per second.", "This is the target:", "Answer:"]

    def kept(side):
        tokenizer.padding_side = side
        tokens = tokenizer(texts, return_tensors="pt", padding=True)
        n = int(tokens.attention_mask[1].sum())
        return tokenizer.decode(tokens.input_ids[1, :n], skip_special_tokens=True)

    assert kept("right") == "This is the target:"
    assert kept("left") == ""   # all padding: the label is gone


# --- greedy answers, adapters before the previous stage -----------------------

def test_answers_are_decoded_greedily_unless_asked_otherwise():
    class Toy:
        def generate(self, batch, max_new_tokens=50, **kwargs):
            return kwargs

    launch.decode_greedy(Toy)
    assert Toy().generate([]) == {"do_sample": False}
    assert Toy().generate([], do_sample=True) == {"do_sample": True}
    launch.cap_answers(64, Toy)                       # both wrappers together
    assert Toy().generate([], max_new_tokens=2000) == {"do_sample": False}


def test_adapters_are_switched_on_before_the_previous_stage_loads():
    calls = []

    class Trainer:
        def _enable_lora_if_needed(self, stage):
            calls.append(("adapters on", stage))

        def _load_previous_stage_model(self, current_stage):
            calls.append(("load previous", current_stage))
            return {"ok": True}

    module = type("cl", (), {"CurriculumTrainer": Trainer})
    launch.keep_previous_adapters(module)
    assert Trainer()._load_previous_stage_model("stage6_hai") == {"ok": True}
    assert calls == [("adapters on", "stage6_hai"), ("load previous", "stage6_hai")]
    calls.clear()
    Trainer()._load_previous_stage_model("stage3_cot")   # OpenTSLM's own stages are left alone
    assert calls == [("load previous", "stage3_cot")]


def test_the_encoder_learning_rate_of_stage6_hai_is_set():
    class Trainer:
        def _train_stage(self, **kwargs):
            return kwargs

    module = type("cl", (), {"CurriculumTrainer": Trainer})
    launch.set_encoder_lr(module, 5e-5)
    assert Trainer()._train_stage(stage_name="stage6_hai", lr_encoder=2e-4)["lr_encoder"] == 5e-5
    assert Trainer()._train_stage(stage_name="stage3_cot", lr_encoder=2e-4)["lr_encoder"] == 2e-4
    assert Trainer()._train_stage(stage_name="stage6_hai") == {"stage_name": "stage6_hai"}
    launch.set_epochs(module, 3)                       # both wrappers together
    got = Trainer()._train_stage(stage_name="stage6_hai", lr_encoder=2e-4, num_epochs=10, lr_projector=1e-4)
    assert got == {"stage_name": "stage6_hai", "lr_encoder": 5e-5, "num_epochs": 3, "lr_projector": 1e-4}


# --- answer cap -------------------------------------------------------------

# --- encoder check -----------------------------------------------------------

def _toy_reader(torch, alive: bool):
    """An encoder and a projector with the shapes launch.encoder_response expects. `alive=False` ignores the curve."""
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):  # [curves, seconds] -> [curves, patches, 2]
            patches = x.reshape(x.size(0), -1, 4).mean(dim=-1, keepdim=True)
            signal = patches if alive else torch.zeros_like(patches)
            return torch.cat([signal * self.scale, torch.ones_like(patches)], dim=-1)

    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.projector = Encoder(), torch.nn.Linear(2, 3)

    torch.manual_seed(0)
    return Reader()


def test_encoder_response_tells_a_working_encoder_from_a_collapsed_one():
    torch = pytest.importorskip("torch")
    assert launch.encoder_response(_toy_reader(torch, alive=True)) > 0.05
    assert launch.encoder_response(_toy_reader(torch, alive=False)) < launch.ENCODER_DEAD_BELOW


def test_encoder_response_leaves_the_training_mode_as_it_was():
    torch = pytest.importorskip("torch")
    reader = _toy_reader(torch, alive=True)
    reader.train()
    launch.encoder_response(reader)
    assert reader.encoder.training and reader.projector.training
    reader.eval()
    launch.encoder_response(reader)
    assert not reader.encoder.training and not reader.projector.training


def test_a_starting_checkpoint_with_a_dead_encoder_is_refused(capsys):
    """The published end-of-curriculum checkpoint is one: two full runs started from it and never read a curve."""
    torch = pytest.importorskip("torch")

    class Trainer:
        def __init__(self, alive):
            self.model, self.saved = _toy_reader(torch, alive), []

        def _get_model(self):
            return self.model

        def _load_previous_stage_model(self, current_stage):
            return {"stage": "before"}

        def _save_loss_history(self, stage, epoch, train_loss, val_loss):
            self.saved.append(epoch)

        def _evaluate_stage(self, name):
            return {"evaluated": name}

    class cl:
        CurriculumTrainer = Trainer

    launch.watch_encoder(cl)
    good = Trainer(alive=True)
    assert good._load_previous_stage_model("stage6_hai") == {"stage": "before"}
    good._save_loss_history("stage6_hai", 2, 0.1, 0.2)
    assert good.saved == [2] and good._evaluate_stage("stage6_hai") == {"evaluated": "stage6_hai"}
    out = capsys.readouterr().out
    assert "at the start" in out and "after epoch 2" in out and "that answers" in out and "DEAD" not in out

    dead = Trainer(alive=False)
    assert dead._load_previous_stage_model("stage3_cot") == {"stage": "before"}  # only stage6_hai is checked
    with pytest.raises(RuntimeError, match="same output for every curve"):
        dead._load_previous_stage_model("stage6_hai")
    dead._save_loss_history("stage6_hai", 3, 0.1, 0.2)  # a collapse while training is reported, not raised
    assert dead.saved == [3] and "DEAD" in capsys.readouterr().out


# --- the model of every pass ----------------------------------------------------

def _pass_trainer(torch, results_dir):
    """A trainer with what keep_passes and watch_encoder use: a model, the three wrapped methods, results_dir."""
    class Model:
        def __init__(self):
            torch.manual_seed(0)
            self.encoder, self.projector = torch.nn.Linear(300, 4), torch.nn.Linear(4, 3)

        def save_lora_state_to_checkpoint(self, checkpoint):
            checkpoint["lora_state"] = {"a": torch.ones(2)}
            checkpoint["lora_enabled"] = True

    class Trainer:
        def __init__(self):
            self.results_dir, self.model, self.calls = str(results_dir), Model(), []

        def _get_model(self):
            return self.model

        def _load_previous_stage_model(self, current_stage):
            return {"stage": "before"}

        def _save_loss_history(self, stage, epoch, train_loss, val_loss):
            self.calls.append((stage, epoch, train_loss, val_loss))

        def _evaluate_stage(self, name):
            return {"evaluated": name}

    return type("cl", (), {"CurriculumTrainer": Trainer})


@pytest.mark.parametrize("watched", ["none", "before", "after"])
def test_the_model_of_every_pass_of_stage6_hai_is_kept(tmp_path, capsys, watched):
    torch = pytest.importorskip("torch")
    cl = _pass_trainer(torch, tmp_path)
    if watched == "before":
        launch.watch_encoder(cl)
    launch.keep_passes(cl)
    if watched == "after":
        launch.watch_encoder(cl)
    trainer = cl.CurriculumTrainer()
    trainer._save_loss_history("stage6_hai", 1, 0.9, 0.8)
    trainer._save_loss_history("stage6_hai", 2, 0.7, 0.6)
    trainer._save_loss_history("stage3_cot", 3, 0.5, 0.4)
    assert trainer.calls == [("stage6_hai", 1, 0.9, 0.8), ("stage6_hai", 2, 0.7, 0.6), ("stage3_cot", 3, 0.5, 0.4)]
    folder = tmp_path / "stage6_hai" / "checkpoints"
    assert sorted(f.name for f in folder.iterdir()) == ["pass_1.pt", "pass_2.pt"]
    assert not (tmp_path / "stage3_cot").exists()
    for epoch, val_loss in ((1, 0.8), (2, 0.6)):
        saved = torch.load(folder / f"pass_{epoch}.pt", weights_only=False)
        assert set(saved) == {"encoder_state", "projector_state", "val_loss", "epoch", "lora_state", "lora_enabled"}
        assert saved["epoch"] == epoch and saved["val_loss"] == val_loss
        assert torch.equal(saved["encoder_state"]["weight"], trainer.model.encoder.weight)
        assert set(saved["projector_state"]) == {"weight", "bias"}
    out = capsys.readouterr().out
    assert f"[launch] kept the model of pass 2: {folder / 'pass_2.pt'}" in out
    assert ("after epoch 2" in out) == (watched != "none")


def test_answer_cap_only_lowers_the_limit():
    class Toy:
        def generate(self, batch, max_new_tokens=50, **kwargs):
            return max_new_tokens

    launch.cap_answers(64, Toy)
    assert Toy().generate([], max_new_tokens=2000) == 64
    assert Toy().generate([]) == 64
    assert Toy().generate([], max_new_tokens=10) == 10


# --- loss scaling -----------------------------------------------------------

def test_loss_scaling_leaves_loss_and_gradients_unchanged():
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.a, self.b = torch.nn.Linear(4, 3), torch.nn.Linear(3, 1)

        def compute_loss(self, batch):
            return self.b(torch.tanh(self.a(batch))).pow(2).mean()

    torch.manual_seed(1)
    x = torch.randn(8, 4)
    plain = Toy()
    plain_loss = plain.compute_loss(x)
    plain_loss.backward()

    launch.scale_loss(4096.0, Toy)
    scaled = Toy()
    scaled_loss = scaled.compute_loss(x)
    scaled_loss.backward()
    assert scaled_loss.item() == pytest.approx(plain_loss.item())
    for p, q in zip(plain.parameters(), scaled.parameters()):
        assert torch.allclose(p.grad, q.grad, rtol=1e-5, atol=1e-8)
    with torch.no_grad():  # validation: no gradient, the loss passes through
        assert scaled.compute_loss(x).item() == pytest.approx(plain_loss.item())


def test_loss_scaling_saves_a_gradient_that_float16_would_round_to_zero():
    torch = pytest.importorskip("torch")

    def make():
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(4))

            def compute_loss(self, batch):
                hidden = self.w.half() * batch.half()   # the float16 part of the network
                return (hidden.float() * 1e-9).sum()    # its gradient, 1e-9, is below what float16 can hold

        return Toy

    x = torch.full((4,), 2.0)
    plain = make()()
    plain.compute_loss(x).backward()
    assert torch.count_nonzero(plain.w.grad) == 0        # lost

    Toy = make()
    launch.scale_loss(4096.0, Toy)
    scaled = Toy()
    scaled.compute_loss(x).backward()
    assert torch.allclose(scaled.w.grad, torch.full((4,), 2e-9), rtol=0.05)   # kept
    assert launch.SCALE["pending"] == 0


def test_loss_scaling_moves_trained_weights_to_float32():
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))

        def compute_loss(self, batch):
            return (self.w * batch).float().sum()

    launch.scale_loss(4096.0, Toy)
    toy = Toy()
    toy.compute_loss(torch.ones(4)).backward()
    assert toy.w.dtype == torch.float32 and toy.w.grad.dtype == torch.float32
    assert torch.allclose(toy.w.grad, torch.ones(4))


def test_loss_scale_halves_after_a_step_that_overflows():
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(4))

        def compute_loss(self, batch):
            hidden = self.w.half() * batch.half()
            return (hidden.float() * 100.0).sum()   # the gradient at the weight is 200: too large for float16 times 512

    launch.scale_loss(4096.0, Toy)
    toy, x = Toy(), torch.full((4,), 2.0)
    toy.compute_loss(x).backward()
    assert torch.count_nonzero(toy.w.grad) == 0 and launch.SCALE["pending"] == 1   # overflowed, set to zero
    for _ in range(4):   # 2048, 1024 and 512 overflow too (200 * 512 > 65504); 256 does not
        toy.w.grad = None
        toy.compute_loss(x).backward()
    assert launch.SCALE["in_use"] == 256.0 and launch.SCALE["bad_steps"] == 4 and launch.SCALE["pending"] == 0
    assert torch.allclose(toy.w.grad, torch.full((4,), 200.0))
    toy.w.grad = None
    toy.compute_loss(x).backward()   # stays at 256 once it fits
    assert launch.SCALE["value"] == 256.0 and torch.allclose(toy.w.grad, torch.full((4,), 200.0))


# --- log readers of the Kaggle scripts --------------------------------------

LOG = [
    "Epoch 1/3:  50%|█████     | 4/8 [00:10<00:10,  2.50s/it, loss=9.1, lr=1e-4]",
    "Epoch 1/3: 100%|██████████| 8/8 [00:20<00:00,  2.50s/it, loss=8.0, lr=1e-4]",
    "Epoch 1/3: 100%|██████████| 8/8 [00:20<00:00,  2.52s/it, loss=8.0, lr=1e-4]",
    "Validating stage6_hai: 100%|██████████| 16/16 [00:04<00:00,  4.00it/s]",
    "Validating stage6_hai: 100%|██████████| 16/16 [00:04<00:00,  3.98it/s]",
    "Epoch 1 — train loss: 9.5000",
    "Epoch 1 — val   loss: 7.2500",
    "Epoch 2/3: 100%|██████████| 8/8 [00:16<00:00,  2.00s/it, loss=6.0, lr=1e-4]",
    "Validating stage6_hai: 100%|██████████| 16/16 [00:04<00:00,  4.00it/s]",
    "Epoch 2 — train loss: 6.1250",
    "Epoch 2 — val   loss: nan",
    "Evaluating stage6_hai: 100%|█████████▉| 1088/1092 [01:01<00:00,  4.00it/s]",
    "Evaluating stage6_hai: 100%|██████████| 16/16 [1:02:03<00:00,  4.00it/s]",
]


def test_bar_seconds_counts_each_bar_once():
    k = load_kaggle_common()
    bars = k.bar_seconds(LOG)
    assert bars == {"Epoch": [20.0, 16.0], "Validating": [4.0, 4.0], "Evaluating": [3723.0]}


def test_losses_are_read_per_epoch():
    k = load_kaggle_common()
    got = k.losses(LOG)
    assert got["train"] == [9.5, 6.125]
    assert got["validation"][0] == 7.25 and got["validation"][1] != got["validation"][1]  # nan


def test_full_run_estimate():
    k = load_kaggle_common()
    assert k.full_run_estimate({"train": 1.0, "validation": None, "answer": 1.0}) is None
    rows = {"train": 3880, "validation": 1092, "test": 1536}
    got = k.full_run_estimate({"train": 0.36, "validation": 0.36, "answer": 3.6}, rows)
    assert got["hours_per_epoch"] == pytest.approx((3880 + 1092) * 0.36 / 3600, abs=0.01)
    assert got["hours_for_test_answers"] == pytest.approx(1536 * 3.6 / 3600, abs=0.01)
    assert k.full_run_estimate({"train": 0.36, "validation": 0.36, "answer": 3.6}) == got  # the default counts


def test_rows_are_counted_from_the_exported_files(tmp_path):
    k = load_kaggle_common()
    (tmp_path / "train.jsonl").write_text("{}\n{}\n\n{}\n")
    (tmp_path / "test.jsonl").write_text("{}\n")
    assert k.count_rows(tmp_path) == {"train": 3, "validation": k.FULL_ROWS["validation"], "test": 1}
