"""Add stage `stage6_hai` to OpenTSLM's curriculum_learning.py.

    python add_stage.py path/to/OpenTSLM/curriculum_learning.py

Edits, each found by an exact anchor text from OpenTSLM commit 2968f4b:
  1. import HAIAttackQADataset
  2. append "stage6_hai" to CURRICULUM_STAGES
  3. a stage6_hai method, copied from stage1_mcq
  4. an `elif stage == "stage6_hai":` branch in run_curriculum
  5. "stage6_hai" in stages_with_lora, in BOTH definitions of _enable_lora_if_needed
  6. the example id in each test prediction, so score.py can join predictions to examples

Running it twice changes nothing. A missing anchor stops it with an error and the file untouched.
"""
import sys
from pathlib import Path

STAGE = "stage6_hai"

IMPORT_ANCHOR = "from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset\n"
IMPORT_LINE = "from opentslm.time_series_datasets.hai.HAIAttackQADataset import HAIAttackQADataset\n"

STAGES_ANCHOR = '    "stage5_ecg_cot",\n]\n'
STAGES_NEW = '    "stage5_ecg_cot",\n    "stage6_hai",\n]\n'

METHOD_ANCHOR = "    def run_curriculum(\n"
METHOD = '''    def stage6_hai(
        self, batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 6: HAI control-loop attacks (attack or normal, start second, one line of text).

        Copied from stage1_mcq. Starts from the stage before it (the seeded checkpoint).
        - Epochs: 10 (early stopping has a patience of EARLY_STOP_PAT)
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4, LoRA on
        - Metric: test loss only. The built-in accuracy counts an empty answer as
          correct; score test_predictions.jsonl with training/score.py in the repository.
        """
        return self._train_stage(
            stage_name="stage6_hai",
            dataset_class=HAIAttackQADataset,
            num_epochs=10,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            eval_only=eval_only,
        )

'''

BRANCH_ANCHOR = '''            else:
                if self.rank == 0:
                    print(f"⚠️  Unknown stage: {stage}, skipping...")
'''
BRANCH = '''            elif stage == "stage6_hai":
                stage_results = self.stage6_hai(
                    batch_size=batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage, stage_results)
'''

LORA_OLD = 'stages_with_lora = ["stage3_cot", "stage4_sleep_cot", "stage5_ecg_cot"]'
LORA_NEW = 'stages_with_lora = ["stage3_cot", "stage4_sleep_cot", "stage5_ecg_cot", "stage6_hai"]'
LORA_DEFINITIONS = 2  # _enable_lora_if_needed is defined twice; the second one wins, edit both

ID_ANCHOR = "                        results.append(result)\n"
ID_LINES = '''                        if stage == "stage6_hai" and "example_id" in sample:
                            result["example_id"] = sample["example_id"]
'''


class AnchorMissing(RuntimeError):
    pass


def _insert_before(src: str, anchor: str, new: str, name: str) -> str:
    if new in src:
        return src
    if src.count(anchor) != 1:
        raise AnchorMissing(f"{name}: expected the anchor once, found it {src.count(anchor)} times:\n{anchor}")
    return src.replace(anchor, new + anchor)


def _insert_after(src: str, anchor: str, new: str, name: str) -> str:
    if anchor + new in src:
        return src
    if src.count(anchor) != 1:
        raise AnchorMissing(f"{name}: expected the anchor once, found it {src.count(anchor)} times:\n{anchor}")
    return src.replace(anchor, anchor + new)


def _replace(src: str, old: str, new: str, count: int, name: str) -> str:
    if src.count(new) == count and old not in src:
        return src
    if src.count(old) != count:
        raise AnchorMissing(f"{name}: expected the anchor {count} times, found it {src.count(old)} times:\n{old}")
    return src.replace(old, new)


def patch(src: str) -> str:
    """Return the patched source. Raises AnchorMissing if the file does not look like the pinned commit."""
    src = _insert_after(src, IMPORT_ANCHOR, IMPORT_LINE, "import")
    src = _replace(src, STAGES_ANCHOR, STAGES_NEW, 1, "CURRICULUM_STAGES")
    src = _insert_before(src, METHOD_ANCHOR, METHOD, "stage method")
    src = _insert_before(src, BRANCH_ANCHOR, BRANCH, "run_curriculum branch")
    src = _replace(src, LORA_OLD, LORA_NEW, LORA_DEFINITIONS, "stages_with_lora")
    src = _insert_before(src, ID_ANCHOR, ID_LINES, "example id in predictions")
    return src


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(__doc__)
    path = Path(argv[0])
    src = path.read_text(encoding="utf-8")
    try:
        new = patch(src)
    except AnchorMissing as e:
        raise SystemExit(f"add_stage.py: {path} not patched. {e}")
    if new == src:
        print(f"{path}: {STAGE} already present, nothing to do")
        return
    path.write_text(new, encoding="utf-8")
    print(f"{path}: added {STAGE}")


if __name__ == "__main__":
    main()
