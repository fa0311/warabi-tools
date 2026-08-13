import random

import pytest
import torch

from ime_context_tools.training.distillation import distillation_loss
from ime_context_tools.training.distill_cross_encoder import TrainPair, sample_list_indices
from ime_context_tools.training.student_configs import STUDENT_SPECS


def test_student_specs_param_counts() -> None:
    from transformers import DistilBertForSequenceClassification

    expected = {"lite": (4.8e6, 5.8e6), "standard": (15e6, 19e6), "max": (46e6, 52e6)}
    for tier, (low, high) in expected.items():
        model = DistilBertForSequenceClassification(STUDENT_SPECS[tier].config())
        params = sum(parameter.numel() for parameter in model.parameters())
        assert low < params < high, (tier, params)


def test_distillation_loss_prefers_teacher_agreement() -> None:
    teacher = torch.tensor([[5.0, 1.0, -1.0, 0.0]])
    targets = torch.tensor([0])
    aligned = distillation_loss(teacher.clone(), teacher, targets)
    shuffled = distillation_loss(teacher.flip(-1), teacher, targets)
    assert aligned < shuffled


def test_distillation_loss_mse_anchors_scale() -> None:
    teacher = torch.tensor([[4.0, 0.0, -4.0, 2.0]])
    targets = torch.tensor([0])
    matched = distillation_loss(teacher.clone(), teacher, targets)
    # same ordering and softmax shape direction, but scale compressed 4x
    compressed = distillation_loss(teacher / 4, teacher, targets)
    assert matched < compressed


def test_distillation_loss_skips_hard_term_when_unlabeled() -> None:
    student = torch.tensor([[1.0, 2.0]])
    teacher = torch.tensor([[1.0, 2.0]])
    loss = distillation_loss(student, teacher, torch.tensor([-1]))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def _pair(target_index, count, logits):
    row = {
        "context": "家に",
        "reading": "よみ",
        "target_index": target_index,
        "candidates": [{"text": str(i)} for i in range(count)],
    }
    record = {"candidate_indices": list(range(count)), "logits": logits}
    return TrainPair.from_row_record(row, record)


def test_sample_list_indices_hard_negatives_are_top_teacher() -> None:
    # teacher scores make indices 9, 8, 7 the hardest non-gold negatives
    pair = _pair(2, 10, [0.0, 1.0, 99.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    indices, target_position = sample_list_indices(
        pair, negatives=5, hard_negatives=3, rng=random.Random(0)
    )
    assert len(indices) == 6
    assert indices[target_position] == 2
    assert {9, 8, 7} <= set(indices)  # hard negatives always present
    assert all(0 <= index < 10 for index in indices)


def test_sample_list_indices_never_duplicates_target() -> None:
    pair = _pair(0, 6, [9.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    indices, target_position = sample_list_indices(
        pair, negatives=5, hard_negatives=2, rng=random.Random(1)
    )
    assert indices.count(0) == 1
    assert indices[target_position] == 0


def test_train_pair_tail_target_positions() -> None:
    row = {
        "context": "",
        "reading": "よみ",
        "target_index": 120,
        "candidates": [{"text": str(i)} for i in range(128)],
    }
    record = {"candidate_indices": [*range(97), 120], "logits": [float(i) for i in range(98)]}
    pair = TrainPair.from_row_record(row, record)
    assert pair.head_len == 97
    assert pair.tail_index == 120
    assert pair.candidate_count == 98
    assert pair.position_of(120) == 97
    assert pair.index_at(97) == 120
    indices, target_position = sample_list_indices(
        pair, negatives=31, hard_negatives=24, rng=random.Random(2)
    )
    assert indices[target_position] == 120


def test_train_pair_unlabeled_row() -> None:
    row = {"context": "", "reading": "よみ", "candidates": [{"text": str(i)} for i in range(40)]}
    record = {"candidate_indices": list(range(40)), "logits": [float(i) for i in range(40)]}
    pair = TrainPair.from_row_record(row, record)
    assert pair.target_index == -1
    indices, target_position = sample_list_indices(
        pair, negatives=31, hard_negatives=24, rng=random.Random(3)
    )
    assert target_position == -1
    assert len(indices) == 32


def test_encode_example_matches_rust_contract() -> None:
    from pathlib import Path

    from transformers import AutoTokenizer

    from ime_context_tools.training.distill_cross_encoder import encode_example

    tokenizer_dir = Path(__file__).parents[1] / "artifacts" / "tokenizers" / "ime-char-8k-v1"
    if not tokenizer_dir.is_dir():
        pytest.skip("tokenizer artifact not built")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    ids = encode_example(
        tokenizer, "家に \n", "かえる", "帰る", max_length=224, context_chars=160
    )
    assert ids[0] == 2  # [CLS]
    assert ids.count(3) == 3  # three [SEP]
    assert ids[-1] == 3
    assert len(ids) == 11  # 2 context (ws dropped) + 3 reading + 2 candidate + 4

    # truncation: context trimmed from the left, reading/candidate intact
    long_context = "あ" * 300
    ids = encode_example(
        tokenizer, long_context, "かえる", "帰る", max_length=32, context_chars=160
    )
    assert len(ids) == 32
    assert ids.count(3) == 3
