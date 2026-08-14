import json

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from ime_context_tools.training.build_char_tokenizer import SPECIAL_TOKENS, write_tokenizer
from ime_context_tools.training.extend_student_vocab import extended_vocab, main


def make_student(tmp_path, chars):
    vocab = SPECIAL_TOKENS + sorted(chars, key=ord)
    model_dir = tmp_path / "student"
    write_tokenizer(model_dir, vocab)
    config = DistilBertConfig(
        vocab_size=len(vocab), max_position_embeddings=256, n_layers=1, dim=32,
        n_heads=2, hidden_dim=64, num_labels=1, problem_type="regression",
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = DistilBertForSequenceClassification(config)
    model.save_pretrained(model_dir, safe_serialization=True)
    (model_dir / "ime_reranker.json").write_text(
        json.dumps({"schema_version": "2.0.0", "max_length": 224, "context_chars": 160}),
        encoding="utf-8",
    )
    return model_dir


def test_extended_vocab_is_codepoint_sorted_superset():
    old = SPECIAL_TOKENS + ["あ", "ん", "家"]
    new = extended_vocab(old, ["い", "🍣"])
    assert new[:5] == SPECIAL_TOKENS
    body = new[5:]
    assert body == sorted(body, key=ord)
    assert set(old[5:]) < set(body)


def test_extension_preserves_logits_and_pads_to_size(tmp_path):
    chars = list("あいうえおかきくけこ家にかえる帰蛙変別空白テスト")
    model_dir = make_student(tmp_path, set(chars))
    out_dir = tmp_path / "extended"
    assert main([
        "--model-dir", str(model_dir),
        "--additions", str(write_additions(tmp_path, ["🍣", "♡", "ヰ"])),
        "--output-dir", str(out_dir),
        "--vocab-size", "64",
    ]) == 0  # the parity gate inside main() enforces drift <= 1e-5
    vocab = (out_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    assert len(vocab) == 64
    assert {"🍣", "♡", "ヰ"} < set(vocab)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["vocab_size"] == 64


def test_rejects_multichar_additions(tmp_path):
    model_dir = make_student(tmp_path, set("あいう"))
    with pytest.raises(SystemExit, match="single characters"):
        main([
            "--model-dir", str(model_dir),
            "--additions", str(write_additions(tmp_path, ["ab"])),
            "--output-dir", str(tmp_path / "x"),
        ])


def write_additions(tmp_path, chars):
    path = tmp_path / "additions.txt"
    path.write_text("\n".join(chars) + "\n", encoding="utf-8", newline="\n")
    return path
