"""Build a teacher-initialized max student (tokenizer transfer).

The max tier shares the teacher's trunk shape (6L x 768 x 12H x 3072), so
every transformer layer, the embedding LayerNorm, and the classification head
copy verbatim. Only the vocabulary differs (32k subwords -> 8k chars):

  - position embeddings: first 256 rows of the teacher's 512 (sinusoidal)
  - char embeddings: if the teacher vocabulary contains the character as a
    single-piece token (`c` or `▁c`), copy that embedding; otherwise use the
    mean embedding of all teacher pieces containing the character; otherwise
    keep random init. Coverage is reported.

Output: a DistilBertForSequenceClassification checkpoint usable with
`distill_cross_encoder --init-from`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DistilBertForSequenceClassification

from .student_configs import STUDENT_SPECS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True, help="student vocab.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student", default="max", choices=sorted(STUDENT_SPECS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = STUDENT_SPECS[args.student]
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher, trust_remote_code=True
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if teacher.config.dim != spec.dim or teacher.config.n_layers != spec.n_layers:
        raise SystemExit(
            f"teacher trunk ({teacher.config.n_layers}L/{teacher.config.dim}) does not "
            f"match tier {spec.tier} ({spec.n_layers}L/{spec.dim}); trunk copy needs equality"
        )

    student = DistilBertForSequenceClassification(spec.config())
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()

    copied, adapted, skipped = [], [], []
    for name, tensor in student_state.items():
        source = teacher_state.get(name)
        if source is None:
            skipped.append(name)
            continue
        if source.shape == tensor.shape:
            student_state[name] = source.clone()
            copied.append(name)
        elif "position_embeddings" in name and source.shape[0] >= tensor.shape[0]:
            student_state[name] = source[: tensor.shape[0]].clone()
            adapted.append(name)
        elif "word_embeddings" not in name:
            skipped.append(name)

    # Character embedding transfer.
    vocab = args.vocab.read_text(encoding="utf-8").splitlines()
    teacher_embeddings = teacher_state["distilbert.embeddings.word_embeddings.weight"]
    piece_vocab = teacher_tokenizer.get_vocab()
    char_matrix = student_state["distilbert.embeddings.word_embeddings.weight"].clone()
    direct = fallback = missed = 0
    # Precompute piece -> chars map once for the mean-fallback path.
    contains: dict[str, list[int]] = {}
    for piece, index in piece_vocab.items():
        stripped = piece.lstrip("▁")
        if 1 <= len(stripped) <= 2:
            for char in set(stripped):
                contains.setdefault(char, []).append(index)
    for row, token in enumerate(vocab):
        if row < 5:  # specials keep random init except [PAD]/[CLS]/[SEP] copy below
            continue
        if len(token) != 1:
            continue
        piece_id = piece_vocab.get(token, piece_vocab.get("▁" + token))
        if piece_id is not None:
            char_matrix[row] = teacher_embeddings[piece_id]
            direct += 1
        elif token in contains:
            char_matrix[row] = teacher_embeddings[contains[token]].mean(dim=0)
            fallback += 1
        else:
            missed += 1
    # Specials: map by role where the teacher has equivalents.
    for student_row, teacher_token in ((0, "<pad>"), (2, "[CLS]"), (3, "[SEP]"), (4, "[MASK]")):
        piece_id = piece_vocab.get(teacher_token)
        if piece_id is not None:
            char_matrix[student_row] = teacher_embeddings[piece_id]
    student_state["distilbert.embeddings.word_embeddings.weight"] = char_matrix

    student.load_state_dict(student_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(args.output_dir, safe_serialization=True)
    report = {
        "trunk_copied_tensors": len(copied),
        "adapted_tensors": adapted,
        "skipped_tensors": skipped,
        "char_embeddings": {
            "direct": direct,
            "mean_fallback": fallback,
            "random": missed,
            "coverage": (direct + fallback) / max(1, len(vocab) - 5),
        },
    }
    (args.output_dir / "teacher_init_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["char_embeddings"], ensure_ascii=False))
    print(f"trunk tensors copied: {len(copied)}, skipped: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
