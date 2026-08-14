"""Initialize a narrow student from a wide cascade teacher via PCA.

The teacher-transplant trick (copy the trunk) needs matching shapes; lite
and standard are narrower than max, so the transferable part is the input
geometry: PCA over max's char embeddings finds the `dim` directions that
carry the vocabulary's structure, and both word and position embeddings are
projected onto that shared basis (they add in input space, so one basis
preserves their sum structure). The trunk and head start from the standard
random init; the cascade KD loss carries the rest.

Output: an HF checkpoint directory usable as `ime-distill --init-from`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import DistilBertForSequenceClassification

from .student_configs import STUDENT_SPECS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, required=True, help="wide checkpoint (max)")
    parser.add_argument("--student", choices=sorted(STUDENT_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def pca_basis(embeddings: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-`dim` principal directions of the (centered) embedding rows."""
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    # V[:, :dim] spans the top principal subspace; project with X @ V.
    _, _, v = torch.linalg.svd(centered, full_matrices=False)
    return mean, v[:dim, :].T


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    teacher = DistilBertForSequenceClassification.from_pretrained(args.teacher).float()
    word = teacher.distilbert.embeddings.word_embeddings.weight.data
    position = teacher.distilbert.embeddings.position_embeddings.weight.data

    spec = STUDENT_SPECS[args.student]
    config = spec.config()
    # The vocabulary belongs to the teacher lineage, not the tier spec: a
    # vocab-extended teacher (v8: 16,384) must produce a matching student.
    config.vocab_size = word.shape[0]
    student = DistilBertForSequenceClassification(config)
    dim = spec.dim

    mean, basis = pca_basis(word, dim)
    projected_word = (word - mean) @ basis
    projected_position = position @ basis  # positions are offsets: no centering
    # Match the random-init scale the trunk expects (~initializer_range) so
    # the projected geometry does not blow past the LayerNorm operating point.
    scale = student.distilbert.embeddings.word_embeddings.weight.data.std() / (
        projected_word.std() + 1e-12
    )
    with torch.no_grad():
        student.distilbert.embeddings.word_embeddings.weight.copy_(projected_word * scale)
        max_positions = student.distilbert.embeddings.position_embeddings.weight.shape[0]
        student.distilbert.embeddings.position_embeddings.weight.copy_(
            projected_position[:max_positions] * scale
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(args.output_dir, safe_serialization=True)
    for name in ("vocab.txt", "tokenizer.json", "tokenizer_config.json"):
        source = args.teacher / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)

    explained = torch.linalg.svdvals(word - word.mean(dim=0))
    ratio = float((explained[:dim] ** 2).sum() / (explained**2).sum())
    report = {
        "student": args.student,
        "dim": dim,
        "teacher": str(args.teacher),
        "pca_explained_variance": round(ratio, 4),
        "embedding_scale": round(float(scale), 4),
    }
    (args.output_dir / "init_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
