"""Extend a trained student's character vocabulary without losing its weights.

Token ids are arbitrary to the model: permuting vocab.txt and the embedding
rows by the same permutation is an exact identity. This tool builds the new
vocabulary as (old chars ∪ additions), sorts the body by codepoint (specials
keep ids 0..4), permutes the word-embedding rows to follow, and initializes
each added row as an exact copy of the trained [UNK] embedding — so the
extended model is functionally IDENTICAL to the original on every input (an
added character scores precisely as [UNK] did), and continuation training
differentiates the copies afterwards. No symmetry-breaking noise is needed:
embedding rows only receive gradient from their own token's occurrences.

A verification gate re-runs the export probe cases through old and new model
and requires bitwise-close logits.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from .build_char_tokenizer import SPECIAL_TOKENS, is_token_char, write_tokenizer
from .export_reranker import PROBE_CASES

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--additions", type=Path, required=True, help="one character per line"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, help="pad the vocab to exactly this size")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_vocab(model_dir: Path) -> list[str]:
    vocab = (model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if vocab[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
        raise SystemExit(f"unexpected special tokens: {vocab[:5]}")
    return vocab


def extended_vocab(old: list[str], additions: list[str]) -> list[str]:
    body = set(old[len(SPECIAL_TOKENS) :])
    for char in additions:
        if len(char) != 1:
            raise SystemExit(f"additions must be single characters, got {char!r}")
        if not is_token_char(char):
            raise SystemExit(f"not a valid token char: U+{ord(char):04X}")
        body.add(char)
    return SPECIAL_TOKENS + sorted(body, key=ord)


@torch.inference_mode()
def probe_logits(model, tokenizer, metadata: dict) -> torch.Tensor:
    from .distill_cross_encoder import batch_encode

    model.eval()
    logits = []
    for case in PROBE_CASES:
        encoded = batch_encode(
            tokenizer,
            {"context": case["context"], "reading": case["reading"],
             "candidates": [{"text": case["candidate"]}]},
            [0],
            max_length=int(metadata["max_length"]),
            context_chars=int(metadata["context_chars"]),
            device=torch.device("cpu"),
        )
        logits.append(model(**encoded).logits.squeeze())
    return torch.stack(logits)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.resolve() == args.model_dir.resolve():
        raise SystemExit("--output-dir must differ from --model-dir")
    torch.manual_seed(args.seed)

    old_vocab = load_vocab(args.model_dir)
    additions = [
        line for line in args.additions.read_text(encoding="utf-8").splitlines() if line
    ]
    new_vocab = extended_vocab(old_vocab, additions)
    if args.vocab_size is not None:
        if len(new_vocab) > args.vocab_size:
            raise SystemExit(f"vocab {len(new_vocab)} exceeds --vocab-size {args.vocab_size}")
        # pad with unused private-use chars so the embedding matrix hits the
        # requested (power-of-two friendly) size exactly
        pua = (chr(cp) for cp in range(0xE000, 0xF8FF))
        taken = set(new_vocab)
        while len(new_vocab) < args.vocab_size:
            char = next(pua)
            if char not in taken:
                new_vocab.append(char)

    metadata = json.loads((args.model_dir / "ime_reranker.json").read_text(encoding="utf-8"))
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir).float()
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    reference = probe_logits(model, old_tokenizer, metadata)

    old_ids = {token: index for index, token in enumerate(old_vocab)}
    unk_row = model.distilbert.embeddings.word_embeddings.weight[old_ids["[UNK]"]]
    new_matrix = torch.empty(
        (len(new_vocab), unk_row.shape[0]), dtype=unk_row.dtype
    )
    added = 0
    for index, token in enumerate(new_vocab):
        source = old_ids.get(token)
        if source is not None:
            new_matrix[index] = model.distilbert.embeddings.word_embeddings.weight[source]
        else:
            new_matrix[index] = unk_row
            added += 1

    model.resize_token_embeddings(len(new_vocab))
    model.distilbert.embeddings.word_embeddings.weight.data.copy_(new_matrix)
    model.config.vocab_size = len(new_vocab)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    write_tokenizer(args.output_dir, new_vocab)
    (args.output_dir / "ime_reranker.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("tokenizer_config.json",):
        source = args.model_dir / name
        if source.is_file() and not (args.output_dir / name).is_file():
            shutil.copy2(source, args.output_dir / name)

    new_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    reloaded = DistilBertForSequenceClassification.from_pretrained(args.output_dir).float()
    drift = (probe_logits(reloaded, new_tokenizer, metadata) - reference).abs().max().item()
    report = {
        "output": str(args.output_dir),
        "vocab_size": len(new_vocab),
        "added_rows": added,
        "permutation_probe_drift": drift,
    }
    print(json.dumps(report, ensure_ascii=False))
    if drift > 1e-5:
        raise SystemExit(f"permutation drift {drift} — the identity was not preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
