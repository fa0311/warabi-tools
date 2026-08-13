"""Train a student cross-encoder from scratch via knowledge distillation.

Data: candidate JSONL rows joined 1:1 (by line) with the teacher-logit
sidecars from `score_teacher_cross_encoder`. Lists of 32 (1 gold + 31
negatives; the hardest 24 negatives are the top teacher-scored non-gold
candidates, the rest sampled from the remaining sidecar head). Loss is
`distillation.distillation_loss` (CE + T^2*KL + MSE).

Input encoding is the student contract:
  [CLS] context [SEP] reading [SEP] candidate [SEP]
with the context pre-trimmed to `--context-chars` chars and then id-level
left-trimmed if the sequence still exceeds `--max-length` (reading and
candidate are never cut) — the same rule as `crates/ime-tokenizer`.

Checkpointing: best dev MRR (pure model logits), `save_pretrained` +
tokenizer artifacts + provisional `ime_reranker.json` (finalized by the
exporter after calibration).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from .distillation import distillation_loss
from .score_teacher_cross_encoder import iter_rows
from .student_configs import CONTEXT_CHARS, MAX_LENGTH, STUDENT_SPECS, TRAINING_DEFAULTS

CLS_ID, SEP_ID, PAD_ID = 2, 3, 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", choices=sorted(STUDENT_SPECS), required=True)
    parser.add_argument(
        "--init-from",
        type=Path,
        help="warm-restart: initialize weights from a previously trained "
        "checkpoint of the same tier instead of random init",
    )
    parser.add_argument(
        "--freeze-trunk",
        action="store_true",
        help="train only the word embeddings (stage A of tokenizer transfer: "
        "adapt the char frontend to a transplanted teacher trunk)",
    )
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument(
        "--train",
        type=Path,
        action="append",
        required=True,
        help="candidate train.jsonl (repeatable); each needs a sidecar in --teacher-logits-dir",
    )
    parser.add_argument("--dev", type=Path, action="append", required=True)
    parser.add_argument("--teacher-logits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--negatives", type=int, default=31)
    parser.add_argument("--hard-negatives", type=int, default=24)
    parser.add_argument("--hard-weight", type=float, default=0.2)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--mse-weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--context-chars", type=int, default=CONTEXT_CHARS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-dev", type=int)
    return parser


def splice_triple(
    context_ids: list[int],
    reading_ids: list[int],
    candidate_ids: list[int],
    *,
    max_length: int,
) -> list[int]:
    """`[CLS] context [SEP] reading [SEP] candidate [SEP]` with the context
    left-trimmed to fit. Mirrors `CharTokenizer::encode_triple` in Rust."""
    overhead = 4 + len(reading_ids) + len(candidate_ids)
    budget = min(max(max_length - overhead, 0), len(context_ids))
    context_ids = context_ids[len(context_ids) - budget :]
    return [CLS_ID, *context_ids, SEP_ID, *reading_ids, SEP_ID, *candidate_ids, SEP_ID]


def encode_example(
    tokenizer: Any,
    context: str,
    reading: str,
    candidate: str,
    *,
    max_length: int,
    context_chars: int,
) -> list[int]:
    return encode_row_lists(
        tokenizer,
        {"context": context, "reading": reading, "candidates": [{"text": candidate}]},
        [0],
        max_length=max_length,
        context_chars=context_chars,
    )[0]


def encode_row_lists(
    tokenizer: Any,
    row: dict[str, Any],
    indices: list[int],
    *,
    max_length: int,
    context_chars: int,
) -> list[list[int]]:
    """Sequences for several candidates of one row: one batched tokenizer
    call (context + reading + all candidates), then pure-Python splicing."""
    texts = [
        str(row["context"])[-context_chars:],
        str(row["reading"]),
        *(str(row["candidates"][index]["text"]) for index in indices),
    ]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    context_ids, reading_ids = encoded[0], encoded[1]
    return [
        splice_triple(context_ids, reading_ids, candidate_ids, max_length=max_length)
        for candidate_ids in encoded[2:]
    ]


def pad_batch(sequences: list[list[int]], device: torch.device) -> dict[str, torch.Tensor]:
    import numpy as np

    width = max(len(sequence) for sequence in sequences)
    input_ids = np.full((len(sequences), width), PAD_ID, dtype=np.int64)
    attention_mask = np.zeros((len(sequences), width), dtype=np.int64)
    for position, sequence in enumerate(sequences):
        input_ids[position, : len(sequence)] = sequence
        attention_mask[position, : len(sequence)] = 1
    return {
        "input_ids": torch.from_numpy(input_ids).to(device),
        "attention_mask": torch.from_numpy(attention_mask).to(device),
    }


def batch_encode(
    tokenizer: Any,
    row: dict[str, Any],
    indices: list[int],
    *,
    max_length: int,
    context_chars: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return pad_batch(
        encode_row_lists(
            tokenizer, row, indices, max_length=max_length, context_chars=context_chars
        ),
        device,
    )


@dataclass(slots=True)
class TrainPair:
    """Memory-compact training row.

    655k rows as raw JSON dicts cost ~45GB and pushed the machine into
    paging; this layout keeps the same information in ~1.5GB. Sidecar
    candidate indices are always a contiguous head plus at most one appended
    out-of-head target, so positions map in O(1) without stored index lists.
    """

    context: str
    reading: str
    target_index: int  # -1 = unlabeled KD-augmentation row
    head_len: int
    tail_index: int  # -1 = no appended out-of-head target
    texts_blob: str  # candidate texts joined by "\x00", index-aligned
    logits: np.ndarray  # float32, one per scored candidate

    @classmethod
    def from_row_record(cls, row: dict[str, Any], record: dict[str, Any]) -> "TrainPair":
        indices = record["candidate_indices"]
        head_len = len(indices)
        tail_index = -1
        if indices and indices[-1] != len(indices) - 1:
            head_len = len(indices) - 1
            tail_index = int(indices[-1])
        if indices[:head_len] != list(range(head_len)):
            raise ValueError("sidecar candidate_indices are not a contiguous head")
        target_value = row.get("target_index")
        return cls(
            context=str(row.get("context", "")),
            reading=str(row["reading"]),
            target_index=int(target_value) if target_value is not None else -1,
            head_len=head_len,
            tail_index=tail_index,
            texts_blob="\x00".join(
                str(candidate["text"]) for candidate in row["candidates"]
            ),
            logits=np.asarray(record["logits"], dtype=np.float32),
        )

    @property
    def candidate_count(self) -> int:
        return self.head_len + (self.tail_index >= 0)

    def index_at(self, position: int) -> int:
        return position if position < self.head_len else self.tail_index

    def position_of(self, index: int) -> int:
        if index < self.head_len:
            return index
        if index == self.tail_index:
            return self.head_len
        raise KeyError(index)


def load_training_pairs(train_files: list[Path], sidecar_dir: Path) -> list[TrainPair]:
    """Rows joined 1:1 with their sidecar records (line order), streamed
    straight into the compact representation."""
    from .score_teacher_cross_encoder import sidecar_path

    pairs: list[TrainPair] = []
    for source in train_files:
        sidecar = sidecar_path(sidecar_dir, source)
        with sidecar.open("rb") as sidecar_handle, source.open("rb") as source_handle:
            header = orjson.loads(sidecar_handle.readline())
            if header.get("kind") != "teacher_logits_header":
                raise SystemExit(f"{sidecar}: missing header")
            for row_line in source_handle:
                if not row_line.strip():
                    continue
                record_line = sidecar_handle.readline()
                if not record_line.strip():
                    raise SystemExit(
                        f"{sidecar}: fewer records than rows — "
                        "finish or --resume the teacher scoring first"
                    )
                row = orjson.loads(row_line)
                record = orjson.loads(record_line)
                if record.get("id") is not None and row.get("id") not in (None, record["id"]):
                    raise SystemExit(f"{sidecar}: id mismatch at row {row.get('id')}")
                pairs.append(TrainPair.from_row_record(row, record))
    return pairs


def sample_list_indices(
    pair: TrainPair,
    *,
    negatives: int,
    hard_negatives: int,
    rng: random.Random,
) -> tuple[list[int], int]:
    """Sampled candidate indices (shuffled) + position of the gold target.

    Unlabeled KD-augmentation rows (`target_index == -1`) sample one extra
    teacher-scored candidate instead of the gold and return position -1;
    the hard CE term skips them (`distillation_loss` masks negative targets).
    """
    target = pair.target_index if pair.target_index >= 0 else None
    order = np.argsort(-pair.logits, kind="stable")
    available = [
        index
        for position in order
        if (index := pair.index_at(int(position))) != target
    ]
    picks = negatives if target is not None else negatives + 1
    selected = available[:hard_negatives]
    remaining = available[hard_negatives:]
    random_count = min(picks - len(selected), len(remaining))
    selected.extend(rng.sample(remaining, random_count))
    indices = [target, *selected] if target is not None else selected
    rng.shuffle(indices)
    return indices, indices.index(target) if target is not None else -1


@torch.inference_mode()
def score_dev(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    context_chars: int,
    use_bfloat16: bool,
) -> dict[str, float]:
    model.eval()
    ranks: list[int] = []
    for row in tqdm(rows, desc="dev", unit="query"):
        indices = list(range(len(row["candidates"])))
        encoded = batch_encode(
            tokenizer,
            row,
            indices,
            max_length=max_length,
            context_chars=context_chars,
            device=device,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            scores = model(**encoded).logits.squeeze(-1).float()
        order = scores.argsort(descending=True).tolist()
        ranks.append(order.index(int(row["target_index"])) + 1)
    count = len(ranks)
    return {
        "top1": sum(rank == 1 for rank in ranks) / count,
        "top5": sum(rank <= 5 for rank in ranks) / count,
        "mrr": sum(1 / rank for rank in ranks) / count,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = STUDENT_SPECS[args.student]
    defaults = TRAINING_DEFAULTS[args.student]
    epochs = args.epochs or defaults["epochs"]
    learning_rate = args.learning_rate or defaults["learning_rate"]
    query_batch_size = args.query_batch_size or defaults["query_batch_size"]
    grad_accum = args.gradient_accumulation or defaults["grad_accum"]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the requested device")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if args.init_from:
        model = DistilBertForSequenceClassification.from_pretrained(args.init_from).to(device)
        if model.config.dim != spec.dim or model.config.n_layers != spec.n_layers:
            raise SystemExit(f"--init-from checkpoint shape does not match tier {spec.tier}")
    else:
        model = DistilBertForSequenceClassification(spec.config()).to(device)
    if args.freeze_trunk:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "word_embeddings" in name
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze-trunk: {trainable} trainable parameters (word embeddings only)")
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    train = load_training_pairs(args.train, args.teacher_logits_dir)
    # Uniform list width keeps every step a single (rows x width) forward;
    # rows too short to fill the list are rare and dropped.
    usable = [pair for pair in train if pair.candidate_count > args.negatives]
    if len(usable) < len(train):
        print(f"dropped {len(train) - len(usable)} rows with < {args.negatives + 1} candidates")
    train = usable
    dev = [row for path in args.dev for row in iter_rows(path)]
    if args.max_train is not None:
        train = train[: args.max_train]
    if args.max_dev is not None:
        dev = dev[: args.max_dev]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = -(-len(train) // query_batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        {
            "student": args.student,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train": len(train),
            "dev": len(dev),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "bfloat16": use_bfloat16,
        },
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_mrr = -1.0
    best_metrics: dict[str, float] = {}
    # Length-bucketed batch construction: the mix spans ~20-token lexical
    # rows to ~220-token note rows, and per-batch padding to the local max
    # wastes most of the compute when they land in the same batch. Sort by
    # context length, shuffle within same-length windows (fresh composition
    # per epoch), then shuffle the batch order for stochasticity.
    def build_batches(rng: random.Random) -> list[list[TrainPair]]:
        order = sorted(range(len(train)), key=lambda i: len(train[i].context))
        window = query_batch_size * 64
        for start in range(0, len(order), window):
            segment = order[start : start + window]
            rng.shuffle(segment)
            order[start : start + window] = segment
        batches = [
            [train[i] for i in order[start : start + query_batch_size]]
            for start in range(0, len(order), query_batch_size)
        ]
        rng.shuffle(batches)
        return batches

    for epoch in range(1, epochs + 1):
        rng = random.Random(args.seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = build_batches(rng)
        width = args.negatives + 1
        # Data prep (sampling + one batched tokenizer call per step) runs one
        # step ahead on a worker thread; the fast tokenizer releases the GIL,
        # so prep genuinely overlaps GPU compute. A single worker preserves
        # `rng` determinism (submissions execute in order).
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        def prepare_step(batch: list) -> tuple[list[list[int]], list[torch.Tensor], list[int]]:
            texts: list[str] = []
            spans: list[tuple[int, int]] = []
            sampled: list[tuple[TrainPair, list[int], int]] = []
            for pair in batch:
                indices, target_position = sample_list_indices(
                    pair,
                    negatives=args.negatives,
                    hard_negatives=args.hard_negatives,
                    rng=rng,
                )
                candidate_texts = pair.texts_blob.split("\x00")
                spans.append((len(texts), len(indices)))
                texts.append(pair.context[-args.context_chars :])
                texts.append(pair.reading)
                texts.extend(candidate_texts[index] for index in indices)
                sampled.append((pair, indices, target_position))
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            sequences: list[list[int]] = []
            teacher_lists: list[torch.Tensor] = []
            targets: list[int] = []
            for (pair, indices, target_position), (start, count) in zip(sampled, spans):
                context_ids, reading_ids = encoded[start], encoded[start + 1]
                sequences.extend(
                    splice_triple(
                        context_ids, reading_ids, candidate_ids, max_length=args.max_length
                    )
                    for candidate_ids in encoded[start + 2 : start + 2 + count]
                )
                positions = [pair.position_of(index) for index in indices]
                teacher_lists.append(torch.from_numpy(pair.logits[positions].copy()))
                targets.append(target_position)
            return sequences, teacher_lists, targets

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = deque(pool.submit(prepare_step, batch) for batch in batches[:2])
            progress = tqdm(total=len(batches), desc=f"epoch {epoch}/{epochs}", unit="batch")
            for step in range(1, len(batches) + 1):
                sequences, teacher_lists, targets = pending.popleft().result()
                if step + 1 < len(batches):
                    pending.append(pool.submit(prepare_step, batches[step + 1]))
                rows = len(teacher_lists)
                encoded = pad_batch(sequences, device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    flat = model(**encoded).logits.squeeze(-1).float()
                student_scores = flat.reshape(rows, width)
                teacher_scores = torch.stack(teacher_lists).to(device)
                loss = distillation_loss(
                    student_scores,
                    teacher_scores,
                    torch.tensor(targets, device=device),
                    hard_weight=args.hard_weight,
                    kl_weight=args.kl_weight,
                    mse_weight=args.mse_weight,
                    temperature=args.temperature,
                )
                (loss / grad_accum).backward()
                total_loss += loss.item() * rows
                if step % grad_accum == 0 or step == len(batches):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                progress.update(1)
            progress.close()

        metrics = score_dev(
            model,
            tokenizer,
            dev,
            device=device,
            max_length=args.max_length,
            context_chars=args.context_chars,
            use_bfloat16=use_bfloat16,
        )
        metrics["loss"] = total_loss / max(1, len(train))
        print({"epoch": epoch, "dev": metrics}, flush=True)
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_metrics = metrics
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            shutil.copy2(args.tokenizer_dir / "vocab.txt", args.output_dir / "vocab.txt")
            _write_metadata(args, spec, best_metrics)

    print({"best_dev": best_metrics, "output": str(args.output_dir)}, flush=True)
    return 0


def _write_metadata(args: argparse.Namespace, spec: Any, metrics: dict[str, float]) -> None:
    metadata = {
        "schema_version": "2.0.0",
        "architecture": "cross_encoder",
        "model_family": "distilbert",
        "tier": spec.tier,
        "input_template": "[CLS] context [SEP] reading [SEP] candidate [SEP]",
        "tokenizer": {
            "type": "character",
            "vocab_file": "vocab.txt",
            "normalization": "none",
            "whitespace_rule": "drop Unicode White_Space",
            "special_token_ids": {"pad": 0, "unk": 1, "cls": 2, "sep": 3, "mask": 4},
        },
        "truncation_order": "context left-trim first; reading/candidate never cut",
        "max_length": args.max_length,
        "context_chars": args.context_chars,
        "distillation": {
            "teacher_logits_dir": str(args.teacher_logits_dir),
            "hard_weight": args.hard_weight,
            "kl_weight": args.kl_weight,
            "mse_weight": args.mse_weight,
            "temperature": args.temperature,
        },
        "dev_metrics": metrics,
        "calibration_pending": True,
    }
    (args.output_dir / "ime_reranker.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
