"""Cache teacher logits for knowledge distillation.

Scores candidate rows with the `ime-context-reranker-v1` cross-encoder teacher
and writes one sidecar JSONL per input file. Students train against these
cached logits (KL + MSE terms) without touching the teacher again.

Sidecar format (`<input stem>.teacher.jsonl` in --output-dir):
  line 1   header: schema, teacher path + weights sha256, head size,
           max_length/context_chars, source file
  line N+1 row N of the source file (1:1, in order):
           {"id": ..., "candidate_indices": [...], "logits": [...]}

`candidate_indices` is the contiguous head (up to --head) plus the target
index; students must sample negatives from this set only, which matches the
runtime shortlist. The teacher is queried in its own input format
(`文脈：{context}\\n読み：{reading}` / `候補：{candidate}`, MeCab tokenizer) —
the student's char-level format is unrelated by design.

`--verify` re-ranks rows by the cached logits and reports top1/top5/MRR, which
must reproduce the teacher's known dev metrics (top1 ~0.948 on its dev mix).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

SCHEMA_VERSION = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, required=True, help="teacher model directory")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="candidate JSONL file (train.jsonl/dev.jsonl); repeatable",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head", type=int, default=97, help="contiguous candidate head to score")
    parser.add_argument(
        "--rows-per-batch", type=int, default=4, help="rows fused into one forward"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="continue a partial sidecar")
    parser.add_argument("--max-rows", type=int, help="score only the first N rows (smoke runs)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="after scoring, re-rank rows from the sidecar and print top1/top5/MRR",
    )
    return parser


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield orjson.loads(line)


def select_indices(row: dict[str, Any], head: int) -> list[int]:
    """Contiguous head plus the gold target (when the row is labeled),
    preserving candidate order. Unlabeled KD-augmentation rows have no
    `target_index` and score the head only."""
    count = len(row["candidates"])
    indices = list(range(min(head, count)))
    target = row.get("target_index")
    if target is not None and int(target) >= len(indices):
        indices.append(int(target))
    return indices


def weights_sha256(teacher_dir: Path) -> str:
    return hashlib.sha256((teacher_dir / "model.safetensors").read_bytes()).hexdigest()


def sidecar_path(output_dir: Path, source: Path) -> Path:
    return output_dir / f"{source.parent.name}--{source.stem}.teacher.jsonl"


def make_header(args: argparse.Namespace, source: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "teacher_logits_header",
        "teacher": {
            "path": str(args.teacher),
            "weights_sha256": weights_sha256(args.teacher),
            "max_length": int(metadata["max_length"]),
            "context_chars": int(metadata["context_chars"]),
        },
        "head": args.head,
        "source": str(source),
    }


class TeacherScorer:
    """Thin wrapper around the HF teacher; one forward per row."""

    def __init__(self, teacher_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        metadata_path = teacher_dir / "ime_reranker.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Schema v2 models (our students) use the spliced-[SEP] input format
        # and a plain fast tokenizer; the original teacher needs its custom
        # MeCab tokenizer via trust_remote_code.
        self.student_format = str(self.metadata.get("schema_version", "")).startswith("2")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            teacher_dir, trust_remote_code=not self.student_format
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            teacher_dir, trust_remote_code=not self.student_format
        ).to(self.device)
        self.model.eval()
        self.max_length = int(self.metadata["max_length"])
        self.context_chars = int(self.metadata["context_chars"])
        self.use_bfloat16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    def encode_chunk(self, chunk: list[tuple[dict[str, Any], list[int]]]) -> Any:
        """Tokenize every pair of several rows in one call (CPU tensors)."""
        if self.student_format:
            return self._encode_chunk_student(chunk)
        queries: list[str] = []
        candidates: list[str] = []
        for row, indices in chunk:
            query = f"文脈：{row['context'][-self.context_chars :]}\n読み：{row['reading']}"
            queries.extend([query] * len(indices))
            candidates.extend(f"候補：{row['candidates'][index]['text']}" for index in indices)
        return self.tokenizer(
            queries,
            candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _encode_chunk_student(self, chunk: list[tuple[dict[str, Any], list[int]]]) -> Any:
        """Student-format models: `[CLS] context [SEP] reading [SEP] cand [SEP]`."""
        from .distill_cross_encoder import encode_row_lists

        torch = self.torch
        sequences: list[list[int]] = []
        for row, indices in chunk:
            sequences.extend(
                encode_row_lists(
                    self.tokenizer,
                    row,
                    indices,
                    max_length=self.max_length,
                    context_chars=self.context_chars,
                )
            )
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for position, sequence in enumerate(sequences):
            input_ids[position, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[position, : len(sequence)] = 1
        from transformers import BatchEncoding

        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    def score_encoded(self, encoded: Any, counts: list[int]) -> list[list[float]]:
        """One forward for a pre-tokenized chunk, logits split back per row."""
        torch = self.torch
        encoded = encoded.to(self.device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_bfloat16,
            ),
        ):
            flat = self.model(**encoded).logits.squeeze(-1).float().cpu()
        result = []
        offset = 0
        for count in counts:
            result.append(flat[offset : offset + count].tolist())
            offset += count
        return result

    def score(self, row: dict[str, Any], indices: list[int]) -> list[float]:
        chunk = [(row, indices)]
        return self.score_encoded(self.encode_chunk(chunk), [len(indices)])[0]


def score_file(
    scorer: TeacherScorer,
    source: Path,
    sidecar: Path,
    header: dict[str, Any],
    *,
    head: int,
    resume: bool,
    max_rows: int | None,
    rows_per_batch: int = 4,
) -> int:
    done = 0
    if resume and sidecar.is_file():
        with sidecar.open("rb") as handle:
            existing_header = orjson.loads(handle.readline())
            if existing_header["teacher"]["weights_sha256"] != header["teacher"]["weights_sha256"]:
                raise SystemExit(f"{sidecar}: teacher weights changed; cannot resume")
            done = sum(1 for line in handle if line.strip())
    mode = "ab" if done else "wb"
    written = 0

    def pending_rows():
        for number, row in enumerate(iter_rows(source)):
            if max_rows is not None and number >= max_rows:
                return
            if number >= done:
                yield row

    def chunks():
        chunk: list[tuple[dict[str, Any], list[int]]] = []
        for row in pending_rows():
            chunk.append((row, select_indices(row, head)))
            if len(chunk) == rows_per_batch:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # Tokenization (slow MeCab pipeline) for the next chunk overlaps the
    # GPU forward of the current one; a single worker keeps ordering.
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with sidecar.open(mode) as out, ThreadPoolExecutor(max_workers=1) as pool:
        if not done:
            out.write(orjson.dumps(header) + b"\n")
        progress = tqdm(desc=source.parent.name, unit="row", initial=done)
        iterator = chunks()
        pending = deque()
        for chunk in iterator:
            pending.append(pool.submit(lambda c=chunk: (c, scorer.encode_chunk(c))))
            if len(pending) >= 2:
                break
        while pending:
            chunk, encoded = pending.popleft().result()
            next_chunk = next(iterator, None)
            if next_chunk is not None:
                pending.append(
                    pool.submit(lambda c=next_chunk: (c, scorer.encode_chunk(c)))
                )
            counts = [len(indices) for _, indices in chunk]
            for (row, indices), logits in zip(chunk, scorer.score_encoded(encoded, counts)):
                out.write(
                    orjson.dumps(
                        {"id": row.get("id"), "candidate_indices": indices, "logits": logits}
                    )
                    + b"\n"
                )
                written += 1
            progress.update(len(chunk))
        progress.close()
    return written


def verify_file(source: Path, sidecar: Path) -> dict[str, float]:
    """Rank rows by cached logits alone; the gold target should come first."""
    ranks: list[int] = []
    with sidecar.open("rb") as handle:
        header = orjson.loads(handle.readline())
        records = (orjson.loads(line) for line in handle if line.strip())
        for row, record in zip(iter_rows(source), records):
            target = int(row["target_index"])
            if target not in record["candidate_indices"]:
                continue
            paired = sorted(
                zip(record["logits"], record["candidate_indices"]),
                key=lambda item: item[0],
                reverse=True,
            )
            position = next(
                rank for rank, (_, index) in enumerate(paired, start=1) if index == target
            )
            ranks.append(position)
    count = len(ranks)
    metrics = {
        "rows": count,
        "top1": sum(rank == 1 for rank in ranks) / count,
        "top5": sum(rank <= 5 for rank in ranks) / count,
        "mrr": sum(1 / rank for rank in ranks) / count,
    }
    print(json.dumps({"source": header["source"], **metrics}, ensure_ascii=False))
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = TeacherScorer(args.teacher, args.device)
    for source in args.input:
        header = make_header(args, source, scorer.metadata)
        written = score_file(
            scorer,
            source,
            sidecar_path(args.output_dir, source),
            header,
            head=args.head,
            resume=args.resume,
            max_rows=args.max_rows,
            rows_per_batch=args.rows_per_batch,
        )
        print(f"{source}: wrote {written} rows", file=sys.stderr)
        if args.verify:
            verify_file(source, sidecar_path(args.output_dir, source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
