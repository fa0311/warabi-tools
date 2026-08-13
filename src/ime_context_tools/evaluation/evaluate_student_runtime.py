"""Runtime evaluation + blend calibration for a distilled student.

Production-faithful pipeline: candidates come from the Rust engine
(`ime-cli convert-dump` over the eval readings), the student scores the model
head (+ late whole-exact entries), and the final order is the blended score
`logit - w * calibrated_cost / 1000` — the same fusion the Rust runtime
applies. Student logits are computed once; the `--weights` grid is applied
afterwards, so calibration sweeps are nearly free.

Reports top1/top5/MRR per (eval file, weight). Gate targets (plan A4):
curated top1 >= 0.72 (lite) / 0.77 (standard) / 0.795 (max).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import orjson
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from ime_context_tools.training.distill_cross_encoder import batch_encode

# Mirror of crates/ime-core/src/fusion.rs
CONVERTED_EXACT = -3_000
SINGLE_CHAR_COPY = -12_000
PARTICLE_COPY = -50_000
PREDICTION = -40_000
PARTICLES = set("がをにのはとへでもやか")
LONG_WHOLE_EXACT_MIN_READING = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--ime-cli", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, action="append", required=True)
    parser.add_argument(
        "--weights",
        default="0.10,0.13,0.16,0.19,0.22,0.25,0.30",
        help="comma-separated dictionary_cost_weight grid",
    )
    parser.add_argument("--model-candidate-count", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, help="write the report JSON here")
    return parser


def dump_candidates(
    ime_cli: Path, pack: Path, readings: list[str], limit: int
) -> dict[str, list[list[Any]]]:
    with tempfile.TemporaryDirectory() as tmp:
        readings_path = Path(tmp) / "readings.txt"
        readings_path.write_text("\n".join(readings) + "\n", encoding="utf-8", newline="\n")
        result = subprocess.run(
            [str(ime_cli), "convert-dump", str(pack), str(readings_path), str(limit)],
            capture_output=True,
            check=True,
        )
    table: dict[str, list[list[Any]]] = {}
    for line in result.stdout.splitlines():
        if line.strip():
            row = orjson.loads(line)
            table[row["reading"]] = row["candidates"]
    return table


def calibrated_cost(candidate: list[Any], reading: str, context: str) -> int:
    text, search_cost, kind, _whole_exact = candidate[0], candidate[1], candidate[2], candidate[3]
    adjustment = 0
    if kind == "exact" and text != reading:
        adjustment += CONVERTED_EXACT
    if len(reading) == 1 and text == reading:
        adjustment += PARTICLE_COPY if context and reading in PARTICLES else SINGLE_CHAR_COPY
    if kind == "prediction":
        adjustment += PREDICTION
    return search_cost + adjustment


def model_indices(candidates: list[list[Any]], reading: str, head: int) -> list[int]:
    indices = list(range(min(head, len(candidates))))
    if len(reading) >= LONG_WHOLE_EXACT_MIN_READING:
        chosen = set(indices)
        indices.extend(
            index
            for index, candidate in enumerate(candidates)
            if candidate[3] and index not in chosen
        )
    return indices


@torch.inference_mode()
def score_rows(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    candidate_table: dict[str, list[list[Any]]],
    *,
    head: int,
    device: torch.device,
    max_length: int,
    context_chars: int,
    use_bfloat16: bool,
) -> list[tuple[list[int], list[float]]]:
    """Per row: (scored candidate indices, student logits)."""
    model.eval()
    scored: list[tuple[list[int], list[float]]] = []
    for row in tqdm(rows, desc="student scoring", unit="row"):
        candidates = candidate_table.get(row["reading"], [])
        if not candidates:
            scored.append(([], []))
            continue
        indices = model_indices(candidates, row["reading"], head)
        encoded = batch_encode(
            tokenizer,
            {
                "context": row["context"],
                "reading": row["reading"],
                "candidates": [{"text": candidates[index][0]} for index in indices],
            },
            list(range(len(indices))),
            max_length=max_length,
            context_chars=context_chars,
            device=device,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            logits = model(**encoded).logits.squeeze(-1).float()
        scored.append((indices, logits.cpu().tolist()))
    return scored


def evaluate_weight(
    rows: list[dict[str, Any]],
    candidate_table: dict[str, list[list[Any]]],
    scored: list[tuple[list[int], list[float]]],
    weight: float,
) -> dict[str, float]:
    ranks: list[int | None] = []
    for row, (indices, logits) in zip(rows, scored):
        candidates = candidate_table.get(row["reading"], [])
        expected = set(row["expected"])
        blended = sorted(
            (
                (
                    logit - weight * calibrated_cost(candidates[index], row["reading"], row["context"]) / 1000.0,
                    index,
                )
                for index, logit in zip(indices, logits)
            ),
            reverse=True,
        )
        ordered = [candidates[index][0] for _, index in blended]
        ordered.extend(
            candidates[index][0]
            for index in range(len(candidates))
            if index not in set(indices)
        )
        rank = next((i + 1 for i, text in enumerate(ordered) if text in expected), None)
        ranks.append(rank)
    count = len(ranks)
    return {
        "rows": count,
        "top1": sum(rank == 1 for rank in ranks if rank) / count,
        "top5": sum(rank is not None and rank <= 5 for rank in ranks) / count,
        "mrr": sum(1 / rank for rank in ranks if rank) / count,
        "recall": sum(rank is not None for rank in ranks) / count,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weights = [float(value) for value in args.weights.split(",")]
    metadata = json.loads((args.student / "ime_reranker.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    model = DistilBertForSequenceClassification.from_pretrained(args.student).to(device)
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    report: dict[str, Any] = {"student": str(args.student), "weights": {}}
    for eval_file in args.eval_file:
        rows = [
            orjson.loads(line)
            for line in eval_file.read_bytes().splitlines()
            if line.strip()
        ]
        readings = sorted({str(row["reading"]) for row in rows})
        candidate_table = dump_candidates(args.ime_cli, args.pack, readings, args.candidate_limit)
        scored = score_rows(
            model,
            tokenizer,
            rows,
            candidate_table,
            head=args.model_candidate_count,
            device=device,
            max_length=int(metadata["max_length"]),
            context_chars=int(metadata["context_chars"]),
            use_bfloat16=use_bfloat16,
        )
        for weight in weights:
            metrics = evaluate_weight(rows, candidate_table, scored, weight)
            report["weights"].setdefault(f"{weight:.2f}", {})[eval_file.stem] = metrics
            print(json.dumps({"file": eval_file.stem, "weight": weight, **metrics}))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
