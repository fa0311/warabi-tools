"""Quantize a static-shape reranker export to QDQ INT8 for NPU providers.

Input: an `ime-export-reranker --static-shape` output directory. Calibration
feeds real materialized rows (context/reading/candidates) encoded exactly as
the runtime does, padded to the static shape. Output directory contains the
quantized model.onnx plus the loader files, with parity_cases regenerated
from the QUANTIZED graph on CPU (the device gate then checks HTP-vs-CPU
integer arithmetic drift, not int8-vs-fp32 quantization error) and a looser
tolerance to absorb accumulator differences across QNN versions.

Quantization error itself is reported (fp32 vs int8 on the probe cases) and
gated separately: rankings live on a ~±10 logit scale, so a mean probe shift
above --max-quant-drift means the calibration set or scheme needs work.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy
import torch
from transformers import AutoTokenizer

from .distill_cross_encoder import batch_encode
from .export_reranker import PROBE_CASES

QUANT_ATOL = 0.35  # device gate: HTP vs quantized-CPU reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True, help="static-shape export")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-file", type=Path, required=True,
        help="materialized rows (train/dev.jsonl) for activation ranges",
    )
    parser.add_argument("--calibration-rows", type=int, default=200)
    parser.add_argument(
        "--max-quant-drift", type=float, default=0.5,
        help="max mean |fp32 - int8| probe logit shift accepted",
    )
    return parser


def encode_padded(tokenizer, metadata, context, reading, candidates):
    batch, seq = metadata["static_shape"]["batch"], metadata["static_shape"]["seq"]
    encoded = batch_encode(
        tokenizer,
        {"context": context, "reading": reading,
         "candidates": [{"text": text} for text in candidates]},
        list(range(len(candidates))),
        max_length=int(metadata["max_length"]),
        context_chars=int(metadata["context_chars"]),
        device=torch.device("cpu"),
    )
    ids = encoded["input_ids"].numpy()
    mask = encoded["attention_mask"].numpy()
    padded_ids = numpy.zeros((batch, seq), dtype=numpy.int64)
    padded_mask = numpy.zeros((batch, seq), dtype=numpy.int64)
    rows = min(ids.shape[0], batch)
    padded_ids[:rows, : ids.shape[1]] = ids[:rows]
    padded_mask[:rows, : mask.shape[1]] = mask[:rows]
    return padded_ids, padded_mask


def main(argv: list[str] | None = None) -> int:
    import orjson
    from onnxruntime import InferenceSession
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    args = build_parser().parse_args(argv)
    if args.output_dir.resolve() == args.model_dir.resolve():
        raise SystemExit("--output-dir must differ from --model-dir")
    metadata = json.loads((args.model_dir / "ime_reranker.json").read_text(encoding="utf-8"))
    if "static_shape" not in metadata:
        raise SystemExit("model is not a static-shape export (run with --static-shape)")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    rows = []
    with args.calibration_file.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows.append(orjson.loads(line))
            if len(rows) >= args.calibration_rows:
                break

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.iterator = iter(rows)

        def get_next(self):
            row = next(self.iterator, None)
            if row is None:
                return None
            ids, mask = encode_padded(
                tokenizer, metadata, row["context"], row["reading"],
                [candidate["text"] for candidate in row["candidates"][:32]],
            )
            return {"input_ids": ids, "attention_mask": mask}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(args.model_dir / "model.onnx"),
        model_output=str(args.output_dir / "model.onnx"),
        calibration_data_reader=Reader(),
        quant_format="QDQ",
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    for name in ("ime_reranker.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json"):
        source = args.model_dir / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)

    # Probe both graphs on CPU: quantization error report + new parity cases.
    fp32 = InferenceSession(
        str(args.model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    int8 = InferenceSession(
        str(args.output_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    drifts, cases = [], []
    for case in PROBE_CASES:
        ids, mask = encode_padded(
            tokenizer, metadata, case["context"], case["reading"], [case["candidate"]]
        )
        feed = {"input_ids": ids, "attention_mask": mask}
        reference = float(fp32.run(None, feed)[0][0].squeeze())
        quantized = float(int8.run(None, feed)[0][0].squeeze())
        drifts.append(abs(quantized - reference))
        cases.append({**case, "logit": quantized})
    (args.output_dir / "parity_cases.json").write_text(
        json.dumps({"atol": QUANT_ATOL, "rtol": 0.05, "cases": cases}, ensure_ascii=False,
                   indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "output": str(args.output_dir),
        "calibration_rows": len(rows),
        "quant_probe_drift_mean": sum(drifts) / len(drifts),
        "quant_probe_drift_max": max(drifts),
    }
    print(json.dumps(report, ensure_ascii=False))
    if report["quant_probe_drift_mean"] > args.max_quant_drift:
        raise SystemExit(f"quantization drift {report['quant_probe_drift_mean']:.3f} too high")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
