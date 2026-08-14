"""Export a trained student for the ONNX Runtime engine.

Input: a `distill_cross_encoder` output directory. Output directory contents:
  model.onnx               the runtime artifact (fp32, opset 17, dynamic
                           batch/seq axes) — Warabi loads this via ort
  model.safetensors        fp32 torch weights (HF interchange / re-export)
  config.json              standard DistilBertConfig
  vocab.txt                character vocabulary (Rust loads this)
  tokenizer.json etc.      HF fast tokenizer (Python-side parity/eval use)
  ime_reranker.json        schema v2 metadata — single source of truth for
                           both runtimes (blend constants filled by
                           calibration; exporter carries them through)
  parity_cases.json        fixed probe inputs + fp32 logits; the Rust loader
                           asserts allclose on these at load time

Probe cases cover: plain conversion, empty context, long context hitting the
max_length truncation rule, fullwidth/halfwidth distinctions, and UNK chars.
The ONNX graph is verified against the torch logits on the probes before the
export is accepted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from .distill_cross_encoder import batch_encode

PROBE_CASES = [
    {"context": "家に", "reading": "かえる", "candidate": "帰る"},
    {"context": "家に", "reading": "かえる", "candidate": "蛙"},
    {"context": "", "reading": "かえる", "candidate": "変える"},
    {"context": "血液検査で", "reading": "いじょう", "candidate": "異常"},
    {"context": "血液検査で", "reading": "いじょう", "candidate": "以上"},
    {"context": "あ" * 300, "reading": "てすと", "candidate": "テスト"},
    {"context": "価格は", "reading": "１えん", "candidate": "１円"},
    {"context": "価格は", "reading": "1えん", "candidate": "1円"},
    {"context": "レア文字🍣と𠮷野家", "reading": "てすと", "candidate": "テスト"},
    {"context": "空白　と\n改行", "reading": "てすと", "candidate": "手酢と"},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dictionary-cost-weight",
        type=float,
        help="calibrated blend weight; omit to keep the value already in ime_reranker.json",
    )
    parser.add_argument("--model-candidate-count", type=int)
    parser.add_argument(
        "--static-shape",
        metavar="BATCHxSEQ",
        help="export with fixed input dims (e.g. 32x224) for NPU execution "
        "providers that reject dynamic axes (QNN HTP, Intel NPU). The engine "
        "pads batches at runtime; metadata records the shape",
    )
    return parser


@torch.inference_mode()
def probe_logits(model, tokenizer, metadata: dict) -> list[float]:
    model.eval()
    logits: list[float] = []
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
        logits.append(float(model(**encoded).logits.squeeze().item()))
    return logits


def export_onnx(model, output_dir: Path, static_shape: tuple[int, int] | None = None) -> None:
    if static_shape is None:
        ids = torch.randint(5, 8000, (2, 16))
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        }
    else:
        ids = torch.randint(5, 8000, static_shape)
        dynamic_axes = None
    mask = torch.ones_like(ids)
    torch.onnx.export(
        model,
        (ids, mask),
        output_dir / "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )


def verify_onnx(
    output_dir: Path,
    tokenizer,
    metadata: dict,
    expected: list[float],
    static_shape: tuple[int, int] | None = None,
) -> float:
    import numpy
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(output_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    drift = 0.0
    for case, reference in zip(PROBE_CASES, expected):
        encoded = batch_encode(
            tokenizer,
            {"context": case["context"], "reading": case["reading"],
             "candidates": [{"text": case["candidate"]}]},
            [0],
            max_length=int(metadata["max_length"]),
            context_chars=int(metadata["context_chars"]),
            device=torch.device("cpu"),
        )
        ids = encoded["input_ids"].numpy()
        mask = encoded["attention_mask"].numpy()
        if static_shape is not None:
            batch, seq = static_shape
            padded_ids = numpy.zeros((batch, seq), dtype=ids.dtype)
            padded_mask = numpy.zeros((batch, seq), dtype=mask.dtype)
            padded_ids[: ids.shape[0], : ids.shape[1]] = ids
            padded_mask[: mask.shape[0], : mask.shape[1]] = mask
            ids, mask = padded_ids, padded_mask
        logit = float(
            session.run(None, {"input_ids": ids, "attention_mask": mask})[0][0].squeeze()
        )
        drift = max(drift, abs(logit - reference))
    return drift


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.resolve() == args.model_dir.resolve():
        # Windows keeps loaded safetensors mmap-locked; in-place rewrites fail.
        raise SystemExit("--output-dir must differ from --model-dir")
    metadata = json.loads((args.model_dir / "ime_reranker.json").read_text(encoding="utf-8"))
    if args.dictionary_cost_weight is not None:
        metadata["dictionary_cost_weight"] = args.dictionary_cost_weight
        metadata["calibration_pending"] = False
    if args.model_candidate_count is not None:
        metadata["model_candidate_count"] = args.model_candidate_count
    metadata.setdefault("model_candidate_count", 32)
    static_shape: tuple[int, int] | None = None
    if args.static_shape:
        batch, seq = (int(part) for part in args.static_shape.lower().split("x"))
        if seq != int(metadata["max_length"]):
            raise SystemExit(f"static seq {seq} must equal max_length {metadata['max_length']}")
        static_shape = (batch, seq)
        metadata["static_shape"] = {"batch": batch, "seq": seq}
    else:
        metadata.pop("static_shape", None)
    if "dictionary_cost_weight" not in metadata:
        print("warning: exporting without a calibrated dictionary_cost_weight (0.19 default)")
        metadata["dictionary_cost_weight"] = 0.19

    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir).float()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_logits = probe_logits(model, tokenizer, metadata)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    export_onnx(model, args.output_dir, static_shape)
    vocab_source = args.model_dir / "vocab.txt"
    if not vocab_source.is_file():
        raise SystemExit(f"vocab.txt not found next to the model: {vocab_source}")
    shutil.copy2(vocab_source, args.output_dir / "vocab.txt")

    cases = [{**case, "logit": logit} for case, logit in zip(PROBE_CASES, fp32_logits)]
    # allclose-style gate: |ort - torch| <= atol + rtol*|torch|. ONNX runs the
    # same fp32 graph, so drift stays ~1e-5; the tolerance keeps headroom for
    # kernel differences across ort versions/platforms while staying two
    # orders below blend decision margins.
    (args.output_dir / "parity_cases.json").write_text(
        json.dumps({"atol": 1e-3, "rtol": 1e-4, "cases": cases}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ime_reranker.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    drift = verify_onnx(args.output_dir, tokenizer, metadata, fp32_logits, static_shape)
    print(json.dumps({"output": str(args.output_dir), "onnx_probe_drift": drift}))
    if drift > 1e-3:
        raise SystemExit(f"onnx drift {drift} exceeds tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
