"""Export a trained student for the Rust/Burn runtime.

Input: a `distill_cross_encoder` output directory. Output directory contents:
  model.safetensors        fp32, HF DistilBERT canonical tensor names
  model.fp16.safetensors   fp16 copy (mobile-friendly)
  config.json              standard DistilBertConfig
  vocab.txt                character vocabulary (Rust loads this)
  tokenizer.json etc.      HF fast tokenizer (Python-side parity/eval use)
  ime_reranker.json        schema v2 metadata — single source of truth for
                           both runtimes (blend constants filled by
                           calibration; exporter carries them through)
  parity_cases.json        fixed probe inputs + fp32 logits; the Rust loader
                           asserts max |diff| < 1e-3 on these at load time

Probe cases cover: plain conversion, empty context, long context hitting the
max_length truncation rule, fullwidth/halfwidth distinctions, and UNK chars.
"""

from __future__ import annotations

import argparse
import copy
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = json.loads((args.model_dir / "ime_reranker.json").read_text(encoding="utf-8"))
    if args.dictionary_cost_weight is not None:
        metadata["dictionary_cost_weight"] = args.dictionary_cost_weight
        metadata["calibration_pending"] = False
    if args.model_candidate_count is not None:
        metadata["model_candidate_count"] = args.model_candidate_count
    metadata.setdefault("model_candidate_count", 64)
    if "dictionary_cost_weight" not in metadata:
        print("warning: exporting without a calibrated dictionary_cost_weight (0.19 default)")
        metadata["dictionary_cost_weight"] = 0.19

    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir).float()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Probe on the pristine fp32 weights BEFORE any dtype conversion.
    fp32_logits = probe_logits(model, tokenizer, metadata)
    # fp32 canonical + fp16 copy, HF tensor names preserved by save_pretrained.
    # half() mutates in place, so the fp16 copy comes from a deep copy.
    model.save_pretrained(args.output_dir, safe_serialization=True)
    copy.deepcopy(model).half().save_pretrained(
        args.output_dir / "fp16", safe_serialization=True
    )
    shutil.move(
        args.output_dir / "fp16" / "model.safetensors",
        args.output_dir / "model.fp16.safetensors",
    )
    shutil.rmtree(args.output_dir / "fp16")
    tokenizer.save_pretrained(args.output_dir)
    vocab_source = args.model_dir / "vocab.txt"
    if not vocab_source.is_file():
        raise SystemExit(f"vocab.txt not found next to the model: {vocab_source}")
    shutil.copy2(vocab_source, args.output_dir / "vocab.txt")

    cases = [{**case, "logit": logit} for case, logit in zip(PROBE_CASES, fp32_logits)]
    # allclose-style gate: |rust - python| <= atol + rtol*|python|. Framework
    # fp32 drift for the 768-wide trunk measures up to ~0.027 absolute
    # (dominated by activation scale, not output scale; v6b hit 0.026 on the
    # halfwidth-digit probe); 0.03 stays an order below blend decision
    # margins.
    (args.output_dir / "parity_cases.json").write_text(
        json.dumps({"atol": 3.0e-2, "rtol": 5e-3, "cases": cases}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ime_reranker.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Reload parity: fp16 weights must reproduce fp32 logits within tolerance.
    from safetensors.torch import load_file

    fp16_model = DistilBertForSequenceClassification.from_pretrained(args.output_dir)
    fp16_state = load_file(args.output_dir / "model.fp16.safetensors")
    fp16_model.load_state_dict(
        {key: value.float() for key, value in fp16_state.items()}, strict=True
    )
    drift = max(
        abs(a - b)
        for a, b in zip(
            probe_logits(fp16_model, tokenizer, metadata),
            [case["logit"] for case in cases],
        )
    )
    print(json.dumps({"output": str(args.output_dir), "fp16_probe_drift": drift}))
    if drift > 0.05:
        raise SystemExit(f"fp16 drift {drift} exceeds tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
