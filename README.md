# ime-context-tools — KeyNako IME model pipeline

## Self-contained pipeline (no paths outside this repo)

All upstream inputs come from Hugging Face; everything else is generated
in-repo. Full rebuild from scratch:

```
# 0. one-time: fetch teacher + dictionary + raw dataset from HF
python -m ime_context_tools.fetch_upstream

# 1. materialize candidate lists with the Rust engine + downloaded dictionary
python -m ime_context_tools.training.materialize_candidates \
  --dataset-file artifacts/upstream/context-dataset/data/*.jsonl \
  --pack artifacts/upstream/dictionary-core/dictionary.sqlite3 \
  --ime-cli <ime-context-core>/target/release/ime-cli \
  --output-dir artifacts/materialized/v1

# 2. character tokenizer from upstream + materialized text
python -m ime_context_tools.training.build_char_tokenizer \
  --candidates-dir artifacts/materialized/v1 \
  --dictionary-tsv artifacts/upstream/dictionary-core/entries.tsv \
  --dictionary-tsv artifacts/upstream/dictionary-core/predictions.tsv \
  --output-dir artifacts/tokenizers/ime-char-8k-v2

# 3. teacher logits (downloaded teacher; sidecars record its SHA-256)
python -m ime_context_tools.training.score_teacher_cross_encoder \
  --teacher artifacts/upstream/teacher \
  --input artifacts/materialized/v1/train.jsonl \
  --input artifacts/materialized/v1/dev.jsonl \
  --output-dir artifacts/teacher-logits-v2 --resume --verify

# 4. distill (lite | standard | max)
python -m ime_context_tools.training.distill_cross_encoder \
  --student lite --tokenizer-dir artifacts/tokenizers/ime-char-8k-v2 \
  --train artifacts/materialized/v1/train.jsonl \
  --dev artifacts/materialized/v1/dev.jsonl \
  --teacher-logits-dir artifacts/teacher-logits-v2 \
  --output-dir artifacts/students/ime-reranker-lite-v1

# 5. evaluate + calibrate the blend weight, then export for Rust
python -m ime_context_tools.evaluation.evaluate_student_runtime ...
python -m ime_context_tools.training.export_burn_reranker ...
```

Evaluation sets live in `datasets/evaluation/` (gitignored — local-only data;
the locked sets are final-gate only and must never be published or used for
calibration).

Self-contained home for the cross-platform IME core:

- `src/ime_context_tools/` — Python distillation pipeline that turns the
  `ime-context-reranker-v1` teacher (68.7M params, CUDA-only, MeCab tokenizer)
  into three Rust-deployable students:
  | tier | layers | hidden | params |
  |---|---|---|---|
  | `ime-reranker-lite` | 6 | 256 | ~6.9M |
  | `ime-reranker-standard` | 8 | 384 | ~17.5M |
  | `ime-reranker-max` | 10 | 512 | ~36M |
- `crates/` — Rust workspace (engine, lattice, lexicon, tokenizer, Burn
  reranker, FFI). Inference: Burn with wgpu (Vulkan/DX12/Metal) + CPU fallback.
- `tests/` — Python tests (`python -m pytest` from this directory).

## Model input format

Character-level tokenizer (vocab 8,192, no normalization, White_Space
dropped), three spliced segments:

```
[CLS] context [SEP] reading [SEP] candidate [SEP]
```

Specials: `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`. The contract is
implemented twice — HF fast tokenizer (`tokenizer.json`, training) and
`crates/ime-tokenizer` (runtime) — and parity-tested.

## Teacher upgrades (v2, v3, ...)

The pipeline is teacher-agnostic: every script takes `--teacher <dir>` and
sidecars/exports record the teacher's SHA-256. When `ime-context-reranker-v2`
lands, the whole refresh is three commands, zero code changes:

```
ime-score-teacher   --teacher <v2-dir> --output-dir artifacts/teacher-logits-v2 ...
ime-distill         --teacher-logits-dir artifacts/teacher-logits-v2 ...
ime-export-reranker --model-dir ... --output-dir ...
```

The Rust runtime never sees the teacher — only exported students.

## Quantization

fp16 ships by default (lite 13.8 MB / standard 35 MB / max 72 MB; measured
probe drift ~1e-3). int8 (Burn per-channel weights) is reserved for the CPU
path if latency demands it and PTQ costs < 0.5pt curated top1. No 4-bit.

## Measured baselines (2026-08-12, this machine)

- Candidate generation (Rust, 512 candidates): ~5.5 ms/reading upper bound;
  first-512 order byte-identical to Python on 48 readings.
- Reranker CPU (lite-shape, 64 cand x 224 tokens, Burn backends):
  ndarray ~1.07 s / CubeCL-cpu ~2.2 s / candle ~0.9 s per batch. CPU alone
  cannot hit the <100 ms budget at full width — wgpu GPU is the primary path;
  CPU fallback uses fewer candidates (+ int8 if needed), weakest devices run
  algorithmic-only.

## Environment notes

- Fully self-contained: every upstream input downloads from Hugging Face
  (`fetch_upstream`), evaluation sets live in `datasets/` (gitignored), and
  no code imports anything outside this repository (plus `ime-context-core`
  for the Rust engine binary).
- Python environment: D: is small, so the venv lives on C:.
  `UV_PROJECT_ENVIRONMENT=C:/Users/yuki/.venvs/ime-context-tools uv sync`
  then run tools with `C:/Users/yuki/.venvs/ime-context-tools/Scripts/python.exe`
  (or `uv run` with the same env var).
