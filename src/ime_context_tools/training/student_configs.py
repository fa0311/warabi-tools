"""Student architecture presets for the three distilled reranker tiers.

All tiers are plain DistilBERT (same family as the teacher, standard blocks
only — implementable in Burn without exotic attention), sinusoidal position
embeddings (zero learnable position parameters), a single regression logit,
and the shared 8,192-character vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import DistilBertConfig

VOCAB_SIZE = 8192
MAX_LENGTH = 224
CONTEXT_CHARS = 160


@dataclass(frozen=True)
class StudentSpec:
    tier: str
    n_layers: int
    dim: int
    n_heads: int
    hidden_dim: int

    def config(self) -> DistilBertConfig:
        return DistilBertConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=256,
            sinusoidal_pos_embds=True,
            n_layers=self.n_layers,
            dim=self.dim,
            n_heads=self.n_heads,
            hidden_dim=self.hidden_dim,
            activation="gelu",
            dropout=0.1,
            attention_dropout=0.1,
            seq_classif_dropout=0.2,
            initializer_range=0.02,
            pad_token_id=0,
            num_labels=1,
            problem_type="regression",
        )


# Depth is the serial latency bottleneck, width is GPU-friendly: lite and
# standard share 4 layers (CPU/mobile and laptop GPU), max keeps 6 layers at
# the teacher's width — it doubles as the quality-ceiling probe and the
# cascade-distillation teacher for the smaller tiers.
STUDENT_SPECS = {
    "lite": StudentSpec("lite", n_layers=4, dim=256, n_heads=4, hidden_dim=1024),
    # depth-vs-width probe: lite's width with max's depth. If it approaches
    # standard's quality at half the parameters, depth (composition steps)
    # is what the character-level task pays for, not per-layer capacity.
    "deep-lite": StudentSpec("deep-lite", n_layers=6, dim=256, n_heads=4, hidden_dim=1024),
    # v9 candidate: the depth verdict's upper tier — deep-lite's depth with
    # room for the knowledge width the std experiment showed dev rewards.
    "deep-mid": StudentSpec("deep-mid", n_layers=6, dim=384, n_heads=6, hidden_dim=1536),
    # min-tier probe: if depth is what the task pays for, six thin layers
    # should beat lite (4x256) at ~0.85x its compute.
    "deep-min": StudentSpec("deep-min", n_layers=6, dim=192, n_heads=3, hidden_dim=768),
    "standard": StudentSpec("standard", n_layers=4, dim=512, n_heads=8, hidden_dim=2048),
    "max": StudentSpec("max", n_layers=6, dim=768, n_heads=12, hidden_dim=3072),
}

# Effective batches match the original 32/36/32-row schedules. The per-step
# forward is capped by VRAM: 32 rows x 32 candidates x ~224 tokens fits the
# lite model, but the wider tiers need gradient accumulation (36 rows in one
# forward took standard down with CUDA unknown-error on a 16GB card).
# Epoch counts assume the full v3 mix (~660k rows); total training tokens
# stay comparable to the earlier 159k-row/10-epoch schedule.
TRAINING_DEFAULTS = {
    "lite": {"epochs": 6, "learning_rate": 3e-4, "query_batch_size": 32, "grad_accum": 1},
    "deep-lite": {"epochs": 6, "learning_rate": 3e-4, "query_batch_size": 32, "grad_accum": 1},
    "deep-mid": {"epochs": 6, "learning_rate": 2.5e-4, "query_batch_size": 24, "grad_accum": 2},
    "deep-min": {"epochs": 6, "learning_rate": 3e-4, "query_batch_size": 32, "grad_accum": 1},
    "standard": {"epochs": 5, "learning_rate": 2.5e-4, "query_batch_size": 24, "grad_accum": 2},
    # 12x3 OOMed on long-context buckets (16GB card); 9x4 keeps the same
    # effective 36 rows with a 25% lower peak.
    "max": {"epochs": 4, "learning_rate": 1.5e-4, "query_batch_size": 9, "grad_accum": 4},
}
