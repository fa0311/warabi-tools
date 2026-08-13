"""Distillation loss for listwise candidate ranking.

Combined objective per query list (1 gold + N negatives):

  L = w_hard * CE(student_scores, target)
    + w_kl   * T^2 * KL(softmax(teacher/T) || softmax(student/T))
    + w_mse  * MSE(student_logits, teacher_logits)

The MSE term anchors the student to the teacher's raw logit scale: the
runtime blend `ai_logit - w * dictionary_cost / 1000` is shift-invariant but
NOT scale-invariant, so a scale-compressed student would silently re-weight
the dictionary prior. KL alone does not constrain scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def distillation_loss(
    student_scores: Tensor,
    teacher_scores: Tensor,
    targets: Tensor,
    *,
    hard_weight: float = 0.2,
    kl_weight: float = 0.5,
    mse_weight: float = 0.3,
    temperature: float = 2.0,
) -> Tensor:
    """`student_scores`/`teacher_scores`: (batch, list_width); `targets`: (batch,).

    Rows whose target is negative (unlabeled augmentation rows) contribute
    only the soft terms; the hard CE skips them.
    """
    if student_scores.shape != teacher_scores.shape:
        raise ValueError(
            f"score shapes differ: {tuple(student_scores.shape)} vs {tuple(teacher_scores.shape)}"
        )
    labeled = targets >= 0
    if labeled.any():
        hard = F.cross_entropy(student_scores[labeled], targets[labeled])
    else:
        hard = student_scores.new_zeros(())
    soft = F.kl_div(
        F.log_softmax(student_scores / temperature, dim=-1),
        F.softmax(teacher_scores / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    mse = F.mse_loss(student_scores, teacher_scores)
    return hard_weight * hard + kl_weight * soft + mse_weight * mse
