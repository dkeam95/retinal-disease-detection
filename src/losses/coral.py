import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralLoss(nn.Module):
    """Consistent Rank Logits (CORAL) Loss for ordinal regression tasks.

    Expects logits of shape [B, num_classes - 1] and targets of shape [B].
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        num_classes_minus_one = logits.size(1)

        # Create binary rank targets: shape [B, K - 1]
        # Example: if target = 2 (for K=5), binary vector becomes [1, 1, 0, 0]
        levels = torch.arange(num_classes_minus_one, device=targets.device)
        levels = levels.unsqueeze(0).repeat(targets.size(0), 1)
        binary_targets = (targets.unsqueeze(1) > levels).float()

        # Compute Binary Cross Entropy for each threshold
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        )

        # Sum losses across thresholds and take batch mean
        return bce_loss.sum(dim=1).mean()


def coral_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Converts CORAL logits [B, K-1] into class probability distributions [B, K]."""
    sigmoid_logits = torch.sigmoid(logits)
    cum_probs = sigmoid_logits

    probs = []
    # P(y = 0) = 1 - P(y > 0)
    probs.append(1.0 - cum_probs[:, 0:1])

    # P(y = k) = P(y > k-1) - P(y > k)
    for i in range(1, cum_probs.shape[1]):
        probs.append(cum_probs[:, i - 1 : i] - cum_probs[:, i : i + 1])

    # P(y = K-1) = P(y > K-2)
    probs.append(cum_probs[:, -1:])

    probs = torch.cat(probs, dim=1)
    return torch.clamp(probs, min=0.0, max=1.0)
