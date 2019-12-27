import numpy as np
import torch
from torch import nn
from torch.nn import functional
from TTS.utils.generic_utils import sequence_mask


class L1LossMasked(nn.Module):
    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        mask = mask.expand_as(x)
        loss = functional.l1_loss(
            x * mask, target * mask, reduction="sum")
        loss = loss / mask.sum()
        return loss


class MSELossMasked(nn.Module):
    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        mask = mask.expand_as(x)
        loss = functional.mse_loss(
            x * mask, target * mask, reduction="sum")
        loss = loss / mask.sum()
        return loss


class AttentionEntropyLoss(nn.Module):
    # pylint: disable=R0201
    def forward(self, align):
        """
        Forces attention to be more decisive by penalizing
        soft attention weights

        TODO: arguments
        TODO: unit_test
        """
        entropy = torch.distributions.Categorical(probs=align).entropy()
        loss = (entropy / np.log(align.shape[1])).mean()
        return loss


class DurationLoss(nn.Module):
    def forward(self, y_hat, y, K=5):
        # y = torch.clamp(y, max=K)
        # loss = functional.cross_entropy(y_hat.view(y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2]), y.view(y.shape[0] * y.shape[1]))
        loss = functional.mse_loss(y_hat, torch.log(y.float() + 1.0))
        # b = torch.zeros(y_hat.shape[0], y_hat.shape[1], 5).to(y_hat.device)
        # b[:] = torch.FloatTensor(list(range(5)))
        # y_hat = (y_hat * b).sum(-1)
        # loss = F.l1_loss(y_hat, y.float())
        return loss