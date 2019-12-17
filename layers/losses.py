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
    def forward(self, x, target, length, seq_len_norm=False):
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
        if seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.mse_loss(
                x * mask, target * mask, reduction='none')
            loss = loss.mul(out_weights.cuda()).sum() 
        else:
            loss = functional.mse_loss(
                x * mask, target * mask, reduction='sum')
            loss = loss / mask.sum()
        return loss


class TTSLoss(nn.Module):
    def forward(self, x, target, length, seq_len_norm=False):
        """
        Computing both l1 and MSE losses together. It enables to compare
        Tacotron and Tacotron2 models which in the original works use different
        loss functions.
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
        if seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss_mse = functional.mse_loss(
                x * mask, target * mask, reduction='none')
            loss_l1 = functional.l1_loss(
                x * mask, target * mask, reduction='none')
            loss_mse = loss_mse.mul(out_weights.cuda()).sum() 
            loss_l1 = loss_l1.mul(out_weights.cuda()).sum() 
        else:
            loss_mse = functional.mse_loss(
                x * mask, target * mask, reduction='sum')
            loss_l1 = functional.l1_loss(
                x * mask, target * mask, reduction='sum')
            loss_mse = loss_mse / mask.sum()
            loss_l1 = loss_l1 / mask.sum()
        return loss_mse, loss_l1


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
