import torch
import math
from torch.distributions.normal import Normal
from torch.nn import functional as F
from TTS.utils.data import pad_list

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)

class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, 5)

    def calculate_durations(self, att_ws, ilens, olens):
        durations = [self._calculate_duration(att_w, ilen, olen) for att_w, ilen, olen in zip(att_ws, ilens, olens)]
        return pad_list(durations, 0)

    @staticmethod
    def _calculate_duration(att_w, ilen, olen):
        return torch.stack([att_w[:olen, :ilen].argmax(-1).eq(i).sum() for i in range(ilen)])

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)
        xs = F.softmax(xs)
        if is_inference:
            # NOTE: calculate in linear domain
            xs = sample_from_gaussian(xs)
#             xs = torch.clamp(xs[:, :, 0], min=0)
#             xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return self._forward(xs, x_masks, True)
    
    def compute_alignment(self, durations):
        T_en = durations.shape[0]
        a_vals = []
        for idx, dur in enumerate(durations.squeeze().long()):
            dur = dur.item()
            if dur == 0:
                continue
            a = torch.zeros(T_en, dur)
            a[idx] = 1
            a_vals.append(a)
        a_vals = torch.cat(a_vals, dim=-1)
        return a_vals

    def length_regulation(self, x, durations):
        T_en = durations.shape[0]
        a_vals = []
        for idx, dur in enumerate(durations.squeeze().long()):
            dur = dur.item()
            if dur == 0:
                continue
            a = [x[idx]] * dur 
            a_vals += a
        a_vals = torch.stack(a_vals)
        return a_vals

        
def discrete_loss(y_hat, y, log_std_min=-7.0):
    y = torch.clamp(y, max=4)
    loss = F.cross_entropy(y_hat.view(y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2]), y.view(y.shape[0] * y.shape[1]))
    # b = torch.zeros(y_hat.shape[0], y_hat.shape[1], 5).to(y_hat.device)
    # b[:] = torch.FloatTensor(list(range(5)))
    # y_hat = (y_hat * b).sum(-1)
    # loss = F.l1_loss(y_hat, y.float())
    return loss


class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.
    The loss value is Calculated in log domain to make it Gaussian.
    """

    def __init__(self, offset=1.0):
        """Initilize duration predictor loss module.
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.
        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)
        Returns:
            Tensor: Mean squared error loss value.
        Note:
            `outputs` is in log domain but `targets` is in linear domain.
        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss