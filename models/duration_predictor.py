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
        super().__init__()
        self.offset = offset
        self.k = 5
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, self.k)
        self.linear2 = torch.nn.Linear(n_chans, self.k)

    def calculate_durations(self, att_ws, ilens, olens):
        durations = [self._calculate_duration(att_w, ilen, olen) for att_w, ilen, olen in zip(att_ws, ilens, olens)]
        return pad_list(durations, 0)

    @staticmethod
    def _calculate_duration(att_w, ilen, olen):
        '''
        attw : batch x outs x ins
        '''
        durations = torch.stack([att_w[:olen, :ilen].argmax(-1).eq(i).sum() for i in range(ilen)])
        return durations

    def calculate_scores(self, att_ws, ilens, olens):
        scores = [self._calculate_scores(att_w, ilen, olen, self.k) for att_w, ilen, olen in zip(att_ws, ilens, olens)]
        return pad_list(scores, 0)

    @staticmethod
    def _calculate_scores(att_w, ilen, olen, k):
        # which input is attended for each output
        scores = [None] * ilen
        values, idxs = att_w[:olen, :ilen].max(-1)
        for i in range(ilen):
            vals = values[torch.where(idxs == i)]
            scores[i] = vals
        scores = [torch.nn.functional.pad(score, (0, k - score.shape[0])) for score in scores]
        return torch.stack(scores)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        durs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax, 5)
        if is_inference:
            # NOTE: calculate in linear domain
            durs = F.softmax(durs, dim=-1)
            # b = torch.zeros(durs.shape[0], durs.shape[1], 5).to(durs.device)
            # b[:] = torch.FloatTensor(list(range(5)))
            # durs = (durs * b).sum(-1)
            durs = durs.argmax(-1)
        
        # compute attention scores
        scores = self.linear2(xs.transpose(1, -1).squeeze(-1))
        # scores = torch.sigmoid(scores)

        if x_masks is not None:
            scores = scores.masked_fill(x_masks, 0.0)
            durs = durs.masked_fill(x_masks, 0.0)

        return durs, scores

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
    
    def compute_alignment_batch(self, durations, scores):
        alignment = [self.compute_alignment(dur, sco).T for dur, sco in zip(durations, scores)]
        alignment = pad_list(alignment, 0)
        return alignment

    def compute_alignment(self, durations, scores=None):
        T_en = durations.shape[0]
        a_vals = []
        for idx, dur in enumerate(durations.squeeze().long()):
            dur = dur.item()
            if dur == 0:
                continue
            a = torch.zeros(T_en, dur)
            if scores is None:
                a[idx] = 1
            else:
                for i in range(dur):
                    a[idx, i] = scores[idx, i]
                    if i + 1 == dur and idx + 2 < durations.shape[0]:
                        a[idx + 1, -1] = 1 - scores[idx + 1, 0]
            a_vals.append(a)
        a_vals = torch.cat(a_vals, dim=-1)
        return a_vals

    def length_regulation_batch(self, xs, durations, scores):
        context = [self.length_regulation(x, dur, sco) for x, dur, sco in zip(xs, durations, scores)]
        context = pad_list(context, 0)
        return context

    def length_regulation(self, x, durations, scores=None):
        # breakpoint()
        # scores = torch.clamp(scores, min=0.3, max=0.99)
        T_en = durations.shape[0]
        a_vals = []
        # durations *= 1.05
        for idx, dur in enumerate(durations.squeeze().long()):
            dur = dur.item()
            if dur == 0:
                dur = 1
            if scores is None:
                a_vals += [x[idx]] * dur 
            else:
                for i in range(dur):
                    if i+1 == dur and idx + 2 < durations.shape[0]:
                        a_vals += [x[idx] * scores[idx, i] + x[idx + 1] * scores[idx+1, 0]]
                    else:
                        a_vals += [x[idx] * scores[idx, i]]
        a_vals = torch.stack(a_vals)
        return a_vals

        
def discrete_loss(y_hat, y):
    y = torch.clamp(y, max=4)
    loss = F.cross_entropy(y_hat.view(y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2]), y.view(y.shape[0] * y.shape[1]))
    # b = torch.zeros(y_hat.shape[0], y_hat.shape[1], 5).to(y_hat.device)
    # b[:] = torch.FloatTensor(list(range(5)))
    # y_hat = (y_hat * b).sum(-1)
    # loss = F.l1_loss(y_hat, y.float())
    return loss


class DurationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dur_pred, dur, score_pred, scores):
        dur_loss = discrete_loss(dur_pred, dur) 
        length_loss = F.l1_loss(dur_pred.argmax(-1).sum(-1).float(), dur.sum(-1).float())
        score_loss = F.l1_loss(score_pred, scores)
        return score_loss + dur_loss + length_loss * 0.01