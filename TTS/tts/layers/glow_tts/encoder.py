import math
import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.utils.generic_utils import sequence_mask

from TTS.tts.layers.glow_tts.glow import ConvGroupNorm
from TTS.tts.layers.glow_tts.transformer import Transformer
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor


class Encoder(nn.Module):
    def __init__(self,
                 num_chars,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 filter_channels_dp,
                 num_heads,
                 num_layers,
                 kernel_size,
                 dropout_p,
                 rel_attn_window_size=None,
                 input_length=None,
                 mean_only=False,
                 use_prenet=True,
                 c_in_channels=0):

        super().__init__()

        self.num_chars = num_chars
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.rel_attn_window_size = rel_attn_window_size
        self.input_length = input_length
        self.mean_only = mean_only
        self.use_prenet = use_prenet
        self.c_in_channels = c_in_channels

        self.emb = nn.Embedding(num_chars, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if use_prenet:
            self.prenet = ConvGroupNorm(hidden_channels,
                                        hidden_channels,
                                        hidden_channels,
                                        kernel_size=5,
                                        num_layers=3,
                                        dropout_p=0.5)
        self.encoder = Transformer(
            hidden_channels,
            filter_channels,
            num_heads,
            num_layers,
            kernel_size,
            dropout_p,
            rel_attn_window_size=rel_attn_window_size,
            input_length=input_length,
        )

        self.mean_layer = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.scale_layer = nn.Conv1d(hidden_channels, out_channels, 1)

        self.duration_predictor = DurationPredictor(
            hidden_channels + c_in_channels, filter_channels_dp, kernel_size,
            dropout_p)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.use_prenet:
            x = self.prenet(x, x_mask)
        x = self.encoder(x, x_mask)

        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        o_mean = self.mean_layer(x) * x_mask
        if not self.mean_only:
            o_scale_log = self.scale_layer(x) * x_mask
        else:
            o_scale_log = torch.zeros_like(o_mean)

        o_dur_log = self.duration_predictor(x_dp, x_mask)
        return o_mean, o_scale_log, o_dur_log, x_mask