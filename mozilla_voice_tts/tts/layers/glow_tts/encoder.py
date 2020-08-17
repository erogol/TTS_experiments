import math
import torch
from torch import nn

from mozilla_voice_tts.tts.layers.glow_tts.transformer import Transformer
from mozilla_voice_tts.tts.utils.generic_utils import sequence_mask
from mozilla_voice_tts.tts.layers.glow_tts.glow import ConvLayerNorm, LayerNorm
from mozilla_voice_tts.tts.layers.glow_tts.duration_predictor import DurationPredictor


class GatedConvBlock(nn.Module):
    """Gated convolutional block as in https://arxiv.org/pdf/1612.08083.pdf
    Args:
        in_out_channels (int): number of input/output channels.
        kernel_size (int): convolution kernel size.
        dropout_p (float): dropout rate.
    """
    def __init__(self, in_out_channels, kernel_size, dropout_p, num_layers):
        super().__init__()
        # class arguments
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        # define layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers += [
                nn.Conv1d(in_out_channels,
                          2 * in_out_channels,
                          kernel_size,
                          padding=kernel_size // 2)
            ]
            self.norm_layers += [LayerNorm(2 * in_out_channels)]

    def forward(self, x, x_mask):
        o = x
        res = x
        for idx in range(self.num_layers):
            o = nn.functional.dropout(o,
                                      p=self.dropout_p,
                                      training=self.training)
            o = self.conv_layers[idx](o * x_mask)
            o = self.norm_layers[idx](o)
            o = nn.functional.glu(o, dim=1)
            o = res + o
            res = o
        return o


class Encoder(nn.Module):
    """Glow-TTS encoder module. It uses Transformer with Relative Pos.Encoding
    as in the original paper or GatedConvBlock as a faster alternative.

    Args:
        num_chars (int): number of characters.
        out_channels (int): number of output channels.
        hidden_channels (int): encoder's embedding size.
        filter_channels (int): transformer's feed-forward channels.
        num_head (int): number of attention heads in transformer.
        num_layers (int): number of transformer encoder stack.
        kernel_size (int): kernel size for conv layers and duration predictor.
        dropout_p (float): dropout rate for any dropout layer.
        mean_only (bool): if True, output only mean values and use constant std.
        use_prenet (bool): if True, use pre-convolutional layers before transformer layers.
        c_in_channels (int): number of channels in conditional input.

    Shapes:
        - input: (B, T, C)
    """
    def __init__(self,
                 num_chars,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 filter_channels_dp,
                 encoder_type,
                 num_heads,
                 num_layers,
                 kernel_size,
                 dropout_p,
                 rel_attn_window_size=None,
                 input_length=None,
                 mean_only=False,
                 use_prenet=False,
                 c_in_channels=0):
        super().__init__()
        # class arguments
        self.num_chars = num_chars
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.mean_only = mean_only
        self.use_prenet = use_prenet
        self.c_in_channels = c_in_channels
        self.encoder_type = encoder_type
        # embedding layer
        self.emb = nn.Embedding(num_chars, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        # init encoder
        if encoder_type.lower() == "transformer":
            # optional convolutional prenet
            if use_prenet:
                self.pre = ConvLayerNorm(hidden_channels,
                                         hidden_channels,
                                         hidden_channels,
                                         kernel_size=5,
                                         num_layers=3,
                                         dropout_p=0.5)
            # text encoder
            self.encoder = Transformer(
                hidden_channels,
                filter_channels,
                num_heads,
                num_layers,
                kernel_size=kernel_size,
                dropout_p=dropout_p,
                rel_attn_window_size=rel_attn_window_size,
                input_length=input_length)
        elif encoder_type.lower() == 'gatedconv':
            self.encoder = GatedConvBlock(hidden_channels,
                                          kernel_size=5,
                                          dropout_p=dropout_p,
                                          num_layers=3 + num_layers)
        # final projection layers
        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        # duration predictor
        self.duration_predictor = DurationPredictor(
            hidden_channels + c_in_channels, filter_channels_dp, kernel_size,
            dropout_p)

    def forward(self, x, x_lengths, g=None):
        # embedding layer
        # [B ,T, D]
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        # [B, D, T]
        x = torch.transpose(x, 1, -1)
        # compute input sequence mask
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)),
                                 1).to(x.dtype)
        # pre-conv layers
        if self.encoder_type == 'transformer':
            if self.use_prenet:
                x = self.pre(x, x_mask)
        # encoder
        x = self.encoder(x, x_mask)
        # set duration predictor input
        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)
        # final projection layer
        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)
        # duration predictor
        logw = self.duration_predictor(x_dp, x_mask)
        return x_m, x_logs, logw, x_mask
