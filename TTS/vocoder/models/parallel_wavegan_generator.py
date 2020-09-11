import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from TTS.vocoder.layers.parallel_wavegan import ResidualBlock
from TTS.vocoder.layers.upsample import ConvUpsample


class ResBlock(nn.Module) :
    def __init__(self, dims) :
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x) :
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class ResNet(nn.Module) :
    def __init__(self, res_blocks, in_channels, hid_channels, out_channels) :
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hid_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm1d(hid_channels)
        self.layers = nn.ModuleList()
        for _ in range(res_blocks) :
            self.layers.append(ResBlock(hid_channels))
        self.conv_out = nn.Conv1d(hid_channels, out_channels, kernel_size=1)

    def forward(self, x) :
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers :
            x = f(x)
        x = self.conv_out(x)
        return x


class ParallelWaveganGenerator(torch.nn.Module):
    """PWGAN generator as in https://arxiv.org/pdf/1910.11480.pdf.
    It is similar to WaveNet with no causal convolution.
        It is conditioned on an aux feature (spectrogram) to generate
    an output waveform from an input noise.
    """
    # pylint: disable=dangerous-default-value
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 num_res_blocks=30,
                 stacks=3,
                 res_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80,
                 dropout=0.0,
                 bias=True,
                 use_weight_norm=True,
                 upsample_factors=[4, 4, 4, 4],
                 sample_rates=None,
                 inference_padding=2):

        super(ParallelWaveganGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.num_res_blocks = num_res_blocks
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.upsample_factors = upsample_factors
        self.sample_rates = sample_rates
        self.sample_rate = None
        self.upsample_scale = np.prod(upsample_factors)

        self.inference_padding = inference_padding

        # check the number of layers and stacks
        assert num_res_blocks % stacks == 0
        layers_per_stack = num_res_blocks // stacks

        # define first convolution
        self.first_conv = torch.nn.Conv1d(in_channels,
                                          res_channels,
                                          kernel_size=1,
                                          bias=True)

        # resnet block
        self.resnet = ResNet(8, 80, 128, 80)

        # define conv + upsampling network
        if isinstance(sample_rates, list):
            self.upsample_net_ids = [(sr, idx) for idx, sr in enumerate(sample_rates)]
            self.upsample_net_ids = dict(self.upsample_net_ids)
            self.upsample_nets = torch.nn.ModuleList([ConvUpsample(upsample_factors=upsample_factors) for sr in sample_rates])
        else:
            self.upsample_net = ConvUpsample(upsample_factors=upsample_factors)

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(num_res_blocks):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                res_channels=res_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList([
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(skip_channels,
                            skip_channels,
                            kernel_size=1,
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(skip_channels,
                            out_channels,
                            kernel_size=1,
                            bias=True),
        ])

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, c):
        """
            c: (B, C ,T').
            o: Output tensor (B, out_channels, T)
        """
        # random noise
        x = torch.randn([c.shape[0], 1, c.shape[2] * self.upsample_scale])
        x = x.to(self.first_conv.bias.device)

        # resnet for c
        c = self.resnet(c)

        # perform upsampling
        if self.sample_rate is not None:
            if c is not None and self.upsample_nets is not None:
                layer_id = self.upsample_net_ids[self.sample_rate]
                c = self.upsample_nets[layer_id](c)
                assert c.shape[-1] == x.shape[
                    -1], f" [!] Upsampling scale does not match the expected output. {c.shape} vs {x.shape}"
        else:
            if c is not None and self.upsample_net is not None:
                c = self.upsample_net(c)
                assert c.shape[-1] == x.shape[
                    -1], f" [!] Upsampling scale does not match the expected output. {c.shape} vs {x.shape}"

        # encode to hidden representation
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    @torch.no_grad()
    def inference(self, c):
        c = c.to(self.first_conv.weight.device)
        c = torch.nn.functional.pad(
            c, (self.inference_padding, self.inference_padding), 'replicate')
        return self.forward(c)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.weight_norm(m)
                # print(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers,
                                  stacks,
                                  kernel_size,
                                  dilation=lambda x: 2**x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        return self._get_receptive_field_size(self.layers, self.stacks,
                                              self.kernel_size)
