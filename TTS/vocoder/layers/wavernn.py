import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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


class MelResNet(nn.Module) :
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad) :
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks) :
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x) :
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers : x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module) :
    def __init__(self, x_scale, y_scale) :
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x) :
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module) :
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad, use_aux_net) :
        super().__init__()
        self.total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * self.total_scale
        self.use_aux_net = use_aux_net
        if use_aux_net:
            self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
            self.resnet_stretch = Stretch2d(self.total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales :
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m) :
        if self.use_aux_net:
            aux = self.resnet(m).unsqueeze(1)
            aux = self.resnet_stretch(aux)
            aux = aux.squeeze(1)
            aux = aux.transpose(1, 2)
        else:
            aux = None
        m = m.unsqueeze(1)
        for f in self.up_layers : m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux


class Upsample(nn.Module):
    def __init__(self, scale, pad, res_blocks, feat_dims, compute_dims, res_out_dims, use_aux_net):
        super().__init__()
        self.scale = scale
        self.pad = pad
        self.indent = pad * scale
        self.use_aux_net = use_aux_net
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)

    def forward(self, m):
        if self.use_aux_net:
            aux = self.resnet(m)
            aux = torch.nn.functional.interpolate(aux, scale_factor= self.scale, mode='linear', align_corners=True)
            aux = aux.transpose(1, 2)
        else:
            aux = None
        m = torch.nn.functional.interpolate(m, scale_factor= self.scale, mode='linear', align_corners=True)
        m = m[:, :, self.indent:-self.indent]
        m = m * 0.045   # empirically found

        return m.transpose(1, 2), aux
