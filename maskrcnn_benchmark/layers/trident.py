import torch
import torch.nn as nn
import torch.nn.functional as F

from . import FrozenBatchNorm2d
from .dcn import ModulatedDeformConv2d, modulated_deform_conv
from .dcn import DeformConv2d, deform_conv


class TridentConv2d(nn.Conv2d):
    def forward(self, inputs):
        assert isinstance(inputs, (tuple, list))

        dilation = self.dilation
        if not isinstance(dilation, list):
            dilation = [dilation for _ in range(len(inputs))]

        padding = self.padding
        if not isinstance(padding, list):
            padding = [padding for _ in range(len(inputs))]

        '''
        print(len(inputs))
        for x in inputs:
            print(x.size())
        print(dilation, padding)
        '''

        out = [
            F.conv2d(x, self.weight, self.bias, self.stride,
                     padding[i], dilation[i], self.groups)
            for i, x in enumerate(inputs)
        ]
        return out


class TridentFrozenBatchNorm2d(FrozenBatchNorm2d):
    def forward(self, inputs):
        assert isinstance(inputs, (tuple, list))

        out = [
            super(TridentFrozenBatchNorm2d, self).forward(x)
            for x in inputs
        ]
        return out


class TridentDeformConv2d(DeformConv2d):
    def forward(self, inputs, offsets):
        assert isinstance(inputs, (tuple, list))
        assert isinstance(offsets, (tuple, list))

        dilation = self.dilation
        if not isinstance(dilation, list):
            dilation = [dilation for _ in range(len(inputs))]

        padding = self.padding
        if not isinstance(padding, list):
            padding = [padding for _ in range(len(inputs))]

        out = [
            deform_conv(x, offsets[i], self.weight, self.stride,
                        padding[i], dilation[i], self.groups,
                        self.deformable_groups)
            for i, x in enumerate(inputs)
        ]
        return out
