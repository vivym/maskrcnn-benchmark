from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import TridentFrozenBatchNorm2d
from maskrcnn_benchmark.layers import TridentDeformConv2d
from maskrcnn_benchmark.layers import TridentConv2d
from maskrcnn_benchmark.layers.dcn import DeformConv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry

from .resnet import BottleneckWithFixedBatchNorm
from .resnet import BottleneckWithGN
from .resnet import StemWithFixedBatchNorm
from .resnet import StemWithGN


# TridentResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index", # Index of the stage, eg 1, 2, ..,. 5
        "block_count", # Numer of residual blocks in the stage
        "return_features", # True => return the last feature map from this stage
        "trident",
    ],
)

# -----------------------------------------------------------------------------
# Trident ResNet models
# -----------------------------------------------------------------------------
# TridentResNet-50 (including all stages)
TridentResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r, trident=t)
    for (i, c, r, t) in (
            (1, 3, False, False),
            (2, 4, False, False),
            (3, 6, False, True),
            (4, 3, True, False),
    )
)

# ResNet-101 (including all stages)
TridentResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r, trident=t)
    for (i, c, r, t) in (
            (1, 3, False, False),
            (2, 4, False, False),
            (3, 23, False, True),
            (4, 3, True, False),
    )
)

# ResNet-50-FPN (including all stages)
TridentResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r, trident=t)
    for (i, c, r, t) in (
            (1, 3, True, False),
            (2, 4, True, False),
            (3, 6, True, True),
            (4, 3, True, False),
    )
)

# ResNet-101-FPN (including all stages)
TridentResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r, trident=t)
    for (i, c, r, t) in (
            (1, 3, True, False),
            (2, 4, True, False),
            (3, 23, True, True),
            (4, 3, True, False),
    )
)

# ResNet-152-FPN (including all stages)
TridentResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r, trident=t)
    for (i, c, r, t) in (
            (1, 3, True, False),
            (2, 8, True, False),
            (3, 36, True, True),
            (4, 3, True, False),
    )
)


class TridentResNet(nn.Module):
    def __init__(self, cfg):
        super(TridentResNet, self).__init__()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]
        trident_transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRIDENT_TRANS_FUNC]

        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS

        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            if stage_spec.trident:
                module = _make_trident_stage(
                    transformation_module,
                    trident_transformation_module,
                    in_channels,
                    bottleneck_channels,
                    out_channels,
                    stage_spec.block_count,
                    num_groups,
                    cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                    first_stride=int(stage_spec.index > 1) + 1,
                )
            else:
                module = _make_resnet_stage(
                    transformation_module,
                    in_channels,
                    bottleneck_channels,
                    out_channels,
                    stage_spec.block_count,
                    num_groups,
                    cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                    first_stride=int(stage_spec.index > 1) + 1,
                )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if isinstance(x, (tuple, list)):
                print('*' * 30)
                print(x[0].size())
                x = torch.cat(x)
                print(x.size())
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


def _make_trident_stage(
        transformation_module,
        trident_transformation_module,
        in_channels,
        bottleneck_channels,
        out_channels,
        block_count,
        num_groups,
        stride_in_1x1,
        first_stride,
        dilation=1,
        num_branches=3,
):
    blocks = []
    stride = first_stride

    for i in range(block_count):
        if i == 0:
            # a normal bottleneck block
            blocks.append(
                transformation_module(
                    in_channels,
                    bottleneck_channels,
                    out_channels,
                    num_groups,
                    stride_in_1x1,
                    stride,
                    dilation=dilation,
                )
            )
            blocks.append(StackBranchInput(num_branches))
        else:
            blocks.append(
                trident_transformation_module(
                    in_channels,
                    bottleneck_channels,
                    out_channels,
                    num_groups,
                    stride_in_1x1,
                    stride,
                    dilation=[1, 2, 3],
                )
            )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


def _make_resnet_stage(
        transformation_module,
        in_channels,
        bottleneck_channels,
        out_channels,
        block_count,
        num_groups,
        stride_in_1x1,
        first_stride,
        dilation=1
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class StackBranchInput(nn.Module):
    def __init__(self, num_branches):
        super(StackBranchInput, self).__init__()
        self.num_branches = num_branches

    def forward(self, x):
        return [x for _ in range(self.num_branches)]


class TridentBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
    ):
        super(TridentBottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                TridentConv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False,
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, TridentConv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = TridentConv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above

        self.conv2_offset = TridentConv2d(
            bottleneck_channels, 72,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )
        self.conv2 = TridentDeformConv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=4,
        )
        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = TridentConv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False,
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv2, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = [F.relu_(o) for o in x]

        offset = self.conv2_offset(x)
        x = self.conv2(x, offset)
        x = self.bn2(x)
        x = [F.relu_(o) for o in x]

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = [F.relu_(o + i) for o, i in zip(x, identity)]

        return x


class TridentBottleneckWithFixedBatchNorm(TridentBottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
    ):
        super(TridentBottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=TridentFrozenBatchNorm2d,
        )


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
    "TridentBottleneckWithFixedBatchNorm": TridentBottleneckWithFixedBatchNorm,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "T-R-50-C5": TridentResNet50StagesTo5,
    "T-R-101-C5": TridentResNet101StagesTo5,
    "T-R-50-FPN": TridentResNet50FPNStagesTo5,
    "T-R-101-FPN": TridentResNet101FPNStagesTo5,
    "T-R-152-FPN": TridentResNet152FPNStagesTo5,
})
