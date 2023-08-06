# rpn.py

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet

from torchplus.nn.modules.common import Empty, Sequential
from torchplus.nn.modules.normalization import GroupNorm
from torchplus.tools import change_default_args

class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        super().__init__()

        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._box_code_size = box_code_size

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters, num_anchor_per_loc * box_code_size, 1)
        
        self.conv_dir_cls = nn.Conv2d(final_num_filters, num_anchor_per_loc * num_direction_bins, 1)
    # end function

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor
    # end function

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes
    # end function

    def forward(self, x):
        ups = []
        stage_outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))

        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        res["out"] = x

        x = res["out"]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc, self._num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        dir_cls_preds = self.conv_dir_cls(x)
        dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(0, 1, 3, 4, 2).contiguous()
        # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_cls_preds": dir_cls_preds
        }

        return preds_dict
    # end function



