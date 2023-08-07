# rpn.py

import numpy as np
import torch
from torch import nn

class RPN(nn.Module):
    def __init__(self,
                 num_classes,
                 layer_nums,
                 layer_strides,
                 num_filters,
                 upsample_strides,
                 num_upsample_filters,
                 num_input_features,
                 num_anchor_per_loc,
                 encode_background_as_zeros,
                 box_code_size,
                 num_direction_bins):
        super().__init__()

        assert len(layer_nums) >= 1, 'layer nums must be at least 1'
        assert len(layer_nums) == len(layer_strides) == len(num_filters) == len(num_upsample_filters) == len(upsample_strides)

        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(layer_strides[:i + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

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
            if i >= 0:
                stride = upsample_strides[i]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=num_out_filters,
                                           out_channels=num_upsample_filters[i],
                                           kernel_size=stride, stride=stride, bias=False),
                        nn.BatchNorm2d(num_features=num_upsample_filters[i],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        nn.Conv2d(num_out_filters, num_upsample_filters[i],
                                  kernel_size=stride, stride=stride, bias=False),
                        nn.BatchNorm2d(num_features=num_upsample_filters[i],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    )
                # end if
                deblocks.append(deblock)
            # end if
        # end for

        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_classes = num_classes
        self._box_code_size = box_code_size

        # current settings:
        # num_anchor_per_loc = 17
        # num_direction_bins = 2
        # num_classes = 9
        # box_code_size = 7

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_classes
        else:
            num_cls = num_anchor_per_loc * (num_classes + 1)
        # end if

        final_num_filters = sum(num_upsample_filters)

        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters, num_anchor_per_loc * box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(final_num_filters, num_anchor_per_loc * num_direction_bins, 1)
    # end function

    def _make_layer(self, inplanes, planes, num_blocks, stride):
        layer_list = []
        layer_list.append(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layer_list.append(nn.BatchNorm2d(num_features=planes, eps=1e-3, momentum=0.01))
        layer_list.append(nn.ReLU())

        for j in range(num_blocks):
            layer_list.append(nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1, bias=False))
            layer_list.append(nn.BatchNorm2d(num_features=planes, eps=1e-3, momentum=0.01))
            layer_list.append(nn.ReLU())
        # end for

        layer_block = nn.Sequential(*layer_list)

        return layer_block, planes
    # end function

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i >= 0:
                ups.append(self.deblocks[i](x))
            # end if
        # end for
        # ups: [(4, 128, 50, 50), (4, 128, 50, 50), (4, 128, 50, 50), (4, 128, 50, 50)], torch.float32
        x = torch.cat(ups, dim=1)
        # (4, 384, 50, 50), torch.float32

        _, _, H, W = x.shape

        # (4, 384, 50, 50)
        cls_preds = self.conv_cls(x)
        # (4, 153, 50, 50)
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc, self._num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
        # (4, 17, 50, 50, 9), 9 => num_classes

        # (4, 384, 50, 50)
        box_preds = self.conv_box(x)
        # (4, 119, 50, 50)
        box_preds = box_preds.view(-1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
        # (4, 17, 50, 50, 7), 7 => x, y, z, w, l, h, yaw

        # (4, 384, 50, 50)
        dir_cls_preds = self.conv_dir_cls(x)
        # (4, 34, 50, 50)
        dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(0, 1, 3, 4, 2).contiguous()
        # (4, 17, 50, 50, 2), 2 =>forward, backward

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_cls_preds": dir_cls_preds
        }
        # cls_preds:     (4, 17, 50, 50, 9), 9 => num_classes
        # box_preds:     (4, 17, 50, 50, 7), 7 => x, y, z, w, l, h, yaw
        # dir_cls_preds: (4, 17, 50, 50, 2), 2 =>forward, backward
        return preds_dict
    # end function
# end class



