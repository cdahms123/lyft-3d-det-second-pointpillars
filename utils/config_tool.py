# config_tool.py

import numpy as np 

def change_detection_range(model_config, new_range):
    assert len(new_range) == 4, "you must provide a list such as [-50, -50, 50, 50]"
    old_pc_range = list(model_config.voxel_generator.point_cloud_range)
    old_pc_range[:2] = new_range[:2]
    old_pc_range[3:5] = new_range[2:]
    model_config.voxel_generator.point_cloud_range[:] = old_pc_range
    for anchor_generator in model_config.target_assigner.anchor_generators:
        a_type = anchor_generator.WhichOneof('anchor_generator')
        if a_type == "anchor_generator_range":
            a_cfg = anchor_generator.anchor_generator_range
            old_a_range = list(a_cfg.anchor_ranges)
            old_a_range[:2] = new_range[:2]
            old_a_range[3:5] = new_range[2:]
            a_cfg.anchor_ranges[:] = old_a_range
        elif a_type == "anchor_generator_stride":
            a_cfg = anchor_generator.anchor_generator_stride
            old_offset = list(a_cfg.offsets)
            stride = list(a_cfg.strides)
            old_offset[0] = new_range[0] + stride[0] / 2
            old_offset[1] = new_range[1] + stride[1] / 2
            a_cfg.offsets[:] = old_offset
        else:
            raise ValueError("unknown")
    old_post_range = list(model_config.post_center_limit_range)
    old_post_range[:2] = new_range[:2]
    old_post_range[3:5] = new_range[2:]
    model_config.post_center_limit_range[:] = old_post_range
# end function

def get_downsample_factor(model_config):
    downsample_factor = np.prod(model_config['rpn']['layer_strides'])
    if len(model_config['rpn']['upsample_strides']) > 0:
        downsample_factor /= model_config['rpn']['upsample_strides'][-1]
    # end if
    downsample_factor *= model_config['middle_feature_extractor']['downsample_factor']
    downsample_factor = int(downsample_factor)
    assert downsample_factor > 0
    return downsample_factor
# end function



