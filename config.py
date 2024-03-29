model = {
    'second': {
        'network_class_name': 'VoxelNet',
        'voxel_generator':  {
            'point_cloud_range' : [-50, -50, -10, 50, 50, 10],
            'voxel_size' : [0.25, 0.25, 20],
            'max_number_of_points_per_voxel' : 60
            },
        'voxel_feature_extractor': {
            'module_class_name': "PillarFeatureNet",
            'num_filters': [64],
            'with_distance': False,
            'num_input_features': 4
            },
        'middle_feature_extractor': {
            'module_class_name': "PointPillarsScatter",
            'downsample_factor': 1,
            'num_input_features': 64
            },
        'rpn': {
            'module_class_name': "RPNV2",
            'layer_nums': [3, 5, 5],
            'layer_strides': [2, 2, 2],
            'num_filters': [64, 128, 256],
            'upsample_strides': [0.25, 0.5, 1],
            'num_upsample_filters': [128, 128, 128],
            'num_input_features': 64
            },
        'loss': {
            'classification_loss': {
                'weighted_sigmoid_focal': {
                    'alpha': 0.25,
                    'gamma': 2.0,
                    'anchorwise_output': True
                }
            },
            'localization_loss': {
                'weighted_smooth_l1': {
                    'sigma': 3.0,
                    'code_weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    }
                },
            'classification_weight': 1.0,
            'localization_weight': 2.0
        },
        'num_point_features': 4, # model's num point feature should be independent of dataset
        # Outputs
        'use_sigmoid_score': True,
        'encode_background_as_zeros': True,
        'encode_rad_error_by_sin': True,
        'sin_error_factor': 1.0,

        'direction_loss_weight': 0.2,
        'num_direction_bins': 2,
        'direction_limit_offset': 0,
        'direction_offset': 0.78,

        # Loss
        'pos_class_weight': 1.0,
        'neg_class_weight': 1.0,

        'loss_norm_type': 'NormByNumPositives',
        # Postprocess
        'post_center_limit_range': [-59.6, -59.6, -10, 59.6, 59.6, 10],
        'nms_class_agnostic': False,   # only valid in multi-class nms
        'box_coder': {
            'ground_box3d_coder': {
                'linear_dim': False,
                'encode_angle_vector': False
            }
        },
        'target_assigner': {
            'class_settings_0': {
                'class_name': 'car',
                'anchor_generator_range': {
                    'sizes': [1.93017717, 4.76718145, 1.72270761],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -1.07, 49.6, 49.6, -1.07],
                    'rotations': [0, 1.57],   # DON'T modify this unless you are very familiar with my code.
                    # custom_values: [0, 0],   # velocity vector base value
                },
                'matched_threshold' : 0.4,
                'unmatched_threshold' : 0.3,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'class_settings_1': {
                'class_name': "bicycle",
                'anchor_generator_range': {
                    'sizes': [0.63058911, 1.76452161, 1.44192197],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -1.07, 49.6, 49.6, -1.07],
                    'rotations': [0, 1.57],   # DON'T modify this unless you are very familiar with my code
                    # custom_values: [0, 0],   # velocity vector base value
                },
                'matched_threshold' : 0.2,
                'unmatched_threshold' : 0.15,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'class_settings_2': {
                'class_name': "animal",
                'anchor_generator_range': {
                    'sizes': [0.36058911, 0.73452161, 0.5192197],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -1.79, 49.6, 49.6, -1.79],
                    'rotations': [0, 1.57],   # DON'T modify this unless you are very familiar with my code
                    # 'custom_values': [0, 0] # velocity vector base value
                },
                'matched_threshold' : 0.2,
                'unmatched_threshold' : 0.15,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'class_settings_3': {
                'class_name': "bus",
                'anchor_generator_range': {
                    'sizes': [2.96046906, 12.3485991, 3.44030982],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -0.35, 49.6, 49.6, -0.35],
                    'rotations': [0, 1.57],   # DON'T modify this unless you are very familiar with my code.
                    # custom_values: [0, 0] # velocity vector base value
                },
                'matched_threshold' : 0.3,
                'unmatched_threshold' : 0.15,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'class_settings_4': {
                'class_name': "emergency_vehicle",
                'anchor_generator_range': {
                    'sizes': [2.45046906, 6.5285991, 2.39030982],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -0.88, 49.6, 49.6, -0.88],
                    'rotations': [0, 1.57],   # DON'T modify this unless you are very familiar with my code.
                    # custom_values: [0, 0] # velocity vector base value
                },
                'matched_threshold' : 0.5,
                'unmatched_threshold' : 0.35,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'class_settings_5': {
                'class_name': "other_vehicle",
                'anchor_generator_range': {
                    'sizes': [2.79050468, 8.20352896, 3.23312415],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -0.62, 49.6, 49.6, -0.62],
                    'rotations': [0, 1.57],   # DON'T modify this unless you are very familiar with my code.
                    # custom_values: [0, 0] # velocity vector base value
                },
                'matched_threshold' : 0.5,
                'unmatched_threshold' : 0.3,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'class_settings_6': {
                'class_name': "motorcycle",
                'anchor_generator_range': {
                    'sizes': [0.96279481, 2.35973778, 1.59403034],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -1.32, 49.6, 49.6, -1.32],
                    'rotations': [0, 1.57],   # DON'T modify this unless you are very familiar with my code.
                    # custom_values: [0, 0] # velocity vector base value
                },
                'matched_threshold' : 0.2,
                'unmatched_threshold' : 0.15,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'class_settings_7': {
                'class_name': "pedestrian",
                'anchor_generator_range': {
                    'sizes': [0.77344886, 0.8156437, 1.78748069],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -0.91, 49.6, 49.6, -0.91],
                    'rotations': [0] # DON'T modify this unless you are very familiar with my code.
                    # custom_values: [0, 0] # velocity vector base value
                },
                'matched_threshold' : 0.2,
                'unmatched_threshold' : 0.15,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'class_settings_8': {
                'class_name': "truck",
                'anchor_generator_range': {
                    'sizes': [2.8460939, 10.24778078, 3.44004906],   # wlh
                    'anchor_ranges': [-49.6, -49.6, -0.30, 49.6, 49.6, -0.30],
                    'rotations': [0, 1.57] # DON'T modify this unless you are very familiar with my code.
                    # custom_values: [0, 0] # velocity vector base value
                },
                'matched_threshold' : 0.5,
                'unmatched_threshold' : 0.3,
                'use_rotate_nms': False,
                'use_multi_class_nms': False,
                'nms_pre_max_size': 1000,
                'nms_post_max_size': 300,
                'nms_score_threshold': 0.05,
                'nms_iou_threshold': 0.3,
                'region_similarity_calculator': 'nearest_iou_similarity'
            },
            'sample_positive_fraction': -1,
            'sample_size': 512,
            'assign_per_class': True
        }
    }
}

train_input_reader = {
    'dataset': {
        'dataset_class_name': "LyftDataset"
    },
    'batch_size': 4,
    'preprocess': {
        'max_number_of_voxels': 20000,
        'shuffle_points': False,
        'num_workers': 3,
        'groundtruth_localization_noise_std': [0, 0, 0],
        'groundtruth_rotation_uniform_noise': [0, 0],
        # groundtruth_localization_noise_std: [0.25, 0.25, 0.25],
        # groundtruth_rotation_uniform_noise: [-0.15707963267, 0.15707963267],
        'global_rotation_uniform_noise': [0, 0],
        'global_scaling_uniform_noise': [0.95, 1.05],
        'global_random_rotation_range_per_object': [0, 0],
        'global_translate_noise_std': [0.2, 0.2, 0.2],
        'anchor_area_threshold': -1,
        'remove_points_after_sample': True,
        'groundtruth_points_drop_percentage': 0.0,
        'groundtruth_drop_max_keep_points': 15,
        'remove_unknown_examples': False,
        'sample_importance': 1.0,
        'random_flip_x': True,
        'random_flip_y': True,
        'remove_environment': False
    }
}

train_config = {
    'optimizer': {
        'adam_optimizer': {
            'learning_rate': {
                'one_cycle': {
                    'lr_max': 3e-3,
                    'moms': [0.95, 0.85],
                    'div_factor': 10.0,
                    'pct_start': 0.4
                }
            },
            'weight_decay': 0.01
        },
        'fixed_weight_decay': True,
        'use_moving_average': False
    },
    'steps': 890670,   # 14065 * 20 (28130 // 2 )
    'steps_per_eval': 33445,  # 14065 * 2
    'save_checkpoints_secs' : 1800,   # half hour
    'save_summary_steps' : 10,
    'enable_mixed_precision': True,
    'loss_scale_factor': -1,
    'clear_metrics_every_epoch': True
}

eval_input_reader = {
    'dataset': {
        'dataset_class_name': "LyftDataset"
    },
    'batch_size': 2,
    'preprocess': {
        'max_number_of_voxels': 30000,
        'shuffle_points': False,
        'num_workers': 2,
        'anchor_area_threshold': -1,
        'remove_environment': False
    }
}



