# pointpillars.py

import time
from enum import Enum
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from protos import voxel_generator_pb2
from protos import target_pb2
from spconv.utils import VoxelGeneratorV2
from core import region_similarity
from core.box_coders import BevBoxCoderTorch, GroundBox3dCoderTorch
from core.target_assigner import TargetAssigner
from core.anchor_generator import AnchorGeneratorStride, AnchorGeneratorRange
# from builder import anchor_generator_builder
# from builder import losses_builder
from protos import second_pb2

from core import losses
from core.ghm_loss import GHMCLoss, GHMRLoss
from protos import losses_pb2

import torchplus
from torchplus.nn.functional import one_hot
from core import box_torch_ops
from core.losses import WeightedSmoothL1LocalizationLoss, WeightedSoftmaxClassificationLoss
from models import pointpillars_aux, rpn
from torchplus import metrics

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss
# end function

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"
# end class

class PointPillars(nn.Module):
    def __init__(self, model_cfg, measure_time=False):
        super().__init__()

        voxel_generator = self.buildVoxelGenerator(model_cfg.voxel_generator)
        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        box_coder = self.buildBoxCoder(model_cfg.box_coder)
        target_assigner_cfg = model_cfg.target_assigner
        target_assigner = self.buildTargetAssigner(target_assigner_cfg, bv_range, box_coder)
        box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim

        if not isinstance(model_cfg, second_pb2.VoxelNet):
            raise ValueError('model_cfg not of type ' 'second_pb2.VoxelNet.')
        vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
        vfe_with_distance = model_cfg.voxel_feature_extractor.with_distance
        grid_size = voxel_generator.grid_size
        dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
        classes_cfg = model_cfg.target_assigner.class_settings
        num_class = len(classes_cfg)
        use_mcnms = [c.use_multi_class_nms for c in classes_cfg]
        use_rotate_nms = [c.use_rotate_nms for c in classes_cfg]
        if len(model_cfg.target_assigner.nms_pre_max_sizes) != 0:
            nms_pre_max_sizes = list(model_cfg.target_assigner.nms_pre_max_sizes)
            assert len(nms_pre_max_sizes) == num_class
        else:
            nms_pre_max_sizes = [c.nms_pre_max_size for c in classes_cfg]
        if len(model_cfg.target_assigner.nms_post_max_sizes) != 0:
            nms_post_max_sizes = list(model_cfg.target_assigner.nms_post_max_sizes)
            assert len(nms_post_max_sizes) == num_class
        else:
            nms_post_max_sizes = [c.nms_post_max_size for c in classes_cfg]
        if len(model_cfg.target_assigner.nms_score_thresholds) != 0:
            nms_score_thresholds = list(model_cfg.target_assigner.nms_score_thresholds)
            assert len(nms_score_thresholds) == num_class
        else:
            nms_score_thresholds = [c.nms_score_threshold for c in classes_cfg]
        if len(model_cfg.target_assigner.nms_iou_thresholds) != 0:
            nms_iou_thresholds = list(model_cfg.target_assigner.nms_iou_thresholds)
            assert len(nms_iou_thresholds) == num_class
        else:
            nms_iou_thresholds = [c.nms_iou_threshold for c in classes_cfg]
        assert all(use_mcnms) or all([not b for b in use_mcnms]), "not implemented"
        assert all(use_rotate_nms) or all([not b for b in use_rotate_nms]), "not implemented"
        if all([not b for b in use_mcnms]):
            assert all([e == nms_pre_max_sizes[0] for e in nms_pre_max_sizes])
            assert all([e == nms_post_max_sizes[0] for e in nms_post_max_sizes])
            assert all([e == nms_score_thresholds[0] for e in nms_score_thresholds])
            assert all([e == nms_iou_thresholds[0] for e in nms_iou_thresholds])
            # @ags: so, we gotta make all pre/pos max, score and iou thresholds equal??

        num_input_features = model_cfg.num_point_features
        loss_norm_type_dict = {
            0: LossNormType.NormByNumExamples,
            1: LossNormType.NormByNumPositives,
            2: LossNormType.NormByNumPosNeg,
            3: LossNormType.DontNorm,
        }
        loss_norm_type = loss_norm_type_dict[model_cfg.loss_norm_type]

        losses = build_losses(model_cfg.loss)
        encode_rad_error_by_sin = model_cfg.encode_rad_error_by_sin
        cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses
        pos_cls_weight = model_cfg.pos_class_weight
        neg_cls_weight = model_cfg.neg_class_weight
        direction_loss_weight = model_cfg.direction_loss_weight
        sin_error_factor = model_cfg.sin_error_factor
        if sin_error_factor == 0:
            sin_error_factor = 1.0
        # end if

        post_center_range = list(model_cfg.post_center_limit_range)
        use_norm = True
        output_shape = dense_shape

        self.name = 'voxelnet'
        self._sin_error_factor = sin_error_factor
        self._num_class = num_class
        self._use_rotate_nms = all(use_rotate_nms)
        self._multiclass_nms = all(use_mcnms)
        self._nms_score_thresholds = nms_score_thresholds
        self._nms_pre_max_sizes = nms_pre_max_sizes
        self._nms_post_max_sizes = nms_post_max_sizes
        self._nms_iou_thresholds = nms_iou_thresholds
        self._use_sigmoid_score = model_cfg.use_sigmoid_score
        self._encode_background_as_zeros = model_cfg.encode_background_as_zeros
        self._use_direction_classifier = model_cfg.use_direction_classifier
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self.target_assigner = target_assigner
        self.voxel_generator = voxel_generator
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._diff_loc_loss_ftor = WeightedSmoothL1LocalizationLoss()
        self._dir_offset = model_cfg.direction_offset
        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_weight
        self._loc_loss_weight = loc_weight
        self._post_center_range = post_center_range or []
        self.measure_time = measure_time
        self._nms_class_agnostic = model_cfg.nms_class_agnostic
        self._num_direction_bins = model_cfg.num_direction_bins
        self._dir_limit_offset = model_cfg.direction_limit_offset

        self.voxel_feature_extractor = pointpillars_aux.PillarFeatureNet(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=vfe_with_distance,
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range)

        self.middle_feature_extractor = pointpillars_aux.PointPillarsScatter(
            output_shape,
            use_norm,
            num_input_features=model_cfg.middle_feature_extractor.num_input_features,
            num_filters_down1=list(model_cfg.middle_feature_extractor.num_filters_down1),
            num_filters_down2=list(model_cfg.middle_feature_extractor.num_filters_down2))

        self.rpn = rpn.RPNV2(
            use_norm=True,
            num_class=num_class,
            layer_nums=list(model_cfg.rpn.layer_nums),
            layer_strides=list(model_cfg.rpn.layer_strides),
            num_filters=list(model_cfg.rpn.num_filters),
            upsample_strides=list(model_cfg.rpn.upsample_strides),
            num_upsample_filters=list(model_cfg.rpn.num_upsample_filters),
            num_input_features=model_cfg.rpn.num_input_features,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=model_cfg.encode_background_as_zeros,
            use_direction_classifier=model_cfg.use_direction_classifier,
            use_groupnorm=model_cfg.rpn.use_groupnorm,
            num_groups=model_cfg.rpn.num_groups,
            box_code_size=target_assigner.box_coder.code_size,
            num_direction_bins=self._num_direction_bins)

        self.rpn_acc = metrics.Accuracy(dim=-1, encode_background_as_zeros=model_cfg.encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=model_cfg.use_sigmoid_score,
            encode_background_as_zeros=model_cfg.encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()

        # initialize global step to zero
        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        # self.register_buffer("global_step", torch.tensor([0], dtype=torch.int64))

        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}
    # end function

    def buildVoxelGenerator(self, voxel_config):
        if not isinstance(voxel_config, voxel_generator_pb2.VoxelGenerator):
            raise ValueError('input_reader_config not of type '
                             'input_reader_pb2.InputReader.')
        voxel_generator = VoxelGeneratorV2(
            voxel_size=list(voxel_config.voxel_size),
            point_cloud_range=list(voxel_config.point_cloud_range),
            max_num_points=voxel_config.max_number_of_points_per_voxel,
            max_voxels=20000,
            full_mean=voxel_config.full_empty_part_with_mean,
            block_filtering=voxel_config.block_filtering,
            block_factor=voxel_config.block_factor,
            block_size=voxel_config.block_size,
            height_threshold=voxel_config.height_threshold)
        return voxel_generator
    # end function

    def buildBoxCoder(self, box_coder_config):
        box_coder_type = box_coder_config.WhichOneof('box_coder')
        if box_coder_type == 'ground_box3d_coder':
            cfg = box_coder_config.ground_box3d_coder
            return GroundBox3dCoderTorch(cfg.linear_dim, cfg.encode_angle_vector)
        elif box_coder_type == 'bev_box_coder':
            cfg = box_coder_config.bev_box_coder
            return BevBoxCoderTorch(cfg.linear_dim, cfg.encode_angle_vector, cfg.z_fixed, cfg.h_fixed)
        else:
            raise ValueError("unknown box_coder type")
        # end if
    # end function

    def buildTargetAssigner(self, target_assigner_config, bv_range, box_coder):
        if not isinstance(target_assigner_config, target_pb2.TargetAssigner):
            raise ValueError('input_reader_config not of type '
                             'input_reader_pb2.InputReader.')
        classes_cfg = target_assigner_config.class_settings
        anchor_generators = []
        classes = []
        feature_map_sizes = []
        for class_setting in classes_cfg:
            # anchor_generator = anchor_generator_builder.build(class_setting)
            anchor_generator = build_anchor_generator(class_setting)
            if anchor_generator is not None:
                anchor_generators.append(anchor_generator)
            else:
                assert target_assigner_config.assign_per_class is False
            classes.append(class_setting.class_name)
            feature_map_sizes.append(class_setting.feature_map_size)
        similarity_calcs = []
        for class_setting in classes_cfg:
            similarity_calcs.append(self.buildSimilarityCalculator(class_setting.region_similarity_calculator))

        positive_fraction = target_assigner_config.sample_positive_fraction
        if positive_fraction < 0:
            positive_fraction = None
        target_assigner = TargetAssigner(
            box_coder=box_coder,
            anchor_generators=anchor_generators,
            feature_map_sizes=feature_map_sizes,
            positive_fraction=positive_fraction,
            sample_size=target_assigner_config.sample_size,
            region_similarity_calculators=similarity_calcs,
            classes=classes,
            assign_per_class=target_assigner_config.assign_per_class)
        return target_assigner
    # end function

    def buildSimilarityCalculator(self, similarity_config):
        similarity_type = similarity_config.WhichOneof('region_similarity')
        if similarity_type == 'rotate_iou_similarity':
            return region_similarity.RotateIouSimilarity()
        elif similarity_type == 'nearest_iou_similarity':
            return region_similarity.NearestIouSimilarity()
        elif similarity_type == 'distance_similarity':
            cfg = similarity_config.distance_similarity
            return region_similarity.DistanceSimilarity(
                distance_norm=cfg.distance_norm,
                with_rotation=cfg.with_rotation,
                rotation_alpha=cfg.rotation_alpha)
        else:
            raise ValueError("unknown similarity type")
        # end if
    # end function

    def start_timer(self, *names):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        for name in names:
            self._time_dict[name] = time.time()
    # end function

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0
    # end function

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()
    # end function

    @contextlib.contextmanager
    def profiler(self):
        old_measure_time = self.measure_time
        self.measure_time = True
        yield
        self.measure_time = old_measure_time
    # end function

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret
    # end function

    def update_global_step(self):
        self.global_step += 1
    # end function

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])
    # end function

    def clear_global_step(self):
        self.global_step.zero_()
    # end function

    def loss(self, example, preds_dict):
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        batch_size_dev = cls_preds.shape[0]
        self.start_timer("loss forward")
        labels = example['labels']
        reg_targets = example['reg_targets']
        importance = example['importance']
        self.start_timer("prepare weight forward")
        cls_weights, reg_weights, cared = prepare_loss_weights(
            labels,
            pos_cls_weight=self._pos_cls_weight,
            neg_cls_weight=self._neg_cls_weight,
            loss_norm_type=self._loss_norm_type,
            dtype=box_preds.dtype)

        cls_targets = labels * cared.type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)
        self.end_timer("prepare weight forward")
        self.start_timer("create_loss forward")
        loc_loss, cls_loss = create_loss(
            self._loc_loss_ftor,
            self._cls_loss_ftor,
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights * importance,
            reg_targets=reg_targets,
            reg_weights=reg_weights * importance,
            num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            encode_background_as_zeros=self._encode_background_as_zeros,
            box_code_size=self._box_coder.code_size,
            sin_error_factor=self._sin_error_factor,
            num_direction_bins=self._num_direction_bins,
        )
        loc_loss_reduced = loc_loss.sum() / batch_size_dev
        loc_loss_reduced *= self._loc_loss_weight
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self._pos_cls_weight
        cls_neg_loss /= self._neg_cls_weight
        cls_loss_reduced = cls_loss.sum() / batch_size_dev
        cls_loss_reduced *= self._cls_loss_weight
        loss = loc_loss_reduced + cls_loss_reduced
        self.end_timer("create_loss forward")
        if self._use_direction_classifier:
            dir_targets = get_direction_target(
                example['anchors'],
                reg_targets,
                dir_offset=self._dir_offset,
                num_bins=self._num_direction_bins)
            dir_logits = preds_dict["dir_cls_preds"].view(
                batch_size_dev, -1, self._num_direction_bins)
            weights = (labels > 0).type_as(dir_logits) * importance
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self._dir_loss_ftor(
                dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size_dev
            loss += dir_loss * self._direction_loss_weight
        self.end_timer("loss forward")
        res = {
            "loss": loss,
            "cls_loss": cls_loss,
            "loc_loss": loc_loss,
            "cls_pos_loss": cls_pos_loss,
            "cls_neg_loss": cls_neg_loss,
            "cls_preds": cls_preds,
            "cls_loss_reduced": cls_loss_reduced,
            "loc_loss_reduced": loc_loss_reduced,
            "cared": cared,
        }
        if self._use_direction_classifier:
            res["dir_loss_reduced"] = dir_loss
        return res
    # end function

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        if len(num_points.shape) == 2:  # multi-gpu
            num_voxel_per_batch = example["num_voxels"].cpu().numpy().reshape(
                -1)
            voxel_list = []
            num_points_list = []
            coors_list = []
            for i, num_voxel in enumerate(num_voxel_per_batch):
                voxel_list.append(voxels[i, :num_voxel])
                num_points_list.append(num_points[i, :num_voxel])
                coors_list.append(coors[i, :num_voxel])
            voxels = torch.cat(voxel_list, dim=0)
            num_points = torch.cat(num_points_list, dim=0)
            coors = torch.cat(coors_list, dim=0)
        batch_anchors = example["anchors"]

        # ToDo: what is batch_size_dev (i.e. how is this different from batch size ?? what does "dev" mean ??)
        batch_size_dev = batch_anchors.shape[0]

        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size_dev)
        preds_dict = self.rpn(spatial_features)


        # need to check size.
        box_preds = preds_dict["box_preds"].view(batch_size_dev, -1, self._box_coder.code_size)
        err_msg = f"num_anchors={batch_anchors.shape[1]}, but num_output={box_preds.shape[1]}. please check size"
        assert batch_anchors.shape[1] == box_preds.shape[1], err_msg
        if self.training:
            return self.loss(example, preds_dict)
        else:
            with torch.no_grad():
                res = self.predict(example, preds_dict)
            # end with
            return res
        # end if
    # end function

    def predict(self, example, preds_dict):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information
            }
        """
        batch_size = example['anchors'].shape[0]
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]
        batch_anchors = example["anchors"].view(batch_size, -1,
                                                example["anchors"].shape[-1])
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1,
                                                   self._num_direction_bins)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        post_center_range = None
        if len(self._post_center_range) > 0:
            post_center_range = torch.tensor(
                self._post_center_range,
                dtype=batch_box_preds.dtype,
                device=batch_box_preds.device).float()
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_anchors_mask, meta_list):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms
            feature_map_size_prod = batch_box_preds.shape[
                1] // self.target_assigner.num_anchors_per_location
            if self._multiclass_nms:
                assert self._encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)

                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []

                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = self._nms_score_thresholds
                pre_max_sizes = self._nms_pre_max_sizes
                post_max_sizes = self._nms_post_max_sizes
                iou_thresholds = self._nms_iou_thresholds
                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                        range(self._num_class),
                        score_threshs,
                        pre_max_sizes, post_max_sizes, iou_thresholds):
                    if self._nms_class_agnostic:
                        class_scores = total_scores.view(
                            feature_map_size_prod, -1,
                            self._num_class)[..., class_idx]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = boxes.view(-1,
                                                     boxes_for_nms.shape[-1])
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        anchors_range = self.target_assigner.anchors_range(class_idx)
                        class_scores = total_scores.view(
                            -1,
                            self._num_class)[anchors_range[0]:anchors_range[1], class_idx]
                        class_boxes_nms = boxes.view(-1,
                            boxes_for_nms.shape[-1])[anchors_range[0]:anchors_range[1], :]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = class_boxes_nms.contiguous().view(
                            -1, boxes_for_nms.shape[-1])
                        class_boxes = box_preds.view(-1,
                            box_preds.shape[-1])[anchors_range[0]:anchors_range[1], :]
                        class_boxes = class_boxes.contiguous().view(
                            -1, box_preds.shape[-1])
                        if self._use_direction_classifier:
                            class_dir_labels = dir_labels.view(-1)[anchors_range[0]:anchors_range[1]]
                            class_dir_labels = class_dir_labels.contiguous(
                            ).view(-1)
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[
                                class_scores_keep]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[
                                class_scores_keep]
                        keep = nms_func(class_boxes_nms, class_scores, pre_ms,
                                        post_ms, iou_th)
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]

                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(
                            torch.full([class_boxes[selected].shape[0]],
                                       class_idx,
                                       dtype=torch.int64,
                                       device=box_preds.device))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(
                                class_dir_labels[selected])
                        selected_scores.append(class_scores[selected])
                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self._use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(
                        total_scores, dim=-1)
                if self._nms_score_thresholds[0] > 0.0:
                    top_scores_keep = top_scores >= self._nms_score_thresholds[0]
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if self._nms_score_thresholds[0] > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_sizes[0],
                        post_max_size=self._nms_post_max_sizes[0],
                        iou_threshold=self._nms_iou_thresholds[0],
                    )
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    period = (2 * np.pi / self._num_direction_bins)
                    dir_rot = box_torch_ops.limit_period(
                        box_preds[..., 6] - self._dir_offset,
                        self._dir_limit_offset, period)
                    box_preds[
                        ...,
                        6] = dir_rot + self._dir_offset + period * dir_labels.to(
                            box_preds.dtype)
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                    torch.zeros([0, box_preds.shape[-1]],
                                dtype=dtype,
                                device=device),
                    "scores":
                    torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                    torch.zeros([0], dtype=top_labels.dtype, device=device),
                    "metadata":
                    meta,
                }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def metrics_to_float(self):
        self.rpn_acc.float()
        #self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self, cls_loss, loc_loss, cls_preds, labels, sampled):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        #prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        #prec = prec.numpy()
        #recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "loss": {
                "cls_loss": float(rpn_cls_loss),
                "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
                'loc_loss': float(rpn_loc_loss),
                "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            },
            "rpn_acc": float(rpn_acc),
            #"pr": {},
        }
        #for i, thresh in enumerate(self.rpn_metrics.thresholds):
        #    ret["pr"][f"prec@{int(thresh*100)}"] = float(prec[i])
        #    ret["pr"][f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        #self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()
    # end function

    @staticmethod
    def convert_norm_to_float(net):
        """
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        """
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            PointPillars.convert_norm_to_float(child)
        return net
    # end function
# end class

def build_anchor_generator(class_cfg):
    ag_type = class_cfg.WhichOneof('anchor_generator')

    if ag_type == 'anchor_generator_stride':
        config = class_cfg.anchor_generator_stride
        ag = AnchorGeneratorStride(
            sizes=list(config.sizes),
            anchor_strides=list(config.strides),
            anchor_offsets=list(config.offsets),
            rotations=list(config.rotations),
            match_threshold=class_cfg.matched_threshold,
            unmatch_threshold=class_cfg.unmatched_threshold,
            class_name=class_cfg.class_name,
            custom_values=list(config.custom_values))
        return ag
    elif ag_type == 'anchor_generator_range':
        config = class_cfg.anchor_generator_range
        ag = AnchorGeneratorRange(
            sizes=list(config.sizes),
            anchor_ranges=list(config.anchor_ranges),
            rotations=list(config.rotations),
            match_threshold=class_cfg.matched_threshold,
            unmatch_threshold=class_cfg.unmatched_threshold,
            class_name=class_cfg.class_name,
            custom_values=list(config.custom_values))
        return ag
    elif ag_type == 'no_anchor':
        return None
    else:
        raise ValueError(" unknown anchor generator type")
    # end if
# end function

def build_losses(loss_config):
    """Build losses based on the config.

    Builds classification, localization losses and optionally a hard example miner
    based on the config.

    Args:
      loss_config: A losses_pb2.Loss object.

    Returns:
      classification_loss: Classification loss object.
      localization_loss: Localization loss object.
      classification_weight: Classification loss weight.
      localization_weight: Localization loss weight.
      hard_example_miner: Hard example miner object.

    Raises:
      ValueError: If hard_example_miner is used with sigmoid_focal_loss.
    """
    classification_loss = _build_classification_loss(loss_config.classification_loss)
    localization_loss = _build_localization_loss(loss_config.localization_loss)
    classification_weight = loss_config.classification_weight
    localization_weight = loss_config.localization_weight
    hard_example_miner = None
    if loss_config.HasField('hard_example_miner'):
        raise ValueError('Pytorch don\'t support HardExampleMiner')
    return (classification_loss, localization_loss, classification_weight, localization_weight, hard_example_miner)
# end function

def _build_localization_loss(loss_config):
    """Builds a localization loss based on the loss config.

    Args:
      loss_config: A losses_pb2.LocalizationLoss object.

    Returns:
      Loss based on the config.

    Raises:
      ValueError: On invalid loss_config.
    """
    if not isinstance(loss_config, losses_pb2.LocalizationLoss):
        raise ValueError('loss_config not of type losses_pb2.LocalizationLoss.')

    loss_type = loss_config.WhichOneof('localization_loss')

    if loss_type == 'weighted_l2':
        config = loss_config.weighted_l2
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedL2LocalizationLoss(code_weight)

    if loss_type == 'weighted_smooth_l1':
        config = loss_config.weighted_smooth_l1
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedSmoothL1LocalizationLoss(config.sigma, code_weight)
    if loss_type == 'weighted_ghm':
        config = loss_config.weighted_ghm
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return GHMRLoss(config.mu, config.bins, config.momentum, code_weight)

    raise ValueError('Empty loss config.')
# end function

def _build_classification_loss(loss_config):
    """Builds a classification loss based on the loss config.

    Args:
      loss_config: A losses_pb2.ClassificationLoss object.

    Returns:
      Loss based on the config.

    Raises:
      ValueError: On invalid loss_config.
    """
    if not isinstance(loss_config, losses_pb2.ClassificationLoss):
        raise ValueError('loss_config not of type losses_pb2.ClassificationLoss.')

    loss_type = loss_config.WhichOneof('classification_loss')

    if loss_type == 'weighted_sigmoid':
        return losses.WeightedSigmoidClassificationLoss()

    if loss_type == 'weighted_sigmoid_focal':
        config = loss_config.weighted_sigmoid_focal
        # alpha = None
        # if config.HasField('alpha'):
        #   alpha = config.alpha
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SigmoidFocalClassificationLoss(gamma=config.gamma, alpha=alpha)
    # end if

    if loss_type == 'weighted_softmax_focal':
        config = loss_config.weighted_softmax_focal
        # alpha = None
        # if config.HasField('alpha'):
        #   alpha = config.alpha
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        # end if
        return losses.SoftmaxFocalClassificationLoss(gamma=config.gamma, alpha=alpha)
    # end if

    if loss_type == 'weighted_ghm':
        config = loss_config.weighted_ghm
        return GHMCLoss(bins=config.bins, momentum=config.momentum)
    # end if

    if loss_type == 'weighted_softmax':
        config = loss_config.weighted_softmax
        return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.logit_scale)
    # end if

    if loss_type == 'bootstrapped_sigmoid':
        config = loss_config.bootstrapped_sigmoid
        return losses.BootstrappedSigmoidClassificationLoss(alpha=config.alpha,
                                                            bootstrap_type=('hard' if config.hard_bootstrap else 'soft'))
    # end if

    raise ValueError('Empty loss config.')
# end function

def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot, factor=1.0):
    if factor != 1.0:
        boxes1_rot = factor * boxes1_rot
        boxes2_rot = factor * boxes2_rot
    rad_pred_encoding = torch.sin(boxes1_rot) * torch.cos(boxes2_rot)
    rad_tg_encoding = torch.cos(boxes1_rot) * torch.sin(boxes2_rot)
    boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]],
                       dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                       dim=-1)
    return boxes1, boxes2
# end function

def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                sin_error_factor=1.0,
                box_code_size=7,
                num_direction_bins=2):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.functional.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        # reg_tg_rot = box_torch_ops.limit_period(
        #     reg_targets[..., 6:7], 0.5, 2 * np.pi / num_direction_bins)
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets, box_preds[..., 6:7], reg_targets[..., 6:7],
                                                    sin_error_factor)

    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses
# end function

def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared
# end function

def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights
# end function

def get_direction_target(anchors,
                         reg_targets,
                         one_hot=True,
                         dir_offset=0,
                         num_bins=2):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = box_torch_ops.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_cls_targets = torchplus.nn.functional.one_hot(
            dir_cls_targets, num_bins, dtype=anchors.dtype)
    return dir_cls_targets
# end function



