# my_lyft_dataset.py

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.eval.detection.mAP_evaluation import recall_precision

import os
import pathlib
import numpy as np
import json
import time
from tqdm import tqdm
import pyquaternion
from multiprocessing import Process
from typing import Dict, List, Union
from collections import defaultdict
import pprint
import torch
from torch.utils.data import Dataset

# local imports
from core import preprocess as prep
from core import box_np_ops
from protos import input_reader_pb2
from utils.config_tool import get_downsample_factor

class MyLyftDataset(Dataset):
    def __init__(self,
                 lyft: LyftDataset,
                 frameIds: List[str],
                 idxToFrameIdDict: Dict[int, str],
                 input_reader_config,
                 model_config,
                 training,
                 voxel_generator,
                 target_assigner):
        print('in MyLyftDataset init')

        self.lyft = lyft
        self.frameIds = frameIds
        self.idxToFrameIdDict = idxToFrameIdDict

        if not isinstance(input_reader_config, input_reader_pb2.InputReader):
            raise ValueError('input_reader_config is not type input_reader_pb2.InputReader.')
        # end if

        prep_cfg = input_reader_config.preprocess
        out_size_factor = get_downsample_factor(model_config)
        assert out_size_factor > 0

        grid_size = voxel_generator.grid_size
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        assert all([n != '' for n in target_assigner.classes]), 'all target_assigner classes must have non-blank names'

        genAnchorsRetVal = target_assigner.generate_anchors(feature_map_size)
        class_names = target_assigner.classes
        anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
        anchors_list = []
        for k, v in anchors_dict.items():
            anchors_list.append(v['anchors'])
        # end for

        anchors = np.concatenate(anchors_list, axis=0)
        anchors = anchors.reshape([-1, target_assigner.box_ndim])
        assert np.allclose(anchors, genAnchorsRetVal['anchors'].reshape(-1, target_assigner.box_ndim))
        matched_thresholds = genAnchorsRetVal['matched_thresholds']
        unmatched_thresholds = genAnchorsRetVal['unmatched_thresholds']
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            'anchors': anchors,
            'anchors_bv': anchors_bv,
            'matched_thresholds': matched_thresholds,
            'unmatched_thresholds': unmatched_thresholds,
            'anchors_dict': anchors_dict,
        }

        self._class_names = class_names
        self._voxel_generator = voxel_generator
        self._target_assigner = target_assigner
        self._training = training
        self._max_voxels = prep_cfg.max_number_of_voxels
        self._remove_unknown = prep_cfg.remove_unknown_examples
        self._gt_rotation_noise = list(prep_cfg.groundtruth_rotation_uniform_noise)
        self._gt_loc_noise_std = list(prep_cfg.groundtruth_localization_noise_std)
        self._global_rotation_noise = list(prep_cfg.global_rotation_uniform_noise)
        self._global_scaling_noise = list(prep_cfg.global_scaling_uniform_noise)
        self._global_random_rot_range = list(prep_cfg.global_random_rotation_range_per_object)
        self._global_translate_noise_std = list(prep_cfg.global_translate_noise_std)
        self._use_group_id = prep_cfg.use_group_id
        self._min_points_in_gt = prep_cfg.min_num_of_points_in_gt
        self._random_flip_x = prep_cfg.random_flip_x
        self._random_flip_y = prep_cfg.random_flip_y
        self._anchor_cache = anchor_cache
    # end function

    def __len__(self):
        return len(self.frameIds)
    # end function

    def __getitem__(self, idx):
        frameId = self.idxToFrameIdDict[idx]

        lyftInfoDict = self.getLyftInfoDict(frameId)

        itemDict = self.getItemDict(lyftInfoDict)

        return itemDict
    # end function

    def getLyftInfoDict(self, frameId: str) -> Dict:
        sampleData = self.lyft.get('sample', frameId)

        lidarTopId: str = sampleData['data']['LIDAR_TOP']

        lidar_path: pathlib.Path = self.lyft.get_sample_data_path(lidarTopId)

        sample_data_record = self.lyft.get('sample_data', sampleData['data']['LIDAR_TOP'])
        cal_sensor_record = self.lyft.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
        pose_record = self.lyft.get('ego_pose', sample_data_record['ego_pose_token'])

        lyftInfoDict = {
            'token': frameId,
            'lidar_path': lidar_path,
            'lidar2ego_rotation': cal_sensor_record['rotation'],
            'lidar2ego_translation': cal_sensor_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'ego2global_translation': pose_record['translation']
        }

        # if in training mode, prep the ground truth data and add to lyftInfoDict
        if self._training:
            boxes: List[Box] = self.lyft.get_boxes(lidarTopId)
            for i in range(len(boxes)):
                # move box to ego vehicle coord system
                boxes[i].translate(-np.array(pose_record["translation"]))
                boxes[i].rotate(pyquaternion.Quaternion(pose_record["rotation"]).inverse)

                # move box to sensor coord system
                boxes[i].translate(-np.array(cal_sensor_record["translation"]))
                boxes[i].rotate(pyquaternion.Quaternion(cal_sensor_record["rotation"]).inverse)
            # end for

            # locs = []
            # dims = []
            # yaws = []
            # for box in boxes:
            #     locs.append(box.center)
            #     dims.append(box.wlh)
            #     yaws.append(box.orientation.yaw_pitch_roll[0])
            # # end for
            #
            # locs = np.array(locs, np.float64)
            # # locs should be n rows x 3 cols, n rows is number of boxes in the frame, cols are x, y, z
            # assert locs.shape[1] == 3, 'error, locs.shape[1] = ' + str(locs.shape[1]) + ', should be 3'
            #
            # dims = np.array(dims, np.float64)
            # # dims should be n rows x 3 cols, n rows is number of boxes in the frame, cols are w, l, h
            # assert dims.shape[1] == 3, 'error, dims.shape[1] = ' + str(dims.shape[1]) + ', should be 3'
            #
            # yaws = np.array(yaws).reshape(-1, 1)
            # # yaws should be n rows x 1 col, n rows is the number of boxes in the frame, col is yaw angle
            # assert yaws.shape[1] == 1, 'error, yaws.shape[1] = ' + str(yaws.shape[1]) + ', should be 1'
            #
            # gt_boxes = np.concatenate([locs, dims, -yaws - np.pi / 2], axis=1)
            # # gt_boxes should be n rows x 7 col, n rows is the number of boxes in the frame, cols are x, y, z, w, l, h, yaw
            # assert gt_boxes.shape[1] == 7, 'error, gt_boxes.shape[1] = ' + str(gt_boxes.shape[1]) + ', should be 7'

            gt_boxes = np.zeros((len(boxes), 7), dtype=np.float64)
            for i, box in enumerate(boxes):
                gt_boxes[i, 0:3] = box.center   # x, y, z
                gt_boxes[i, 3:6] = box.wlh      # w, l, h
                yaw = box.orientation.yaw_pitch_roll[0]
                # ToDo: need to understand this yaw conversion better
                yaw = -yaw - np.pi / 2
                gt_boxes[i, 6] = yaw            # yaw
            # end for

            lyftInfoDict['gt_boxes'] = gt_boxes

            names = [b.name for b in boxes]
            names = np.array(names)
            lyftInfoDict['gt_names'] = names
        # end if

        return lyftInfoDict
    # end function

    def getItemDict(self, lyftInfoDict: Dict) -> Union[Dict, None]:
        # there is a bad data point in the Lyft dataset, file name host-a011_lidar1_1233090652702363606.bin,
        # sample_data_token 25cca7dd22b1f0a2c7664ddfa285694193a80d033896d4df70cb0e29a3d466e2, which will cause a crash
        # when read in when reshaped to (-1, 5), so when this data point is encountered return None
        lidarFileName = str(lyftInfoDict['lidar_path']).split('/')[-1]
        if lidarFileName == 'host-a011_lidar1_1233090652702363606.bin':
            return None
        # end if

        lidarPointCloud: LidarPointCloud = LidarPointCloud.from_file(lyftInfoDict['lidar_path'])
        points = lidarPointCloud.points.transpose()
        # points is now n rows x 4 cols (x, y, z, intensity)
        # set the 4th column (intensity) to all zeros
        points[: 3] = 0

        # this if block has to go before voxel_generator.generate(points, . . .) call below b/c this if block modifies points
        if self._training:
            gt_dict = {
                'gt_boxes': lyftInfoDict['gt_boxes'],
                'gt_names': lyftInfoDict['gt_names'],
                'gt_importance': np.ones([lyftInfoDict['gt_boxes'].shape[0]], dtype=lyftInfoDict['gt_boxes'].dtype)
            }

            selected = self.drop_arrays_by_name(gt_dict['gt_names'], ['DontCare'])
            self._dict_select(gt_dict, selected)

            class_names = self._target_assigner.classes
            gt_boxes_mask = np.array([n in class_names for n in gt_dict['gt_names']], dtype=np.bool_)

            prep.noise_per_object_v3_(
                gt_dict['gt_boxes'],
                points,
                gt_boxes_mask,
                rotation_perturb=self._gt_rotation_noise,
                center_noise_std=self._gt_loc_noise_std,
                global_random_rot_range=self._global_random_rot_range,
                group_ids=None,
                num_try=100)

            self._dict_select(gt_dict, gt_boxes_mask)
            gt_classes = np.array([class_names.index(n) + 1 for n in gt_dict['gt_names']], dtype=np.int32)
            gt_dict['gt_classes'] = gt_classes
            gt_dict['gt_boxes'], points = prep.random_flip(gt_dict['gt_boxes'], points, 0.5, self._random_flip_x, self._random_flip_y)
            gt_dict['gt_boxes'], points = prep.global_rotation_v2(gt_dict['gt_boxes'], points, *self._global_rotation_noise)
            gt_dict['gt_boxes'], points = prep.global_scaling_v2(gt_dict['gt_boxes'], points, *self._global_scaling_noise)
            prep.global_translate_(gt_dict['gt_boxes'], points, self._global_translate_noise_std)
            bv_range = self._voxel_generator.point_cloud_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range_by_center(gt_dict['gt_boxes'], bv_range)
            self._dict_select(gt_dict, mask)
            # limit rad to [-pi, pi]
            gt_dict['gt_boxes'][:, 6] = box_np_ops.limit_period(gt_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi)
        # end if

        voxelDict = self._voxel_generator.generate(points, self._max_voxels)
        itemDict = {
            'voxels': voxelDict['voxels'],
            'num_points': voxelDict['num_points_per_voxel'],
            'coordinates': voxelDict['coordinates'],
            'num_voxels': np.array([voxelDict['voxels'].shape[0]], dtype=np.int64),
            'metrics': {},
            'anchors': self._anchor_cache['anchors'],
            'metadata': {'token': lyftInfoDict['token']}
        }

        # if in test mode (no ground truths) we're done !!
        if not self._training:
            return itemDict
        else:
            # noinspection PyUnboundLocalVariable
            itemDict['gt_names'] = gt_dict['gt_names']

            anchors_dict = self._anchor_cache['anchors_dict']
            matched_thresholds = self._anchor_cache['matched_thresholds']
            unmatched_thresholds = self._anchor_cache['unmatched_thresholds']

            targets_dict = self._target_assigner.assign(
                itemDict['anchors'],
                anchors_dict,
                gt_dict['gt_boxes'],
                anchors_mask=None,
                gt_classes=gt_dict['gt_classes'],
                gt_names=gt_dict['gt_names'],
                matched_thresholds=matched_thresholds,
                unmatched_thresholds=unmatched_thresholds,
                importance=gt_dict['gt_importance'])

            itemDict.update({'labels': targets_dict['labels'],
                             'reg_targets': targets_dict['bbox_targets'],
                             'importance': targets_dict['importance']})

            return itemDict
        # end if
    # end function

    def _dict_select(self, dict_, inds):
        for k, v in dict_.items():
            if isinstance(v, dict):
                self._dict_select(v, inds)
            else:
                dict_[k] = v[inds]
            # end if
        # end for
    # end function

    def drop_arrays_by_name(self, gt_names, used_classes):
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds
    # end function

    def evaluation_lyft(self, detections, output_dir):
        mapped_class_names = self._class_names

        # make predictions (list of dictionaries), which is really just a re-formatted version
        # of detections (which was the net output)
        print('preparing pred_data list ..')
        predictions: List[Dict] = []
        for det in tqdm(detections):
            boxes = second_det_to_lyft_box(det)

            frameId = det['metadata']['token']

            lyftInfoDict = self.getLyftInfoDict(frameId)

            boxes = lidar_lyft_box_to_global(lyftInfoDict, boxes)

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                lyft_anno = {
                    'sample_token': det['metadata']['token'],
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'name': name,
                    'score': box.score,
                }
                predictions.append(lyft_anno)
            # end for
        # end for

        # delete detections to save memory (no longer needed now that we have predictions)
        del detections

        startTime = time.time()

        os.makedirs(output_dir, exist_ok=True)

        # make ground truth data (list of dictionaries)
        print('preparing gt_data list . . .')
        gt_data: List[Dict] = []
        for idx in tqdm(range(len(self.frameIds))):
            frameId = self.idxToFrameIdDict[idx]
            lyftInfoDict = self.getLyftInfoDict(frameId)

            sample_token = lyftInfoDict['token']
            sample = self.lyft.get('sample', sample_token)
            for ann_token in sample['anns']:
                ann_record = self.lyft.get('sample_annotation', ann_token)
                try:
                    data = {
                        'sample_token': sample_token,
                        'translation': ann_record['translation'],
                        'size': ann_record['size'],
                        'rotation': ann_record['rotation'],
                        'name': ann_record['category_name']
                    }
                    gt_data.append(data)
                except Exception as ex:
                    print('error, ex: ' + str(ex))
                # end try
            # end for
        # end for

        print('\n' + 'gt_data: ')
        print('type(gt_data) = ' + str(type(gt_data)))
        print('len(gt_data) = ' + str(len(gt_data)))
        print('gt_data[0]: ')
        pprint.pprint(gt_data[0], sort_dicts=False)

        print('\n' + 'predictions: ')
        print('type(predictions) = ' + str(type(predictions)))
        print('len(predictions) = ' + str(len(predictions)))
        print('predictions[0]: ')
        pprint.pprint(predictions[0], sort_dicts=False)

        gt_by_class_name = group_by_key(gt_data, 'name')
        pred_by_class_name = group_by_key(predictions, 'name')

        print('\n' + 'gt_by_class_name.keys(): ')
        print(gt_by_class_name.keys())

        print('\n' + 'pred_by_class_name.keys(): ')
        print(pred_by_class_name.keys())
        print('')

        # Note: Take caution with the size of this array!! If this is too big it will crash on
        #       some computers, especially ones without huge memory!! A 32GB machine can't seem
        #       to handle more than 4 or 5 numbers in this array, and in some cases even that
        #       will cause a crash in the save_ap function due to all memory being used up.
        #       If you have a machine with 128GB memory you may be able to get away with as many
        #       as 10 numbers in this array.
        iouThresholds = np.array([0.5, 0.8], dtype=np.float64)

        # order doesn't matter here
        class_names = ['animal', 'bicycle', 'bus', 'car', 'emergency_vehicle', 'motorcycle', 'other_vehicle',
                       'pedestrian', 'truck']

        # must use multi-processing here or this step takes forever
        processes = []
        for iou_threshold in iouThresholds:
            process = Process(target=save_ap, args=(gt_data, predictions, class_names, iou_threshold, output_dir))
            processes.append(process)
        # end for

        for process in processes:
            process.start()
        # end for

        for process in processes:
            process.join()
            print('joined save_ap process')
        # end for

        # get overall metrics
        metric, overall_ap = get_metric_overall_ap(iouThresholds, output_dir, class_names)

        mAP = np.mean(overall_ap)

        metric['overall'] = dict()
        for idx, class_name in enumerate(class_names):
            metric['overall'][class_name] = overall_ap[idx]
        # end for

        metric['mAP'] = mAP

        summary_path = os.path.join(output_dir, 'metric_summary.json')
        with open(str(summary_path), 'w') as f:
            json.dump(metric, f, indent=4)
        # end with

        elapsedTime = time.time() - startTime
        print('eval_main time taken: ' + str(elapsedTime) + ' seconds')
    # end function

# end class

def save_ap(gt: List[Dict], predictions: List[Dict], class_names: List[str], iou_threshold: float, output_dir):
    print('entering save_ap, calling get_average_precisions with iou_threshold = ' + str(iou_threshold))
    startTime = time.time()
    ap = get_average_precisions(gt, predictions, class_names, iou_threshold)
    print('get_average_precisions with iou_threshold = ' + str(iou_threshold) + ' took ' + '{:.2f}'.format(time.time() - startTime) + ' seconds')

    metric = dict()
    for idx, class_name in enumerate(class_names):
        metric[class_name] = ap[idx]
    # end for

    print('\n' + 'type(metric) = ' + str(type(metric)))
    print('len(metric) = ' + str(len(metric)))
    pprint.pprint(metric)

    summary_path = os.path.join(output_dir, 'metric_summary_' + str(iou_threshold) + '.json')
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f)
    # end with
# end function

def get_average_precisions(gt: List[Dict], predictions: List[Dict], class_names: List[str], iou_threshold: float) -> np.array:
    """
    format for gt and predictions:

    gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
    }]

    predictions = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
        'name': 'car',
        'score': 0.3077029437237213
    }]
    """
    assert 0.0 <= iou_threshold <= 1.0

    gt_by_class_name = group_by_key(gt, 'name')
    pred_by_class_name = group_by_key(predictions, 'name')

    average_precisions = np.zeros(len(class_names))

    for class_id, class_name in enumerate(class_names):
        if class_name in gt_by_class_name and class_name in pred_by_class_name:
            recalls, precisions, average_precision = recall_precision(gt_by_class_name[class_name],
                                                                      pred_by_class_name[class_name],
                                                                      iou_threshold)
            average_precisions[class_id] = average_precision
        else:
            average_precisions[class_id] = 0.0
        # end if
    # end for
    return average_precisions
# end function

def group_by_key(detections, key):
    groups = defaultdict(list)
    for detection in detections:
        groups[detection[key]].append(detection)
    return groups
# end function

def get_metric_overall_ap(iouThresholds, output_dir, class_names):
    metric = {}
    overall_ap = np.zeros(len(class_names))
    for iou_threshold in iouThresholds:
        summary_path = os.path.join(output_dir, 'metric_summary_' + str(iou_threshold) + '.json')
        with open(str(summary_path), 'r') as f:
            data = json.load(f)
            metric[iou_threshold] = data
            overall_ap += np.array([data[c] for c in class_names])
        # end with
    # end for

    overall_ap /= len(iouThresholds)
    return metric, overall_ap
# end function

def example_convert_to_torch(batch: Dict, float_dtype, device) -> dict:
    torchBatch = {}
    for key, v in batch.items():
        if key in ['voxels', 'anchors', 'reg_targets', 'importance']:
            torchBatch[key] = torch.tensor(v, dtype=torch.float32, device=device).to(float_dtype)
        elif key in ['num_points', 'coordinates', 'labels']:
            torchBatch[key] = torch.tensor(v, dtype=torch.int32, device=device)
        elif key in ['num_voxels']:
            torchBatch[key] = torch.tensor(v)
        elif key in ['metrics', 'gt_names', 'metadata']:
            torchBatch[key] = v
        else:
            raise ValueError('error in example_convert_to_torch, key ' + str(key) + ' is not recognized')
        # end if
    # end for
    return torchBatch
# end function

def my_collate_fn(itemDictList: List[Dict]):
    """
    This function is a bit confusing and requires some explanation.

    The input parameter to this function is a list of itemDicts as returned by MyLyftDataset __getitem__,
    and itemDictList's length will be equal to the batch size.

    If this function was not used, the main training loop would crash when the DataLoaders are iterated
    through because the size of the value of 'voxels' is not the same from one itemDict to the next, therefore
    if this function was not used, the PyTorch built in batching algorithm, which calls torch.stack, would be
    used and a crash would occur.

    For the input itemDictList, supposing batch size = 4, itemDictList looks like this:
    itemDictList = [ itemDict0, itemDict1, itemDict2, itemDict3 ]
    where each itemDict has the following keys:
    voxels, num_points, coordinates, num_voxels, metrics, anchors, gt_names, labels, reg_targets, importance, metadata
    """

    # The first for loop changes the format from a list of dictionaries
    # to a single dictionary where each value is a list of 4 items:
    itemDictOfLists = dict()
    for itemDict in itemDictList:
        if itemDict is not None:
            for key, val in itemDict.items():
                if key not in itemDictOfLists: itemDictOfLists[key] = list()
                itemDictOfLists[key].append(val)
            # end for
        # end if
    # end for

    # This 2nd for loop changes the format to the necessary format to pass into the SECOND net
    batch = {}
    for key, elems in itemDictOfLists.items():
        if key in ['voxels', 'num_points', 'gt_names']:
            batch[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            batch[key] = np.concatenate(coors, axis=0)
        elif key in ['metrics', 'metadata']:
            batch[key] = elems
        elif key in ['num_voxels', 'anchors', 'labels', 'reg_targets', 'importance']:
            batch[key] = np.stack(elems, axis=0)
        else:
            raise ValueError('error in my_collate_fn, key ' + str(key) + ' is not recognized')
        # end if
    # end for

    # batch as returned here is exactly the same as batch provided by DataLoaders in the main training loop
    return batch
# end function

def second_det_to_lyft_box(detection):
    box3d = detection['box3d_lidar'].detach().cpu().numpy()
    scores = detection['scores'].detach().cpu().numpy()
    labels = detection['label_preds'].detach().cpu().numpy()
    box3d[:, 6] = -box3d[:, 6] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        velocity = (np.nan, np.nan, np.nan)
        if box3d.shape[1] == 9:
            velocity = (*box3d[i, 7:9], 0.0)
        # end if
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list
# end function

def lidar_lyft_box_to_global(info, boxes):
    import pyquaternion
    box_list = []
    for box in boxes:
        # lidar -> ego
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))

        # ego -> global
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    # end for
    return box_list
# end function



