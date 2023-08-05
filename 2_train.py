# 2_train.py

# installed imports
import copy
import os
import pprint
import time
import numpy as np
import torch
import torch.utils.data
import pyquaternion
import pickle
import json
import math
import random
import pandas as pd
from google.protobuf import text_format
from tqdm import tqdm
import termcolor
from typing import List, Dict
import psutil
import warnings
from termcolor import colored
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.utils.geometry_utils import quaternion_yaw

# local imports
import config
import my_lyft_dataset
from my_lyft_dataset import MyLyftDataset
from models.pointpillars import PointPillars
# from protos import pipeline_pb2
from utils.log_tool import SimpleModelLog

warnings.filterwarnings('ignore')

CONFIG_FILE_LOC = os.path.join(os.getcwd(), 'configs', 'settings.config')
RESULTS_LOC = os.path.join(os.getcwd(), 'results')
DISPLAY_STEP = 500
MEASURE_TIME = False
GRAPH_NAME = 'graph.pt'

NUM_EPOCHS = 2

TRAIN_DATA_ROOT_PATH = '../../LyftObjDetDataset/train'
TEST_DATA_ROOT_PATH = '../../LyftObjDetDataset/test'

TRAIN_VERSION = 'v1.0-trainval'
TEST_VERSION = 'v1.0-test'

# inputs in dataset dir
SAMPLE_SUBMISSION_LOC = '../../LyftObjDetDataset/sample_submission.csv'
# inputs in results dir
GRAPH_LOC = os.path.join(os.getcwd(), 'results', 'graph.pt')
# output
TEST_RESULTS_LOC = os.path.join(os.getcwd(), 'results', 'test_results.pkl')
VAL_PREDS_FILE_LOC = os.path.join(os.getcwd(), 'results', 'val_preds.csv')
OUTPUT_SUB_FILE_LOC = os.path.join(os.getcwd(), 'results', 'kaggle_sub.csv')

TEST_BATCH_SIZE = 2
PRED_THRESHOLD = 0.2

OVERALL_CAR_MAP_ACCEPTABLE_THRESH = 0.21    # 0.16 for 1/2, 0.21 for full
MAKE_SUBMISSION = True

# ToDo: move loss out of PointPillars class, move loss into separate file

# ToDo: move voxel_generator out of the dataset class

def main():
    np.set_printoptions(suppress=True)

    # check GPU availability
    if torch.cuda.is_available():
        device = 'cuda'
        print(termcolor.colored('\n' + 'using GPU' + '\n', 'green'))
    else:
        device = 'cpu'
        print(termcolor.colored('\n' + 'GPU does not seem to be available, using CPU' + '\n', 'red'))
    # end if

    # # check if the results directory already exists, if so show an error and bail
    # if os.path.isdir(RESULTS_LOC):
    #     print(termcolor.colored('RESULTS_LOC ' + str(RESULTS_LOC) + ' already exists !!', 'red'))
    #     print(termcolor.colored('either delete this directory or move it somewhere else to save it before training again', 'red'))
    #     print('')
    #     quit()
    # # end if

    os.makedirs(RESULTS_LOC, exist_ok=True)

    # # read in the config
    # config = pipeline_pb2.TrainEvalPipelineConfig()
    # with open(CONFIG_FILE_LOC, 'r') as f:
    #     proto_str = f.read()
    #     text_format.Merge(proto_str, config)
    # # end with

    # break out the various sub-parts of the config
    train_input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model['second']
    train_cfg = config.train_config

    # instantiate the net
    net = PointPillars(model_cfg, MEASURE_TIME)
    net = net.to(device)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    print('num parameters:', len(list(net.parameters())))

    optimizer = torch.optim.Adam(net.parameters())
    float_dtype = torch.float32

    lyftTrainVal = LyftDataset(data_path=TRAIN_DATA_ROOT_PATH, json_path=os.path.join(TRAIN_DATA_ROOT_PATH, 'data'))

    # train/val split

    allTrainValSceneIds = []
    for scene in lyftTrainVal.scene:
        allTrainValSceneIds.append(scene['token'])
    # end for

    random.shuffle(allTrainValSceneIds)

    splitIdx = round(len(allTrainValSceneIds) * 0.8)
    trainSceneIds = allTrainValSceneIds[0:splitIdx]
    valSceneIds = allTrainValSceneIds[splitIdx:]

    print('\n' + 'len(allTrainValSceneIds) = ' + str(len(allTrainValSceneIds)))
    print('len(trainSceneIds) = ' + str(len(trainSceneIds)))
    print('len(valSceneIds) = ' + str(len(valSceneIds)) + '\n')

    trainSceneIds = set(trainSceneIds)
    valSceneIds = set(valSceneIds)

    trainFrameIds = []
    valFrameIds = []
    for sample in lyftTrainVal.sample:
        if sample['scene_token'] in trainSceneIds:
            trainFrameIds.append(sample['token'])
        elif sample['scene_token'] in valSceneIds:
            valFrameIds.append(sample['token'])
        else:
            print(colored('sample[\'scene_token\'] = ' + str(sample['scene_token']) + ' not found in trainSceneIds or valSceneIds', 'red'))
            return
    # end for

    assert len(trainFrameIds) + len(valFrameIds) == len(lyftTrainVal.sample)

    trainIdxToFrameIdDict = dict()
    for i, trainFrameId in enumerate(trainFrameIds):
        trainIdxToFrameIdDict[i] = trainFrameId
    # end for

    valIdxToFrameIdDict = dict()
    for i, valFrameId in enumerate(valFrameIds):
        valIdxToFrameIdDict[i] = valFrameId
    # end for

    trainDataset = MyLyftDataset(
        lyftTrainVal,
        trainFrameIds,
        trainIdxToFrameIdDict,
        train_input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    evalDataset = MyLyftDataset(
        lyftTrainVal,
        valFrameIds,
        valIdxToFrameIdDict,
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    trainDataLoader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=train_input_cfg['batch_size'],
        shuffle=True,
        collate_fn=my_lyft_dataset.my_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=True)
    evalDataLoader = torch.utils.data.DataLoader(
        evalDataset,
        batch_size=eval_input_cfg['batch_size'],
        shuffle=False,
        collate_fn=my_lyft_dataset.my_collate_fn)

    model_logging = SimpleModelLog(RESULTS_LOC)
    model_logging.open()
    model_logging.log_text('dont know' + '\n', 0, tag='config', print_to_terminal=False)

    # log starting step
    print('starting from step ' + str(net.get_global_step()) + '\n')

    t = time.time()
    clear_metrics_every_epoch = train_cfg['clear_metrics_every_epoch']

    optimizer.zero_grad()

    step_times = []

    for epoch in range(1, NUM_EPOCHS + 1):

        if clear_metrics_every_epoch:
            net.clear_metrics()
        # end if

        # training loop
        # for each batch in batches . . .
        for i, batch in enumerate(tqdm(trainDataLoader)):
            if batch is None:
                print(termcolor.colored('batch is None !!', 'yellow', attrs=['bold']))
                net.update_global_step()
                continue
            elif isinstance(batch, dict) and len(batch) == 0:
                print(termcolor.colored('batch is a dictionary with zero items !!', 'yellow', attrs=['bold']))
                net.update_global_step()
                continue
            # end if

            time_metrics = batch['metrics']
            batch.pop('metrics')

            batch_size = batch['anchors'].shape[0]

            # convert batch from a dictionary of non-tensors to a dictionary of tensors
            batch = my_lyft_dataset.example_convert_to_torch(batch, float_dtype, device)

            # run batch through the net
            ret_dict = net(batch)

            # break out net results
            cls_preds = ret_dict['cls_preds']
            loss = ret_dict['loss'].mean()
            cls_loss_reduced = ret_dict['cls_loss_reduced'].mean()
            loc_loss_reduced = ret_dict['loc_loss_reduced'].mean()
            cls_pos_loss = ret_dict['cls_pos_loss'].mean()
            cls_neg_loss = ret_dict['cls_neg_loss'].mean()
            loc_loss = ret_dict['loc_loss']
            cls_loss = ret_dict['cls_loss']

            cared = ret_dict['cared']
            labels = batch['labels']

            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

            optimizer.step()
            optimizer.zero_grad()

            net.update_global_step()
            net_metrics = net.update_metrics(cls_loss_reduced, loc_loss_reduced, cls_preds, labels, cared)
            step_time = (time.time() - t)
            step_times.append(step_time)
            t = time.time()
            metrics = {}
            num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
            num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
            if 'anchors_mask' not in batch:
                num_anchors = batch['anchors'].shape[1]
            else:
                num_anchors = int(batch['anchors_mask'][0].sum())
            # end if

            global_step = net.get_global_step()

            # if due for displaying results again . . .
            if i % DISPLAY_STEP == 0:
                if MEASURE_TIME:
                    for name, val in net.get_avg_time_dict().items():
                        print('avg ' + str(name) + ' time = ' + '{:.3f}'.format(val * 1000) + ' ms')
                    # end for
                # end if

                loc_loss_elem = [
                    float(loc_loss[:, :, j].sum().detach().cpu().numpy() / batch_size) for j in range(loc_loss.shape[-1])
                ]
                metrics['runtime'] = {
                    'step': global_step,
                    'steptime': np.mean(step_times),
                }
                metrics['runtime'].update(time_metrics[0])
                step_times = []
                metrics.update(net_metrics)
                metrics['loss']['loc_elem'] = loc_loss_elem
                metrics['loss']['cls_pos_rt'] = float(cls_pos_loss.detach().cpu().numpy())
                metrics['loss']['cls_neg_rt'] = float(cls_neg_loss.detach().cpu().numpy())
                if model_cfg['use_direction_classifier']:
                    dir_loss_reduced = ret_dict['dir_loss_reduced'].mean()
                    metrics['loss']['dir_rt'] = float(dir_loss_reduced.detach().cpu().numpy())
                # end if

                metrics['misc'] = {
                    'num_vox': int(batch['voxels'].shape[0]),
                    'num_pos': int(num_pos),
                    'num_neg': int(num_neg),
                    'num_anchors': int(num_anchors),
                    'mem_usage': psutil.virtual_memory().percent
                }
                model_logging.log_metrics(metrics, global_step)
            # end if
        # end for

        # validation takes a long time, so only do every other epic
        if epoch % 2 == 0:
            net.eval()

            with torch.no_grad():
                result_path_step = os.path.join(RESULTS_LOC, 'step_' + str(net.get_global_step()))
                os.makedirs(result_path_step, exist_ok=True)

                global_step = net.get_global_step()
                model_logging.log_text('#################################', global_step)
                model_logging.log_text('# EVAL', global_step)
                model_logging.log_text('#################################', global_step)

                t = time.time()
                detections = []
                net.clear_timer()

                print('generating eval detections')
                for batch in tqdm(evalDataLoader):

                    if batch is None:
                        print(termcolor.colored('batch is None !!', 'yellow', attrs=['bold']))
                        net.update_global_step()
                        continue
                    elif isinstance(batch, dict) and len(batch) == 0:
                        print(termcolor.colored('batch is a dictionary with zero items !!', 'yellow', attrs=['bold']))
                        net.update_global_step()
                        continue
                    # end if

                    batch = my_lyft_dataset.example_convert_to_torch(batch, float_dtype, device)

                    currentDetections = net(batch)

                    detections += currentDetections
                # end for

                print('\n' + 'building validation dataframe to write val preds .csv file . . . ' + '\n')
                df = pd.DataFrame(columns=['Id', 'PredictionString'])
                for idx, det in enumerate(tqdm(detections)):

                    pred = copy.deepcopy(det)

                    # we can't write PyTorch Tensors on the GPU to a dataframe, so convert to cpu/numpy
                    pred['box3d_lidar'] = pred['box3d_lidar'].detach().cpu().numpy()
                    pred['scores'] = pred['scores'].detach().cpu().numpy()
                    pred['label_preds'] = pred['label_preds'].detach().cpu().numpy()

                    # only use predictions that pass threshold
                    pred = getPredsThatPassThreshold(pred, PRED_THRESHOLD)
                    # get the frame ID
                    frameId = pred['metadata']['token']
                    # get the prediction string
                    pred_str = getPredString(net.target_assigner.classes, evalDataset, pred, frameId)
                    # append current frame ID and prediction string to the end of the dataframe
                    df.loc[len(df.index)] = [frameId, pred_str]
                # end for
                print('\n' + 'writing val preds to .csv . . .')
                df.to_csv(VAL_PREDS_FILE_LOC, index=False)

                print('starting evaluation on predictions . . .')
                evalDataset.evaluation_lyft(detections, result_path_step)

                summary_path = os.path.join(result_path_step, 'metric_summary.json')
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                # end with

                for threshold, ap_metrics in summary.items():
                    model_logging.log_text('\n' + 'threshold: ' + str(threshold), global_step)
                    model_logging.log_text(json.dumps(ap_metrics, indent=2), global_step)

                    if threshold == 'overall':
                        print('type(ap_metrics): ')
                        print(type(ap_metrics))
                        print('ap_metrics: ')
                        print(ap_metrics)

                        overallCarAp = float(ap_metrics['car'])

                        if overallCarAp > OVERALL_CAR_MAP_ACCEPTABLE_THRESH:
                            print(termcolor.colored('\n' + 'overall car mAP = ' + str(overallCarAp) + ' is at least decent' + '\n', 'green', attrs=['bold']))
                        else:
                            print(termcolor.colored('\n' + 'warning, overall car mAP = ' + str(overallCarAp) + ' is below an acceptable level, result is suspect' + '\n', 'red', attrs=['bold']))
                        # end if
                    # end if
                # end for
            # end with

            # set net back to training mode as we are now exiting eval block
            net.train()
        # end if

    # end for

    torch.save(net.state_dict(), os.path.join(RESULTS_LOC, GRAPH_NAME))

    print('\n' + 'train/val complete')

    # if make submission is turned off, we're done
    if not MAKE_SUBMISSION: return

    ### begin test ########################################
    print('beginning testing, A.K.A. making submission . . .')

    net.eval()

    test_input_cfg = config.eval_input_reader

    print('\n' + 'test_input_cfg: ')
    print(test_input_cfg)
    print('')

    lyftTest = LyftDataset(data_path=TEST_DATA_ROOT_PATH, json_path=os.path.join(TEST_DATA_ROOT_PATH, 'data'))

    testFrameIds = []
    for sample in lyftTest.sample:
        testFrameIds.append(sample['token'])
    # end for

    testIdxToFrameIdDict = dict()
    for i, testFrameId in enumerate(testFrameIds):
        testIdxToFrameIdDict[i] = testFrameId
    # end for

    testDataset = MyLyftDataset(
        lyftTest,
        testFrameIds,
        testIdxToFrameIdDict,
        test_input_cfg,
        config.model.second,
        training=False,
        voxel_generator=net.voxel_generator,
        target_assigner=net.target_assigner)
    testDataLoader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        collate_fn=my_lyft_dataset.my_collate_fn)

    print('\n' + 'performing test detections . . . ' + '\n')
    detections = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testDataLoader)):

            if len(batch.items()) <= 0:
                print(termcolor.colored('\n' + 'examples is empty' + '\n', 'red'))
                continue
            # end if

            batch = my_lyft_dataset.example_convert_to_torch(batch, float_dtype, device)

            dets = net(batch)

            for det in dets:
                det['box3d_lidar'] = det['box3d_lidar'].detach().cpu().numpy()
                det['scores'] = det['scores'].detach().cpu().numpy()
                det['label_preds'] = det['label_preds'].detach().cpu().numpy()
                detections.append(det)
            # end for
        # end for
    # end with

    # write detections to test results .pkl file, if it does not exist already
    if not os.path.exists(TEST_RESULTS_LOC):
        # write out the detections file
        print('writing ' + TEST_RESULTS_LOC)
        with open(TEST_RESULTS_LOC, 'wb') as f:
            pickle.dump(detections, f)
        # end with
    # end if

    # make the .csv file for Kaggle submission

    # read the sample submission file to get an initial dataframe
    # the sample submission file has the following format:
    # Id,PredictionString
    # [test_frame_ID_0],
    # [test_frame_ID_1],
    # . . .
    # [test_frame_ID_n-1],
    df = pd.read_csv(SAMPLE_SUBMISSION_LOC)

    # output .csv format is the same, except the predictions for each frame as listed after the frame ID on each line,
    # the format for each prediction is:
    # confidence centerX centerY centerZ width length height yaw classification
    # this repeats for as many predictions as there are for that frame on the line for the frame ID
    # note that there is only a comma after the frame ID, the results for that frame are space delimited

    print('\n' + 'building dataframe to write .csv file . . . ' + '\n')
    classes: List[str] = net.target_assigner.classes
    for idx, pred in enumerate(tqdm(detections)):
        pred = getPredsThatPassThreshold(pred, PRED_THRESHOLD)
        frameId = pred['metadata']['token']
        pred_str = getPredString(classes, testDataset, pred, frameId)
        index = df[df['Id'] == frameId].index[0]
        df.loc[index, 'PredictionString'] = pred_str
    # end for

    # print the first 5 dataframe rows
    df.head()

    # finally we can write dataframe out to CSV file
    print('\n' + 'writing .csv file . . . ')
    df.to_csv(OUTPUT_SUB_FILE_LOC, index=False)

    print('\n' + 'saved output .csv file to: ' + str(OUTPUT_SUB_FILE_LOC) + ', this can be submitted to the ')
    print('Kaggle competition at https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/submit' + '\n')
# end function

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print('WORKER ' + str(worker_id) + ' seed: ' + str(np.random.get_state()[1][0]))
# end function

def getPredsThatPassThreshold(pred: Dict, threshold: float) -> Dict:
    box3d = pred['box3d_lidar']
    scores = pred['scores']
    labels = pred['label_preds']

    # find indices where score is greater than threshold
    passingIdxs = np.where(scores > threshold)[0]

    # filter out low score detections
    box3d = box3d[passingIdxs, :]
    labels = np.take(labels, passingIdxs)
    scores = np.take(scores, passingIdxs)

    # put threshold passing detections back into pred dictionary and return it
    pred['box3d_lidar'] = box3d
    pred['scores'] = scores
    pred['label_preds'] = labels
    return pred
# end function

def getPredString(classes: List[str], testDataset: MyLyftDataset, pred: Dict, frameId: str):
    boxes_lidar = pred['box3d_lidar']
    boxes_class = pred['label_preds']
    scores = pred['scores']
    preds_classes = [classes[x] for x in boxes_class]
    box_centers = boxes_lidar[:, :3]
    box_yaws = boxes_lidar[:, -1]
    box_wlh = boxes_lidar[:, 3:6]

    lyftInfoDict = testDataset.getLyftInfoDict(frameId)

    pred_str = ''
    for idx in range(len(boxes_lidar)):
        translation = box_centers[idx]
        yaw = - box_yaws[idx] - math.pi / 2
        size = box_wlh[idx]
        name = preds_classes[idx]
        detection_score = scores[idx]
        quat = pyquaternion.Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        box = Box(
            center=box_centers[idx],
            size=size,
            orientation=quat,
            score=detection_score,
            name=name,
            token=frameId
        )
        box = lidarSpaceToGlobalSpace(box, lyftInfoDict)
        pred = str(box.score) + ' ' + str(box.center[0]) + ' ' + \
               str(box.center[1]) + ' ' + str(box.center[2]) + ' ' + \
               str(box.wlh[0]) + ' ' \
               + str(box.wlh[1]) + ' ' + str(box.wlh[2]) + ' ' + str(quaternion_yaw(box.orientation)) + ' ' \
               + str(name) + ' '
        pred_str += pred
    # end for

    return pred_str.strip()
# end function

def lidarSpaceToGlobalSpace(box: Box, lyftInfoDict: Dict):
    # lidar -> ego
    box.rotate(pyquaternion.Quaternion(lyftInfoDict['lidar2ego_rotation']))
    box.translate(np.array(lyftInfoDict['lidar2ego_translation']))
    # ego -> global
    box.rotate(pyquaternion.Quaternion(lyftInfoDict['ego2global_rotation']))
    box.translate(np.array(lyftInfoDict['ego2global_translation']))

    return box
# end function

if __name__ == '__main__':
    main()



