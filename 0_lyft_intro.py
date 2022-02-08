# 0_lyft_intro.py

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box

import os
import pathlib
import numpy as np
import pprint
from typing import List, Dict

LYFT_TRAIN_DATASET_LOC = os.path.join(os.path.expanduser('~'), 'LyftObjDetDataset', 'train')

def main():
    # good general sources for Lyft Level 5 Object Detection Dataset info:
    # https://medium.com/wovenplanetlevel5/lyft-level-5-self-driving-dataset-competition-now-open-97493e9f154a
    # https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles
    # https://github.com/lyft/nuscenes-devkit
    # https://github.com/lyft/nuscenes-devkit/blob/master/notebooks/tutorial_lyft.ipynb

    # To be clear, for Lyft Level 5, now Woven Planet Level 5, the Motion Prediction dataset is entirely
    # different, both the dataset and the associated GitHub API.  Only the Object Detection set and API
    # will be covered here

    # The Lyft Object Detection GitHub API was forked from the NuScenes GitHub API when it was made therefore
    # the Lyft Object Detection API is very similar to the NuScenes API

    # The Lyft Object Detection Dataset is divided into train (ground truth annotations provided) and
    # test (ground truth annotations not provided), so if you need a validation split you will have to
    # manually split it out from the training data

    # In the Lyft Object Detection set, as with NuScenes, trips are called "scenes", frames are called "samples",
    # and alphanumeric unique identifiers are usually called "tokens"

    # instantiate the LyftDataset object like this, note you need to choose to load either the train or test set
    lyft = LyftDataset(data_path=LYFT_TRAIN_DATASET_LOC, json_path=os.path.join(LYFT_TRAIN_DATASET_LOC, 'data'))

    print('\n' + 'type(lyft): ')
    print(type(lyft))

    # the scene property is actually a list of all the scenes (i.e. trips)
    print('\n' + 'type(lyft.scene): ')
    print(type(lyft.scene))
    print('len(lyft.scene): ')
    print(len(lyft.scene))

    # you can index the scene list to get the data dictionary for a specific scene (i.e. trip)
    tripId = 0   # choose any number 0 through 179 because there are 180 trips
    tripData: Dict = lyft.scene[tripId]
    print('\n' + 'tripData: ')
    pprint.pprint(tripData, sort_dicts=False)

    # "token" is the term in the Lyft Level 5 Dataset for an alphanumeric unique ID,
    # for example you can get the "token" (frame ID) for a frame like this:
    frameId: str = tripData['first_sample_token']
    print('\n' + 'frameId: ')
    print(frameId)

    # given the "token" (unique ID) for something you can use the get function to get the data dictionary
    # for it, for example we have the frame ID for the 1st frame of our trip, so we can get the
    # frame data dictionary like this:
    frameData: Dict = lyft.get('sample', frameId)
    print('\n' + 'tripData: ')
    pprint.pprint(frameData, sort_dicts=False)

    # you will commonly need to go from a frame ID to the lidar points for that frame,
    # which is usually done like this:
    sampleData: dict = lyft.get('sample', frameId)
    lidarTopId: str = sampleData['data']['LIDAR_TOP']
    lidarFilePathObj: pathlib.Path = lyft.get_sample_data_path(lidarTopId)
    lidarPointCloud: LidarPointCloud = LidarPointCloud.from_file(lidarFilePathObj)
    lidarPoints: np.ndarray = lidarPointCloud.points

    print('\n' + 'lidarPoints.shape: ')
    print(lidarPoints.shape)

    # you will also commonly need to get the ground truth bounding boxes to match a lidar frame,
    # which is usually done like this:
    gndTrBoxes: List[Box] = lyft.get_boxes(lidarTopId)

    print('\n' + 'type(gndTrBoxes): ')
    print(type(gndTrBoxes))
    print('len(gndTrBoxes): ')
    print(len(gndTrBoxes))

    print('\n' + 'type(gndTrBoxes[0])')
    print(type(gndTrBoxes[0]))
    print('gndTrBoxes[0]: ')
    pprint.pprint(gndTrBoxes[0], sort_dicts=False)

    # see the tutorial for more info:
    # https://github.com/lyft/nuscenes-devkit/blob/master/notebooks/tutorial_lyft.ipynb

    print('\n')
# end function

if __name__ == '__main__':
    main()



