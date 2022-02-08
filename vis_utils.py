# vis_utils.py

from lyft_dataset_sdk.lyftdataset import LyftDataset, Box

import pandas as pd
import numpy as np
import pyquaternion
from typing import List

def getPredBoxes(df: pd.DataFrame, idx: int) -> List[Box]:

    frameId = df.iloc[idx]['Id']

    predString = df.iloc[idx]['PredictionString']

    if not isinstance(predString, str) or predString == '':
        return []
    # end if

    predTokens = predString.split(' ')

    assert len(predTokens) % 9 == 0

    predBoxes = []
    for i in range(0, len(predTokens), 9):
        confi = float(predTokens[i + 0])
        x = float(predTokens[i + 1])
        y = float(predTokens[i + 2])
        z = float(predTokens[i + 3])
        width = float(predTokens[i + 4])
        length = float(predTokens[i + 5])
        height = float(predTokens[i + 6])
        yaw = float(predTokens[i + 7])
        classification = str(predTokens[i + 8])

        box = Box(center=[x, y, z],
                  size=[width, length, height],
                  orientation=pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw),
                  score=confi,
                  name=classification,
                  token=frameId)
        predBoxes.append(box)
    # end for

    return predBoxes
# end function

def moveBoxFromWorldSpaceToSensorSpace(level5data: LyftDataset, box: Box, lidarTopData: dict) -> Box:

    box = box.copy()

    # world space to car space
    egoPoseData: dict = level5data.get('ego_pose', lidarTopData['ego_pose_token'])
    box.translate(-np.array(egoPoseData['translation']))
    box.rotate(pyquaternion.Quaternion(egoPoseData['rotation']).inverse)

    # car space to sensor space
    calSensorData: dict = level5data.get('calibrated_sensor', lidarTopData['calibrated_sensor_token'])
    box.translate(-np.array(calSensorData['translation']))
    box.rotate(pyquaternion.Quaternion(calSensorData['rotation']).inverse)

    return box
# end function

def addLineToPlotlyLines(point1, point2, xLines: List, yLines: List, zLines: List) -> None:
    xLines.append(point1[0])
    xLines.append(point2[0])
    xLines.append(None)

    yLines.append(point1[1])
    yLines.append(point2[1])
    yLines.append(None)

    zLines.append(point1[2])
    zLines.append(point2[2])
    zLines.append(None)
# end function




















