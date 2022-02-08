# 1_visualize_dataset.py

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box

import os
import pathlib
import numpy as np
import pyquaternion
from typing import List, Dict
import plotly.graph_objects as PlotlyGraphObjects
import pprint

import vis_utils

# must use the train dataset if going to show ground truth boxes because test does not have ground truth boxes
LYFT_TRAIN_DATASET_LOC = os.path.join(os.path.expanduser('~'), 'LyftObjDetDataset', 'train')
TRIP_ID = 0   # can set this to any valid trip ID

SHOW_PLOTLY_MOUSEOVERS = False

def main():
    # suppress numpy printing in scientific notation
    np.set_printoptions(suppress=True)

    # load training data
    print('\n' + 'loading lyft training data . . . ' + '\n')
    level5data = LyftDataset(data_path=LYFT_TRAIN_DATASET_LOC, json_path=os.path.join(LYFT_TRAIN_DATASET_LOC, 'data'), verbose=True)

    # get the trip data
    tripData: Dict = level5data.scene[TRIP_ID]
    print('\n' + 'tripData: ')
    pprint.pprint(tripData, sort_dicts=False)
    print('')

    # optionally could iterate through frames in the trip and show any chosen frame,
    # but to keep things simple, let's just use the first frame in the trip
    frameId: str = tripData['first_sample_token']

    # go from frame ID to lidar points
    sampleData: dict = level5data.get('sample', frameId)
    lidarTopId: str = sampleData['data']['LIDAR_TOP']
    lidarFilePathObj: pathlib.Path = level5data.get_sample_data_path(lidarTopId)
    lidarPointCloud: LidarPointCloud = LidarPointCloud.from_file(lidarFilePathObj)
    lidarPoints: np.ndarray = lidarPointCloud.points

    # for lidar points, first 3 rows are x, y, z, 4th row is intensity which is always 100, so remove it
    lidarPoints: np.ndarray = lidarPoints[:3, :]

    # go from lidar top ID to ground truth boxes
    lidarTopData: dict = level5data.get('sample_data', lidarTopId)
    gndTrBoxes: List[Box] = level5data.get_boxes(lidarTopId)

    ### 3D visualization ######################################################

    s3dPoints = PlotlyGraphObjects.Scatter3d(x=lidarPoints[0], y=lidarPoints[1], z=lidarPoints[2], mode='markers', marker={'size': 1})

    # 3 separate lists for the x, y, and z components of each line
    xLines = []
    yLines = []
    zLines = []
    for box in gndTrBoxes:

        box = vis_utils.moveBoxFromWorldSpaceToSensorSpace(level5data, box, lidarTopData)

        corners = box.corners()

        # see here for documentation of Box:
        # https://github.com/lyft/nuscenes-devkit/blob/master/lyft_dataset_sdk/utils/data_classes.py#L622
        # when getting corners, the first 4 corners are the ones facing forward, the last 4 are the ones facing rearwards

        corners = corners.transpose()

        # 4 lines for front surface of box
        vis_utils.addLineToPlotlyLines(corners[0], corners[1], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[1], corners[2], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[2], corners[3], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[3], corners[0], xLines, yLines, zLines)

        # 4 lines between front points and read points
        vis_utils.addLineToPlotlyLines(corners[0], corners[4], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[1], corners[5], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[2], corners[6], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[3], corners[7], xLines, yLines, zLines)

        # 4 lines for rear surface of box
        vis_utils.addLineToPlotlyLines(corners[4], corners[7], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[5], corners[4], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[6], corners[5], xLines, yLines, zLines)
        vis_utils.addLineToPlotlyLines(corners[7], corners[6], xLines, yLines, zLines)

    # end for

    s3dGndTrBoxLines = PlotlyGraphObjects.Scatter3d(x=xLines, y=yLines, z=zLines, mode='lines', name='gnd trs')

    # make and show a plotly Figure object
    plotlyFig = PlotlyGraphObjects.Figure(data=[s3dPoints, s3dGndTrBoxLines])
    plotlyFig.update_layout(scene_aspectmode='data')

    if not SHOW_PLOTLY_MOUSEOVERS:
        plotlyFig.update_layout(hovermode=False)
        plotlyFig.update_layout(scene=dict(xaxis_showspikes=False,
                                           yaxis_showspikes=False,
                                           zaxis_showspikes=False))
    # end if

    plotlyFig.show()

# end function

if __name__ == '__main__':
    main()


