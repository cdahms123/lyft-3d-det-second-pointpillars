# 3_visualize_val.py

from lyft_dataset_sdk.lyftdataset import LyftDataset, Box
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

import os
import pathlib
import pandas as pd
import cv2
import numpy as np
import copy
import pyquaternion
import plotly.graph_objects as PlotlyGraphObjects
from tqdm import tqdm
from typing import List

import vis_utils

TRAIN_DATA_ROOT_PATH = '../../LyftObjDetDataset/train'
VAL_PREDS_FILE_LOC = os.path.join(os.getcwd(), 'results', 'val_preds.csv')

SHOW_PLOTLY_MOUSEOVERS = False

def main():
    # suppress numpy printing in scientific notation
    np.set_printoptions(suppress=True)

    # read in validation predictions
    valPredsDataFrame = pd.read_csv(VAL_PREDS_FILE_LOC)

    print(valPredsDataFrame)

    # load training data
    print('\n' + 'loading train data (note this also includes the val split) . . . ' + '\n')
    level5data = LyftDataset(data_path=TRAIN_DATA_ROOT_PATH, json_path=os.path.join(TRAIN_DATA_ROOT_PATH, 'data'))

    print('\n' + 'building boxes lists . . . ' + '\n')
    allPredBoxes: List[List[Box]] = []
    for idx in tqdm(range(len(valPredsDataFrame))):
        predBoxes: List[Box] = vis_utils.getPredBoxes(valPredsDataFrame, idx)
        allPredBoxes.append(predBoxes)
    # end function

    print('press [Enter] to show next frame, or Ctrl + \ to end program')

    for idx, row in valPredsDataFrame.iterrows():
        # get the frame ID for the current frame
        frameId = valPredsDataFrame.iloc[idx]['Id']
        # get the pred boxes for the current frame
        predBoxes = allPredBoxes[idx]

        sample: dict = level5data.get('sample', frameId)
        lidarTopId: str = sample['data']['LIDAR_TOP']
        lidarFilePathObj: pathlib.Path = level5data.get_sample_data_path(lidarTopId)
        lidarPointCloud: LidarPointCloud = LidarPointCloud.from_file(lidarFilePathObj)
        lidarPoints: np.ndarray = lidarPointCloud.points

        print('frameId = ' + str(frameId) + ', lidarPoints.shape = ' + str(lidarPoints.shape))

        # intensity is always 100, so remove it
        lidarPoints: np.ndarray = lidarPoints[:3, :]

        lidarTopData: dict = level5data.get('sample_data', lidarTopId)
        gndTrBoxes: List[Box] = level5data.get_boxes(lidarTopId)

        ### 3D visualization ############################################

        s3dPoints = PlotlyGraphObjects.Scatter3d(x=lidarPoints[0], y=lidarPoints[1], z=lidarPoints[2], mode='markers',
                                                 marker={'size': 1})

        # 3 separate lists for the x, y, and z components of each line
        predXLines = []
        predYLines = []
        predZLines = []
        for predBox in predBoxes:
            predBox = vis_utils.moveBoxFromWorldSpaceToSensorSpace(level5data, predBox, lidarTopData)

            corners = predBox.corners()

            # see here for documentation of Box:
            # https://github.com/lyft/nuscenes-devkit/blob/master/lyft_dataset_sdk/utils/data_classes.py#L622
            # when getting corners, the first 4 corners are the ones facing forward, the last 4 are the ones facing rearwards

            corners = corners.transpose()

            # 4 lines for front surface of box
            vis_utils.addLineToPlotlyLines(corners[0], corners[1], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[1], corners[2], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[2], corners[3], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[3], corners[0], predXLines, predYLines, predZLines)

            # 4 lines between front points and read points
            vis_utils.addLineToPlotlyLines(corners[0], corners[4], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[1], corners[5], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[2], corners[6], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[3], corners[7], predXLines, predYLines, predZLines)

            # 4 lines for rear surface of box
            vis_utils.addLineToPlotlyLines(corners[4], corners[7], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[5], corners[4], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[6], corners[5], predXLines, predYLines, predZLines)
            vis_utils.addLineToPlotlyLines(corners[7], corners[6], predXLines, predYLines, predZLines)

        # end for

        s3dPredBoxLines = PlotlyGraphObjects.Scatter3d(x=predXLines, y=predYLines, z=predZLines, mode='lines',
                                                       name='preds')

        # 3 separate lists for the x, y, and z components of each line
        gndTrXLines = []
        gndTrYLines = []
        gndTrZLines = []
        for gndTrBox in gndTrBoxes:
            gndTrBox = vis_utils.moveBoxFromWorldSpaceToSensorSpace(level5data, gndTrBox, lidarTopData)

            corners = gndTrBox.corners()

            # see here for documentation of Box:
            # https://github.com/lyft/nuscenes-devkit/blob/master/lyft_dataset_sdk/utils/data_classes.py#L622
            # when getting corners, the first 4 corners are the ones facing forward, the last 4 are the ones facing rearwards

            corners = corners.transpose()

            # 4 lines for front surface of box
            vis_utils.addLineToPlotlyLines(corners[0], corners[1], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[1], corners[2], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[2], corners[3], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[3], corners[0], gndTrXLines, gndTrYLines, gndTrZLines)

            # 4 lines between front points and read points
            vis_utils.addLineToPlotlyLines(corners[0], corners[4], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[1], corners[5], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[2], corners[6], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[3], corners[7], gndTrXLines, gndTrYLines, gndTrZLines)

            # 4 lines for rear surface of box
            vis_utils.addLineToPlotlyLines(corners[4], corners[7], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[5], corners[4], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[6], corners[5], gndTrXLines, gndTrYLines, gndTrZLines)
            vis_utils.addLineToPlotlyLines(corners[7], corners[6], gndTrXLines, gndTrYLines, gndTrZLines)

        # end for

        s3dGndTrBoxLines = PlotlyGraphObjects.Scatter3d(x=gndTrXLines, y=gndTrYLines, z=gndTrZLines, mode='lines',
                                                        name='gnd trs')

        # make and show a plotly Figure object
        plotlyFig = PlotlyGraphObjects.Figure(data=[s3dPoints, s3dPredBoxLines, s3dGndTrBoxLines])
        plotlyFig.update_layout(scene_aspectmode='data')

        if not SHOW_PLOTLY_MOUSEOVERS:
            plotlyFig.update_layout(hovermode=False)
            plotlyFig.update_layout(scene=dict(xaxis_showspikes=False,
                                               yaxis_showspikes=False,
                                               zaxis_showspikes=False))
        # end if

        plotlyFig.show()

        # pause here until the user presses enter
        input()
    # end for

# end function

if __name__ == '__main__':
    main()



