'''
Project: 
Wind Gust Detection using Physical Sensors in Quadcopters

Author: 
Suwen Gu and Menghao Lin

Description:
This file is mainly responsible for loading data. 

Since data files contain data for more than one sensors,
this file also decomposes the data into a collection of sensory data.

'''

import pandas as pd
import numpy as np
from scipy import signal

# the following variables are responsible for
# constructing file paths of data
PATH = "data/"
LABEL_0 = "label_0/"
LABEL_1 = "label_1/"
LABEL_2 = "label_2/"
LABEL_2_5 = "label_2.5/"
LABEL_3 = "label_3/"
LABEL_4 = "label_4/"
FILE_PREFIX = "data_set_label_"
FILE_MIDDLE_PACKET = "_packet_"
FILE_MIDDLE_FACING = "_facing_"
FILE_SUFFIX = ".csv"


def load_data(label, total_files, project='project1', drone="drone1", direction=""):
    """Load data files into a dataframe for a specified project and drone.

    Parameters
    ----------
    label: int
        the label of a specific wind speed
    total_files: int
        the number of files consist of the data for one class label
    project: str
        the project name, default is project1
    drone: str
        the name of the drone, default is drone1
    direction: str
        the direction of the label, default is "" which stands for front wind.

    Returns
    -------
    DataFrame
        raw sensory data set
    """

    # generate file path for a class label
    path = PATH + project + '/' + drone + '/'

    if label == 0:
        path += LABEL_0
    elif label == 1:
        path += LABEL_1
    elif label == 2:
        path += LABEL_2
    elif label == 2.5:
        path += LABEL_2_5
    elif label == 3:
        path += LABEL_3
    elif label == 4:
        path += LABEL_4

    # add additional info to the path if it is the data of directional detection
    if direction != '':
        path = path[0:-1]+FILE_MIDDLE_FACING+direction+"/"
        direction = "_"+direction

    columns = ["timestamp_start", "timestamp_end", "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw", "gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z", "mag.x", "mag.y", "mag.z","label"]
    data = pd.DataFrame(data=[], columns=columns)

    for i in range(total_files):
        fileName = FILE_PREFIX+str(label)+direction+FILE_MIDDLE_PACKET+str(i)+FILE_SUFFIX
        temp_data = pd.read_csv(path+fileName, index_col=0)
        temp_data["label"] = label
        # cut the first and last 100 = 1s after taking off and 1s before landing
        temp_data = temp_data.iloc[100:-100, :]
        # only need 6000 data points, which is equivalent to 1 min of data
        temp_data = temp_data.iloc[:6000, :]
        data = data.append(temp_data, ignore_index=True)

    return data

def separate_data_based_on_apparatus(data):
    """Seperates data into a collection of data

    Use a dictionary to store the raw data, in which
    names of the sensors as the keys,
    corresponding data as the values.

    Parameters
    ----------
    data: DataFrame
        raw sensory data

    Returns
    -------
    list
        data stored in a dictionary
    """

    acc = data.iloc[:, 0:3]
    gyro = data.iloc[:, 3:6]
    mag = data.iloc[:, 7:10]
    stabilizer = data.iloc[:, 10:13]

    data_collection = {
        "acc": acc,
        "gyro": gyro,
        "mag": mag,
        "stabilizer": stabilizer
    }

    return data_collection