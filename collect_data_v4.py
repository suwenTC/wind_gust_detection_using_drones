
'''
Project: 
Wind Gust Detection using Physical Sensors in Quadcopters

Author: 
Suwen Gu and Menghao Lin

Description:
This file is used to collect gyroscope and stabilizer data.

Code version: 4
'''

import sys
import logging
import time
import pandas as pd
import numpy as np
from collections import OrderedDict
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander
import warnings
warnings.filterwarnings("ignore")

# drone channel
# URI = 'radio://0/80/2M/E7E7E7E701'
# URI = 'radio://0/85/2M/E7E7E7E702'
# URI = 'radio://0/90/2M/E7E7E7E7DF'
# URI = 'radio://0/80/2M/E7E7E7E7E7'
# URI = 'radio://0/86/2M/E7E7E7E7E6'
URI = 'radio://0/70/2M/E7E7E7E701'
# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

def write_to_file(data, class_label, packet_num):
    """ Writes the log data to a file

    Parameters
    ----------
    data: list
        the list of collected of data for one flight
    class_label: int
        the number that indicates the direction of wind
    packet_num: int
        tthe number that indcates the current file number

    Returns:
    ----------
    """
    data = pd.DataFrame(data)
    data['label'] = [class_label]*data.shape[0]

    data.to_csv("data/drone1/data_set_label_"+class_label+"_packet_"+packet_num+".csv")

if __name__ == '__main__':
    # Initialize the low-level drivers (don't list the debug drivers)

    cflib.crtp.init_drivers(enable_debug_driver=False)

    data = []
        
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        lg = LogConfig(name='sta_gyro', period_in_ms=10)
        lg.add_variable('gyro.x', 'float')
        lg.add_variable('gyro.y', 'float')
        lg.add_variable('gyro.z', 'float')
        lg.add_variable('stabilizer.roll', 'float')
        lg.add_variable('stabilizer.yaw', 'float')
        lg.add_variable('stabilizer.pitch', 'float')

        class_label = sys.argv[1]
        packet_num = sys.argv[2]
            
        endTime = time.time() + 20

        with SyncLogger(scf, lg) as logger:
            with MotionCommander(scf) as mc:
                # drone takes off
                mc.up(0.1)
                while time.time() < endTime:
                    # let the drone move forward at rate of 0.001.
                    mc.forward(0.001)

                    # log data
                    for log_entry in logger:
                        row = log_entry[1]
                        data.append(row)
                        break

        # after a flight, write data to a file
        write_to_file(data, str(class_label), packet_num)