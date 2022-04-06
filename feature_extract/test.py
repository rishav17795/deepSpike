import glob
import os
import numpy as np
import matplotlib
from matplotlib import cm

if __name__ == '__main__':
    path=os.path.join('data_tensors_saccades', '20_20_rot_data')
    samples_sb = glob.glob(f'{path}{os.sep}sugar_box{os.sep}yarp_data_ROT_EVENTS_042_adjustable_wrench_degree_0\.314159*{os.sep}spike_tensor.pt')
    print(samples_sb)