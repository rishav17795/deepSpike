import math
import os
import glob
from matplotlib import animation, pyplot as plt
import numpy as np
import re
from numpy import random
import torch
from torch.utils.data import Dataset
import read_events
import lava.lib.dl.slayer as slayer

class ROTDataset(Dataset):
    """
    """
    def __init__(
        self, path=os.path.join('data', '20_20_rot_data'), sampling_time = 1, 
        num_time_bins = 500, transform = None, w_in=96, train = False
    ):
        super(ROTDataset, self).__init__()
        self.path = path
        self.samples = glob.glob(f'{path}{os.sep}*{os.sep}*{os.sep}data.log')
        random.seed(42)
        random.shuffle(self.samples)
        self.w_in = w_in
        self.sampling_time = sampling_time
        self.num_time_bins = num_time_bins
        self.transform = transform
        self.all_labels = next(os.walk(path + os.sep + '.'))[1]
        self.all_labels = sorted(self.all_labels)

        if train is True:
            self.samples = self.samples[
                    :int(len(self.samples) * 0.8)
                ] 
        else:
            self.samples = self.samples[
                    -int(len(self.samples) * 0.2):
                ]

    def __getitem__(self, index: int):
        filename = self.samples[index]
        label = filename.split(os.sep)[-3]

        d_names = next(os.walk(self.path + os.sep + label + os.sep + '.'))[1]
        number = sorted(d_names).index(filename.split(os.sep)[-2])
        
        roi_event_nparray = read_events.prepare_test_image(
                                read_events.load_sample(label,number), 
                                label
                            )
        roi_events = slayer.io.Event(
                        roi_event_nparray[:, 0],
                        roi_event_nparray[:, 1],
                        roi_event_nparray[:, 3],
                        roi_event_nparray[:, 2]
                    )
        spike = roi_events.fill_tensor(
                    torch.zeros(2, self.w_in, self.w_in, self.num_time_bins),
                    sampling_time=self.sampling_time
                )
        return spike, self.all_labels.index(label)

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    training_set = ROTDataset()
    filename = training_set.samples[47]
    label = filename.split(os.sep)[-3]
    d_names = next(os.walk(training_set.path + os.sep + label + os.sep + '.'))[1]
    number = sorted(d_names).index(filename.split(os.sep)[-2])
    roi_event_nparray = read_events.prepare_test_image(
                                read_events.load_sample(label,number), 
                                label
                            )
    roi_events = slayer.io.Event(
                    roi_event_nparray[:, 0],
                    roi_event_nparray[:, 1],
                    roi_event_nparray[:, 3],
                    roi_event_nparray[:, 2]
                )
    print(training_set.all_labels.index(label))
    anim = roi_events.anim(plt.figure(figsize=(5, 5)), frame_rate=4800)
    cwd = os.getcwd()
    anim.save(f'{cwd}{os.sep}gifs/{label}_{number}.gif', animation.PillowWriter(fps=5), dpi=300)