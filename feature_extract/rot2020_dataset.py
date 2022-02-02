import math
import os,sys

cwd = os.getcwd()
lava_dl_path = f"{cwd}{os.sep}..{os.sep}lava-dl{os.sep}src"
sys.path.insert(0,lava_dl_path)
lava_path = f"{cwd}{os.sep}..{os.sep}lava{os.sep}src"
sys.path.insert(0,lava_path)
sys.path.insert(0,cwd)

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
        num_time_bins = 500, transform = None, w_in=96, train = False, device = torch.device('cpu')
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
        self.device = device

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
                                read_events.load_sample(label,number,self.device), 
                                label,self.device
                            ).to(torch.device('cpu'))
        roi_events = slayer.io.Event(
                        roi_event_nparray[:, 0],
                        roi_event_nparray[:, 1],
                        roi_event_nparray[:, 3],
                        roi_event_nparray[:, 2]
                    )
        # roi_events_aug = self.augment(roi_events)
        spike = roi_events.fill_tensor(
                    torch.zeros(2, self.w_in, self.w_in, self.num_time_bins).to(self.device),
                    sampling_time=self.sampling_time
                )
        print("Events converted to tensor.")
        # spike_aug = roi_events_aug.fill_tensor(
        #             torch.zeros(2, self.w_in, self.w_in, self.num_time_bins).to(self.device),
        #             sampling_time=self.sampling_time
        #         )
        return spike, self.all_labels.index(label)
        
        # return torch.cat((spike,spike_aug),dim=0), self.all_labels.index(label)

    def __len__(self):
        return len(self.samples)

    def augment(self,event):
        x_shift = 4
        y_shift = 4
        theta = 10
        xjitter = np.random.randint(2*x_shift) - x_shift
        yjitter = np.random.randint(2*y_shift) - y_shift
        ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
        sin_theta = np.sin(ajitter)
        cos_theta = np.cos(ajitter)
        event.x = event.x * cos_theta - event.y * sin_theta + xjitter
        event.y = event.x * sin_theta + event.y * cos_theta + yjitter
        return event

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