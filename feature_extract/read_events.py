import time
import numpy as np
import os,sys
import re
import math
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import animation
import torch

cwd = os.getcwd()
lava_dl_path = f"{cwd}{os.sep}..{os.sep}lava-dl{os.sep}src"
sys.path.insert(0,lava_dl_path)
lava_path = f"{cwd}{os.sep}..{os.sep}lava{os.sep}src"
sys.path.insert(0,lava_path)
sys.path.insert(0,cwd)


import lava.lib.dl.slayer as slayer


Event = namedtuple('Event', ['x', 'y', 'ts', 'polarity'])
event_dtype = np.dtype([('x', np.uint16), ('y', np.uint16), ('ts', np.float64), ('polarity', np.int8)])

def unwarp(ts):
    wrap_indices = torch.where(torch.diff(ts) < 0)
    for i in wrap_indices[0]:
        ts[i + 1:] += 2 ** 30
    return ts


def load_sample(name, number, device=torch.device('cpu')):
    """Load data.log files into numpy array with events."""
    data_dir = f"data{os.sep}20_20_rot_data{os.sep}"
    eventPattern = re.compile('(\d+) (\d+\.\d+) ([A-Z]+) \((.*)\)')
    obj = name
    for root, d_names, f_names in os.walk(data_dir + obj):
        sub_dir = sorted(d_names)[number]
        # print(os.path.join(data_dir, name, sub_dir, 'data.log'))
        with open(os.path.join(data_dir, name, sub_dir, 'data.log'), 'r') as f:
            parsedContent = eventPattern.findall(f.read())
        # ts, events = np.concatenate([x[-1].split(' ') for x in parsedContent]).reshape(-1, 2).swapaxes(0, 1).astype(
        #     np.int64)
        ts, events = torch.cat([torch.tensor(np.asarray(x[-1].split(' ')).astype(np.int64)) for x in parsedContent]).reshape(-1, 2).swapaxes(0, 1).to(device)
        
        ts = unwarp(ts)
        test_img = torch.zeros((len(events), 4)).to(device)
        test_img[:, 0] = events >> 1 & 0x3FF
        test_img[:, 1] = events >> 12 & 0x1FF
        test_img[:, 3] = events & 0x01
        test_img[:, 2] = (ts - ts[0]) * 80e-9
        break
    # print("Events read from file")
    # sample_events = slayer.io.Event(test_img[:, 0],test_img[:, 1],test_img[:, 3],test_img[:, 2])
    return test_img


def prepare_test_image(test_img, name, device=torch.device('cpu'), w_in=96):
    """Split events into six equally sized chunks (six saccades), only consider events in the region of interest (ROI)
     and sample down to 5500 events per saccade."""

    # test_img = all_events.to_tensor()
    # prepare regions of interest
    widthROI = w_in
    heightROI = w_in

    ROIs = {"sugar_box": [290, 240], "banana": [290, 260], "bowl": [290, 255], "mini_soccer_ball": [300, 245],
            "large_clamp": [290, 260],
            "hammer": [290, 260], "j_cups": [290, 260], "orange": [290, 260], "phillips_screwdriver": [290, 260],
            "mug": [290, 255],
            "adjustable_wrench": [290, 270], "pudding_box": [290, 250], "tuna_fish": [290, 260],
            "mustard_bottle": [290, 240],
            "tomato_soup_can": [290, 250], "plate": [290, 265], "rubiks_cube": [290, 260], "scissors": [290, 265],
            "sponge": [290, 270], "tennis_ball": [290, 260]}

    CROP_XL = ROIs[name][0]
    CROP_XU = CROP_XL + widthROI
    CROP_YL = ROIs[name][1]
    CROP_YU = CROP_YL + heightROI
    np.random.seed(42)
    # divide into saccades
    number_of_events = test_img.size(dim=0)
    events_per_sample = math.floor(number_of_events / 6)
    # print(events_per_sample)
    # saccades = [test_img[i*events_per_sample:(i+1)*events_per_sample, :] for i in range(6)]
    saccades = torch.split(test_img, events_per_sample, dim=0)

    results = torch.empty((1, 4)).to(device)
    for events in saccades:
       # cut to region of interest
        # print(len(events))
        events_idx = range(len(events))
        events_inside_region = torch.nonzero(torch.logical_and(events[:,0]>= CROP_XL, 
                                    torch.logical_and(events[:,0] < CROP_XU,
                                        torch.logical_and(events[:,1] >= CROP_YL,events[:,1] < CROP_YU) ) ))
        roi_events = events[events_inside_region,:].reshape(-1,4)
        # print(roi_events.shape)
        # for idx, row in enumerate(events):
        #     if row[0] < CROP_XL or row[0] >= CROP_XU or row[1] < CROP_YL or row[1] >= CROP_YU:
        #         events_outside_region.append(idx)
        # roi_events_idx = np.delete(events_idx, events_outside_region, 0)
        # roi_events = events[roi_events_idx,:]
 
        # randomly downsample to 5500 samples per saccade
        n_left_over_events = len(roi_events) - 5500
        if n_left_over_events > 1:
            events_to_keep = np.random.default_rng().choice(len(roi_events), 5500, replace=False)
            roi_events = roi_events[events_to_keep,:]
        roi_events[:, 0] -= CROP_XL
        roi_events[:, 1] -= CROP_YL
        results = torch.cat([results,roi_events]).to(device)
    # print("Events processed")
    return results

if __name__ == '__main__':
    name = "large_clamp"
    number = 1
    start_time = time.time()
    sample_events = load_sample(name=name, number=number)
    # print(sample_events)
    results = prepare_test_image(sample_events, name, w_in=96)
    # print(type(results))
    print("--- %s seconds ---" % (time.time() - start_time))
    final_sample = slayer.io.Event(results[:, 0],results[:, 1],results[:, 3],results[:, 2])
    anim = final_sample.anim(plt.figure(figsize=(5, 5)), frame_rate=4800)
    cwd = os.getcwd()
    anim.save(f'{cwd}{os.sep}gifs/{name}_{number}_withtensor_mod.gif', animation.PillowWriter(fps=5), dpi=300)

