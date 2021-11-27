import numpy as np
import os
import re
import math
from hats import HATS
from collections import namedtuple
Event = namedtuple('Event', ['x', 'y', 'ts', 'polarity'])
event_dtype = np.dtype([('x', np.uint16), ('y', np.uint16), ('ts', np.float64), ('polarity', np.int8)])

def unwarp(ts):
    wrap_indices = np.where(np.diff(ts) < 0)
    for i in wrap_indices[0]:
        ts[i + 1:] += 2 ** 30


def load_sample(name, number):
    """Load data.log files into numpy array with events."""
    data_dir = "/home/sveamari/benchmarking/masterthesis2/data/20_20_rot_data/"
    eventPattern = re.compile('(\d+) (\d+\.\d+) ([A-Z]+) \((.*)\)')
    obj = name
    for root, d_names, f_names in os.walk(data_dir + obj):
        sub_dir = sorted(d_names)[number]
        print(os.path.join(data_dir, name, sub_dir, 'data.log'))
        with open(os.path.join(data_dir, name, sub_dir, 'data.log'), 'r') as f:
            parsedContent = eventPattern.findall(f.read())
        ts, events = np.concatenate([x[-1].split(' ') for x in parsedContent]).reshape(-1, 2).swapaxes(0, 1).astype(
            np.int)
        unwarp(ts)
        test_img = np.zeros((len(events), 4))
        test_img[:, 0] = events >> 1 & 0x3FF
        test_img[:, 1] = events >> 12 & 0x1FF
        test_img[:, 3] = events & 0x01
        test_img[:, 2] = (ts - ts[0]) * 80e-9
        break
    return test_img


def prepare_test_image(test_img, name, w_in=96):
    """Split events into six equally sized chunks (six saccades), only consider events in the region of interest (ROI)
     and sample down to 5500 events per saccade."""

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

    # divide into saccades
    number_of_events = len(test_img)
    events_per_sample = math.floor(number_of_events / 6)
    saccades = [test_img[i*events_per_sample:(i+1)*events_per_sample, :] for i in range(6)]

    results = []
    for events in saccades:
       # cut to region of interest
        events_outside_region = []
        for idx, row in enumerate(events):
            if row[0] < CROP_XL or row[0] >= CROP_XU or row[1] < CROP_YL or row[1] >= CROP_YU:
                events_outside_region.append(idx)
        roi_events = np.delete(events, events_outside_region, 0)

        # randomly downsample to 5500 samples per event
        n_left_over_events = len(roi_events) - 5500
        if n_left_over_events > 1:
            events_to_keep = np.random.default_rng().choice(len(roi_events), 5500, replace=False)
            roi_events = roi_events[events_to_keep,:]
        roi_events[:, 0] -= CROP_XL
        roi_events[:, 1] -= CROP_YL
        results.append(roi_events)

    return results

if __name__ == '__main__':
    print(os.getcwd())
    my_sample = load_sample(name="large_clamp", number=6)
    final_sample = prepare_test_image(my_sample, "large_clamp", w_in=96)
    for sample in final_sample:
        hats = HATS(temp_window=0.1, width=96, height=96, delta_t=0.1, tau=0.5, R=8, K=8)
        events = [Event(*ev) for ev in sample]
        events_np = np.array(events, dtype=event_dtype)
        hats.reset()
        hats.process_all(events_np)

