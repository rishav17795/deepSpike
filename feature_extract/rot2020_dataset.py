import math
import os,sys
from pathlib import Path

cwd = os.getcwd()
lava_dl_path = f"{cwd}{os.sep}..{os.sep}lava-dl{os.sep}src"
sys.path.insert(0,lava_dl_path)
lava_path = f"{cwd}{os.sep}..{os.sep}lava{os.sep}src"
sys.path.insert(0,lava_path)
sys.path.insert(0,cwd)

import time
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
        self, path=os.path.join('data_tensors_saccades', '20_20_rot_data'), sampling_time = 0.1, 
        num_time_bins = 40, loss = None, transform = None, w_in=96, train = False, 
        device = torch.device('cpu'), from_tensor = True
    ):
        super(ROTDataset, self).__init__()
        self.path = path
        self.samples = glob.glob(f'{path}{os.sep}*{os.sep}*{os.sep}spike_tensor.pt')
        random.seed(42)
        random.shuffle(self.samples)
        self.w_in = w_in
        self.sampling_time = sampling_time
        self.num_time_bins = num_time_bins
        self.transform = transform
        self.all_labels = next(os.walk(path + os.sep + '.'))[1]
        self.all_labels = sorted(self.all_labels)
        self.device = device
        self.loss = loss
        self.from_tensor = from_tensor

        if train is True:
            self.samples = self.samples[
                    :int(len(self.samples) * 0.8)
                ] 
        else:
            self.samples = self.samples[
                    -int(len(self.samples) * 0.2):
                ]

    def __getitem__(self, index: int):

        if self.from_tensor:
            if self.loss is None:
                filename = self.samples[index]
                a_label = filename.split(os.sep)[-3]
                # print(filename)
                a_spike = torch.load(filename)
                a_spike = torch.cat((a_spike,torch.zeros(2,96,96,396)),dim=3)
                return a_spike, self.all_labels.index(a_label)

            else:
                filename = self.samples[index]
                a_label = filename.split(os.sep)[-3]
                # print(filename)
                a_spike = torch.load(filename)
                # a_spike_load_img = a_spike[1,:,:,1].reshape(96,96)
                # plt.imshow(a_spike_load_img)
                # plt.show()
                # a_spike = torch.cat((a_spike[:,:,:,0].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                      a_spike[:,:,:,1].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                      a_spike[:,:,:,2].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                      a_spike[:,:,:,3].reshape((2,96,96,1)),torch.zeros(2,96,96,9)),dim=3)
                
                
                d_names = next(os.walk(self.path + os.sep + a_label + os.sep + '.'))[1]
                a_number = sorted(d_names).index(filename.split(os.sep)[-2])
                all_index = np.arange(len(d_names))
                p_numbers = all_index[np.where(all_index != a_number)]
                p_number = p_numbers[np.random.default_rng().choice(len(p_numbers),1)].item()
                # print(self.path + os.sep + a_label + os.sep + sorted(d_names)[p_number] + os.sep + 'spike_tensor.pt')
                p_spike = torch.load(self.path + os.sep + a_label + os.sep + sorted(d_names)[p_number] + os.sep + 'spike_tensor.pt')
                # p_spike = torch.cat((p_spike[:,:,:,0].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                     p_spike[:,:,:,1].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                     p_spike[:,:,:,2].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                     p_spike[:,:,:,3].reshape((2,96,96,1)),torch.zeros(2,96,96,9)),dim=3)
                
                n_labels = np.array(self.all_labels)[np.where(np.array(self.all_labels) != np.array(a_label))[0]]
                n_label = n_labels[np.random.default_rng().choice(len(n_labels),1)].item()
                n_d_names = next(os.walk(self.path + os.sep + n_label + os.sep + '.'))[1]
                n_number = np.random.default_rng().choice(len(n_d_names),1)[0]
                # print(self.path + os.sep + n_label + os.sep + sorted(n_d_names)[n_number] + os.sep + 'spike_tensor.pt')
                n_spike = torch.load(self.path + os.sep + n_label + os.sep + sorted(n_d_names)[n_number] + os.sep + 'spike_tensor.pt')
                # n_spike = torch.cat((n_spike[:,:,:,0].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                     n_spike[:,:,:,1].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                     n_spike[:,:,:,2].reshape((2,96,96,1)),torch.zeros(2,96,96,9),
                #                     n_spike[:,:,:,3].reshape((2,96,96,1)),torch.zeros(2,96,96,9)),dim=3)
                
                return torch.cat((a_spike,p_spike,n_spike),dim=0), self.all_labels.index(a_label)
        else:
            if self.loss is None:
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
                spike = roi_events.fill_tensor(
                            torch.zeros(2, self.w_in, self.w_in, self.num_time_bins).to(self.device),
                            sampling_time=self.sampling_time
                        )
                print(f'Completed Loading: {label} from file ---------------------------------------', end='\r')
                
                return spike, self.all_labels.index(label)
                
                # return torch.cat((spike,spike_aug),dim=0), self.all_labels.index(label)
            else:
                filename = self.samples[index]
                a_label = filename.split(os.sep)[-3]
                n_labels = np.array(self.all_labels)[np.where(np.array(self.all_labels) != np.array(a_label))[0]]
                n_label = n_labels[np.random.default_rng().choice(len(n_labels),1)].item()

                d_names = next(os.walk(self.path + os.sep + a_label + os.sep + '.'))[1]
                a_number = sorted(d_names).index(filename.split(os.sep)[-2])
                roi_event_nparray = read_events.prepare_test_image(
                                        read_events.load_sample(a_label,a_number,self.device), 
                                        a_label,self.device
                                    ).to(torch.device('cpu'))
                roi_events = slayer.io.Event(
                                roi_event_nparray[:, 0],
                                roi_event_nparray[:, 1],
                                roi_event_nparray[:, 3],
                                roi_event_nparray[:, 2]
                            )
                a_spike = roi_events.fill_tensor(
                            torch.zeros(2, self.w_in, self.w_in, self.num_time_bins).to(self.device),
                            sampling_time=self.sampling_time
                        )
                print(f'Completed Loading: {a_label} from file ---------------------------------------', end='\r')
                all_index = np.arange(len(d_names))
                p_numbers = all_index[np.where(all_index != a_number)]
                p_number = p_numbers[np.random.default_rng().choice(len(p_numbers),1)].item()
                roi_event_nparray = read_events.prepare_test_image(
                                        read_events.load_sample(a_label,p_number,self.device), 
                                        a_label,self.device
                                    ).to(torch.device('cpu'))
                roi_events = slayer.io.Event(
                                roi_event_nparray[:, 0],
                                roi_event_nparray[:, 1],
                                roi_event_nparray[:, 3],
                                roi_event_nparray[:, 2]
                            )
                p_spike = roi_events.fill_tensor(
                            torch.zeros(2, self.w_in, self.w_in, self.num_time_bins).to(self.device),
                            sampling_time=self.sampling_time
                        )

                n_d_names = next(os.walk(self.path + os.sep + n_label + os.sep + '.'))[1]
                n_number = np.random.default_rng().choice(len(n_d_names),1)[0]
                roi_event_nparray = read_events.prepare_test_image(
                                        read_events.load_sample(n_label,n_number,self.device), 
                                        n_label,self.device
                                    ).to(torch.device('cpu'))
                roi_events = slayer.io.Event(
                                roi_event_nparray[:, 0],
                                roi_event_nparray[:, 1],
                                roi_event_nparray[:, 3],
                                roi_event_nparray[:, 2]
                            )
                n_spike = roi_events.fill_tensor(
                            torch.zeros(2, self.w_in, self.w_in, self.num_time_bins).to(self.device),
                            sampling_time=self.sampling_time
                        )
                return torch.cat((a_spike,p_spike,n_spike),dim=0), self.all_labels.index(a_label)
                


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
    # roughly 40 time bins sampled at 0.1 sampling rate
    # scale issue in time domain
    num_time_bins=120
    sampling_time=4/num_time_bins
    training_set = ROTDataset(num_time_bins=num_time_bins, sampling_time=sampling_time, train=True)
    path=os.path.join('data', '20_20_rot_data')
    samples = glob.glob(f'{path}{os.sep}*{os.sep}*{os.sep}data.log')
    filename = samples[145]
    # print(filename)
    a_label = filename.split(os.sep)[-3]

    # n_labels = np.array(training_set.all_labels)[np.where(np.array(training_set.all_labels) != np.array(a_label))[0]]
    
    # n_label = n_labels[np.random.default_rng().choice(len(n_labels),1)].item()

    d_names = next(os.walk(path + os.sep + a_label + os.sep + '.'))[1]
    a_number = sorted(d_names).index(filename.split(os.sep)[-2])
    # # print(sorted(d_names)[a_number])
    # # all_index = np.arange(len(d_names))
    # # p_numbers = all_index[np.where(all_index != a_number)]
    # # p_number = p_numbers[np.random.default_rng().choice(len(p_numbers),1)].item()
    # # print(sorted(d_names)[p_number])
    # # n_d_names = next(os.walk(training_set.path + os.sep + n_label + os.sep + '.'))[1]
    # # n_number = np.random.default_rng().choice(len(n_d_names),1)[0]
    # # print(sorted(n_d_names)[n_number])
    # # d_names = next(os.walk(training_set.path + os.sep + label + os.sep + '.'))[1]
    # # number = sorted(d_names).index(filename.split(os.sep)[-2])
    # t0 = time.time()
    roi_event_nparray = read_events.prepare_test_image(
                                read_events.load_sample(a_label,a_number), 
                                a_label
                            )
    roi_events = slayer.io.Event(
                    roi_event_nparray[:, 0],
                    roi_event_nparray[:, 1],
                    roi_event_nparray[:, 3],
                    roi_event_nparray[:, 2]
                )
    # t1 = time.time()
    # print(roi_event_nparray[:, 2].min())
    # print(f'load_from_file_t = {t1-t0}')

    a_spike = roi_events.fill_tensor(
                        torch.zeros(2, training_set.w_in, training_set.w_in, num_time_bins).to(training_set.device),
                        sampling_time=sampling_time
                    )
    # a_spike, _, _, _, _, _ = torch.split(a_spike,int(a_spike.shape[-1]/6),dim=3)
    
    # print(a_spike.shape)
    # torch.save(a_spike.clone(), 'spike_tensor.pt')

    # t0 = time.time()
    ix = 1
    for s in range(6):
        a_spike_load = torch.load(f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_{s}{os.sep}spike_tensor.pt')
        for i in range(a_spike_load.shape[-1]):
            a_spike_load_img = a_spike_load[1,:,:,i].reshape(96,96)
            ax = plt.subplot(6, a_spike_load.shape[-1], ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(a_spike_load_img)
            ix += 1
    # show the figure
    plt.show()
    # print(training_set.all_labels.index(label))
    # ----------------------following section does not work------------------------------------------------------
    # anim = events.anim(plt.figure(figsize=(5, 5)), frame_rate=40)
    # cwd = os.getcwd()
    # anim.save(f'{cwd}{os.sep}gifs/{a_label}_{a_number}_experiments.gif', animation.PillowWriter(fps=5), dpi=300)
    # -----------------------------------------------------------------------------------------------------------
    # path=os.path.join('data', '20_20_rot_data')
    # samples = glob.glob(f'{path}{os.sep}*{os.sep}*{os.sep}data.log')
    # for i in range(len(samples)):
    #     filename = samples[i]
    #     print(f'Converting:{i+1}/{len(samples)}')
    #     a_label = filename.split(os.sep)[-3]
    #     d_names = next(os.walk(path + os.sep + a_label + os.sep + '.'))[1]
    #     a_number = sorted(d_names).index(filename.split(os.sep)[-2])
    #     roi_event_nparray = read_events.prepare_test_image(
    #                             read_events.load_sample(a_label,a_number), 
    #                             a_label
    #                         )
    #     roi_events = slayer.io.Event(
    #                     roi_event_nparray[:, 0],
    #                     roi_event_nparray[:, 1],
    #                     roi_event_nparray[:, 3],
    #                     roi_event_nparray[:, 2]
    #                 )
    #     a_spike = roi_events.fill_tensor(
    #                     torch.zeros(2, training_set.w_in, training_set.w_in, num_time_bins),
    #                     sampling_time=sampling_time
    #                 )
    #     Path(f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_0').mkdir(parents=True, exist_ok=True)
    #     Path(f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_1').mkdir(parents=True, exist_ok=True)
    #     Path(f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_2').mkdir(parents=True, exist_ok=True)
    #     Path(f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_3').mkdir(parents=True, exist_ok=True)
    #     Path(f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_4').mkdir(parents=True, exist_ok=True)
    #     Path(f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_5').mkdir(parents=True, exist_ok=True)
    #     a_spike_s0, a_spike_s1, a_spike_s2, a_spike_s3, a_spike_s4, a_spike_s5 = torch.split(a_spike,int(a_spike.shape[-1]/6),dim=3)
    #     torch.save(a_spike_s0.clone(), f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_0{os.sep}spike_tensor.pt')
    #     torch.save(a_spike_s1.clone(), f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_1{os.sep}spike_tensor.pt')
    #     torch.save(a_spike_s2.clone(), f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_2{os.sep}spike_tensor.pt')
    #     torch.save(a_spike_s3.clone(), f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_3{os.sep}spike_tensor.pt')
    #     torch.save(a_spike_s4.clone(), f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_4{os.sep}spike_tensor.pt')
    #     torch.save(a_spike_s5.clone(), f'data_tensors_saccades{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}_sac_5{os.sep}spike_tensor.pt')
    
    # filename = training_set.samples[129]
    # print(filename)
    # a_spike_load = torch.load(f'data_tensors_2{os.sep}{filename.split(os.sep)[1]}{os.sep}{filename.split(os.sep)[2]}{os.sep}{filename.split(os.sep)[3]}{os.sep}spike_tensor.pt')
    # a_spike_load_img = a_spike_load[1,:,:,3].reshape(96,96)
    
    
        

