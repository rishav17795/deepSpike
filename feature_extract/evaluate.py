import os,sys
import numpy as np

cwd = os.getcwd()
lava_dl_path = f"{cwd}{os.sep}..{os.sep}lava-dl{os.sep}src"
sys.path.insert(0,lava_dl_path)
lava_path = f"{cwd}{os.sep}..{os.sep}lava{os.sep}src"
sys.path.insert(0,lava_path)
sys.path.insert(0,cwd)

import torch
from torch.utils.data.dataloader import DataLoader
import rot2020_dataset 
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from torchsummary import summary 
import warnings

import train_triplet
import loss

class Evaluator(torch.nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()

        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 0.25,
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : False,
            }
        
        neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Conv(
                    neuron_params_drop, in_features = 2, out_features = 8,
                    kernel_size = (3,3) , stride = (2,2), padding = 1,
                    weight_norm=True, delay=False
                )
            ])

    def forward(self, spike):
        count = []
        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)
        ).to(spike.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    
if __name__ == '__main__':

    trained_folder = 'Evaluation'
    os.makedirs(trained_folder, exist_ok=True)
    device = torch.device('cpu')

    net = train_triplet.Network().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    # # net = hdf5.Network(net_config=f'trained{os.sep}network.net')

    # # print(net.state_dict()["blocks.0.synapse.weight_v" ])

    # # net.load_state_dict(torch.load(f'trained{os.sep}network.pt', map_location=device))
    # # net.load_state_dict( torch.load(f'Trained{os.sep}network.pt',map_location=device))
    load_dict = torch.load(f'Trained{os.sep}network.pt',map_location=device)
    print("Model's state_dict:")
    for param_tensor in load_dict:
        print(param_tensor, "\t", load_dict[param_tensor].size())
    net_new_sd = net.state_dict()
    net_new_sd["blocks.0.neuron.current_decay" ] = load_dict["blocks.0.neuron.current_decay"]
    net_new_sd["blocks.0.neuron.voltage_decay" ] = load_dict["blocks.0.neuron.voltage_decay"]
    net_new_sd["blocks.0.synapse.weight_v" ] = load_dict["blocks.0.synapse.weight"]
    net_new_sd["blocks.1.neuron.current_decay" ] = load_dict["blocks.1.neuron.current_decay"]
    net_new_sd["blocks.1.neuron.voltage_decay" ] = load_dict["blocks.1.neuron.voltage_decay"]
    net_new_sd["blocks.1.synapse.weight_v" ] = load_dict["blocks.1.synapse.weight"]
    net_new_sd["blocks.3.neuron.current_decay" ] = load_dict["blocks.3.neuron.current_decay"]
    net_new_sd["blocks.3.neuron.voltage_decay" ] = load_dict["blocks.3.neuron.voltage_decay"]
    net_new_sd["blocks.3.synapse.weight_v" ] = load_dict["blocks.3.synapse.weight"]

    net.load_state_dict(net_new_sd)

    error = loss.TripletLossWithMining()

    training_set = rot2020_dataset.ROTDataset(train=True,device=device,loss=error)
    testing_set = rot2020_dataset.ROTDataset(train=False,device=device,loss=error)

    train_loader = DataLoader(
            dataset=training_set, batch_size=4, shuffle=True
        )
    test_loader = DataLoader(dataset=testing_set, batch_size=4, shuffle=True)

    # # # # stats = slayer.utils.LearningStats()
    # # # # assistant = slayer.utils.Assistant(
    # # # #         net, error, optimizer, stats, count_log=True
    # # # #     )

    # test_features = torch.empty((1,128)).cpu()
    # labels = torch.empty(1,1).cpu()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # for i, (input, label) in enumerate(test_loader):  # training loop
                
        #         a_input, p_input, n_input = torch.split(input,2,dim=1)
        #         input = torch.cat((a_input,p_input, n_input),dim=0).to(device)
                
        #         output, count = net(input)
        #         label = label.reshape((-1,1)).detach().cpu()
        #         spike_rate = slayer.classifier.Rate.rate(output)
        #         a_spike_rate, p_spike_rate, n_spike_rate = torch.split(spike_rate,int(spike_rate.shape[0]/3),dim=0)
        #         test_features = torch.cat((test_features,a_spike_rate.detach().cpu()),dim=0)
        #         labels = torch.cat((labels,label),dim=0)

        # test_features = np.array(test_features)
        # labels = np.array(labels)
        # tsne = TSNE(2)
        # tsne_proj = tsne.fit_transform(test_features)
        # # Plot those points as a scatter plot and label them based on the pred labels
        # cmap = cm.get_cmap('tab20')
        # fig, ax = plt.subplots(figsize=(8,8))
        # num_categories = len(training_set.all_labels)
        # for lab in range(num_categories):
        #     indices = labels==lab
        #     indices = indices.reshape((1,-1)).nonzero()
        #     ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        # ax.legend(labels = training_set.all_labels, fontsize='large', markerscale=2)
        # plt.savefig(f'{trained_folder}{os.sep}tsne_epoch.png')

        # *******************************************************************************************************************
        # filters = net.state_dict()["blocks.0.synapse.weight_v" ]
        # print(filters.shape)
        # f_min, f_max = filters.min(), filters.max()
        # filters = (filters - f_min) / (f_max - f_min)
        # n_filters, ix = 8, 1

        # for i in range(n_filters):
        #     # get the filter
        #     f = filters[i, :, :, :, :].reshape((2,3,3))
        #     # plot each channel separately
        #     for j in range(2):
        #         # specify subplot and turn of axis
        #         ax = plt.subplot(n_filters, 2, ix)
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         # plot filter channel in grayscale
        #         plt.imshow(f[j, :, :], cmap='gray')
        #         ix += 1
        # # show the figure
        # plt.show()
        # *******************************************************************************************************************

        eval_net = Evaluator().to(device=device)
        
        eval_new = eval_net.state_dict()
        eval_new["blocks.0.neuron.current_decay" ] = load_dict["blocks.0.neuron.current_decay"]
        eval_new["blocks.0.neuron.voltage_decay" ] = load_dict["blocks.0.neuron.voltage_decay"]
        eval_new["blocks.0.synapse.weight_v" ] = load_dict["blocks.0.synapse.weight"]
        eval_net.load_state_dict(eval_new)
        

        for i, (input, label) in enumerate(test_loader):  # training loop
                
            a_input, p_input, n_input = torch.split(input,2,dim=1)
            # input = torch.cat((a_input,p_input, n_input),dim=0).to(device)
            print(a_input.shape)
            output, count = eval_net(a_input)
            output = output.detach().numpy()
            f_min, f_max = output.min(), output.max()
            output_sc = (output - f_min) / (f_max - f_min)
            ax = plt.subplot(3, 3, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(a_input[0, 1, :, :, 10].reshape((96,96)), cmap='gray')
            for ix in range(8):
                    # specify subplot and turn of axis
                    ax = plt.subplot(3, 3, ix+2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(output_sc[0, ix, :, :, 2], cmap='gray')
            # show the figure
            plt.show()
            
            break
            
