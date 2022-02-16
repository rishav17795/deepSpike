from doctest import OutputChecker
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
import h5py
import loss
from sklearn.manifold import TSNE

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

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
                    weight_norm=True, delay=True
                ),
                slayer.block.cuba.Pool(
                    neuron_params_drop, kernel_size = (2,2), stride = (2,2), padding = 0,
                    weight_norm=True, delay=True
                ),
                slayer.block.cuba.Flatten(),
                slayer.block.cuba.Dense(
                    neuron_params, in_neurons = 8*24*24, out_neurons = 128,
                    weight_norm=True
                ),
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

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

if __name__ == '__main__':
    trained_folder = 'Trained'
    os.makedirs(trained_folder, exist_ok=True)

    device = torch.device('cpu')
    # device = torch.device('cuda')

    net = Network().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    error = loss.TripletLossWithMining().to(device)

    training_set = rot2020_dataset.ROTDataset(train=True,device=device,loss=error)
    testing_set = rot2020_dataset.ROTDataset(train=False,device=device,loss=error)

    train_loader = DataLoader(
            dataset=training_set, batch_size=4, shuffle=True
        )
    test_loader = DataLoader(dataset=testing_set, batch_size=4, shuffle=True)

    # error = slayer.loss.SpikeRate(
    #         true_rate=0.2, false_rate=0.03, reduction='sum'
    #     ).to(device)
    # error = slayer.loss.SpikeMax(mode='logsoftmax').to(device)

    

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
            net, error, optimizer, stats, count_log=True
        )

    epochs = 20

    for epoch in range(epochs):
        for i, (input, label) in enumerate(train_loader):  # training loop
            # print(''.join( [f'{training_set.all_labels[label[k].item()]} ' for k in range(len(label))] ))
            a_input, p_input, n_input = torch.split(input,2,dim=1)
            input = torch.cat((a_input,p_input, n_input),dim=0).to(device)
            label = label.reshape((-1,1)).to(device)           
            # print(label)
            output, count = assistant.train(input, label)
            # print(output)
            header = [
                    'Event rate : ' +
                    ', '.join([f'{c.item():.4f}' for c in count.flatten()])
                ]
            stats.print(epoch, iter=i, header=header, dataloader=train_loader)
        
        # test_features = torch.empty((1,128)).cpu()
        # labels = torch.empty(1,1).cpu()
        for i, (input, label) in enumerate(test_loader):  # testing loop

            a_input, p_input, n_input = torch.split(input,2,dim=1)
            input = torch.cat((a_input,p_input, n_input),dim=0).to(device)

            output, count = assistant.test(input, label)
            # label = label.reshape((-1,1)).detach().cpu()
            # spike_rate = slayer.classifier.Rate.rate(output)
            # a_spike_rate, p_spike_rate, n_spike_rate = torch.split(spike_rate,int(spike_rate.shape[0]/3),dim=0)
            # test_features = torch.cat((test_features,a_spike_rate.detach().cpu()),dim=0)
            # labels = torch.cat((labels,label),dim=0)
            header = [
                    'Event rate : ' +
                    ', '.join([f'{c.item():.4f}' for c in count.flatten()])
                ]
            stats.print(epoch, iter=i, header=header, dataloader=test_loader)

        if stats.testing.best_loss:
            torch.save(net.state_dict(), trained_folder + os.sep + 'network.pt')
            # test_features = np.array(test_features)
            # print(test_features.shape)
            # tsne = TSNE(2, verbose=1)
            # tsne_proj = tsne.fit_transform(test_features)
            # print(tsne_proj.shape)
            # # Plot those points as a scatter plot and label them based on the pred labels
            # cmap = cm.get_cmap('tab20')
            # fig, ax = plt.subplots(figsize=(8,8))
            # num_categories = len(training_set.all_labels)
            # for lab in range(num_categories):
            #     indices = labels==lab
            #     indices = indices.reshape((1,-1))
            #     print(tsne_proj[indices,0])
            #     ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
            # ax.legend(fontsize='large', markerscale=2)
            # plt.savefig(f'{trained_folder}{os.sep}tsne_epoch_{epoch}.png')

        stats.update()
        stats.save(trained_folder + os.sep)
        stats.plot(path=trained_folder + os.sep)
        net.grad_flow(trained_folder + os.sep)