import os
import torch
from torch.utils.data.dataloader import DataLoader
import rot2020_dataset 
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
import h5py
import loss


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

    # device = torch.device('cpu')
    device = torch.device('cuda')

    net = Network().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    training_set = rot2020_dataset.ROTDataset(train=True)
    testing_set = rot2020_dataset.ROTDataset(train=False)

    train_loader = DataLoader(
            dataset=training_set, batch_size=16, shuffle=True
        )
    test_loader = DataLoader(dataset=testing_set, batch_size=16, shuffle=True)

    # error = slayer.loss.SpikeRate(
    #         true_rate=0.2, false_rate=0.03, reduction='sum'
    #     ).to(device)
    # error = slayer.loss.SpikeMax(mode='logsoftmax').to(device)

    error = loss.SupConLoss().to(device)

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
            net, error, optimizer, stats,
            classifier=slayer.classifier.Rate.predict, count_log=True
        )

    epochs = 20

    for epoch in range(epochs):
        for i, (input, label) in enumerate(train_loader):  # training loop
            # print(''.join( [f'{training_set.all_labels[label[k].item()]} ' for k in range(len(label))] ))
            input, input_aug = torch.split(input,2,dim=1)
            input = torch.cat((input,input_aug),dim=0)
            label = torch.cat((label,label),dim=0)
            output, count = assistant.train(input, label)
            header = [
                    'Event rate : ' +
                    ', '.join([f'{c.item():.4f}' for c in count.flatten()])
                ]
            stats.print(epoch, iter=i, header=header, dataloader=train_loader)

        for i, (input, label) in enumerate(test_loader):  # testing loop
            input, input_aug = torch.split(input,2,dim=1)
            input = torch.cat((input,input_aug),dim=0)
            label = torch.cat((label,label),dim=0)
            output, count = assistant.test(input, label)
            header = [
                    'Event rate : ' +
                    ', '.join([f'{c.item():.4f}' for c in count.flatten()])
                ]
            stats.print(epoch, iter=i, header=header, dataloader=test_loader)

        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), trained_folder + os.sep + 'network.pt')
        stats.update()
        stats.save(trained_folder + os.sep)
        stats.plot(path=trained_folder + os.sep)
        net.grad_flow(trained_folder + os.sep)