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

import train_triplet
import loss

if __name__ == '__main__':

    device = torch.device('cpu')

    net = train_triplet.Network().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    error = loss.TripletLossWithMining()

    training_set = rot2020_dataset.ROTDataset(train=True,device=device,loss=error)
    testing_set = rot2020_dataset.ROTDataset(train=False,device=device,loss=error)

    train_loader = DataLoader(
            dataset=training_set, batch_size=4, shuffle=True
        )
    test_loader = DataLoader(dataset=testing_set, batch_size=4, shuffle=True)

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
            net, error, optimizer, stats, count_log=True
        )

    test_features = torch.empty((1,128)).cpu()
    labels = torch.empty(1,1).cpu()
    for i, (input, label) in enumerate(test_loader):  # training loop
            
            a_input, p_input, n_input = torch.split(input,2,dim=1)
            input = torch.cat((a_input,p_input, n_input),dim=0).to(device)
            
            output, count = assistant.test(input, label)
            label = label.reshape((-1,1)).detach().cpu()
            spike_rate = slayer.classifier.Rate.rate(output)
            a_spike_rate, p_spike_rate, n_spike_rate = torch.split(spike_rate,int(spike_rate.shape[0]/3),dim=0)
            test_features = torch.cat((test_features,a_spike_rate.detach().cpu()),dim=0)
            labels = torch.cat((labels,label),dim=0)

    test_features = np.array(test_features)
    labels = np.array(labels)
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_features)
    print(tsne_proj.shape)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = len(training_set.all_labels)
    for lab in range(num_categories):
        indices = labels==lab
        indices = indices.reshape((1,-1)).nonzero()
        print(tsne_proj[[1,2,3,4],0])
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig(f'trained{os.sep}tsne_epoch_{1}.png')
