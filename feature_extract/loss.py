
from __future__ import print_function

import torch
import torch.nn as nn

import os,sys

cwd = os.getcwd()
lava_dl_path = f"{cwd}{os.sep}..{os.sep}lava-dl{os.sep}src"
sys.path.insert(0,lava_dl_path)
lava_path = f"{cwd}{os.sep}..{os.sep}lava{os.sep}src"
sys.path.insert(0,lava_path)
sys.path.insert(0,cwd)

import lava.lib.dl.slayer as slayer
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, input, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        input = input.reshape(input.shape[0], -1, input.shape[-1])
        
        features = torch.mean(input,dim=-1)
        # print(features.shape)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # contrast_count = features.shape[1]
        # contrast_count required for number of patches from image in the reference algorithm
        # may be useful when doing augmentation
        contrast_count = 1
        contrast_feature = features
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T).to(device),
            self.temperature).to(device)
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast
        print(logits.shape)
        # tile mask
        # print(mask.shape)
        # mask = mask.repeat(anchor_count, contrast_count)
        
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ).to(device)
        
        mask = mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        print(log_prob)
        # print(log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print(loss)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class TripletLossWithMining(nn.Module):
    def __init__(
        self, 
        moving_window=None, reduction='mean'
    ):
        super(TripletLossWithMining, self).__init__()
        
        self.reduction = reduction
        if moving_window is not None:
            self.window = slayer.classifier.MovingWindow(moving_window)
        else:
            self.window = None

    def forward(self, input, label):
        """Forward computation of loss.
        """
        input = input.reshape(input.shape[0], -1, input.shape[-1])
        if self.window is None:  # one label for each sample in a batch
            
            spike_rate = slayer.classifier.Rate.rate(input)
            a_spike_rate, p_spike_rate, n_spike_rate = torch.split(spike_rate,int(spike_rate.shape[0]/3),dim=0)
            return F.triplet_margin_loss(
                a_spike_rate,
                p_spike_rate,
                n_spike_rate,
                swap=True,
                reduction=self.reduction
            )

if __name__ == '__main__':

    loss = SupConLoss()
    input = torch.rand(4,32,25)
    print(input)
    print(loss.forward(input=input, labels=torch.tensor([0,1,1,0])))