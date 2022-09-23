import imp
import torch
import torch.nn as nn
from loss import JSD


class SingleBranchContrast(nn.Module):
    def __init__(self, **kwargs):
        super(SingleBranchContrast, self).__init__()
        self.kwargs = kwargs

    def sample(self, anchor, sample, neg_sample=None, use_gpu=True, *args, **kwargs):
        # sample: pos, neg_sample: neg, the mask is used to computer in loss.py
        num_graphs = anchor.shape[0]  # M
        num_nodes = sample.shape[0]   # N
        device = sample.device

        if neg_sample is not None:
            assert num_graphs == 1  # only one graph, explicit negative samples are needed
            assert sample.shape == neg_sample.shape
            pos_mask1 = torch.ones((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask0 = torch.zeros((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)     # M * 2N
            sample = torch.cat([sample, neg_sample], dim=0)         # 2N * D

        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask

    def forward(self, h, g, hn):
        anchor, sample, pos_mask, neg_mask = self.sample(anchor=g, sample=h, neg_sample=hn)
        return anchor, sample, pos_mask, neg_mask
