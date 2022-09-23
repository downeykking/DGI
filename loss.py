import numpy as np
import torch.nn.functional as F
import torch


class JSD():
    def __init__(self, discriminator=lambda x, y: x @ y.t()):
        # see paper 'https://arxiv.org/pdf/1808.06670.pdf' formula(4)
        self.discriminator = discriminator

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_pos = pos_mask.int().sum()
        num_neg = neg_mask.int().sum()

        similarity = self.discriminator(anchor, sample)

        E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
        E_pos /= num_pos

        neg_sim = similarity * neg_mask
        E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)).sum()
        E_neg /= num_neg

        return -(E_pos - E_neg)

    def __call__(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        loss = self.compute(anchor, sample, pos_mask, neg_mask)
        return loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCE():
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        pos_sim = exp_sim * pos_mask
        pos_sim = pos_sim.sum(dim=1)
        loss = pos_sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        return -loss.mean()

    def __call__(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        loss = self.compute(anchor, sample, pos_mask, neg_mask)
        return loss


