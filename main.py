import torch
import torch_geometric.transforms as T

import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid

import os
from loss import JSD, InfoNCE
import numpy as np

from model import SingleBranchContrast
from eval import get_split, LREvaluator
from evaluate_embedding import evaluate_embedding
from torch_geometric import seed_everything


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index, edge_weight)
            z = act(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn

    def get_embeddings(self, data):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        emb = []
        y = []
        with torch.no_grad():
            data = data.to(device)
            z, _, _ = self.forward(data.x, data.edge_index)
            emb.append(z.cpu().numpy())
            y.append(data.y.cpu().numpy())
        emb = np.concatenate(emb, 0)
        y = np.concatenate(y, 0)
        return emb, y


def train(encoder_model, contrast_model, data, criterion, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    anchor, sample, pos_mask, neg_mask = contrast_model(h=z, g=g, hn=zn)
    loss = criterion(anchor, sample, pos_mask, neg_mask)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cuda:1')
    homedir = os.environ['HOME']
    data_dir = "dataset"
    dataset_name = "Cora"
    path = os.path.join(homedir, data_dir, dataset_name)

    dataset = Planetoid(path, name=dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    seed_everything(22)

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=512).to(device)
    contrast_model = SingleBranchContrast().to(device)
    criterion = JSD()

    optimizer = Adam(encoder_model.parameters(), lr=0.001)

    best_loss = 1e9
    best_t = 0
    patience = 20

    for epoch in range(1, 10000):
        loss = train(encoder_model, contrast_model, data, criterion, optimizer)

        print(f'Train: Epoch {epoch:02d}, Loss: {loss:.4f}')

        if loss < best_loss:
            best_loss = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(encoder_model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

    print('Loading {}th epoch'.format(best_t))
    encoder_model.load_state_dict(torch.load('best_dgi.pkl'))

    embeds, y = encoder_model.get_embeddings(data)
    leg_mean, leg_std = evaluate_embedding(embeds, y, 10)
    print(f'leg mean: {leg_mean:.4f}, leg std: {leg_std:.4f}')

if __name__ == '__main__':
    main()
