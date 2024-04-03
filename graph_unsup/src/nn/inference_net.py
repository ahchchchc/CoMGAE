import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
from torch_geometric import seed_everything

from ..utils import log_max


class InfNet(nn.Module):
    def __init__(self, in_dim, hid_dims, dropout, gconv_bias=False, **kwargs) -> None:
        super(InfNet, self).__init__()
        self.dropout = dropout
        
        dims = [in_dim] + list(hid_dims)
        # 1 additional dim to store the value of `kappa`, the shape parameter of weibull distribution
        dims[-1] = dims[-1] + 1 

        self.GConvLayers = nn.ModuleList(
            [GCNConv(F_in, F_out, bias=gconv_bias) for (F_in, F_out) in zip(dims[:-1], dims[1:])]
        )
        # self.GConvLayers = nn.ModuleList(
        #     [GraphSAGE(F_in, F_out, F_out) for (F_in, F_out) in zip(dims[:-1], dims[1:])]
        # )

        self.random_encoder = kwargs['random_encoder'] if 'random_encoder' in kwargs else False
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
    

    def reparameterize(self, lbd, kappa):
        '''
            weibull reparameterization: z = lbd * (- ln(1 - u)) ^ (1/kappa), u ~ uniform(0,1)
            z: node-community affiliation.
            lbd: scale parameter, kappa: shape parameter
        '''
        if self.random_encoder and self.training:
            u = torch.rand_like(lbd)
            z = lbd * (- log_max(1 - u)).pow(1 / kappa)
        else:
            z = lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))

        return z

    def mask_edge(self, g, mask_rate=0.8):
        edge_index = g.edge_index
        num_edges = edge_index.shape[1]
        perm = torch.randperm(num_edges, device=edge_index.device)

        num_mask_edge = int(mask_rate * num_edges)
        mask_edges = perm[: num_mask_edge]
        keep_edges = perm[num_mask_edge: ]

        edge_index[:, mask_edges] = 0
        return edge_index, (mask_edges, keep_edges)

    def encoding_mask_node(self, g, mask_rate=0.8, replace_rate=0.):
        num_nodes = g.num_nodes  # g.num_nodes()
        x = g.x
        perm = torch.randperm(num_nodes, device=g.x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if replace_rate > 0:
            num_noise_nodes = int(replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int((1 - replace_rate) * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        if mask_rate == 0:
            mask_nodes = None

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, data, mask_rate):
        # mask_data, mask_x, (mask_nodes, _) = self.encoding_mask_noise(data, data.x, mask_rate)
        # x, edge_index = mask_x, mask_data.edge_index
        x, edge_index = data.x, data.edge_index
        # gcn forwarding
        mask_data, mask_x, (self.mask_nodes, _) = self.encoding_mask_node(data, mask_rate)
        # mask_edge_index, (_, _) = self.mask_edge(data, mask_rate)
        h = F.softplus(self.GConvLayers[0](mask_x, edge_index))
        for gconv in self.GConvLayers[1:]:
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.softplus(gconv(h, edge_index))
        # split the output of gcn into lbd and kappa, kappa is a scalar for each node.
        lbd, kappa = h.split([h.size(1)-1, 1], dim=1)   # shared between node and edge partitioner

        z = self.reparameterize(lbd, kappa + 0.1)
        return z, lbd, kappa + 0.1, self.mask_nodes