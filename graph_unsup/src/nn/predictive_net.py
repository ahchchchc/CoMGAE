import copy
import random
import time
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

import numpy as np
import math
from easydict import EasyDict as edict

from .layers import GINConv
from ..utils import create_activation


class ReconNet(nn.Module):
    def __init__(self, N_coms, in_dim, emb_dim, N_classes, x_loss_lamb, random_loss_lamb, requires_chunk=True, get_emb_type='all', recomp_type='linear', **kwargs):
        super(ReconNet, self).__init__()
        self.kwargs = kwargs
        self.N_coms = N_coms
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.N_classes = N_classes
        self._x_lambda = x_loss_lamb
        self._random_lambda = random_loss_lamb
        self.requires_chunk = requires_chunk
        self.get_emb_type = get_emb_type
        self.recomp_type = recomp_type

        chunks = np.array_split(np.ones(emb_dim), N_coms)
        com_emb_dims = list(map(lambda x: int(x.sum()), chunks))
        self.com_emb_dim = com_emb_dims[0]

        # modules to get node representations
        self.EdgePart = EdgePart(**self.configs().EdgePart)
        self.NodePart = NodePart(**self.configs().NodePart)

        self.ComGNNBank = ComGNNBank(**self.configs().ComGNNBank)

        if recomp_type == 'linear':
            self.RepComposer = nn.Linear(emb_dim, emb_dim)
        else:
            self.RepComposer = RepComposer(**self.configs().RepComposer)


        # node to graph (n2g) summarization
        self.N_layers = 1 + self.configs().ComGNNBank.N_layers + self.configs().RepComposer.N_layers


        self.decoder = make_gin_conv(emb_dim, in_dim, train_eps=False)


    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    def forward(self, z, data, mask_rate, fixed_mask=False):
        '''
        args:
            z: node-community affiliation, obtained from the InfNet; masked hidden
            data: unmask input
        '''
        x, edge_index = data.x, data.edge_index

        edge_weight_list = self.EdgePart(z, edge_index)  # torch.unbind(torch.ones((edge_index.shape[-1], self.N_coms), device=z.device), dim=1)
        x_commun_list, commun_indices_list = self.NodePart(x, z, self.requires_chunk)


        # mask1 -- mask inside x_part
        mask_x_list, mask_node_list = [], []
        for x_com, indices in zip(x_commun_list, commun_indices_list):
            _x = torch.clone(x_com)
            mask_part_x, (mask_part_node, _) = self.encoding_mask_noise(_x, indices, mask_rate)
            mask_x_list.append(mask_part_x)
            mask_node_list.append(mask_part_node)

        # mask2 -- mask whole x
        mask_node = torch.unique(torch.cat(mask_node_list))
        if fixed_mask:
            mask_x, (_, _) = self.encoding_mask_noise(x, mask_rate=mask_rate, mask_nodes=mask_node)
        else:
            mask_x, (mask_node, _) = self.encoding_mask_noise(x, mask_rate=mask_rate)


        hs = self.ComGNNBank(edge_index, edge_weight_list, mask_x, mask_x_list)
        hh = hs[-1].clone()  # final GNN output of all the communities
        if self.recomp_type == 'linear':
            hs += [self.RepComposer(hh)]
        else:
            hs += self.RepComposer(hh, edge_index)

        all_mask_node = torch.unique(torch.cat(mask_node_list)) # remask hidden
        for i in range(len(hs)):
            hs[i][all_mask_node] = 0

        # random mask on whole x

        h_random = self.ComGNNBank(edge_index, tuple(None for _ in range(self.N_coms)), mask_x)  #
        hh = h_random[-1].clone()
        if self.recomp_type == 'linear':
            h_random += [self.RepComposer(hh)]
        else:
            h_random += self.RepComposer(hh, edge_index)
        for i in range(len(h_random)):
            h_random[i][mask_node] = 0

        h_init = hs[-1]
        _, (mask_hidden, _) = self.encoding_mask_noise(h_init, mask_rate=mask_rate)
        # h_random_rep = self.RepComposer(h_random, edge_index)[0]
        # h_random_rep[mask_hidden] = 0   # remask

        h_init = h_random[-1]
        _, (mask_hidden, _) = self.encoding_mask_noise(h_init, mask_rate=mask_rate)

        # Recon x
        recon_x = torch.zeros_like(x)
        for h in hs:
            recon_x += self.decoder(h, edge_index)
        x_init = x[all_mask_node]
        x_rec = recon_x[all_mask_node]
        loss_x = self.sce_loss(x_rec, x_init)

        # Recon X random
        recon_x = torch.zeros_like(x)
        for h in h_random:
            recon_x += self.decoder(h, edge_index)
        x_init = x[mask_node]
        x_rec = recon_x[mask_node]
        loss_random = self.sce_loss(x_rec, x_init)

        return self._x_lambda * loss_x + self._random_lambda * loss_random

    def get_emb(self, z, data):
        x, edge_index = data.x, data.edge_index

        edge_weight_list = self.EdgePart(z, edge_index)  # tuple(None for _ in range(self.N_coms))  torch.unbind(torch.ones((edge_index.shape[-1], self.N_coms), device=z.device), dim=1)
        x_commun_list, _ = self.NodePart(x, z)

        hs = self.ComGNNBank(edge_index, edge_weight_list, x, x_commun_list)
        hh = hs[-1]
        if self.recomp_type == 'linear':
            hs += [self.RepComposer(hh)]
        else:
            hs += self.RepComposer(hh, edge_index)

        h_random = self.ComGNNBank(edge_index, tuple(None for _ in range(self.N_coms)), x)  # todo
        hh = h_random[-1]
        if self.recomp_type == 'linear':
            h_random += [self.RepComposer(hh)]
        else:
            h_random += self.RepComposer(hh, edge_index)
        h_random = h_random[1: ]

        hs = hs + h_random # only used for NCI1, IMDB-BINARY,

        if self.get_emb_type == 'all':
            use_emb = hs
        else: # part
            use_emb = [hs[0] + hs[-1]]

        return use_emb # [hs[0] + hs[-1]]  # hs[-(self.configs.RepComposer.N_layers + 1):]

    def get_emb_indices(self, z, data):
        x, edge_index = data.x, data.edge_index

        edge_weight_list = self.EdgePart(z, edge_index)  # tuple(None for _ in range(self.N_coms))  torch.unbind(torch.ones((edge_index.shape[-1], self.N_coms), device=z.device), dim=1)
        x_commun_list, indices_list = self.NodePart(x, z)
        # x_commun_list = [x for _ in range(self.N_coms)]

        hs = self.ComGNNBank(edge_index, edge_weight_list, x, x_commun_list)
        hh = hs[-1]
        if self.recomp_type == 'linear':
            hs += [self.RepComposer(hh)]
        else:
            hs += self.RepComposer(hh, edge_index)
        com_emb = copy.deepcopy(hs)

        h_random = self.ComGNNBank(edge_index, edge_weight_list, x)  # todo
        hh = h_random[-1]
        if self.recomp_type == 'linear':
            h_random += [self.RepComposer(hh)]
        else:
            h_random += self.RepComposer(hh, edge_index)
        global_emb = copy.deepcopy(h_random)
        h_random[0] = 0

        hs = [i + j for i, j in zip(hs, h_random)]
        if self.get_emb_type == 'all':
            use_emb = hs
        else:
            use_emb = [hs[0] + hs[-1]]

        return com_emb, global_emb, use_emb, indices_list # hs[-(self.configs.RepComposer.N_layers + 1):]

    def get_commun(self, z, data):
        x, edge_index = data.x, data.edge_index

        edge_weight_list = self.EdgePart(z, edge_index)  # tuple(None for _ in range(self.N_coms))  torch.unbind(torch.ones((edge_index.shape[-1], self.N_coms), device=z.device), dim=1)
        x_commun_list, indices_list = self.NodePart(x, z)
        # x_commun_list = [x for _ in range(self.N_coms)]

        commun_emb = self.ComGNNBank.get_commun(edge_index, edge_weight_list, x, x_commun_list)
        # hh = hs[-1]
        # if self.recomp_type == 'linear':
        #     hs += [self.RepComposer(hh)]
        # else:
        #     hs += self.RepComposer(hh, edge_index)

        # h_random = self.ComGNNBank(edge_index, edge_weight_list, x)  # todo
        # hh = h_random[-1]
        # if self.recomp_type == 'linear':
        #     h_random += [self.RepComposer(hh)]
        # else:
        #     h_random += self.RepComposer(hh, edge_index)
        # h_random[0] = 0

        # hs = [i + j for i, j in zip(hs, h_random)]
        # if self.get_emb_type == 'all':
        #     use_emb = hs
        # else:
        #     use_emb = [hs[0] + hs[-1]]

        return commun_emb # hs[-(self.configs.RepComposer.N_layers + 1):]

    def inter_community_mask(self, x_commun, mask_rate=0.5):
        device = x_commun.device
        num_hidden = len(x_commun)

        mask_num = math.ceil(num_hidden * mask_rate)  # at least one community will be masked
        perm = torch.randperm(num_hidden, device=device)
        mask_indices = perm[:mask_num]
        keep_indices = perm[mask_num:]

        mask_commun = torch.zeros_like(x_commun)
        mask_commun[keep_indices] = x_commun[keep_indices]

        return mask_commun, mask_indices

    def encoding_mask_noise(self, x, indices=None, mask_rate=0.5, replace_rate=0., mask_nodes=None):
        if indices is not None:
            num_nodes = len(indices)
            perm = torch.randperm(num_nodes, device=x.device)
            indices_shuffle = indices[perm]
        else:
            num_nodes = len(x)  # g.num_nodes()
            indices_shuffle = torch.randperm(num_nodes, device=x.device)

        if mask_nodes is not None:
            num_mask_nodes = mask_nodes.shape[0]
            keep_nodes = indices_shuffle[~torch.isin(indices_shuffle, mask_nodes.to(x.device))]
        else:
            # random masking
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = indices_shuffle[: num_mask_nodes]
            keep_nodes = indices_shuffle[num_mask_nodes:]

        if replace_rate > 0:
            num_noise_nodes = max(int(self._replace_rate * num_mask_nodes), 1)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(g.num_nodes(), device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        # out_x[token_nodes] += self.enc_mask_token

        if mask_rate == 0:
            mask_nodes = None

        return out_x, (mask_nodes, keep_nodes)

    # @property
    def configs(self):
        _configs = edict()

        # EdgePart Configurations
        _configs.EdgePart = edict()
        _configs.EdgePart.N_coms = self.N_coms
        _configs.EdgePart.dropout = self._set_default('ep_dropout', .5)
        _configs.EdgePart.tau = self._set_default('ep_tau', 1.)

        # NodePart Configurations
        _configs.NodePart = edict()
        _configs.NodePart.N_coms = self.N_coms
        _configs.NodePart.dropout = self._set_default('ep_dropout', .5)

        # ComGNNBank Configurations
        _configs.ComGNNBank = edict()
        _configs.ComGNNBank.N_coms = self.N_coms
        _configs.ComGNNBank.in_dim = self.in_dim
        _configs.ComGNNBank.emb_dim = self.emb_dim
        _configs.ComGNNBank.N_layers = self._set_default('N_layers_h', 2)
        _configs.ComGNNBank.dropout = self._set_default('dropout', .5)

        # RepComposer Configurations
        _configs.RepComposer = edict()
        _configs.RepComposer.emb_dim = self.emb_dim
        _configs.RepComposer.N_layers = self._set_default('N_layers_t', 1)
        _configs.RepComposer.dropout = self._set_default('rc_dropout', .5)

        # InterMaskEnc Configurations
        _configs.InterMaskEnc = edict()
        _configs.InterMaskEnc.in_dim = self.com_emb_dim  # self.emb_dim
        _configs.InterMaskEnc.emb_dim = self.com_emb_dim
        _configs.InterMaskEnc.N_layers = self._set_default('N_layers_t', 1)
        _configs.InterMaskEnc.dropout = self._set_default('rc_dropout', .5)

        # # n2g configuration
        # _configs.n2g = edict()
        # _configs.n2g.coverage = self.n2g_coverage
        # _configs.n2g.final_dropout = self._set_default('final_dropout', .5)

        return _configs

    def _set_default(self, key, default):
        if key in self.kwargs:
            return self.kwargs[key]
        else:
            return default




def create_predictor(emb_dim, N_classes, dropout=0.):
    return nn.Sequential(
        nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(emb_dim, N_classes)
    )


###################################### Edge Partitioner ######################################

class EdgePart(nn.Module):
    def __init__(self, N_coms, dropout, tau=1., max_edges=10000):
        '''
            Partition the edges according to node-community affiliation (s).
            args:
                N_coms: number of communities.
                dropout: dropout on (s) before computing the partition weights.
                tau: softmax temperature along the community dimension.
                max_edges: the edge `chunk` size. (Splitting edges into chunks to prevent OOM.)
        '''
        super(EdgePart, self).__init__()
        self.N_coms = N_coms
        self.dropout = dropout
        self.tau = tau
        self.max_edges = max_edges
        # self.BN = nn.BatchNorm1d(N_coms)

    def chunk_computer(self, z, edge_index):
        row, col = edge_index
        z_start, z_end = z[row], z[col]
        return torch.sum(z_start * z_end, dim=1)

    def edge_weigher(self, z, edge_index):
        edge_index_chunks = edge_index.split(self.max_edges, dim=-1)
        return torch.cat([self.chunk_computer(z, indices) for indices in edge_index_chunks])

    def forward(self, z, edge_index):
        '''
            Compute the partition weights for all edges.
            Args:
                z: node-community affiliation matrix (denoted as Z in paper).
                edge_index: sparse adj edge indices.
            Output: a list of edge weights, with each element corresponding to weights in one community graph.
        '''
        z = F.dropout(z, self.dropout, training=self.training)
        z_chunks = z.tensor_split(self.N_coms, dim=-1)

        edge_weight_unnorm = torch.stack(
            [self.edge_weigher(z_k, edge_index) for z_k in z_chunks])  # .transpose(1, 0)  # shape = [N_coms, N_edges]
        # edge_weight_unnorm = self.BN(edge_weight_unnorm).transpose(1, 0)
        edge_weight_list = torch.unbind(F.softmax(edge_weight_unnorm / self.tau, dim=0), dim=0)
        return edge_weight_list


class NodePart(nn.Module):
    def __init__(self, N_coms, dropout, threshold=1.0, max_nodes=10000):
        '''
            Partition the edges according to node-community affiliation (s).
            args:
                N_coms: number of communities.
                dropout: dropout on (s) before computing the partition weights.
                threshold: threshold to separate the community
        '''
        super(NodePart, self).__init__()
        self.N_coms = N_coms
        self.dropout = dropout
        self.thershold = threshold
        self.max_nodes = max_nodes
        # self.BN = nn.BatchNorm1d(N_coms)

    #
    def chunk_computer(self, phi):
        # node_num = phi.shape[0]
        # device = phi.device
        # m_count = torch.einsum('ij,lm->ijml', phi, phi)
        #
        # _m_1 = torch.sum(m_count, dim=2)
        # mask_1 = torch.triu(torch.ones((node_num, node_num), device=device), diagonal=1).unsqueeze(1)  # left lower triangle = 0 (contrain diagonal)
        # m_1 = torch.sum(_m_1 * mask_1, dim=2)
        # m_1_norm = F.softmax(m_1, dim=1)
        #
        # _m_2 = torch.sum(m_count, dim=1)
        # mask_2 = mask_1 # torch.unsqueeze( torch.tril(torch.ones((node_num, node_num), device=device), diagonal=0) - torch.eye(node_num,device=device),dim=1)  # upper right = 0
        # m_2 = torch.sum(_m_2 * mask_2, dim=2)
        # m_2_norm = F.softmax(m_2, dim=1)
        # node_weight = F.softmax(m_1_norm + m_2_norm, dim=1)

        node_num = phi.shape[0]
        device = phi.device
        phi = F.softmax(phi, dim=0)
        m_count = torch.einsum('ij,lm->ijml', phi, phi)

        _m_1 = torch.sum(m_count, dim=2)
        mask_1 = torch.triu(torch.ones((node_num, node_num), device=device), diagonal=1).unsqueeze(1)  # left lower triangle = 0 (contrain diagonal)
        m_1 = torch.sum(_m_1 * mask_1, dim=2)

        _m_2 = torch.sum(m_count, dim=1)
        mask_2 = mask_1 #torch.unsqueeze(torch.tril(torch.ones((node_num, node_num), device=device), diagonal=0) - torch.eye(node_num, device=device),dim=1)  # upper right = 0
        m_2 = torch.sum(_m_2 * mask_2, dim=0).transpose(1, 0)
        node_weight = m_1 + m_2 #F.softmax((m_1 + m_2) / self.tau, dim=0)
        # node_weight = F.softmax(unnode_weight, dim=1)
        return node_weight

    #
    def node_chunker(self, phi):
        phi_chunks = phi.split(self.max_nodes, dim=0)
        return torch.cat([self.chunk_computer(phi) for phi in phi_chunks])

    def forward(self, x, z, requires_chunk=False):
        '''
            Compute the partition weights for all edges.
            Args:
                z: node-community affiliation matrix (denoted as Z in paper).
                edge_index: sparse adj edge indices.
            Output: a list of edge weights, with each element corresponding to weights in one community graph.
        '''
        node_num = z.shape[0]
        z = F.dropout(z, self.dropout, training=self.training)
        z_chunks = z.tensor_split(self.N_coms, dim=-1)

        # t0 = time.time()
        phi = torch.empty((node_num, self.N_coms), device=z.device)
        for i, chunk in enumerate(z_chunks):
            phi[:, i] = torch.mean(chunk, dim=1)
            # phi[:, i] = torch.sum(chunk, dim=1)
        # print(f' first {time.time()-t0: .4f} s.')

        if requires_chunk:
            node_weight = self.node_chunker(phi)
        else:
            node_weight = self.chunk_computer(phi)
        # node_weight = self.BN(node_weight)

        max_value = torch.max(node_weight, dim=1).values.unsqueeze(1).repeat(1, self.N_coms)
        node_mask_1 = torch.eq(node_weight, max_value)
        node_mask_2 = torch.where(node_weight >= self.thershold, True, False)
        node_mask = torch.logical_or(node_mask_1, node_mask_2)
        x_commun_list = []
        indices_list = []
        # t0 = time.time()
        for i in range(self.N_coms):
            mask = node_mask[:, i]
            indices = torch.where(mask == True)[0]

            x_part = torch.zeros_like(x)
            x_part[indices, :] = x[indices, :]

            x_commun_list.append(x_part)
            indices_list.append(indices)
        # print(f'mask_list  {time.time() - t0: .4f} s.')
        return x_commun_list, indices_list


###################################### Multi-community GNN ######################################
class ComGNNBank(nn.Module):
    def __init__(self, N_coms, in_dim, emb_dim, N_layers, dropout, train_eps=False):
        super(ComGNNBank, self).__init__()
        self.train_eps = train_eps
        self.N_coms = N_coms
        self.emb_dim = emb_dim

        # compute emb_dim for each ComGNN
        chunks = np.array_split(np.ones(emb_dim), N_coms)
        com_emb_dims = list(map(lambda x: int(x.sum()), chunks))

        # self.ComGNNs = nn.ModuleList([ComGNN(in_dim, ED, N_layers, dropout, self.train_eps) for ED in com_emb_dims])
        self.ComGNN = ComGNN(in_dim, com_emb_dims[0], N_layers, dropout, self.train_eps)

        self.input_encoder = nn.Linear(in_dim, emb_dim)

    def forward(self, edge_index, edge_weight_list, x, x_list=None):
        # get node reps by community
        outs = []
        if x_list is not None:
            for k in range(self.N_coms):
                outs.append(self.ComGNN(x_list[k], edge_index, edge_weight_list[k]))
                # outs.append(self.ComGNNs[k](x_list[k], edge_index, edge_weight_list[k]))
        else:
            for k in range(self.N_coms):
                outs.append(self.ComGNN(x, edge_index, edge_weight_list[k]))

        # re-arrange node reps, gather them by the output layer
        # concatenate the outputs from the same layer (<CAVEAT> not including the input layer)
        outs = list(zip(*outs))
        outs = list(map(lambda tup: torch.cat(tup, dim=-1), outs))
        # 不同community都会经过两层GNN, 最后将不同community在同一层GNN的表示concat到一起
        # 无论community数量,len(outs) = GNN layer number
        return [self.input_encoder(x)] + outs
    def get_commun(self, edge_index, edge_weight_list, x, x_list=None):
        # get node reps by community
        outs = []
        if x_list is not None:
            for k in range(self.N_coms):
                outs.append(self.ComGNN(x_list[k], edge_index, edge_weight_list[k]))
                # outs.append(self.ComGNNs[k](x_list[k], edge_index, edge_weight_list[k]))
        else:
            for k in range(self.N_coms):
                outs.append(self.ComGNN(x, edge_index, edge_weight_list[k]))
        return outs # sorted by community


def make_gin_conv(Fi_dim, Fo_dim, train_eps):
    return GINConv(nn.Sequential(nn.Linear(Fi_dim, Fo_dim), nn.ReLU(), nn.Linear(Fo_dim, Fo_dim)), train_eps=train_eps)


class ComGNN(nn.Module):
    def __init__(self, in_dim, emb_dim, N_layers, dropout, train_eps=False):
        super(ComGNN, self).__init__()
        self.N_layers = N_layers
        self.dropout = dropout

        self.gconvs = nn.ModuleList([make_gin_conv(in_dim, emb_dim, train_eps)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim)])
        for i in range(N_layers - 1):
            self.gconvs.append(make_gin_conv(emb_dim, emb_dim, train_eps))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_weight=None):
        hs = [x]
        for i in range(self.N_layers):
            h = self.gconvs[i](hs[i], edge_index, edge_weight=edge_weight)
            if h.dim() == 3:
                h = F.relu(self.batch_norms[i](h.permute(0, 2, 1))).permute(0, 2, 1)
            else:
                h = F.relu(self.batch_norms[i](h))
            hs.append(F.dropout(h, self.dropout, training=self.training))
        return hs[1:]


#################################### Representation Composer ####################################

class RepComposer(nn.Module):
    def __init__(self, emb_dim, N_layers, dropout, train_eps=False):
        super(RepComposer, self).__init__()
        self.N_layers = N_layers
        self.dropout = dropout

        self.gconvs = nn.ModuleList([make_gin_conv(emb_dim, emb_dim, train_eps)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim)])
        for i in range(N_layers - 1):
            self.gconvs.append(make_gin_conv(emb_dim, emb_dim, train_eps))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, h, edge_index):
        hs = [h]
        for i in range(self.N_layers):
            h = self.gconvs[i](hs[i], edge_index)
            h = F.relu(self.batch_norms[i](h))
            hs.append(F.dropout(h, self.dropout, training=self.training))
        return hs[1:]


class InterMaskEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim, N_layers, dropout, train_eps=False):
        super(InterMaskEncoder, self).__init__()
        self.N_layers = N_layers
        self.dropout = dropout

        self.gconvs = nn.ModuleList([make_gin_conv(in_dim, emb_dim, train_eps)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim)])
        for i in range(N_layers - 1):
            self.gconvs.append(make_gin_conv(emb_dim, emb_dim, train_eps))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, h, edge_index):
        hs = [h]
        for i in range(self.N_layers):
            h = self.gconvs[i](hs[i], edge_index)
            # t = time.time()
            h = F.relu(self.batch_norms[i](h.permute(0, 2, 1)))
            # print(f'F.relu(self.batch_norms[i](h.permute(0, 2, 1))) takes {time.time() - t} s.')
            hs.append(F.dropout(h.permute(0, 2, 1), self.dropout, training=self.training))
        return hs[1:]