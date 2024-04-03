import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np

import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def log_max(input, SMALL=1e-10):
    device = input.device
    input_ = torch.max(input, torch.tensor([SMALL]).to(device))
    return torch.log(input_)

class UnsupGAE(object):
    '''
        computed the unsupervised loss (graph reconstruction + kl) for the batch.
    '''

    def __init__(self, dataparallel=False):
        self.dp = dataparallel
        self.alpha = torch.tensor([1.])
        self.beta = torch.tensor([1.])

    def _align_device_with(self, align_tensor):
        setattr(self, 'device', align_tensor.device)
        self.alpha = self.alpha.to(self.device)
        self.beta = self.beta.to(self.device)

    def GraphRecon(self, s, data):
        '''
            compute graph reconstruction loss for ONE graph.
            args:
                s: node-community affiliation.
                data: a graph data.
        '''

        def _BerPo(s):
            return 1 - torch.exp(- torch.mm(s, s.t()))

        def _get_pw(data):
            N = data.num_nodes()
            Npos = data.num_edges()
            Nneg = N * (N - 1) - Npos

            # pos_w: augment positive observations (edges) to the same amount of negative observations (non-edges)
            # if Npos = 0 (scatters) or Nneg = 0 (complete graph), set pos_wt = 1,
            # otherwise, it should be (Nneg / Npos).
            pos_w = float(Nneg / Npos) if (Nneg * Npos) > 0 else 1.

            return pos_w

        preds = _BerPo(s)
        pos_w = _get_pw(data)

        # remove self-loops from the edge_index
        data = data.remove_self_loop()
        edge_index = torch.stack(data.edges())

        # create the graph label (a binary adjacency matrix)
        adj_label = torch.sparse_coo_tensor(
            edge_index, torch.ones((data.num_edges(),)),
            torch.Size([data.num_nodes(), data.num_nodes()]), device=self.device
        ).to_dense()

        # compute weighted nll
        pos_labels = pos_w * adj_label
        neg_labels = 1. - adj_label - torch.eye(data.num_nodes(), device=self.device)

        LL = pos_labels * log_max(preds) + neg_labels * log_max(1. - preds)
        return - LL.sum() / (pos_labels + neg_labels).sum()

    def KLD(self, lbd, kappa):
        '''
            compute the KL divergence on one graph.
        '''
        eulergamma = 0.5772
        N = lbd.size(0)

        KL_Part1 = eulergamma * (1 - kappa.pow(-1)) + log_max(lbd / kappa) + 1 + self.alpha * torch.log(self.beta)
        KL_Part2 = - torch.lgamma(self.alpha) + (self.alpha - 1) * (log_max(lbd) - eulergamma * kappa.pow(-1))
        KL_Part3 = - self.beta * lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))

        nKL = KL_Part1 + KL_Part2 + KL_Part3
        return - nKL.mean() / N

    def __call__(self, z, lbd, kappa, data_batch):
        # only calculate masked nodes todo
        # z, lbd, kappa = z[mask_nodes], lbd[mask_nodes], kappa[mask_nodes]
        # data_batch.x = data_batch.x[mask_nodes]

        # "de-batch" the data batch
        if self.dp:
            data_list = data_batch
        else:
            try:
                data_list = data_batch.to_data_list()
            except:
                data_list = [data_batch]
        self._align_device_with(z)

        # z[mask_nodes, :] = 0
        # de-batch z, lbd and kappa
        chunk_sizes = list(map(lambda data: data.num_nodes(), data_list))
        z_list = z.split(chunk_sizes, dim=0)
        lbd_list = lbd.split(chunk_sizes, dim=0)
        kappa_list = kappa.split(chunk_sizes, dim=0)

        # compute graph reconstruction loss for the batch
        loss_gre = [self.GraphRecon(z, data) for (z, data) in zip(z_list, data_list)]  # mask
        loss_gre = torch.stack(loss_gre).mean()

        # compute kl divergence for the batch
        loss_kld = [self.KLD(lbd, kappa) for (lbd, kappa) in zip(lbd_list, kappa_list)]
        loss_kld = torch.stack(loss_kld).mean()

        return loss_kld + loss_gre


class EdgePart(nn.Module):
    def __init__(self, N_coms, dropout=0., **kwargs):
        super(EdgePart, self).__init__()
        # define number of communities
        self.K = N_coms
        self.dropout = dropout
        self.tau = kwargs['tau'] if 'tau' in kwargs else 1.

    def sector(self, embeddings, num_split):
        """ partition the dimension of Phi to communities """
        dim = embeddings.shape[-1]
        # create split sections: [len+1, len+1, ..., len+1, len, len, ..., len]
        rr = dim % num_split
        coms_per_meta = np.ones(num_split) * (dim // num_split)
        coms_per_meta += np.append(np.ones(rr), np.zeros(num_split - rr))
        coms_per_meta = coms_per_meta.astype(np.int64).tolist()
        return embeddings.split(coms_per_meta, dim=-1)

    def epart(self, phi, adj):
        """ generate (sparse, differentiable) community-specific social adjacency matrices """
        indices = adj.indices()

        def _edge_weight(phi, ind):
            row_ind, col_ind = ind
            phi_row, phi_col = phi[row_ind].unsqueeze(1), phi[col_ind].unsqueeze(2)
            return torch.bmm(phi_row, phi_col).flatten()

        def edge_weight(phi, indices):
            """ due to memory constraint, not to compute edge weights all at once """
            max_pairs = 5000
            ind_ = torch.split(indices, max_pairs, dim=1)
            return torch.cat([_edge_weight(phi, ind) for ind in ind_])

        # compute community graph factors
        phi_ = self.sector(phi, self.K)
        eg_counts = torch.stack([edge_weight(phi, indices) for phi in phi_], dim=0)
        eg_weight = F.softmax(eg_counts / self.tau, dim=0)
        eg_weight_ = torch.unbind(eg_weight, dim=0)

        adj_K_ = list(map(
            lambda x: torch.sparse_coo_tensor(adj.indices(), x, adj.shape, requires_grad=True), eg_weight_
        ))
        # row-normalize sparse community factor graph adjacency mats
        tau_row = 0.5
        adj_K_ = [torch.sparse.softmax(adj_k / tau_row, dim=1) for adj_k in adj_K_]

        return adj_K_

    def forward(self, phi, g):
        adj = g.adjacency_matrix()
        phi = F.dropout(phi, self.dropout, self.training)
        return self.epart(phi, adj)

# class EdgePart(nn.Module):
#     def __init__(self, N_coms, dropout, tau=1., max_edges=10000):
#         '''
#             Partition the edges according to node-community affiliation (s).
#             args:
#                 N_coms: number of communities.
#                 dropout: dropout on (s) before computing the partition weights.
#                 tau: softmax temperature along the community dimension.
#                 max_edges: the edge `chunk` size. (Splitting edges into chunks to prevent OOM.)
#         '''
#         super(EdgePart, self).__init__()
#         self.N_coms = N_coms
#         self.dropout = dropout
#         self.tau = tau
#         self.max_edges = max_edges
#         # self.BN = nn.BatchNorm1d(N_coms)
#
#     def chunk_computer(self, z, edge_index):
#         row, col = edge_index
#         z_start, z_end = z[row], z[col]
#         return torch.sum(z_start * z_end, dim=1)
#
#     def edge_weigher(self, z, edge_index):
#         edge_index_chunks = edge_index.split(self.max_edges, dim=-1)
#         return torch.cat([self.chunk_computer(z, indices) for indices in edge_index_chunks])
#
#     def forward(self, z, data):
#         '''
#             Compute the partition weights for all edges.
#             Args:
#                 z: node-community affiliation matrix (denoted as Z in paper).
#                 edge_index: sparse adj edge indices.
#             Output: a list of edge weights, with each element corresponding to weights in one community graph.
#         '''
#         z = F.dropout(z, self.dropout, training=self.training)
#         z_chunks = z.tensor_split(self.N_coms, dim=-1)
#         edge_index = torch.stack(data.edges())
#
#         edge_weight_unnorm = torch.stack([self.edge_weigher(z_k, edge_index) for z_k in z_chunks])# .transpose(1, 0)  # shape = [N_coms, N_edges]
#         # edge_weight_unnorm = self.BN(edge_weight_unnorm).transpose(1, 0)
#         edge_weight_list = torch.unbind(F.softmax(edge_weight_unnorm / self.tau, dim=0), dim=0)
#
#         return edge_weight_list


class NodePart(nn.Module):
    def __init__(self, N_coms, dropout, threshold=1, tau=1., max_nodes=10000):
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
        self.tau = tau
        # self.BN = nn.BatchNorm1d(N_coms)
    #
    # def node_chunk_computer(self, phi):
    #     node_num, N_coms = phi.shape
    #     # phi = F.softmax(phi, dim=0)
    #     m_ik1k2j = torch.einsum('ij,lm->ijml', phi, phi)
    #
    #     m_1 = torch.sum(m_ik1k2j, dim=2)
    #     m_2 = torch.sum(m_ik1k2j, dim=1)
    #
    #     M = torch.zeros(node_num, N_coms)
    #     for i in range(node_num):
    #         for k in range(N_coms):
    #             M[i, k] = torch.sum(m_1[i, k, i+1:]) + torch.sum(m_2[: i, k, i])
    #     return F.softmax(M / self.tau, dim=1)

    def chunk_computer(self, phi):
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
        # node_weight = F.softmax(node_weight/self.tau, dim=1)
        return node_weight
    #
    def node_chunker(self, phi):
        phi_chunks = phi.split(self.max_nodes, dim=0)
        return torch.cat([self.chunk_computer(phi) for phi in phi_chunks])

    def forward(self, x, z):
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
        phi = torch.zeros((node_num, self.N_coms), device=z.device)
        for i, chunk in enumerate(z_chunks):
            phi[:, i] = torch.mean(chunk, dim=1)
        # print(f' first {time.time()-t0: .4f} s.')

        node_weight = self.node_chunker(phi)
        # node_weight = self.BN(node_weight)

        max_value = torch.max(node_weight, dim=1).values.unsqueeze(1).repeat(1, self.N_coms)
        node_mask_1 = torch.eq(node_weight, max_value)
        node_mask_2 = torch.where(node_weight>=self.thershold, True, False)
        node_mask = node_mask_1 # torch.logical_or(node_mask_1, node_mask_2)
        x_commun_list = []
        indices_list = []
        # t0 = time.time()
        for i in range(self.N_coms):
            mask = node_mask[:, i]
            indices = torch.where(mask==True)[0]

            x_part = torch.zeros_like(x)
            x_part[indices, :] = x[indices, :]

            x_commun_list.append(x_part)
            indices_list.append(indices)
        # print(f'mask_list  {time.time() - t0: .4f} s.')
        return x_commun_list, indices_list

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def f_1(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    return f1_score(y_true.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='micro')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument('--N_coms', type=int, default=4)

    parser.add_argument("--loss_z", type=float, default=1)
    parser.add_argument("--use_mlp", type=bool, default=False)
    parser.add_argument("--combine_type", type=str, default="concat")
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true", default=True)
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)


    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
