import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree as tg_degree
from torch_geometric.utils import remove_self_loops, add_self_loops
from glob import glob
import os
import numpy as np
import copy

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from collections import Counter


import logging
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader
SMALL = 1e-10
NINF = -1e10

def split_dataset(dataset):
    assert dataset.dir_name.startswith("ogbn")

    num_nodes = dataset.data.num_nodes
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes, ), False).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes, ), False).index_fill_(0, test_idx, True)

    data = copy.deepcopy(dataset.data)
    data['train_mask'] = train_mask
    data['val_mask'] = val_mask
    data['test_mask'] = test_mask

    return [data]
    # _trainset = copy.deepcopy(dataset.data)
    # _trainset.num_nodes = len(train_idx)
    # _trainset.edge_index = _trainset.edge_index[:, train_idx]
    # _trainset.node_year = _trainset.node_year[train_idx, :]
    # _trainset.y = _trainset.y[train_idx, :]
    # _trainset.x = _trainset.x[train_idx, :]
    # trainset = [_trainset]
    #
    # _valset = copy.deepcopy(dataset.data)
    # _valset.num_nodes = len(val_idx)
    # _valset.edge_index = _valset.edge_index[:, val_idx]
    # _valset.node_year = _valset.node_year[val_idx, :]
    # _valset.y = _valset.y[val_idx, :]
    # _valset.x = _valset.x[val_idx, :]
    # valset = [_valset]
    #
    # _testset = copy.deepcopy(dataset.data)
    # _testset.num_nodes = len(test_idx)
    # _testset.edge_index = _testset.edge_index[:, test_idx]
    # _testset.node_year = _testset.node_year[test_idx, :]
    # _testset.y = _testset.y[test_idx, :]
    # _testset.x = _testset.x[test_idx, :]
    # testset = [_testset]





def degree_as_feature_dataset(dataset):
    print("Using degree as node features")
    feature_dim = 0
    degrees = []
    # for g, _ in data_batch:
    #     feature_dim = max(feature_dim, g.in_degrees().max().item())
    #     degrees.extend(g.in_degrees().tolist())

    for g in dataset:
        feature_dim = max(feature_dim, tg_degree(g.edge_index[0]).max().item())
        degrees.extend(tg_degree(g.edge_index[0]).tolist())

    MAX_DEGREES = 400

    oversize = 0
    for d, n in Counter(degrees).items():
        if d > MAX_DEGREES:
            oversize += n
    # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
    # feature_dim = feature_dim * 2
    feature_dim = min(feature_dim, MAX_DEGREES)

    feature_dim += 1
    # for g, l in data_batch:
    #     degrees = g.in_degrees()
    #     degrees[degrees > MAX_DEGREES] = MAX_DEGREES
    #
    #     feat = F.one_hot(degrees, num_classes=feature_dim).float()
    #     g.ndata["attr"] = feat

    # x = []
    new_dataset = []
    for i, g in enumerate(dataset):
        degrees = tg_degree(g.edge_index[0]).long()
        degrees[degrees > MAX_DEGREES] = MAX_DEGREES

        feat = F.one_hot(degrees, num_classes=int(feature_dim)).float()
        g['x'] = feat
        g['edge_index'] = add_self_loops(remove_self_loops(g.edge_index)[0])[0]
        new_dataset.append(g)
    # x = torch.concat(x, dim=0)
    # dataset.data.x = x
    # dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y, _ in dataset.data]
    # dataset.num_features = feature_dim
    return new_dataset, feature_dim

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
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def degree_as_feature(data):
    print("Using degree as node features")
    feature_dim = 0
    degrees = []
    # for g, _ in data_batch:
    #     feature_dim = max(feature_dim, g.in_degrees().max().item())
    #     degrees.extend(g.in_degrees().tolist())
    feature_dim = max(feature_dim, tg_degree(data.edge_index[0]).max().item())
    degrees.extend(tg_degree(data.edge_index[0]).tolist())
    # for data_batch in data:
    #     feature_dim = max(feature_dim, tg_degree(data_batch.edge_index[0]).max().item())
    #     degrees.extend(tg_degree(data_batch.edge_index[0]).tolist())

    MAX_DEGREES = 400

    oversize = 0
    for d, n in Counter(degrees).items():
        if d > MAX_DEGREES:
            oversize += n
    # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
    feature_dim = min(feature_dim, MAX_DEGREES)

    feature_dim += 1
    # for g, l in data_batch:
    #     degrees = g.in_degrees()
    #     degrees[degrees > MAX_DEGREES] = MAX_DEGREES
    #
    #     feat = F.one_hot(degrees, num_classes=feature_dim).float()
    #     g.ndata["attr"] = feat

    degrees = tg_degree(data.edge_index[0]).long()
    degrees[degrees > MAX_DEGREES] = MAX_DEGREES

    feat = F.one_hot(degrees, num_classes=int(feature_dim)).float()
    data['x'] = feat
    # dataset.num_features = feature_dim
    return data

def load_small_dataset(dataset_name):
    GRAPH_DICT = {
        "cora": CoraGraphDataset,
        "citeseer": CiteseerGraphDataset,
        "pubmed": PubmedGraphDataset,
        "ogbn-arxiv": DglNodePropPredDataset,
    }
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)

def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


def scale_feats(x):
    logging.info("### scaling features ###")
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def log_max(input, SMALL=SMALL):
    device = input.device
    input_ = torch.max(input, torch.tensor([SMALL]).to(device))
    return torch.log(input_)


def drop_softmax(input, p, dim):
    '''
        first performing dropout on the input, then apply softmax along `dimension`.
    '''
    _device = input.device
    if p < 0 or p >= 1:
        raise ValueError(f'domain `p` out of range: expects to be within [0,1), receives {p}')
    else:
        s = 1. / (1. - p)
    mask_ = torch.from_numpy(np.random.binomial(1, (1. - p), size=input.shape).astype(np.float32)).to(_device)
    # mask_ = torch.from_numpy(np.random.randint(2, size=input.shape).astype(np.float32)).to(_device)
    input = (1. - mask_) * s * input + NINF * mask_
    return F.softmax(input, dim=dim)
            
            
class Logger(object):
    def __init__(self, logdir, logfile):
        super(Logger, self).__init__()
        self.logdir = logdir
        self.logfile = logfile
        if not os.path.exists(logdir):
            os.makedirs(logdir)  
        self.logpath = os.path.join(self.logdir, self.logfile)
    
    def record(self, msg):
        msg = msg + '\n'
        with open(self.logpath, 'a') as f:
            f.write(msg)
        print(msg)
    
    def record_args(self, args):
        for attr, value in sorted(vars(args).items()):
            self.record(f'{attr.upper()}: {value}\n')


