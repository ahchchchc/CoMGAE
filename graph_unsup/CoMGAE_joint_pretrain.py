import copy
import math
import os
import time

from easydict import EasyDict as edict
import numpy as np
from glob import glob
from tqdm import tqdm
import torch.nn as nn

# pytorch
import torch
from torch.optim import Adam
import torch.nn.functional as F

# pyg
from torch_geometric.datasets import TUDataset, Reddit, PPI, WikiCS
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric import seed_everything
from torch_geometric.nn import DataParallel
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, add_self_loops

# vepm
from src.nn import InfNet, ReconNet
from src.losses import UnsupGAE
from src.config import config, load_best_configs
from src.utils import degree_as_feature_dataset

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def pretrain_full_batch(dataset, models, optim, loss_inf, mask_rate, device=None):
    models.infnet.train()
    models.reconet.train()
    epoch_losses = []

    data = dataset.data
    if not isinstance(data, list):
        data = data.to(device)

    # process node features
    if not isinstance(data, list):
        if data.x is None:
            # data_batch = degree_as_feature(data_batch, device)
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=device)
    else:
        for _data in data:
            if _data.x is None:
                _data.x = torch.ones((_data.num_nodes, 1), dtype=torch.float32, device=_data.edge_index.device)

    s, lbd, kappa = models.infnet(data, mask_rate)
    _loss_inf = loss_inf(s, lbd, kappa, data)
    mask_nodes = models.infnet.mask_nodes

    data.x = torch.cat([data.x, F.softmax(s, dim=-1)], dim=-1)
    _loss_recon = models.reconet(s, data, mask_nodes)

    loss = _loss_inf + _loss_recon
    optim.zero_grad()
    loss.backward()
    optim.step()

    epoch_losses.append(loss.detach().cpu().numpy())

    return np.mean(epoch_losses)

def pretrain(loader, models, optim, loss_inf, mask_rate, cur_ep, pt_ep, device=None):
    models.infnet.train()
    models.reconet.train()
    epoch_losses = []

    # mask_rate = np.min([math.pow(cur_ep / pt_ep, 2) * 1, 0.5])

    for data_batch in loader:
        # mask_batch, (mask_nodes, _) = encoding_mask_noise(data_batch, mask_rate)
        if not isinstance(data_batch, list):
            data_batch = data_batch.to(device)

        # process node features
        if not isinstance(data_batch, list):
            if data_batch.x is None:
                data_batch.x = torch.ones((data_batch.num_nodes, 1), dtype=torch.float32, device=device)
        else:
            for data in data_batch:
                if data.x is None:
                    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=data.edge_index.device)
        # data_batch.edge_index = add_self_loops(remove_self_loops(data_batch.edge_index)[0])[0]

        s, lbd, kappa, mask_nodes = models.infnet(data_batch, mask_rate=0.9)
        _loss_inf = loss_inf(s, lbd, kappa, data_batch, mask_nodes)

        # mask_nodes = models.infnet.mask_nodes
        # o_x = data_batch.x
        data_batch.x = torch.cat([data_batch.x, F.softmax(s, dim=-1)], dim=-1)
        _loss_recon = models.reconet(s, data_batch, mask_rate)

        if cur_ep <= int(pt_ep * 0.5):
            loss = _loss_inf + _loss_recon
        else:
            loss = _loss_inf + _loss_recon
        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_losses.append(loss.detach().cpu().numpy())

    return np.mean(epoch_losses)


def main(args):

    # set random seed
    seed_everything(args.seed)

    # load data, intialize dataloader
    DL = DataListLoader if args.parallel_ptInfNet else DataLoader
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        # transform = T.Compose([
        #     # T.RandomNodeSplit(num_val=500, num_test=500),
        #     T.TargetIndegree(),
        # ])
        path = os.path.join('./data/datasets/GraphDatasets', args.dataset)
        dataset = Planetoid(path, args.dataset)
        dataloader = DL(dataset, batch_size=len(dataset), shuffle=True) # full batch
        # graph, (num_features, num_classes) = load_small_dataset(args.dataset)
    elif args.dataset in ['Reddit']:
        dataset = Reddit(root=args.datapath)
        dataloader = DL(dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset in ['PPI']:
        dataset = PPI(root=args.datapath, split='train')
        dataloader = DL(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    elif args.dataset in ['WikiCS']:
        dataset = WikiCS(root=args.datapath, is_undirected=True)    # use undirected graph
        dataloader = DL(dataset, batch_size=len(dataset), shuffle=True)
    # elif args.dataset.startswith("ogbn"):
    #     dataset = PygNodePropPredDataset(name=args.dataset, root=args.datapath).shuffle()
        # trainset, _, _ = split_dataset(dataset)
        # dataloader = DL(dataset, batch_size=len(dataset), shuffle=True)
    else:
        dataset = TUDataset(root=args.datapath, name=args.dataset)
        dataloader = DL(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # degree as feature
    feature_dim = dataset.num_features
    num_classes = dataset.num_classes
    if 'x' not in dataset[0]:
        dataset, feature_dim = degree_as_feature_dataset(dataset)
        # # data = degree_as_feature(dataset.data)
        # # dataset.data = data
        dataloader = DL(dataset, batch_size=args.batch_size, shuffle=True)

    # initialize the model, optimizer
    kwargs = edict()
    kwargs.models = edict()

    models = edict()

    print(f'dataset {args.dataset}')
    print(f'feature_dim: {feature_dim}')
    print(f'number of class {num_classes: d}')

    # the infnet
    inf_hid_dim = args.InfNet_hid_dims # [64, 32]
    kwargs.models.infnet = {
        'in_dim':  int(max(feature_dim, 1)),
        'hid_dims': inf_hid_dim,
        'dropout': args.InfNet_dropout,
        'random_encoder': True
    }
    models.infnet = InfNet(**kwargs.models.infnet).to(args.device)
    loss_inf = UnsupGAE(dataparallel=args.parallel_ptInfNet)

    # the reconet
    in_dim = int(max(feature_dim, 1) + inf_hid_dim[-1])
    kwargs.models.reconet = {
        'N_coms': args.N_coms,
        'in_dim': in_dim,
        'dec_dim': in_dim,
        'emb_dim': args.PredNet_edim,
        'N_classes': num_classes,
        'x_loss_lamb': args.pt_x_lambda,
        'random_loss_lamb': args.pt_random_lambda,
        'requires_chunk': args.requires_chunk,
        'ep_tau': args.EdgePart_tau,
        'N_layers_h': args.PredNet_Nlayers_CGBank,
        'N_layers_t': args.PredNet_Nlayers_REComp,
        'type': args.gnn_type,
        'n_heads': args.n_heads,
        'get_emb_type': args.get_emb_type,
        'recomp_type': args.recomp_type
    }
    models.reconet = ReconNet(**kwargs.models.reconet).to(args.device)
    # optims.reconet = Adam(models.reconet.parameters(), **kwargs.optims.reconet)

    optim = Adam([{'params': models.infnet.parameters()},
                  {'params': models.reconet.parameters()},],
                                      lr=args.JointNet_lr, weight_decay=args.JointNet_l2)

    # load the latest pretrained model
    curr_epoch = 0
    model_infer = glob(os.path.join(args.jointpretrainmodelpath, args.dataset, f'mask_rate{args.pt_mask_rate}', f'seed{args.seed}','InfNet',
                               f'InfNet_joint_*_{args.dataset}.pt'))
    model_recon = glob(os.path.join(args.jointpretrainmodelpath, args.dataset, f'mask_rate{args.pt_mask_rate}', f'seed{args.seed}','ReconNet',
                               f'ReconNet_joint_*_{args.dataset}.pt'))
    get_epoch = lambda f: int(f.split('ep')[0].split('_')[-1])
    if model_infer and model_recon:
        pt_epochs_inf = list(map(get_epoch, model_infer))
        pt_epochs_rec = list(map(get_epoch, model_recon))
        if pt_epochs_inf == pt_epochs_rec:
            curr_epoch = np.max(pt_epochs_rec)

            saved_path_inf = os.path.join(args.jointpretrainmodelpath, args.dataset, f'mask_rate{args.pt_mask_rate}', f'seed{args.seed}', 'InfNet',
                                      f'InfNet_joint_{curr_epoch}ep_{args.dataset}.pt')
            models.infnet.load_state_dict(torch.load(saved_path_inf, map_location=args.device))
            saved_path_rec = os.path.join(args.jointpretrainmodelpath, args.dataset, f'mask_rate{args.pt_mask_rate}',
                                          f'seed{args.seed}', 'ReconNet',
                                          f'ReconNet_joint_{curr_epoch}ep_{args.dataset}.pt')
            models.reconet.load_state_dict(torch.load(saved_path_rec, map_location=args.device))

            print('==========')
            print(f'load InfNet from {saved_path_inf}')
            print(f'load ReconNet from {saved_path_rec}')
            print('==========')


    if args.parallel_ptInfNet:
        device_ids = (np.arange(torch.cuda.device_count()) + args.gpu_id) % torch.cuda.device_count()
        models.infnet = DataParallel(models.infnet, device_ids=device_ids.tolist(), output_device=args.device)
        models.reconet = DataParallel(models.reconet, device_ids=device_ids.tolist(), output_device=args.device)

    # lr_f, wd_f, max_ep_f = args.lp_lr, args.lp_weight_decay, args.lp_max_epoch
    # best_test_acc, best_estp_acc = 0, 0
    # start pretraining!
    print(f'Pretraining Seed {args.seed} for mask rate {args.pt_mask_rate}')
    with tqdm(total=(args.pt_epochs - curr_epoch), desc='(T)') as pbar:
        for ep in range(curr_epoch + 1, args.pt_epochs + 1):
            ls = pretrain(dataloader, models, optim, loss_inf, args.pt_mask_rate, ep, args.pt_epochs, device=args.device)
            pbar.set_postfix_str(f'loss: {ls:.4f}')
            pbar.update()

            if ep % args.save_iter == 0:
                # from VEPM_linear_probing_single_seed import Trainer
                # trainer = Trainer(args.device)
                # final_acc, (estp_acc, estp_epoch) = trainer.linear_probing_node_classification_unshuffle(dataloader, models, lr_f, wd_f, max_ep_f)
                # if best_test_acc < final_acc:
                #     best_test_acc = final_acc
                # if best_estp_acc < estp_acc:
                #     best_estp_acc = estp_acc
                #
                # print(
                #     f"Pretrained epoch {ep} Linear Probing result: final test acc {final_acc * 100:.4f}\t estp acc {estp_acc * 100:.4f}\tat epoch {estp_epoch:d}")

                saved_path_infer = os.path.join(args.jointpretrainmodelpath,
                                                args.dataset, f'mask_rate{args.pt_mask_rate}', f'seed{args.seed}', f'lr {args.JointNet_lr}','InfNet')
                saved_path_recon = os.path.join(args.jointpretrainmodelpath,
                                                args.dataset, f'mask_rate{args.pt_mask_rate}', f'seed{args.seed}', f'lr {args.JointNet_lr}','ReconNet')
                if not os.path.exists(saved_path_infer):
                    os.makedirs(saved_path_infer)
                if not os.path.exists(saved_path_recon):
                    os.makedirs(saved_path_recon)
                save_name_infer = os.path.join(saved_path_infer, f'InfNet_joint_{ep}ep_{args.dataset}.pt')
                save_name_recon = os.path.join(saved_path_recon, f'ReconNet_joint_{ep}ep_{args.dataset}.pt')

                if args.parallel_ptInfNet:
                    torch.save(models.infnet.module.state_dict(), save_name_infer)
                    torch.save(models.reconet.module.state_dict(), save_name_recon)
                else:
                    torch.save(models.infnet.state_dict(), save_name_infer)
                    torch.save(models.reconet.state_dict(), save_name_recon)

        # print('Best test accurarcy {:.4f}\t Best estp acc {:.4f}'.format(best_test_acc * 100, best_estp_acc * 100))
if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    args = config()
    if args.best_cfg:
        args = load_best_configs(args, "best_configs.yml")
    main(args)