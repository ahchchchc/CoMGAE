import os
from easydict import EasyDict as edict
import numpy as np
import torch.nn as nn
from glob import glob
from tqdm import tqdm
import time

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

# pytorch
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# pyg
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.utils import degree
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# vepm
from src.nn import InfNet, ReconNet
from src.losses import UnsupGAE
from src.config import config, load_best_configs
from src.utils import degree_as_feature_dataset

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Trainer(object):
    def __init__(self, device):
        self.device = device
        self.unsuper_loss = UnsupGAE()

    def encoding_mask_noise(self, g, x, mask_rate=0.8):
        num_nodes = len(g.batch)    # g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        enc_dim = x.shape[-1]

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        enc_mask_token = nn.Parameter(torch.zeros(1, enc_dim)).to(out_x.device)
        out_x[token_nodes] += enc_mask_token

        g.x[mask_nodes] = 0.0
        use_g = g

        if mask_rate == 0:
            mask_nodes = None

        return use_g, out_x, (mask_nodes, keep_nodes)

    def finetune_predaftnet(self, loader, models, optims, mask_rate):
        '''
            Keep the infnet fixed, only train the prednet.
        '''
        models.infnet.eval()
        models.reconet.eval()
        models.prednet.train()

        epoch_losses = []

        for data in loader:
            data = data.to(self.device)

            # process node features
            if data.x is None:
                # data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=self.device)
                row, col = data.edge_index
                deg = degree(col, data.num_nodes, dtype=torch.float32)
                data.x = deg.view(data.num_nodes, 1).to(self.device)
            with torch.no_grad():
                z, _, _ = models.infnet(data)
            data.x = torch.cat([data.x, F.softmax(z, dim=-1)], dim=-1)

            mask_data, mask_z, (mask_nodes, keep_nodes) = self.encoding_mask_noise(data, z, mask_rate)
            with torch.no_grad():
                hs = models.reconet.get_emb(mask_z, data, mask_nodes=mask_nodes)
            y_hat = models.prednet(hs, data)
            loss = F.nll_loss(y_hat.to(torch.float32), data.y)

            # # compute nll loss (classification task)
            # yhat = models.predaftnet(mask_z, mask_data, mask_nodes)
            # loss = F.nll_loss(yhat.to(torch.float32), data.y)

            # gradient descent
            optims.prednet.zero_grad()
            loss.backward()
            optims.prednet.step()

            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)


    def finetune_infnet(self, loader, models, optims):
        '''
            Keep the prednet fixed, only train the infnet.
        '''
        models.infnet.train()
        models.prednet.eval()

        epoch_losses = []
        
        for data in loader:
            data = data.to(self.device)

            # process node features
            if data.x is None:
                # data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=self.device)
                row, col = data.edge_index
                deg = degree(col, data.num_nodes, dtype=torch.float32)
                data.x = deg.view(data.num_nodes, 1).to(self.device)
            z, lbd, kappa = models.infnet(data)
            data.x = torch.cat([data.x, F.softmax(z, dim=-1)], dim=-1)

            # compute nll loss (classification task)
            yhat = models.prednet(z, data)
            loss_super = 0.1 * F.nll_loss(yhat.to(torch.float32), data.y)

            # compute unsuper loss
            loss_unsup = self.unsuper_loss(z, lbd, kappa, data)

            loss = 0.1 * (loss_super + loss_unsup)

            # gradient descent
            optims.infnet.zero_grad()
            loss.backward()
            optims.infnet.step()

            epoch_losses.append(loss.item())

        return np.mean(epoch_losses)
    

    def finetune_joint(self, loader, models, optims):
        '''
            Joint finetune.
        '''
        models.infnet.train()
        models.prednet.train()

        epoch_losses = []

        for data in loader:
            data = data.to(self.device)

            # process node features
            if data.x is None:
                # data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=self.device)
                row, col = data.edge_index
                deg = degree(col, data.num_nodes, dtype=torch.float32)
                data.x = deg.view(data.num_nodes, 1).to(self.device)
            z, lbd, kappa = models.infnet(data)
            data.x = torch.cat([data.x, F.softmax(z, dim=-1)], dim=-1)

            # compute nll loss (classification task)
            yhat = models.prednet(z, data)
            loss_super = F.nll_loss(yhat.to(torch.float32), data.y)

            # compute unsuper loss
            loss_unsup = self.unsuper_loss(z, lbd, kappa, data)

            loss = loss_super + 0.01 * loss_unsup

            # gradient descent
            optims.infnet.zero_grad(); optims.prednet.zero_grad()
            loss.backward()
            optims.infnet.step(); optims.prednet.step()

            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)


    def evaluate_svm(self, loader, models):
        def evaluate_graph_embeddings_using_svm(embeddings, labels):
            f1_result = []
            acc_result = []
            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

            pbar = tqdm(kf.split(embeddings, labels))
            for train_index, test_index in pbar:
                x_train = embeddings[train_index]
                x_test = embeddings[test_index]
                y_train = labels[train_index]
                y_test = labels[test_index]
                params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
                svc = SVC(random_state=42)
                clf = GridSearchCV(svc, params)
                # t0 = time.time()
                clf.fit(x_train, y_train)
                # t1 = time.time()
                # print(f'SVM fitting time {t1-t0:.4f}s')

                preds = clf.predict(x_test)
                f1 = f1_score(y_test, preds, average="micro") # accuracy_score(y_test, preds)
                acc = accuracy_score(y_test, preds)
                pbar.set_postfix_str(f'f1: {f1 * 100: .4f}\t acc: {acc * 100: .4f}')
                pbar.update()
                f1_result.append(f1)
                acc_result.append(acc)
            test_f1, test_acc = np.mean(f1_result), np.mean(acc_result)
            test_std_f1, test_std_acc = np.std(f1_result), np.std(acc_result)

            return test_f1, test_std_f1, test_acc, test_std_acc

        models.infnet.eval()
        models.reconet.eval()

        x_list = []
        y_list = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                data = data.to(self.device)

                if not isinstance(data, list):
                    data = data.to(self.device)

                    # process node features
                if not isinstance(data, list):
                    if data.x is None:
                        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32, device=self.device)
                else:
                    for item in data:
                        if item.x is None:
                            item.x = torch.ones((item.num_nodes, 1), dtype=torch.float32, device=item.edge_index.device)


                z, _, _, _ = models.infnet(data, mask_rate=0)
                data.x = torch.cat([data.x, F.softmax(z, dim=-1)], dim=-1)
                hs = models.reconet.get_emb(z, data)

                if args.pooltype == 'mean':
                    hgs = [global_mean_pool(h, data.batch) for h in hs]
                elif args.pooltype == 'sum':
                    hgs = [global_add_pool(h, data.batch) for h in hs]
                else:
                    hgs = [global_max_pool(h, data.batch) for h in hs]
                hs = np.array([h.detach().cpu().numpy() for h in hgs])
                h_sum = np.sum(hs, axis=0)
                # hgs = np.array([h.detach().cpu().numpy() for h in hs])
                # h_mean = np.sum(hgs, axis=0)
                y_list.append(data.y.detach().cpu().numpy())
                x_list.append(h_sum)
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        test_f1, test_std_f1, test_acc, test_std_acc = evaluate_graph_embeddings_using_svm(x, y)

        return test_f1, test_std_f1, test_acc, test_std_acc



def main(args):

    # set random seed
    seed_everything(args.seed)

    # initialize logger
    # LOGPATH = os.path.join(args.logpath, args.dataset, f'{args.n2g_coverage}_{args.PredNet_Nlayers_CGBank}_{args.PredNet_Nlayers_REComp}_{args.pooltype}')
    # if args.enable_logger:
    #     logger = Logger(LOGPATH, f'val{args.fold_id}.log')
    # else:
    #     logger = None

    # load dataset, create 9 train folds and 1 val fold
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv']:
        transform = T.Compose([
            T.RandomNodeSplit(num_val=500, num_test=500),
            T.TargetIndegree(),
        ])
        path = os.path.join('./data/datasets/GraphDatasets', args.dataset)
        dataset = Planetoid(path, args.dataset, transform=transform)
        # graph, (num_features, num_classes) = load_small_dataset(args.dataset)
    else:
        dataset = TUDataset(root=args.datapath, name=args.dataset)

    # indices = np.random.RandomState(seed=args.seed).permutation(len(dataset))
    # tenfold = np.array_split(indices, 10)
    # val_indices = tenfold.pop(args.fold_id)
    # trn_indices = np.concatenate(tenfold, axis=0)
    #
    # loader_trn = DataLoader(dataset[trn_indices], batch_size=args.batch_size)
    # loader_val = DataLoader(dataset[val_indices], batch_size=args.batch_size)

    # # record the settings
    # if logger is not None:
    #     logger.record_args(args)

    feature_dim = dataset.num_features
    num_classes = dataset.num_classes
    if dataset.data.x is None:
        dataset, feature_dim = degree_as_feature_dataset(dataset)
        # data = degree_as_feature(dataset.data)
        # dataset.data = data

    # initialize the model, optimizer
    kwargs = edict()
    kwargs.models = edict()
    kwargs.optims = edict()

    models = edict()
    optims = edict()

    print(feature_dim)

    ## the infnet
    inf_hid_dim = args.InfNet_hid_dims
    kwargs.models.infnet = {
        'in_dim': max(int(feature_dim), 1),
        'hid_dims': inf_hid_dim,
        'dropout': args.InfNet_dropout
    }
    models.infnet = InfNet(**kwargs.models.infnet).to(args.device)


    # RecoNet
    in_dim = max(int(feature_dim), 1) + inf_hid_dim[-1]
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

    # load pretrained infnet
    # load_path = os.path.join(args.modelpath, args.dataset, f'mask_rate{args.pt_mask_rate}', f'seed{args.seed}', f'InfNet_{args.load_epoch}ep_{args.dataset}.pt')
    # models.infnet.load_state_dict(torch.load(load_path, map_location=args.device))
    # load_path = os.path.join(args.maskpretrain_modelpath, args.dataset, f'mask_rate{args.pt_mask_rate}', f'seed{args.seed}', f'ReconNet_250ep_{args.dataset}.pt')
    # models.reconet.load_state_dict(torch.load(load_path, map_location=args.device))

    load_path = os.path.join(args.jointpretrainmodelpath, f'{args.dataset}', f'mask_rate{args.pt_mask_rate}',
                             f'seed{args.seed}', f'lr {args.JointNet_lr}') # args.jointpretrainmodelpath
    print('==========')
    print(f'load model {load_path}')

    # train the model
    trainer = Trainer(args.device)
    best_eval_f1 = 0.
    best_epoch = 0

    loader_val = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)    # use all samples to evaluate when unsupervising
    print('==========')
    print(f'finetune mode: SVM\t pooling type:{args.pooltype}')
    print(f'Seed {args.seed} mask rate {args.pt_mask_rate} for dataset {args.dataset}')
    for i in range(args.ft_start_cnt, args.ft_end_cnt):
        ep = i * args.load_iter # * 25
        load_fp = os.path.join(load_path, 'InfNet', f'InfNet_joint_{ep}ep_{args.dataset}.pt')
        models.infnet.load_state_dict(torch.load(load_fp, map_location=args.device))
        load_fp = os.path.join(load_path, 'ReconNet', f'ReconNet_joint_{ep}ep_{args.dataset}.pt')
        models.reconet.load_state_dict(torch.load(load_fp, map_location=args.device))

        eval_f1, eval_std_f1, eval_acc, eval_std_acc = trainer.evaluate_svm(loader_val, models)
        if best_eval_f1 < eval_f1:
            best_eval_f1 = eval_f1
            best_epoch = ep

        print(f"Epoch {ep}\t#Test_f1: {eval_f1 * 100:.4f}±{eval_std_f1 * 100:.4f}")
        print(f"Epoch {ep}\t#Test_acc: {eval_acc * 100:.4f}±{eval_std_acc * 100:.4f}")
    print(f'N_coms: {args.N_coms:d}\tpooling type:{args.pooltype}\t Seed {args.seed} mask rate {args.pt_mask_rate} for dataset {args.dataset}')
    print('Best Evaluation accurarcy {:.4f} at Epoch {:d}'.format(best_eval_f1 * 100, best_epoch))

if __name__ == '__main__':
    args = config()
    if args.best_cfg:
        args = load_best_configs(args, "best_configs.yml")
    args.device = 'cpu'
    main(args)


