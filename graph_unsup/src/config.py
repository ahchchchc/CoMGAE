import argparse
import torch
import os
import yaml

def config():
    parser = argparse.ArgumentParser(description='VEPM hyperparameters.')

    # global parameters
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--fold-id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='IMDB-BINARY',
        help='dataset name (default: IMDB-BINARY)')
    parser.add_argument('--pt-epochs', type=int, default=1500)
    parser.add_argument('--pt-mask-rate', type=float, default=0.25)
    parser.add_argument('--pt-x-lambda', type=float, default=1)
    parser.add_argument('--pt-random-lambda', type=float, default=1)
    parser.add_argument('--requires-chunk', type=bool, default=True)
    parser.add_argument('--best-cfg', type=bool, default=True)

    # GNN parameters
    parser.add_argument('--gnn-type', type=str, default='gin')
    parser.add_argument('--n-heads', type=int, default=4)

    # batch (pre)training
    parser.add_argument('--parallel-ptInfNet', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--save-iter', type=int, default=100)
    parser.add_argument('--recomp-type', type=str, default='gnn')

    # finetune parameter
    parser.add_argument("--get-emb-type", type=str, default='all')
    parser.add_argument('--ft-start-cnt', type=int, default=15)
    parser.add_argument('--ft-end-cnt', type=int, default=16)
    parser.add_argument('--load-iter', type=int, default=100)
    parser.add_argument('--lp-lr', type=float, default=0.01)
    parser.add_argument('--lp-weight-decay', type=float, default=1e-4)
    parser.add_argument('--lp-max-epoch', type=int, default=300)
    
    # number of communities
    parser.add_argument('--N-coms', type=int, default=2, help='2, 4')

    # InfNet parameters
    parser.add_argument('--InfNet-dropout', type=float, default=0.2)
    parser.add_argument('--InfNet-hid-dims', type=int, nargs='+', default=[256, 128])

    # JointNet parameters
    parser.add_argument('--JointNet-lr', type=float, default=0.001)
    parser.add_argument('--JointNet-l2', type=float, default=0.)
    parser.add_argument('--JointNet-dropout', type=float, default=0.2)
    
    # PredNet parameters
    parser.add_argument('--PredNet-Nlayers-CGBank', type=int, default=2)
    parser.add_argument('--PredNet-Nlayers-REComp', type=int, default=1)
    parser.add_argument('--PredNet-edim', type=int, default=512)
    parser.add_argument('--EdgePart-dropout', type=float, default=0.)
    parser.add_argument('--pooltype', type=str, default='mean',
        help='`mean` or `sum` or `max`')
    parser.add_argument('--EdgePart-tau', type=float, default=1)

    # some paths
    parser.add_argument('--datapath', type=str, default='./data/datasets/GraphDatasets/TU-pyg',
        help='/work/06083/ylhe/Data/TU-pyg, /data/datasets/GraphDatasets/TU-pyg')
    parser.add_argument('--jointpretrainmodelpath', type=str, default='./joint_pretrain_model')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda:' + str(args.gpu_id) if args.cuda else 'cpu'

    if not os.path.exists(args.datapath):
        os.makedirs(args.datapath)

    return args


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        print("Best args not found")
        return args

    print("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        if 'InfNet_hid_dims' in k:
            try:
                v = [int(i) for i in v.split()]
            except:
                v = [int(v)]
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

    
    