from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from comgae.utils import create_norm, drop_edge, EdgePart, NodePart, UnsupGAE, log_max


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            random_encoder: bool = True,
            combine_type: str = 'concat',
            N_coms = 8,
            use_mlp = False,
            loss_z = 1.
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        self._random_encoder = random_encoder
        self._combine_type = combine_type
        self._use_mlp = use_mlp
        self._loss_z = loss_z

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0

        inf_num_hidden = num_hidden + 1
        infNet_type = 'gcn'

        enc_in_dim = in_dim #+ inf_num_hidden - 1
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        enc_com_num_hidden = enc_num_hidden // N_coms # cora
        # enc_com_num_hidden = enc_num_hidden


        if self._combine_type == 'concat':
            dec_in_dim = num_hidden * 2 # cora
        else:
            dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        self.EdgePart = EdgePart(N_coms=N_coms, dropout=feat_drop, tau=1)
        self.NodePart = NodePart(N_coms=N_coms, dropout=feat_drop, tau=100)
        self.loss_inf = UnsupGAE()

        if self._use_mlp:
            self.drop_mlp = nn.Dropout(0.5)
            self.enc_mlp = nn.Sequential(
                nn.Linear(enc_in_dim, num_hidden),
                nn.GELU()
            )
            self.dec_mlp = nn.Linear(num_hidden, enc_in_dim)

        self.z_inferNet = setup_module(
            m_type=infNet_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=256, #inf_num_hidden, # 256 16 for cora / 512 16 for citeseer and cora
            out_dim=16 + 1, #inf_num_hidden,
            num_layers=2, #num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation='softplus',
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build encoder
        self.encoder_com = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=enc_in_dim,
            num_hidden=enc_com_num_hidden,
            out_dim=enc_com_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # self.Recomposer = setup_module(
        #     m_type=encoder_type,
        #     enc_dec="encoding",
        #     in_dim=num_hidden,
        #     num_hidden=enc_num_hidden,
        #     out_dim=enc_num_hidden,
        #     num_layers=1,
        #     nhead=enc_nhead,
        #     nhead_out=enc_nhead,
        #     concat_out=True,
        #     activation=activation,
        #     dropout=feat_drop,
        #     attn_drop=attn_drop,
        #     negative_slope=negative_slope,
        #     residual=residual,
        #     norm=norm,
        # )

        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=enc_in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=enc_in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token_z = nn.Parameter(torch.zeros(1, in_dim))
        self.enc_mask_token = nn.Parameter(torch.zeros(1, enc_in_dim))

        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3, indices=None, mask_nodes=None, encoding_z=False):
        num_nodes = g.num_nodes()
        # perm = torch.randperm(num_nodes, device=x.device)
        # num_mask_nodes = int(mask_rate * num_nodes)

        if indices is not None:
            num_nodes = len(indices)
            perm = torch.randperm(num_nodes, device=x.device)
            indices_shuffle = indices[perm]
        else:
            indices_shuffle = torch.randperm(num_nodes, device=x.device)

        if mask_nodes is not None:
            num_mask_nodes = mask_nodes.shape[0]
            keep_nodes = indices_shuffle[~torch.isin(indices_shuffle, mask_nodes.to(x.device))]
        else:
            # random masking
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = indices_shuffle[: num_mask_nodes]
            keep_nodes = indices_shuffle[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = max(int(self._replace_rate * num_mask_nodes), 1)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(g.num_nodes(), device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        if encoding_z:
            out_x[token_nodes] += self.enc_mask_token_z
        else:
            out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x):
        z, lbd, kappa = self.get_z(g, x, self._mask_rate)
        # loss for z recon todo
        loss_inf = self.loss_inf(z, lbd, kappa, g)
        # ---- attribute reconstruction ----

        loss = self._loss_z * loss_inf + self.mask_attr_prediction(g, x, z) #  3 for citeseer, 1 for cora
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def get_z(self, g, x, mask_rate=0.):

        pre_use_g, use_x, (_, _) = self.encoding_mask_noise(g, x, mask_rate, encoding_z=True)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        h = self.z_inferNet(use_g, use_x)
        lbd, kappa = h.split([h.size(1) - 1, 1], dim=1)
        z = self.reparameterize(lbd, kappa + 0.1)
        return z, lbd, kappa + 0.1

    def mask_attr_prediction(self, g, x, z):
        adj_list = self.EdgePart(z, g)
        x_commun_list, commun_indices_list = self.NodePart(x, z)


        mask_x_list, mask_node_list = [], []
        for x_com, indices in zip(x_commun_list, commun_indices_list):
            _x = torch.clone(x_com)
            _, mask_part_x, (mask_part_node, _) = self.encoding_mask_noise(g, x_com, self._mask_rate, indices=indices)
            mask_x_list.append(mask_part_x)
            mask_node_list.append(mask_part_node)

        mask_nodes = torch.unique(torch.cat(mask_node_list))
        pre_use_g, use_x, (_, _) = self.encoding_mask_noise(g, x, mask_rate=self._mask_rate, mask_nodes=mask_nodes)
        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        com_rep = []
        for i, x_com in enumerate(mask_x_list):
            com_enc_rep, all_hidden = self.encoder_com(use_g, x_com, adj_list[i], return_hidden=True)
            if self._concat_hidden:
                com_enc_rep = torch.cat(all_hidden, dim=1)
            # com_enc_rep = torch.sparse.mm(adj_list[i], com_enc_rep)
            com_rep.append(com_enc_rep)#F.normalize(com_enc_rep, p=2, dim=1))
        com_rep = torch.cat(com_rep, dim=-1) # cora

        global_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            global_rep = torch.cat(all_hidden, dim=1)

        if self._combine_type == 'concat':
            enc_rep = torch.cat([com_rep, global_rep], dim=-1) # cora
        else:
            enc_rep = com_rep + global_rep

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        if self._use_mlp:
            x_mlp = self.dec_mlp(self.enc_mlp(self.drop_mlp(x)))
            loss = loss + 2. * self.criterion(x_mlp, x)
        return loss

    def embed(self, g, x):
        z, lbd, kappa = self.get_z(g, x) #self._mask_rate)

        adj_list = self.EdgePart(z, g)
        x_commun_list, _ = self.NodePart(x, z)
        com_rep = []
        for i, x_com in enumerate(x_commun_list):
            com_enc_rep, all_hidden = self.encoder_com(g, x_com, adj_list[i], return_hidden=True)
            # com_enc_rep = torch.sparse.mm(adj_list[i], com_enc_rep)
            com_rep.append(com_enc_rep)#F.normalize(com_enc_rep, p=2, dim=1))
        com_rep = torch.cat(com_rep, dim=-1)
        # com_rep = self.Recomposer(g, com_rep)
        # com_rep = torch.sum(torch.stack(com_rep, dim=0), dim=0)
        global_rep = self.encoder(g, x)

        rep = torch.cat([com_rep, global_rep], dim=-1)# cora citeseer
        # rep = global_rep + com_rep

        if self._use_mlp:
            mlp_rep = self.enc_mlp(x)
            # rep = rep + mlp_rep
            rep = torch.cat([rep, mlp_rep], dim=-1)

        return rep

    def com_emb(self, g, x):
        z, lbd, kappa = self.get_z(g, x, 0)

        adj_list = self.EdgePart(x, g)
        x_commun_list, indices_list = self.NodePart(x, z)
        com_rep = []
        for i, x_com in enumerate(x_commun_list):
            com_enc_rep, all_hidden = self.encoder_com(g, x_com, adj_list[i], return_hidden=True)
            # com_enc_rep = torch.sparse.mm(adj_list[i], com_enc_rep)
            com_rep.append(com_enc_rep)  # F.normalize(com_enc_rep, p=2, dim=1))
        com_rep = torch.cat(com_rep, dim=-1)
        # com_rep = torch.sum(torch.stack(com_rep, dim=0), dim=0)
        global_rep = self.encoder(g, x)

        rep = torch.cat([com_rep, global_rep], dim=-1)  # cora citeseer
        # rep = self.encoder_to_decoder(rep)
        # rep = global_rep + com_rep

        if self._use_mlp:
            mlp_rep = self.enc_mlp(x)
            # rep = rep + mlp_rep
            rep = torch.cat([rep, mlp_rep], dim=-1)

        return com_rep, global_rep, rep, indices_list, z

    def reparameterize(self, lbd, kappa, sample_num=1):
        '''
            weibull reparameterization: z = lbd * (- ln(1 - u)) ^ (1/kappa), u ~ uniform(0,1)
            z: node-community affiliation.
            lbd: scale parameter, kappa: shape parameter
        '''

        if self._random_encoder and self.training:
            u = torch.rand_like(lbd)
            z = lbd * (- log_max(1 - u)).pow(1 / kappa)
        else:
            z = lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))
        return z

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
