from torch_geometric.nn import GINConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, is_torch_sparse_tensor
from torch_geometric.utils.sparse import set_sparse_value
import torch
import torch.nn.functional as F

from torch_geometric.typing import (
    SparseTensor,
    torch_sparse,
)


###################################### GIN conv layer ######################################
############################################################################################

class GINConv(GINConv):
    def __init__(self, nn, eps: float = 0, train_eps: bool = False, **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)

    def forward(self, x, edge_index, edge_weight=None):
        '''
            MLP( (1+eps) x_v + \sum_{u} x_u ), as in Eq 4.1 of https://openreview.net/pdf?id=ryGs6iA5Km
            <CAVEAT> edge_index should NOT include self-loops, so is the edge_weight
        '''
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out + (1 + self.eps) * x


        return self.nn(out)
    
    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        else:
            return x_j


###################################### GAT conv layer ######################################
############################################################################################

class GATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads, concat=True,
        **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, **kwargs)

    def forward(self, x, edge_index, edge_weight=None, size = None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, torch.Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_weight = remove_self_loops(
                    edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, torch.SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_weight)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)#, edge_weight=edge_weight)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, torch.Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    # def message(self, x_j, alpha, edge_weight):
    #     if edge_weight is not None:
    #         return edge_weight.view(-1, 1, 1) * alpha.unsqueeze(-1) * x_j
    #     else:
    #         return alpha.unsqueeze(-1) * x_j



