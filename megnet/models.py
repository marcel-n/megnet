from turtle import forward
from numpy import block
import torch
from torch.nn import Module, Linear, ModuleList, Identity, Softplus
from dgl.nn import Set2Set

from .types import *
from .layers import MegNetBlock


class MLP(Module):

    def __init__(
        self,
        dims: List[int],
        activation: Callable[[Tensor], Tensor] = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True))

                if activation is not None:
                    self.layers.append(activation)
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last))

                if activation is not None and activate_last:
                    self.layers.append(activation)

    def __repr__(self):
        dims = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f'{layer.in_features} \u2192 {layer.out_features}')
            else:
                dims.append(layer.__class__.__name__)

        return f'MLP({", ".join(dims)})'

    @property
    def last_linear(self) -> Linear:
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def in_features(self) -> int:
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x


class MegNet(Module):

    def __init__(
        self,
        in_dim: int,
        num_blocks: int,
        hiddens: List[int],
        conv_hiddens: List[int],
        s2s_num_layers: int,
        s2s_num_iters: int,
        output_hiddens: List[int],
        is_classification: bool = True,
        node_embed: Optional[Module] = None,
        edge_embed: Optional[Module] = None,
        attr_embed: Optional[Module] = None,
        dropout: Optional[float] = None,
        ) -> None:
        super().__init__()

        self.edge_embed = edge_embed if edge_embed else Identity()
        self.node_embed = node_embed if node_embed else Identity()
        self.attr_embed = attr_embed if attr_embed else Identity()
        
        dims = [in_dim] + hiddens
        self.edge_encoder = MLP(dims, Softplus(), activate_last=True)
        self.node_encoder = MLP(dims, Softplus(), activate_last=True)
        self.attr_encoder = MLP(dims, Softplus(), activate_last=True)

        blocks_in_dim = hiddens[-1]
        block_out_dim = conv_hiddens[-1]
        block_args = dict(conv_hidden=conv_hiddens, dropout=dropout, skip=True)
        blocks = []
        # first block
        blocks.append(MegNetBlock(dims=[blocks_in_dim], **block_args))
        # other blocks
        for _ in range(num_blocks-1):
            blocks.append(
                MegNetBlock(dims=[block_out_dim]+hiddens, **block_args)
            )
        self.blocks = ModuleList(blocks)

        self.edge_s2s = Set2Set(block_out_dim, s2s_num_iters, s2s_num_layers)
        self.node_s2s = Set2Set(block_out_dim, s2s_num_iters, s2s_num_layers)

        self.output_proj = MLP(
            dims=[block_out_dim]+output_hiddens+[1],
            activation=Softplus(),
            activate_last=False
        )

        self.is_classification = is_classification

    def forward(
        self,
        graph: DGLGraph, 
        edge_feat: Tensor, 
        node_feat: Tensor, 
        graph_attr: Tensor,
    ) -> None:

        edge_feat = self.edge_encoder(self.edge_embed(edge_feat))
        node_feat = self.node_encoder(self.node_embed(node_feat))
        graph_attr = self.attr_encoder(self.attr_embed(graph_attr))

        for block in self.blocks:
            output = block(graph, edge_feat, node_feat, graph_attr)
            edge_feat, node_feat, graph_attr = output
        
        node_vec = self.node_s2s(graph, node_feat)
        edge_vec = edge_feat  # TODO(marcel): need a s2s version

        vec = torch.hstack([node_vec, edge_vec, graph_attr])

        if self.dropout:
            vec = self.dropout(vec)

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        return output
        