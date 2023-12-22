import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.dropout = self.config.dropout
        self.transformations = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model, bias=False)
                for _ in range(config.num_relation)
            ]
        )
        self.self_transformations = nn.Linear(
            config.d_model, config.d_model, bias=False
        )

    def forward(self, x, matrix):
        self_x = self.self_transformations(x)
        flatten_matrix = matrix.view(-1, matrix.size(-2), matrix.size(-1))
        progression_x = torch.stack([trans(x)
                                    for trans in self.transformations], dim=1)
        flatten_progression_x = progression_x.view(
            -1, progression_x.size(-2), progression_x.size(-1)
        )
        flatten_neigh_x = flatten_matrix.bmm(flatten_progression_x)
        neigh_x = flatten_neigh_x.view(
            self_x.size(0), -
            1, flatten_neigh_x.size(-2), flatten_neigh_x.size(-1)
        ).sum(dim=1)
        norm = matrix.sum(dim=1).sum(dim=-1, keepdim=True)
        norm = torch.max(norm, torch.ones_like(norm))
        neigh_x = neigh_x / norm
        x = self_x + neigh_x
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return torch.relu(x)


class RGCN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_nodes = nn.Embedding(
            config.num_node + 1, config.d_model, padding_idx=config.num_node
        )
        self.layers = nn.ModuleList(
            [RGCNLayer(config) for _ in range(config.num_rgcnlayer)]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, nodes, matrix):
        matrix = (matrix > 0).float()
        nodes = nodes.masked_fill(nodes == -100, self.config.num_node).long()
        x = self.embed_nodes(nodes)
        for layer in self.layers:
            x = layer(x, matrix)
        return x
