import torch.nn as nn
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder
import torch

class GraphormerEncoder(nn.Module):
    def __init__(self,
                embedding_dim,
                hidden_dim, 
                out_dim,
                num_heads=4,
                attn_drop=0.1,
                num_layers=2,
                max_degree=512,
                num_spatial=511,
                multi_hop_max_dist=5,
                pre_layernorm=True,
                activation_fn=nn.GELU(),
                edge_dim=1,
        ):
        super(GraphormerEncoder, self).__init__()
        self.dropout = nn.Dropout(p=attn_drop)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.graph_token = nn.Embedding(1, embedding_dim)

        self.degree_encoder = DegreeEncoder(
            max_degree=max_degree,
            embedding_dim=embedding_dim
        )

        self.path_encoder = PathEncoder(
            max_len=multi_hop_max_dist,
            feat_dim=edge_dim,
            num_heads=num_heads,
        )

        self.spatial_encoder = SpatialEncoder(
            max_dist=num_spatial,
            num_heads=num_heads
        )
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=hidden_dim,
                    num_heads=num_heads,
                    dropout=attn_drop,
                    activation=activation_fn,
                    norm_first=pre_layernorm,
                )
                for _ in range(num_layers)
            ]
        )

        # map graph_rep to out_dim
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(self, node_feat, in_degree, out_degree, attn_mask, path_data, dist):
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))

        # node feature + degree encoding as input
        node_feat = node_feat + deg_emb
        graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
            num_graphs, 1, 1
        )
        x = torch.cat([graph_token_feat, node_feat], dim=1)

        # spatial encoding and path encoding serve as attention bias
        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        # spatial encoding of the virtual node
        t = self.graph_token_virtual_distance.weight.reshape(
            1, 1, self.num_heads
        )
        # Since the virtual node comes first, the spatial encodings between it
        # and other nodes will fill the 1st row and 1st column (omit num_graphs
        # and num_heads dimensions) of attn_bias matrix by broadcasting.
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        x = self.emb_layer_norm(x)

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, attn_mask=attn_mask)

        graph_rep = x[:, 0, :]
        graph_rep = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias

        return graph_rep