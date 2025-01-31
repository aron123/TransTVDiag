
from torch import nn
from core.model.backbone.graphormer import GraphormerEncoder

class Encoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 graph_hidden_dim, 
                 out_dim,
                 attn_drop=0.1,
                 num_heads=8,
                 num_layers=2):
        super(Encoder, self).__init__()

        self.graph_encoder = GraphormerEncoder(
            hidden_dim=graph_hidden_dim, 
            out_dim=out_dim,
            attn_drop=attn_drop,
            num_heads=num_heads,
            num_layers=num_layers,
            max_degree=512,
            num_spatial=511,
            multi_hop_max_dist=5,
            edge_dim=1,
            embedding_dim=128,
            pre_layernorm=True,
            activation_fn=nn.GELU(),
        )

    def forward(self, node_feat, in_degree, out_degree, attn_mask, path_data, dist):
        return self.graph_encoder(node_feat, in_degree, out_degree, attn_mask, path_data, dist)