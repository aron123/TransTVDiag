import torch
from torch import nn

from core.model.Classifier import Classifyer
from core.model.Voter import Voter
from core.model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()

        self.args = args

        self.metric_encoder = Encoder(in_dim=args.embedding_dim,
                                      attn_drop=args.attn_drop,
                                      num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out)
        self.trace_encoder = Encoder(in_dim=args.embedding_dim,
                                      attn_drop=args.attn_drop,
                                      num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out)
        self.log_encoder = Encoder(in_dim=args.embedding_dim,
                                      attn_drop=args.attn_drop,
                                      num_heads=args.num_heads,
                                      num_layers=args.num_layers,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out)
        fuse_dim = 3 * args.graph_out

        self.locator = Voter(fuse_dim, 
                                  hiddens=args.linear_hidden,
                                  out_dim=args.N_I)
        self.typeClassifier = Classifyer(in_dim=fuse_dim,
                                         hiddens=args.linear_hidden,
                                         out_dim=args.N_T)

    def forward(self, metric_feat, trace_feat, log_feat, in_degree, out_degree, dist, path_data, attn_mask=None):
        x_m = metric_feat
        x_t = trace_feat
        x_l = log_feat
        
        f_m = self.metric_encoder(x_m, in_degree, out_degree, attn_mask, path_data, dist)
        f_t = self.trace_encoder(x_t, in_degree, out_degree, attn_mask, path_data, dist)
        f_l = self.log_encoder(x_l, in_degree, out_degree, attn_mask, path_data, dist)

        f = torch.cat((f_m, f_t, f_l), dim=1)
        # failure type identification
        type_logit = self.typeClassifier(f)
        # root cause localization
        root_logit = self.locator(f)

        return (f_m, f_t, f_l), root_logit, type_logit