import dgl
import torch as th

# Prepares data for processing with Graphormer

def prepare_for_graphormer(graphs, labels):
    for g in graphs:
        spd, path = dgl.shortest_dist(g, root=None, return_paths=True)
        g.ndata["spd"] = spd
        g.ndata["path"] = path

    num_graphs = len(graphs)
    num_nodes = [g.num_nodes() for g in graphs]
    max_num_nodes = max(num_nodes)

    # Graphormer adds a virual node to the graph, which is connected to
    # all other nodes and supposed to represent the graph embedding. So
    # here +1 is for the virtual node.
    attn_mask = th.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
    metric_feat = []
    trace_feat = []
    log_feat = []
    in_degree, out_degree = [], []
    path_data = []
    
    # Since shortest_dist returns -1 for unreachable node pairs and padded
    # nodes are unreachable to others, distance relevant to padded nodes
    # use -1 padding as well.
    dist = -th.ones(
        (num_graphs, max_num_nodes, max_num_nodes), dtype=th.long
    )

    for i in range(num_graphs):
        # A binary mask where invalid positions are indicated by True.
        # Avoid the case where all positions are invalid.
        attn_mask[i, :, num_nodes[i] + 1 :] = 1

        # +1 to distinguish padded non-existing nodes from real nodes
        metric_feat.append(graphs[i].ndata["metrics"] + 1)
        trace_feat.append(graphs[i].ndata["traces"] + 1)
        log_feat.append(graphs[i].ndata["logs"] + 1)
        
        # 0 for padding
        in_degree.append(
            th.clamp(graphs[i].in_degrees() + 1, min=0, max=512)
        )
        out_degree.append(
            th.clamp(graphs[i].out_degrees() + 1, min=0, max=512)
        )

        # Path padding to make all paths to the same length "max_len".
        path = graphs[i].ndata["path"]
        path_len = path.size(dim=2)
        # shape of shortest_path: [n, n, max_len]
        max_len = 5
        if path_len >= max_len:
            shortest_path = path[:, :, :max_len]
        else:
            p1d = (0, max_len - path_len)
            # Use the same -1 padding as shortest_dist for
            # invalid edge IDs.
            shortest_path = th.nn.functional.pad(path, p1d, "constant", -1)
        pad_num_nodes = max_num_nodes - num_nodes[i]
        p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
        shortest_path = th.nn.functional.pad(shortest_path, p3d, "constant", -1)
        # +1 to distinguish padded non-existing edges from real edges
        edata = graphs[i].edata["traces"] + 1

        # shortest_dist pads non-existing edges (at the end of shortest
        # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
        # for all padded edge features.
        edata = th.cat(
            (edata, th.zeros(1, edata.shape[1]).to(edata.device)), dim=0
        )
        path_data.append(edata[shortest_path])

        dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

    # node feat padding
    metric_feat = th.nn.utils.rnn.pad_sequence(metric_feat, batch_first=True)
    trace_feat = th.nn.utils.rnn.pad_sequence(trace_feat, batch_first=True)
    log_feat = th.nn.utils.rnn.pad_sequence(log_feat, batch_first=True)

    # degree padding
    in_degree = th.nn.utils.rnn.pad_sequence(in_degree, batch_first=True)
    out_degree = th.nn.utils.rnn.pad_sequence(out_degree, batch_first=True)

    return (
        labels.reshape(num_graphs, -1),
        metric_feat,
        trace_feat,
        log_feat,
        in_degree,
        out_degree,
        attn_mask,
        th.stack(path_data),
        dist,
    )