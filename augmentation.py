import os
import copy
import numpy as np
import networkx as nx
from time import perf_counter as t
import random
import torch


# drop edges according to EBC, node similarity add edges
def sample_graph_own(dataset, feature, edge_index, remove_pct, k=None):
    edges_index = np.asarray(edge_index).T   # ndarray: (m, 2) cora: (10556, 2)
    n = feature.shape[0]

    G = nx.Graph()
    G.add_edges_from(edges_index)

    nodelist = [node for node in range(n)]
    G.add_nodes_from(nodelist)
    print('There are %d nodes.' % G.number_of_nodes())  # n_nodes = G.number_of_nodes()
    print('There are %d edges.' % G.number_of_edges())  # n_edges = G.number_of_edges()
    g_edges = np.asarray(G.edges)

    savepath = 'data/load_data'
    # compute edge betweenness and save for time consuming
    if k:
        if os.path.exists('{}/ebc_{}_{}knn.npy'.format(savepath, dataset, k)):
            print("Loading edge betweenness of knn graph.....")
            edge_b = np.load('{}/ebc_{}_{}knn.npy'.format(savepath, dataset, k), allow_pickle=True).item()
        else:
            start = t()
            edge_b = nx.edge_betweenness_centrality(G)
            print("compute edge betweenness of {}nn-graph cost time {}\n".format(k, (t() - start)))
            np.save('{}/ebc_{}_{}knn'.format(savepath, dataset, k), edge_b)
    else:
        if os.path.exists('{}/ebc_{}.npy'.format(savepath, dataset)):
            print("Loading edge betweenness.....")
            edge_b = np.load('{}/ebc_{}.npy'.format(savepath, dataset), allow_pickle=True).item()
        else:
            start = t()
            edge_b = nx.edge_betweenness_centrality(G)
            print("compute edge betweenness cost time {}\n".format(t() - start))
            np.save('{}/ebc_{}'.format(savepath, dataset), edge_b)

    eb_value = [value for key, value in edge_b.items()]

    if remove_pct:
        n_remove = int(G.number_of_edges() * remove_pct)
        eb_probs = np.array(eb_value)
        e_index_remove = np.argpartition(eb_probs, -n_remove)[-n_remove:]

        mask = np.ones(len(g_edges), dtype=bool)
        mask[e_index_remove] = False
        edges_new = g_edges[mask]   # array
    else:
        edges_new = g_edges

    return edges_new


# MERIT mask node feature
def aug_feature_dropout(input_feat, drop_percent=0.2):
    aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat


# GRACE drop feature
def drop_feature(x, drop_prob):
    # 使节点的某一维度为0
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
