import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor, WikiCS
import numpy as np
import torch
import scipy.sparse as sp
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph(dataset):

    graph = './data/{}/{}_graph.txt'.format(dataset, dataset)
    print("Loading path:", graph)
    data = './data/{}/{}.txt'.format(dataset, dataset)
    dataset = np.loadtxt(data, dtype=float)
    n, _ = dataset.shape
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(graph, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, edges




def load_npy(dataset):

    graph = './data/{}/{}_graph.txt'.format(dataset, dataset)
    print("Loading path:", graph)

    data = './data/{}/{}_feat.npy'.format(dataset, dataset)
    dataset = np.load(data)
    n, _ = dataset.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(graph, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, edges


def load_graph_3(dataset):

    graph = './data/{}/{}3_graph.txt'.format(dataset, dataset)
    print("Loading path:", graph)
    data = './data/{}/{}.txt'.format(dataset, dataset)
    dataset = np.loadtxt(data, dtype=float)
    n, _ = dataset.shape
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(graph, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, edges


def load_cora(dataset):

    path = './data/cora/'
    data_name = 'cora'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    x = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return x.toarray(), labels, adj, edges


def get_dataset(name):

    assert name in ['cora', 'cite', 'acm', 'dblp', 'hhar', 'usps', 'reut', 'amap', 'wisc', 'wiki', 'texas']

    if name == 'cora':
        x, y, _, edges_index = load_cora(name)

    elif name in ['usps', 'hhar', 'reut']:
        x_p = './data/{}/{}.txt'.format(name, name)
        y_p = './data/{}/{}_label.txt'.format(name, name)
        _, edges_index = load_graph_3(name)
        x = np.loadtxt(x_p)
        y = np.loadtxt(y_p)

    elif name in ['amap', 'wisc', 'wiki', 'texas']:
        x_p = './data/{}/{}_feat.npy'.format(name, name)
        y_p = './data/{}/{}_label.npy'.format(name, name)
        _, edges_index = load_npy(name)
        x = np.load(x_p)
        y = np.load(y_p)

    else:
        x_p = './data/{}/{}.txt'.format(name, name)
        y_p = './data/{}/{}_label.txt'.format(name, name)
        _, edges_index = load_graph(name)
        x = np.loadtxt(x_p)
        y = np.loadtxt(y_p)

        # edge_1 = './data/{}/cite_coarse_graph_edges_scale_0.txt'.format(name, name)
        # edge_2 = './data/{}/cite_coarse_graph_edges_scale_1.txt'.format(name, name)
        # edge_3 = './data/{}/cite_coarse_graph_edges_scale_2.txt'.format(name, name)
        # edge_1 = np.loadtxt(edge_1)
        # edge_2 = np.loadtxt(edge_2)
        # edge_3 = np.loadtxt(edge_3)
        # return x, y, _, edges_index, edge_1, edge_2, edge_3
    return x, y, _, edges_index