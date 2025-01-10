import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.cluster import KMeans

def compute_edge_weights(G, node_features):
    for u, v in G.edges():
        # 获取两个节点的特征向量
        feature_u = node_features[u]
        feature_v = node_features[v]

        # 计算余弦相似度
        weight = F.cosine_similarity(feature_u.unsqueeze(0), feature_v.unsqueeze(0)).item()

        # 将余弦相似度作为边的权重
        G[u][v]['weight'] = weight

    return G

def multi_scale_graph_coarsening(G, node_features, scales=[0.2, 0.1]):
    coarse_graphs = []
    previous_cluster_centers = None  # 用于存储前一层的聚类中心

    for scale in scales:
        target_size = max(10, int(G.number_of_nodes() * scale))

        # 动态设置 k 值
        n_clusters = max(3, int(G.number_of_nodes() * scale * 0.5))

        # 传递前一层的聚类中心到 fast_graph_coarsening
        G_coarse, cluster_centers = fast_graph_coarsening(
            G, target_size=target_size, min_edges=5, node_features=node_features,
            n_clusters=n_clusters, previous_cluster_centers=previous_cluster_centers)

        previous_cluster_centers = cluster_centers  # 保存本层的聚类中心，供下一层使用

        print(f"Generated coarse graph with scale {scale}: {G_coarse.number_of_nodes()} nodes, {G_coarse.number_of_edges()} edges")

        if G_coarse.number_of_edges() > 0:
            coarse_graphs.append(G_coarse)
        else:
            print(f"Skipping this scale due to insufficient edges.")

    return coarse_graphs
from sklearn.metrics.pairwise import cosine_similarity
def fast_graph_coarsening(G, target_size, min_edges=1, node_features=None, n_clusters=3, previous_cluster_centers=None):
    """
    使用k-means对图进行粗化，并结合前一层的聚类结果影响本层的距离
    G: 输入图
    target_size: 粗化后目标节点数量
    min_edges: 最小边数
    node_features: 每个节点的特征（需要为每个节点准备特征向量）
    n_clusters: 当前粗化阶段的簇数量
    previous_cluster_centers: 前一层的聚类中心，用于影响本层的距离计算
    """
    # 复制原始图
    G_coarse = G.copy()

    # 获取节点特征
    if node_features is None:
        raise ValueError("需要提供节点特征才能进行KMeans聚类")

    while G_coarse.number_of_nodes() > target_size:
        # 获取节点列表
        nodes = list(G_coarse.nodes())

        # 将特征从 GPU 转换为 CPU，然后转为 numpy 数组
        node_feat_matrix = np.array([node_features[n].cpu().numpy() for n in nodes])

        # 确保节点数大于簇数
        if len(nodes) < n_clusters:
            print(f"Warning: Number of nodes ({len(nodes)}) is less than number of clusters ({n_clusters}). Skipping KMeans.")
            break

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(node_feat_matrix)

        # 如果有前一层的聚类中心，加入额外的距离惩罚项
        if previous_cluster_centers is not None:
            for i, node in enumerate(nodes):
                current_node_feat = node_feat_matrix[i]
                previous_center = previous_cluster_centers[clusters[i]]

                # 计算当前节点与前一层聚类中心的距离
                previous_distance = np.linalg.norm(current_node_feat - previous_center)

                # 将前一层的距离作为惩罚项，加入当前的聚类分配中
                clusters[i] = (clusters[i] + previous_distance) * 0.5

        # 合并节点：同一个簇的节点会合并为一个新节点
        for cluster_id in range(n_clusters):
            cluster_nodes = [nodes[i] for i in range(len(nodes)) if clusters[i] == cluster_id]
            if len(cluster_nodes) < 2:
                continue

            # 创建新节点，合并这些节点的特征和边
            new_node = "-".join(map(str, cluster_nodes))
            G_coarse.add_node(new_node)

            # 合并邻居
            for node in cluster_nodes:
                for neighbor in list(G_coarse.neighbors(node)):
                    if neighbor not in cluster_nodes:  # 忽略簇内部的邻居
                        new_weight = G_coarse[node][neighbor].get('weight', 1)
                        if G_coarse.has_edge(new_node, neighbor):
                            G_coarse[new_node][neighbor]['weight'] += new_weight
                        else:
                            G_coarse.add_edge(new_node, neighbor, weight=new_weight)

            # 删除原来的节点
            G_coarse.remove_nodes_from(cluster_nodes)

        # 检查是否满足最小边数
        if G_coarse.number_of_edges() < min_edges:
            print(f"Warning: Coarsened graph has fewer than {min_edges} edges. Stopping further coarsening.")
            break

    # 转换节点标签为整数，确保后续处理一致性
    G_coarse = nx.convert_node_labels_to_integers(G_coarse)

    # 检查 kmeans 是否已定义
    if 'kmeans' in locals():
        cluster_centers = kmeans.cluster_centers_  # 当前层的聚类中心
    else:
        cluster_centers = None

    return G_coarse, cluster_centers

def compute_normalized_laplacian(similarity_matrix):
    # 计算度矩阵 D
    degree = similarity_matrix.sum(dim=1)
    degree_matrix = torch.diag(degree)

    # 计算未归一化的拉普拉斯矩阵 L = D - S
    laplacian_matrix = degree_matrix - similarity_matrix

    # 计算 D^{-1/2}
    d_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-10))

    # 计算归一化的拉普拉斯矩阵 L_norm = D^{-1/2} L D^{-1/2}
    laplacian_norm = d_inv_sqrt @ laplacian_matrix @ d_inv_sqrt

    return laplacian_norm

from scipy.cluster.hierarchy import linkage, fcluster

def contrastive_loss_batch(z1, z2, temperature=1, lambda_reg=0.1, n_clusters=3,
                                                        n_super_clusters=1):
    batch_size = 2000
    z1 = F.normalize(z1, dim=-1, p=2)
    z2 = F.normalize(z2, dim=-1, p=2)
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes)
    losses = []

    # Step 1: Perform KMeans clustering on z1 and z2
    kmeans_1 = KMeans(n_clusters=n_clusters).fit(z1.cpu().detach().numpy())  # Cluster z1
    kmeans_2 = KMeans(n_clusters=n_clusters).fit(z2.cpu().detach().numpy())  # Cluster z2

    # Get cluster centroids and labels for both views
    centroids_2 = torch.tensor(kmeans_2.cluster_centers_, device=z2.device)  # Centroids for z2
    labels_2 = torch.tensor(kmeans_2.labels_, device=z2.device)  # Cluster labels for z2

    # Step 2: Perform hierarchical clustering on the centroids of z2 (for super clusters)
    Z = linkage(kmeans_2.cluster_centers_, method='ward')
    super_labels_2 = torch.tensor(fcluster(Z, t=n_super_clusters, criterion='maxclust') - 1,
                                  device=z2.device)  # Get super cluster labels (hierarchical)

    # Compute the super centroids (higher level clusters)
    super_centroids_2 = torch.stack([centroids_2[super_labels_2 == i].mean(0) for i in range(n_super_clusters)])

    # Step 3: Compute positive pairs (z1[i], z2[i], z2's cluster centroid, z2's super cluster centroid)
    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]

        # Original positive pairs: z1[mask] and z2[mask]
        inter_sim = f(torch.mm(z1[mask], z2.t()))
        pos_sim = inter_sim[:, mask]  # z1[i] and z2[i]
        pos_sim_diag = pos_sim.diag()  # Diagonal elements representing z1[i] and z2[i]

        # Add z2's cluster centroid as positive pair
        centroids_for_batch = centroids_2[labels_2[mask]]  # Get centroids for z2[mask]
        inter_sim_with_centroid = f(torch.mm(z1[mask], centroids_for_batch.t()))  # z1[i] and z2's centroid
        pos_sim_with_centroid = inter_sim_with_centroid.diag()  # Diagonal elements for z1[i] and cluster centroids

        # Add z2's super cluster centroid as an additional positive pair
        super_centroids_for_batch = super_centroids_2[super_labels_2[labels_2[mask]]]  # Get super centroids
        inter_sim_with_super_centroid = f(
            torch.mm(z1[mask], super_centroids_for_batch.t()))  # z1[i] and z2's super centroid
        pos_sim_with_super_centroid = inter_sim_with_super_centroid.diag()  # Diagonal elements for z1[i] and super centroids

        # Combine the positive similarities (original + centroid + super centroid)
        combined_pos_sim = pos_sim_diag + pos_sim_with_centroid + pos_sim_with_super_centroid

        # Step 4: Compute negative pairs (same as before)
        intra_sim_11 = f(torch.mm(z1[mask], z1.t()))
        intra_sim_22 = f(torch.mm(z2[mask], z2.t()))

        epsilon = 1e-8
        denom_12 = intra_sim_11.sum(1) + inter_sim.sum(1) - intra_sim_11[:, mask].diag() + epsilon
        denom_21 = intra_sim_22.sum(1) + inter_sim.sum(1) - intra_sim_22[:, mask].diag() + epsilon

        # Step 5: Compute contrastive loss
        loss_12 = -torch.log((combined_pos_sim + epsilon) / denom_12)
        loss_21 = -torch.log((combined_pos_sim + epsilon) / denom_21)
        losses.append(loss_12 + loss_21)

    # Compute final contrastive loss
    contrastive_loss = torch.cat(losses).mean()

    # Step 6: Laplacian regularization remains the same
    inter_sim_full_1 = f(torch.mm(z1, z1.t()))
    inter_sim_full_2 = f(torch.mm(z2, z2.t()))
    laplacian_norm_1 = compute_normalized_laplacian(inter_sim_full_1)
    laplacian_norm_2 = compute_normalized_laplacian(inter_sim_full_2)

    reg_loss_z1 = torch.trace(z1.t() @ laplacian_norm_1 @ z1)
    reg_loss_z2 = torch.trace(z2.t() @ laplacian_norm_2 @ z2)
    reg_loss = (reg_loss_z1 + reg_loss_z2) / (num_nodes * 2)

    total_loss = contrastive_loss + lambda_reg * reg_loss
    return total_loss


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden, activation='relu', base_model=GCNConv):
        super(Encoder, self).__init__()
        self.base_model = base_model

        self.gcn1 = base_model(in_channels, hidden)
        self.gcn2 = base_model(hidden, out_channels)

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.activation(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


class Contra(Module):
    def __init__(self,
                 encoder,
                 hidden_size,
                 projection_size,
                 projection_hidden_size,
                 n_cluster,
                 v=1):
        super().__init__()

        # backbone encoder
        self.encoder = encoder

        # projection layer for representation contrastive
        self.rep_projector = MLP(hidden_size, projection_size, projection_hidden_size)
        # t-student cluster layer for clustering
        self.cluster_layer = nn.Parameter(torch.Tensor(n_cluster, hidden_size), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def kl_cluster(self, z1: torch.Tensor, z2: torch.Tensor):
        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z1.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)  # q1 n*K
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z2.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return q1, q2

    def forward(self, feat, adj):
        h = self.encoder(feat, adj)
        z = self.rep_projector(h)

        return h, z
