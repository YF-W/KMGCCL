import argparse
from sklearn.preprocessing import normalize
from evaluation import eva
from module import *
from utils import get_dataset, setup_seed
from augmentation import *
from logger import Logger, metrics_info, record_info
import datetime
import warnings

warnings.filterwarnings("ignore")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_multi_scale(model, adj, x, drop_feature_rate, label, epochs):
    clu = []
    best_acc = 0
    best_epoch = 0
    metrics = [' acc', ' nmi', ' ari', ' f1']
    logger = Logger(args.dataset + '==' + nowtime)
    logger.info(model)
    logger.info(args)
    logger.info(metrics_info(metrics))
    # 生成多尺度图
    G = nx.Graph()
    G.add_edges_from(adj.cpu().numpy().T)
    node_features = x.cpu()
    G = compute_edge_weights(G, node_features)
    scales = [0.5, 0.25, 0.1]  # 适当调整尺度比例
    coarse_graphs = multi_scale_graph_coarsening(G, node_features=node_features, scales=scales)

    # 将多尺度图转换为 PyTorch 的 edge_index 格式
    adj_coarse_scales = []
    for G_coarse in coarse_graphs:
        if G_coarse.number_of_edges() > 0:
            edge_index = torch.tensor(list(G_coarse.edges()), dtype=torch.long).t().contiguous()
            if edge_index.size(1) > 0:
                adj_coarse_scales.append(edge_index.to(adj.device))

    # 检查是否有有效的图用于训练
    if not adj_coarse_scales:
        print("Error: All coarse graphs are empty. Please check the graph coarsening process.")
        return clu

    for epoch in range(epochs):
        model.train()

        x_aug = drop_feature(x, drop_feature_rate)
        # x_aug_1 = drop_feature(x, drop_feature_rate)
        # x_aug_2 = drop_feature(x, drop_feature_rate)
        # x_aug_3 = drop_feature(x, drop_feature_rate)

        h1, z1 = model(x, adj)
        # h2, z2 = model(x_aug, adj)
        h2_0, z2_0 = model(x_aug, adj_coarse_scales[0])
        h2_1, z2_1 = model(x_aug, adj_coarse_scales[1])
        h2_2, z2_2 = model(x_aug, adj_coarse_scales[2])
        h2 = (0.5 * h2_0 + 0.25 * h2_1 + 0.1 * h2_2) / 3
        z2 = (0.5 * z2_0 + 0.25 * z2_1 + 0.1 * z2_2) / 3
        # h2 = (0.5 * h2_0 + 0.25 * h2_1) / 2
        # z2 = (0.5 * z2_0 + 0.25 * z2_1) / 2

        # l_z_1 = contrastive_loss_batch(z2_0, z2_1)
        # pro_loss_1 = 0.5 * l_z_1.mean()
        #
        # l_z_2 = contrastive_loss_batch(z2_0, z2_2)
        # pro_loss_2 = 0.5 * l_z_2.mean()
        #
        # l_z_3 = contrastive_loss_batch(z2_2, z2_1)
        # pro_loss_3 = 0.5 * l_z_3.mean()

        q1, q2 = model.kl_cluster(h1, h2)

        q1_pred = q1.detach().cpu().numpy().argmax(1)
        acc, nmi, ari, f1 = eva(label, q1_pred, 'Q1_self_cluster', True)
        logger.info("epoch%d%s:\t%s" % (epoch, ' Q1', record_info([acc, nmi, ari, f1])))

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

        if epoch % args.update_p == 0:
            p1 = target_distribution(q1.data)
            # p_pred = p1.detach().cpu().numpy().argmax(1)
            # eva(label, p_pred, 'P_self_cluster', True)

        kl1 = F.kl_div(q1.log(), p1, reduction='batchmean')
        kl2 = F.kl_div(q2.log(), p1, reduction='batchmean')
        con = F.kl_div(q2.log(), q1, reduction='batchmean')
        clu_loss = kl1 + kl2 + con

        l_h = contrastive_loss_batch(h1, h2)
        enc_loss = 0.5 * l_h.mean()

        l_z = contrastive_loss_batch(z1, z2)
        pro_loss = 0.5 * l_z.mean()

        loss = args.rep * enc_loss + args.pro * pro_loss + args.clu * clu_loss #+ (pro_loss_1 + pro_loss_2 + pro_loss_3) / 3
        clu.append((acc, nmi, ari, f1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Best accuracy: {best_acc:.4f} at epoch {best_epoch}')

    return clu, logger, best_epoch


if __name__ == '__main__':
    setup_seed(2018)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='acm')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--pro_hid', type=int, default=1024)

    parser.add_argument('--mask', type=float, default=0.2)

    parser.add_argument('--rep', type=float, default=1)
    parser.add_argument('--clu', type=float, default=1)
    parser.add_argument('--pro', type=float, default=1)
    parser.add_argument('--update_p', type=int, default=1)

    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    # parser.add_argument('--neg', type=bool, default=False)

    args = parser.parse_args()
    if args.dataset == 'acm':
        args.n_clusters = 3
        args.n_input = 1870
    elif args.dataset == 'dblp':
        args.n_clusters = 4
        args.n_input = 334
    elif args.dataset == 'cite':
        args.n_clusters = 6
        args.n_input = 3703
    elif args.dataset == 'cora':
        args.n_clusters = 7
        args.n_input = 1433
    elif args.dataset == 'hhar':
        args.n_clusters = 6
        args.n_input = 561
    elif args.dataset == 'reut':
        args.n_clusters = 4
        args.n_input = 2000
    elif args.dataset == 'usps':
        args.n_clusters = 10
        args.n_input = 256
    elif args.dataset == 'wisc':
        args.n_clusters = 5
        args.n_input = 1703
    elif args.dataset == 'wiki':
        args.n_clusters = 17
        args.n_input = 4973
    elif args.dataset == 'texas':
        args.n_clusters = 5
        args.n_input = 1703
    elif args.dataset == 'amap':
        args.n_clusters = 8
        args.n_input = 745
    print(args)

    # torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x, labels, _, edge_index = get_dataset(args.dataset)
    x = x.astype(float)
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = edge_index.T
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    features = x

    n_cluster = args.n_clusters

    features = features.to(device)
    edge_index = edge_index.to(device)

    feature_drop = args.mask

    # model
    encoder = Encoder(in_channels=args.n_input,
                      out_channels=args.out_dim,
                      hidden=args.hidden,
                      activation='relu',  # 这里使用 ReLU 作为激活函数
                      base_model=GCNConv).to(device)

    model = Contra(encoder=encoder,
                   hidden_size=args.out_dim,
                   projection_hidden_size=args.pro_hid,
                   projection_size=args.pro_hid,
                   n_cluster=n_cluster).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # load pre-train for clustering initialization
    save_model = torch.load('pretrain/{}_contra.pkl'.format(args.dataset), map_location='cpu')
    # 使用学习率调度器，逐步减小学习率

    model.encoder.load_state_dict(save_model)

    # 读取保存的 adj_coarse 文件
    adj_coarse = torch.load('adj_coarse.pt')

    # 将 adj_coarse 移动到需要的设备（如 GPU）
    adj_coarse = adj_coarse.to(device)

    with torch.no_grad():
        h_o, z_o = model(features, adj_coarse)
    h_o_normalized = normalize(h_o.data.cpu().numpy(), norm='l2')
    kmeans = KMeans(n_clusters=n_cluster, n_init=20)
    clu_pre = kmeans.fit_predict(h_o_normalized)
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # 初始化 KMedoids 算法，使用余弦相似度作为度量方式
    # kmedoids = KMedoids(n_clusters=n_cluster, metric='cosine', init='k-medoids++', max_iter=300, random_state=42)
    # # 拟合数据并预测聚类结果
    # clu_pre = kmedoids.fit_predict(h_o.data.cpu().numpy())
    # # 将聚类中心更新到模型中
    # model.cluster_layer.data = torch.tensor(kmedoids.cluster_centers_).to(device)
    eva(labels, clu_pre, 'Initialization')

    clu_acc, logger, best_epoch = train_multi_scale(model, edge_index, features, feature_drop, labels, args.epochs)
    clu_q_max = np.max(np.array(clu_acc), 0)
    logger.info("%sepoch%d:\t%s" % ('Best Acc is at ', best_epoch, record_info(clu_q_max)))
    # clu_q_max = np.max(np.array(clu_acc), 0)
    clu_q_final = clu_acc[-1]