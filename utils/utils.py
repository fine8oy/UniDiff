import numpy as np
import os
import logging
import torch

import networkx as nx
from node2vec import Node2Vec

import matplotlib.pyplot as plt


def plot_compare_sequences(gen_sig, ori_sig, time_steps=288, save_path="comparison.png"):
    """
    绘制8个变量的前288个时间步对比图（竖排）

    参数:
        gen_sig: 生成信号 (8, 23104)
        ori_sig: 原始信号 (8, 23104)
        time_steps: 要显示的时间步数（默认288）
        save_path: 图片保存路径
    """

    # 创建画布（8个子图竖排）
    fig, axes = plt.subplots(8, 1, figsize=(12, 16), dpi=100)
    plt.subplots_adjust(hspace=0.5)  # 调整子图间距

    # 为每个变量绘制对比曲线
    for i in range(8):
        ax = axes[i]
        ax.plot(ori_sig[0, i, :time_steps], label="Original", linewidth=1, alpha=0.8)
        ax.plot(gen_sig[0, i, :time_steps], label="Generated", linewidth=1, alpha=0.8)
        ax.set_title(f"Variable {i + 1} (First {time_steps} steps)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.6)

    # 保存图片
    plt.savefig(save_path, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    plt.close()


# thu evaluate
class MaximumMeanDiscrepancy3D(object):
    """calculate MMD"""

    def __init__(self):
        super(MaximumMeanDiscrepancy3D, self).__init__()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = source.shape[0] + target.shape[0]
        total = np.concatenate([source, target], axis=0)  # 合并在一起
        total0 = np.expand_dims(total, axis=0)
        total0 = np.tile(total0, (total.shape[0], 1, 1))

        total1 = np.expand_dims(total, axis=1)
        total1 = np.tile(total1, (1, total.shape[0], 1))

        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = np.sum(L2_distance) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [np.exp(-L2_distance / bandwidth_temp) for
                      bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # 将多个核合并在一起

    def __call__(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n = source.shape[0]
        m = target.shape[0]

        kernels = self.guassian_kernel(
            source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        # K_ss矩阵，Source<->Source
        XX = np.divide(XX, n * n).sum(axis=1).reshape(1, -1)
        # K_st矩阵，Source<->Target
        XY = np.divide(XY, -n * m).sum(axis=1).reshape(1, -1)

        # K_ts矩阵,Target<->Source
        YX = np.divide(YX, -m * n).sum(axis=1).reshape(1, -1)
        # K_tt矩阵,Target<->Target
        YY = np.divide(YY, m * m).sum(axis=1).reshape(1, -1)

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss

def cal_smape(p_pred, p_real, eps=0.00000001):
    out = np.mean(np.abs(p_real - p_pred) / ((np.abs(p_real) + np.abs(p_pred)) / 2 + eps))
    return out

def adjacency_matrix_to_graph(adj_matrix):
    """将邻接矩阵转换为 networkx 图"""
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0:  # 如果有边
                G.add_edge(i, j, weight=adj_matrix[i, j])
    return G


def generate_node2vec_embeddings(adj_tensor, embedding_dim=128):
    """生成 node2vec 嵌入"""
    num_graphs = adj_tensor.shape[0]
    num_nodes = adj_tensor.shape[1]
    embeddings = torch.zeros(num_graphs, num_nodes, embedding_dim)  # 用于存储所有图的嵌入

    for i in range(num_graphs):
        # 将邻接矩阵转换为图
        adj_matrix = adj_tensor[i].numpy()
        G = adjacency_matrix_to_graph(adj_matrix)

        # 运行 node2vec
        node2vec = Node2Vec(G, dimensions=embedding_dim, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # 获取所有节点的嵌入
        for node_id in range(num_nodes):
            embeddings[i, node_id] = torch.tensor(model.wv[str(node_id)])

    return embeddings


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def save_checkpoint(ckpt_dir, state, epoch, ema_model=None):
    saved_state = {
        'epoch': epoch,
        'model': state['model'].state_dict(),
    }
    if ema_model is not None:
        saved_state['ema_model'] = ema_model.state_dict()
    torch.save(saved_state, ckpt_dir)


def restore_checkpoint(ckpt_dir, state, device, ema_model=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['epoch'] = loaded_state['epoch']
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        if 'ema_model' in loaded_state and ema_model is not None:
            ema_model.load_state_dict(loaded_state['ema_model'])
        logging.info(f'Successfully loaded previous state')
        return state


def log_config_and_tags(args, logger, name):
    logger.log_name_params('config/hyperparameters', vars(args))
    logger.log_name_params('config/name', name)
    logger.add_tags(args.tags)
    logger.add_tags([args.dataset])


def create_model_name_and_dir(args):
    name = (f'domain={args.target_domain}-'
            f'model={args.model}-'
            f'bs={args.batch_size}-'
            f'-lr={args.learning_rate:.4f}-'
            f'ch_mult={args.ch_mult}-'
            f'attn_res={args.attn_resolution}-'
            f'unet_ch={args.unet_channels}'
            )
    args.log_dir = '%s/%s/%s' % (args.log_dir, args.dataset, name)
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    return name


def restore_state(args, state,ema_model=None):
    logging.info("restoring checkpoint from: {}".format(args.log_dir))
    restore_checkpoint(args.log_dir, state, args.device, ema_model = ema_model)
    init_epoch = state['epoch']
    return init_epoch


def print_model_params(logger, model):
    params_num = sum(param.numel() for param in model.parameters())
    logging.info("number of model parameters: {}".format(params_num))
    logger.log_name_params('config/params_num', params_num)


# --- extrapolation and interpolation --- #
# get the mask and x for the time series
def get_x_and_mask(args, data):
    if args.dataset in ['climate', 'physionet']:
        # in the case of these datasets, the 'data_to_predict' is the same as 'observed_data
        if args.task == 'extrapolation':
            # concat the observed and predicted data
            x_ts = torch.cat([data['observed_data'], data['data_to_predict']], dim=1).to(args.device)
            # the predicted mask is opposite. the 1s are observed in the mask so it needed to be flipped in our case
            mask_ts = torch.cat([data['observed_mask'],  1 - data['mask_predicted_data']], dim=1).to(args.device)
        else:
            x_ts = data['observed_data'].to(args.device)
            mask_ts = data['mask_predicted_data'].to(args.device)
    else:
        if args.task == 'extrapolation':
            x_ts = data[0].float().to(args.device)
            # half ones and half zeros
            mask_ts = torch.zeros_like(x_ts)
            mask_ts[:, :x_ts.shape[1] // 2] = 1
        else:
            x_ts = data[0].float().to(args.device)
            # --- generate random mask and mask x as it time series --- #
            B, T, N = x_ts.shape
            mask_ts = torch.rand((B, T, N)).to(args.device)
            mask_ts[mask_ts <= args.mask_rate] = 0  # masked
            mask_ts[mask_ts > args.mask_rate] = 1  # remained

    return mask_ts, x_ts
