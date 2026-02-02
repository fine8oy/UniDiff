import numpy as np
import torchaudio.transforms as transforms
import os
import sys
import torch
import torch.utils.data as Data
import pandas as pd
import datetime
import pickle
import geopandas as gpd
import re


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_kg(data_dir):
    ent2id, rel2id = {}, {}
    kg_data_str = []

    with open(data_dir, 'r') as f:
        for line in f.readlines():
            if data_dir == "/data/wangziang/KGDiff/kg_nyc_bike.txt":
                h, r, t = line.strip().split(' ')
            else:
                h, r, t = line.strip().split('\t\t')
            kg_data_str.append((h, r, t))

    # 获取所有实体并排序
    ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])),
                  key=lambda x: [int(text) if text.isdigit() else text.lower()
                                 for text in re.split('([0-9]+)', x)])

    # 获取所有关系并排序
    rels = sorted(list(set([x[1] for x in kg_data_str])))

    # 更高效地创建实体映射
    ent2id = {ent: idx for idx, ent in enumerate(ents)}

    # 创建关系映射
    rel2id = {rel: idx for idx, rel in enumerate(rels)}

    # 转换为索引表示的三元组
    kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]

    return ent2id, rel2id, kg_data


def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, data):
        return data * (self.std + 1e-6) + self.mean


def gen_dataloader(args):
    def load_region_split(split_rule, split_ratio):
        split_name = f"{split_rule}_{int(round(split_ratio * 100))}_{int(round((1 - split_ratio) * 100))}"
        split_dir = '/data/wangziang/UniDiff/data/region_splits/'
        with open(os.path.join(split_dir, f"{split_name}_train.pkl"), 'rb') as f:
            train_ids = pickle.load(f)
        with open(os.path.join(split_dir, f"{split_name}_test.pkl"), 'rb') as f:
            test_ids = pickle.load(f)
        return train_ids, test_ids

    def load_domain_data(domain, dataset_name):
        path_map = {
            'bike_flow': f'/data/wangziang/UniDiff/data/{dataset_name}_bike_flow_matrix.npy',
            'taxi_flow': f'/data/wangziang/UniDiff/data/{dataset_name}_taxi_flow_matrix.npy',
            'service': f'/data/wangziang/UniDiff/data/{dataset_name}_service_matrix.npy',
            'crime': f'/data/wangziang/UniDiff/data/{dataset_name}_crime_matrix.npy',
        }
        arr = np.load(path_map[domain])
        tensor = torch.tensor(arr, dtype=torch.float, device=args.device)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        return tensor[:, :, :1]  # (num_nodes, time_len, feat_dim)

    def split_by_day(tensor, day_length, start_date, keep_weekdays=True):
        num_nodes, total_len, feat_dim = tensor.shape
        num_days = total_len // day_length
        day_tensor = tensor[:, :num_days * day_length, :].reshape(
            num_nodes, num_days, day_length, feat_dim
        )

        if keep_weekdays:
            weekday_mask = []
            for i in range(num_days):
                d = start_date + datetime.timedelta(days=i)
                weekday_mask.append(d.weekday() < 5)
            weekday_mask = np.array(weekday_mask)
            day_tensor = day_tensor[:, weekday_mask, :, :]

        return day_tensor.permute(1, 0, 2, 3)  # (num_days, num_nodes, day_len, feat_dim)

    def build_dataset(day_data_dict, node_indices=None):
        dataset_list, node_list, domain_list = [], [], []
        domains = list(day_data_dict.keys())

        for domain_idx, domain in enumerate(domains):
            data = day_data_dict[domain]  # (D, N, L, C)
            if node_indices is not None:
                data = data[:, node_indices, :, :]
            else:
                node_indices = range(data.shape[1])

            D, N, _, _ = data.shape
            dataset_list.append(data)

            node_list.append(
                torch.tensor(node_indices, device=args.device).unsqueeze(0).expand(D, -1)
            )
            domain_list.append(
                torch.full((D, N), domain_idx, device=args.device)
            )

        dataset = torch.cat(dataset_list, dim=1)
        node_ids = torch.cat(node_list, dim=1)
        domain_ids = torch.cat(domain_list, dim=1)
        return dataset, node_ids, domain_ids

    # =========================================================
    # 0. 准备 domain 全集 + 自动 source domain
    # =========================================================
    all_domains = ['bike_flow', 'taxi_flow', 'service', 'crime']
    # all_domains = ['bike_flow', 'taxi_flow']
    target_domain = args.target_domain
    source_domains = [d for d in all_domains if d != target_domain]

    day_lengths = {
        'bike_flow': args.seq_len_long,
        'taxi_flow': args.seq_len_long,
        'service': args.seq_len_short,
        'crime': args.seq_len_short
    }

    # =========================================================
    # 1. 区域划分
    # =========================================================
    train_ids, test_ids = load_region_split(args.split_rule, args.split_ratio)
    df_selected = pd.read_csv('/data/wangziang/UniDiff/data/bike_flow_area_sum.csv')['area_id'].tolist()
    id_to_index = {id_: i for i, id_ in enumerate(df_selected)}
    train_indices = [id_to_index[i] for i in train_ids]
    test_indices = [id_to_index[i] for i in test_ids]

    start_date = datetime.date(2024, 1, 1)

    # =========================================================
    # 2. 加载全部 domain 数据（不 split，用来做 scaler fit）
    # =========================================================
    raw_data = {}
    for domain in all_domains:
        raw_data[domain] = load_domain_data(domain, args.dataset).transpose(0, 1)
        # shape (N, T, 1)

    # =========================================================
    # 3. 计算 StandardScaler（四个 domain 都返回）
    # =========================================================
    # =========================================================
    # 计算 global scaler（四个 domain 的 T 不同也没关系）
    # =========================================================

    all_values = []

    for domain in all_domains:
        tensor = raw_data[domain]  # (N, T, 1)
        flat = tensor.reshape(-1, 1)  # 所有值拉成一列: (N*T, 1)
        all_values.append(flat)

    # 拼接所有域的所有值
    concat = torch.cat(all_values, dim=0)  # (sum(N*T), 1)

    # 计算全局均值和方差
    global_mean = concat.mean(dim=0, keepdim=True)  # (1, 1)
    global_std = concat.std(dim=0, keepdim=True)  # (1, 1)

    scaler = StandardScaler(global_mean, global_std)

    # 对每个 domain 使用同一个 scaler
    for domain in all_domains:
        raw_data[domain] = scaler.transform(raw_data[domain])

    # =========================================================
    # 4. split_by_day + 构建 pretrain/fine-tune/test Loader
    # =========================================================

    # ---------- Pretrain (source domains, all nodes) ----------
    source_day_dict = {
        domain: split_by_day(raw_data[domain], day_lengths[domain], start_date)
        for domain in source_domains
    }
    pretrain_loaders = {}
    for domain in source_domains:
        day_tensor = source_day_dict[domain]  # (D, N, L, C)
        data, nid, did = build_dataset({domain: day_tensor})
        pretrain_loaders[domain] = Data.DataLoader(
            Data.TensorDataset(data, nid, did),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

    pretrain_loaders_test = {}
    for domain in source_domains:
        day_tensor = source_day_dict[domain]  # (D, N, L, C)
        data, nid, did = build_dataset({domain: day_tensor})
        pretrain_loaders_test[domain] = Data.DataLoader(
            Data.TensorDataset(data, nid, did),
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers
        )

    # ---------- Fine-tune (target domain, train nodes) ----------
    ft_day = split_by_day(raw_data[target_domain], day_lengths[target_domain], start_date)
    ft_data, ft_nid, ft_did = build_dataset({target_domain: ft_day}, node_indices=train_indices)
    ft_loader = Data.DataLoader(
    # pretrain_loaders[target_domain] = Data.DataLoader(
        Data.TensorDataset(ft_data, ft_nid, ft_did),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    ft_loader_test = Data.DataLoader(
    # pretrain_loaders_test[target_domain] = Data.DataLoader(
        Data.TensorDataset(ft_data, ft_nid, ft_did),
        batch_size=1, shuffle=True, num_workers=args.num_workers
    )

    # ---------- Test (target domain, test nodes) ----------
    test_day = split_by_day(raw_data[target_domain], day_lengths[target_domain], start_date)
    test_data, test_nid, test_did = build_dataset({target_domain: test_day}, node_indices=test_indices)
    test_loader = Data.DataLoader(
        Data.TensorDataset(test_data, test_nid, test_did),
        batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    # ---------- Test for Prediction (target domain, all nodes) ----------
    test_pre_day = split_by_day(raw_data[target_domain], day_lengths[target_domain], start_date)
    test_pre_data, test_pre_nid, test_pre_did = build_dataset({target_domain: test_pre_day})
    test_pre_loader = Data.DataLoader(
        Data.TensorDataset(test_pre_data, test_pre_nid, test_pre_did),
        batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    # =========================================================
    # 返回三个 dataloader + 四个 scaler
    # =========================================================
    return pretrain_loaders, ft_loader, test_loader, scaler, id_to_index, pretrain_loaders_test, ft_loader_test, test_pre_loader
    # return pretrain_loaders, test_loader, scaler, id_to_index, pretrain_loaders_test
