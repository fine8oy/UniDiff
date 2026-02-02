import os, sys
import torch
import numpy as np
import torch.multiprocessing
import logging
from tqdm import tqdm
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger

from UniDiff_model.model import UniDiff
from UniDiff_model.sampler import DiffusionProcess

from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags, plot_compare_sequences, MaximumMeanDiscrepancy3D, cal_smape, adjacency_matrix_to_graph, \
    generate_node2vec_embeddings

from utils.utils_data import gen_dataloader, StandardScaler
from utils.utils_args import parse_args_uncond

from sklearn import metrics

from utils.utils_vis import prepare_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


import datetime


class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, "w", buffering=1)  # 行缓冲
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def enable_global_logging():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"train_log_{timestamp}.txt"
    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout
    print(f"[Global Logger Enabled] All console output is saved to {log_path}")


def run_eval(model, data_loader, kg_emb, img_emb, text_emb, scaler, args, dname=None):
    model.eval()
    gen_sig, real_sig = [], []
    if dname is None:
        img_resolution = args.img_resolution_short if args.target_domain in ['service', 'crime'] else args.img_resolution_long
    else:
        img_resolution = args.img_resolution_short if dname in ['service', 'crime'] else args.img_resolution_long

    with torch.no_grad():
        with model.ema_scope():

            process = DiffusionProcess(args, model.net, (args.input_channels, img_resolution, img_resolution))

            for data in tqdm(data_loader):
                x_ts, node_id, domain_id = data
                node_id = node_id.to(args.device)
                domain_id = domain_id.to(args.device)

                kg_cond = torch.stack([kg_emb[int(nid)] for nid in node_id.flatten()], dim=0).to(args.device)
                img_cond = torch.stack([img_emb[int(nid)] for nid in node_id.flatten()], dim=0).to(args.device)
                text_cond = text_emb[domain_id].to(args.device)
                text_cond = text_cond.reshape(-1, text_cond.shape[2], text_cond.shape[3])

                x_img_gen = process.sampling(kg_cond, img_cond, text_cond, sampling_number=x_ts.shape[1])
                x_ts_gen = model.img_to_ts(x_img_gen).to(args.device)

                x_ts_gen = scaler.inverse_transform(x_ts_gen)
                x_ts_ori = scaler.inverse_transform(x_ts)

                gen_sig.append(x_ts_gen.unsqueeze(0).cpu().numpy())
                real_sig.append(x_ts_ori.cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    ori_sig = np.vstack(real_sig)

    ori_sig_mean = np.mean(ori_sig, axis=0)
    gen_sig_mean = np.mean(gen_sig, axis=0)

    rmse = metrics.mean_squared_error(gen_sig_mean.flatten(), ori_sig_mean.flatten(), squared=False)
    mae = metrics.mean_absolute_error(gen_sig_mean.flatten(), ori_sig_mean.flatten())
    smape = cal_smape(gen_sig_mean.flatten(), ori_sig_mean.flatten())

    mmd3D = MaximumMeanDiscrepancy3D()
    tmpmmds = []
    for i in range(gen_sig.shape[1]):
        realflow = ori_sig[:, i, :, :]
        genflow = gen_sig[:, i, :, :]
        data_1 = realflow.reshape(realflow.shape[0], -1)
        data_2 = genflow.reshape(genflow.shape[0], -1)
        tmpmmds.append(mmd3D(data_1, data_2))
    mmd = np.mean(tmpmmds)

    jsd = jensen_shannon_divergence_nodes(ori_sig, gen_sig)

    return mae, rmse, smape, mmd, jsd


def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    ckpt_dir = os.path.join(args.log_dir, name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        if args.dataset in ['nyc']:
            pretrain_loaders, ft_loader, test_loader, scaler, id_to_index, pretrain_loaders_test, ft_loader_test, test_pre_loader = gen_dataloader(args)

        logging.info(args.dataset + 'dataset is ready.')

        model = UniDiff(args).to(args.device)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        init_epoch = 0

        # ------------------------
        # Load embeddings (stage1 results)
        # ------------------------
        kg_dict = torch.load(f"/data/wangziang/UniDiff/logs_spatial/exp_20251121_110633/kg_emb_epoch{args.mllm_emb_epoch}.pt", map_location=args.device)  # shape (num_nodes, dim)
        img_dict = torch.load(f"/data/wangziang/UniDiff/logs_spatial/exp_20251121_110633/img_emb_epoch{args.mllm_emb_epoch}.pt", map_location=args.device)
        kg_emb = {id_to_index[id_]: tensor.to(args.device) for id_, tensor in kg_dict.items()}
        img_emb = {id_to_index[id_]: tensor.to(args.device) for id_, tensor in img_dict.items()}
        text_emb = torch.load("/data/wangziang/UniDiff/logs_spatial/exp_20251121_110633/text_emb.pt", map_location=args.device)  # shape (num_domains, dim)

        # print model parameters
        print_model_params(logger, model)

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        # -------------------------
        # -------- PRETRAIN -------
        # -------------------------
        print("===> Start Pretraining on Source Domains")
        for epoch in range(args.pretrain_epochs):
            model.train()
            for domain, loader in pretrain_loaders.items():
                for data in loader:
                    x_ts, node_id, domain_id = data
                    B, N, T, C = x_ts.shape
                    x_ts = x_ts.to(args.device).reshape(-1, T, C)   # B*N T C
                    node_id = node_id.to(args.device)
                    domain_id = domain_id.to(args.device)

                    # ---- load 3 conditions ----
                    kg_cond = torch.stack([kg_emb[int(nid)] for nid in node_id.flatten()], dim=0).to(args.device)  # (N, token, dim)
                    img_cond = torch.stack([img_emb[int(nid)] for nid in node_id.flatten()], dim=0).to(args.device)
                    text_cond = text_emb[domain_id].to(args.device)  # (B, N, token, dim)
                    text_cond = text_cond.reshape(-1, text_cond.shape[2], text_cond.shape[3])

                    # ---- convert ts -> img ----
                    x_img = model.ts_to_img(x_ts)
                    loss = model.loss_fn(x_img, kg_cond, img_cond, text_cond)
                    if isinstance(loss, tuple):
                        loss, to_log = loss
                        for key, value in to_log.items():
                            logger.log(f'train/{key}', value, epoch)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()
                    model.on_train_batch_end()

            print(f"Pretrain Epoch {epoch}: loss={loss.item():.4f}")
            if (epoch + 1) % args.test_epoch == 0:
                print(f"[Pretrain] Running eval at epoch {epoch}")

                # 1) 测试集（目标域）
                mae, rmse, smape, mmd, jsd = run_eval(model, test_loader, kg_emb, img_emb, text_emb, scaler, args)
                print(f"[Pretrain-EVAL TestSet] MAE={mae:.4f} RMSE={rmse:.4f} SMAPE={smape:.4f} MMD={mmd:.4f} JSD={jsd:.4f}")

                # 2) 三个源域的训练数据
                for dname, loader in pretrain_loaders_test.items():
                    mae, rmse, smape, mmd, jsd = run_eval(model, loader, kg_emb, img_emb, text_emb, scaler, args, dname)
                    print(
                        f"[Pretrain-EVAL-Source] {dname}:  MAE={mae:.4f} RMSE={rmse:.4f} SMAPE={smape:.4f} MMD={mmd:.4f} JSD={jsd:.4f}")

        # -------------------------
        # -------- FINETUNE -------
        # -------------------------
        print("===> Start Fine-tuning on Target Domain")
        for epoch in range(args.ft_epochs):
            model.train()
            for data in ft_loader:
                x_ts, node_id, domain_id = data
                B, N, T, C = x_ts.shape
                x_ts = x_ts.to(args.device).reshape(-1, T, C)   # B*N T C
                node_id = node_id.to(args.device)
                domain_id = domain_id.to(args.device)

                kg_cond = torch.stack([kg_emb[int(nid)] for nid in node_id.flatten()], dim=0).to(
                    args.device)  # (N, token, dim)
                img_cond = torch.stack([img_emb[int(nid)] for nid in node_id.flatten()], dim=0).to(args.device)
                text_cond = text_emb[domain_id].to(args.device)  # (B, N, token, dim)
                text_cond = text_cond.reshape(-1, text_cond.shape[2], text_cond.shape[3])

                x_img = model.ts_to_img(x_ts)
                loss = model.loss_fn(x_img, kg_cond, img_cond, text_cond)
                if isinstance(loss, tuple):
                    loss, to_log = loss
                    for key, value in to_log.items():
                        logger.log(f'train/{key}', value, epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
                model.on_train_batch_end()

            print(f"Finetune Epoch {epoch}: loss={loss.item():.4f}")
            if (epoch + 1) % args.test_epoch == 0:
                print(f"[Finetune] Running eval at epoch {epoch}")

                # 1) 测试集
                mae, rmse, smape, mmd, jsd = run_eval(model, test_loader, kg_emb, img_emb, text_emb, scaler, args)
                print(f"[Finetune-EVAL TestSet] MAE={mae:.4f} RMSE={rmse:.4f} SMAPE={smape:.4f} MMD={mmd:.4f}  JSD={jsd:.4f}")

                # 2) 目标域训练/验证集
                mae, rmse, smape, mmd, jsd = run_eval(model, ft_loader_test, kg_emb, img_emb, text_emb, scaler, args)
                print(f"[Finetune-EVAL-FinetuneSet] MAE={mae:.4f} RMSE={rmse:.4f} SMAPE={smape:.4f} MMD={mmd:.4f} JSD={jsd:.4f}")

        final_ckpt_path = os.path.join(ckpt_dir, "unidiff_finetune.pt")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "scaler": scaler,
        }, final_ckpt_path)

        print(f"[Checkpoint] Finetuned model saved to {final_ckpt_path}")

        # ------------------------
        # -------- TEST -----------
        # ------------------------
        print("===> Start Testing")
        model.eval()
        gen_sig, real_sig = [], []

        # sampling
        if args.target_domain in ['service', 'crime']:
            img_resolution = args.img_resolution_short
        else:
            img_resolution = args.img_resolution_long

        with torch.no_grad():
            with model.ema_scope():
                process = DiffusionProcess(args, model.net, (args.input_channels, img_resolution, img_resolution))
                for data in tqdm(test_loader):
                    x_ts, node_id, domain_id = data
                    node_id = node_id.to(args.device)
                    domain_id = domain_id.to(args.device)

                    kg_cond = torch.stack([kg_emb[int(nid)] for nid in node_id.flatten()], dim=0).to(args.device)  # (N, token, dim)
                    img_cond = torch.stack([img_emb[int(nid)] for nid in node_id.flatten()], dim=0).to(args.device)
                    text_cond = text_emb[domain_id].to(args.device)  # (B, N, token, dim)
                    text_cond = text_cond.reshape(-1, text_cond.shape[2], text_cond.shape[3])

                    x_img_gen = process.sampling(kg_cond, img_cond, text_cond, sampling_number=x_ts.shape[1], is_test=True)

                    x_ts_gen = model.img_to_ts(x_img_gen).to(args.device)
                    x_ts_gen = scaler.inverse_transform(x_ts_gen)
                    x_ts_ori = scaler.inverse_transform(x_ts)

                    gen_sig.append(x_ts_gen.unsqueeze(0).cpu().numpy())
                    real_sig.append(x_ts_ori.cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                ori_sig = np.vstack(real_sig)
                # 按天平均
                ori_sig_mean = np.mean(ori_sig, axis=0)
                gen_sig_mean = np.mean(gen_sig, axis=0)
                rmse = metrics.mean_squared_error(gen_sig_mean.flatten(), ori_sig_mean.flatten(), squared=False)
                mae = metrics.mean_absolute_error(gen_sig_mean.flatten(), ori_sig_mean.flatten())
                smape = cal_smape(gen_sig_mean.flatten(), ori_sig_mean.flatten())

                # 节点维度平均
                mmd3D = MaximumMeanDiscrepancy3D()
                tmpmmds = []
                for i in range(gen_sig.shape[1]):
                    realflow = ori_sig[:, i, :, :]
                    genflow = gen_sig[:, i, :, :]
                    data_1 = realflow.reshape(realflow.shape[0], -1)
                    data_2 = genflow.reshape(genflow.shape[0], -1)
                    tmpmmds.append(mmd3D(data_1, data_2))
                mmd = np.mean(tmpmmds)

                print('MAE%.4f\tRMSE%.4f\tSMAPE%.4f\tMMD%.4f' % (mae, rmse, smape, mmd))

        logging.info("Testing is complete")

if __name__ == '__main__':
    enable_global_logging()
    args = parse_args_uncond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
