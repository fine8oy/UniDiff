# train_stage1_llava.py
# Stage1: align KG embeddings and satellite tile embeddings into a unified LLM embedding space
# Offline mode: expects local copies of weights at VISION_PATH, LLM_PATH, LLAVA_PATH

import os
import sys
import math
from tqdm import tqdm
from datetime import datetime
import random
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    logging as hf_logging,
    AutoModel,
)

hf_logging.set_verbosity_error()

from torch_geometric.nn import GATConv
from torch_geometric.data import Data as GeoData

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import setproctitle
from llava.model import *

# ------------------------------
# === USER CONFIG / PATHS ====
# ------------------------------
DEVICE_KG = torch.device('cuda:xxx')      # GAT, CLIP, Coord encoder, training part
DEVICE_LLM = torch.device('cuda:xxx')     # LLaVA/Vicuna large model


# Local weight directories (you confirmed)
VISION_PATH = "/data/wangziang/UniDiff/llava_cpt/clip-vit-large-patch14"  # CLIP vision model local repo
LLM_PATH = "/data/wangziang/UniDiff/llava_cpt/vicuna-7b-v1.1"  # LLM local repo (only for hidden dim & tokenizer)
LLAVA_PATH = "/data/wangziang/UniDiff/llava_cpt/LLaVA-7B-v1"  # optional: multimodal projector weights (safetensors or pt)

# Data / resources (adjust to your project)
IMAGE_ROOT = 'UniDiff/ny_tiles/'
TARGET_AREA_CSV = 'UniDiff/data/bike_flow_area_sum.csv'
POI_CSV = 'UniDiff/UrbanKG_data/Processed_data/NYC/NYC_poi.csv'
URBKG_TXT = 'UniDiff/UrbanKG_data/UrbanKG/NYC/UrbanKG_NYC.txt'
ENTITY2ID = 'UniDiff/UrbanKG_data/UrbanKG/NYC/entity2id_NYC.txt'
REL2ID = 'UniDiff/UrbanKG_data/UrbanKG/NYC/relation2id_NYC.txt'
AREA_SHP = 'UniDiff/UrbanKG_data/Meta_data/NYC/Administrative_data/Area/Area.shp'

# Training config
BATCH_SIZE = 151  # 151 regions
LR = 1e-4
EPOCHS = 999
CLIP_FINETUNE = False
PROJECTOR_INIT_FROM_LLAVA = True  # try to load LLAVA projector if available
SAVE_EVERY = 100
TSNE_EVERY = 100
EARLY_STOP_PATIENCE = 50

BETA = 0.95     # semantic anchoring strength
TEMP = 0.07

# Logging dir
exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"./logs_spatial/exp_beta={BETA}_{exp_id}"
os.makedirs(EXP_DIR, exist_ok=True)
print("Experiment dir:", EXP_DIR)


# ------------------------------
# === Utility helpers ===
# ------------------------------
def plot_heatmap(matrix, title, fname):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_tsne(e_kg_all, e_img_all, epoch, fname):
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    z_kg = tsne.fit_transform(e_kg_all)
    z_img = tsne.fit_transform(e_img_all)
    plt.figure(figsize=(6, 6))
    plt.scatter(z_kg[:, 0], z_kg[:, 1], c='blue', alpha=0.6, label='KG')
    plt.scatter(z_img[:, 0], z_img[:, 1], c='red', alpha=0.6, label='IMG')
    for i in range(len(z_kg)):
        plt.plot([z_kg[i, 0], z_img[i, 0]], [z_kg[i, 1], z_img[i, 1]], c='gray', alpha=0.2)
    plt.legend()
    plt.title(f"t-SNE Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# ------------------------------
# === Data preparation ===
# ------------------------------
print("Loading area shapefile...")
area_df = gpd.read_file(AREA_SHP, engine="pyogrio").to_crs('EPSG:4326')[['OBJECTID', 'geometry']]
target_area = pd.read_csv(TARGET_AREA_CSV)
target_ids = set(target_area['area_id'].tolist())
area_df = area_df[area_df['OBJECTID'].isin(target_ids)].reset_index(drop=True)
print(f"Total target areas: {len(area_df)}")

# tile -> area matching (center-in-polygon), fallback to nearest tile for missing areas
print("Finding tiles and matching to areas...")
tile_paths = [f for f in os.listdir(IMAGE_ROOT) if f.endswith('.jpg')]
tile_records = []
# parse filenames like "15_40.476203_-73.701782.jpg" or similar
for fname in tqdm(tile_paths, desc="scanning tiles"):
    p = os.path.join(IMAGE_ROOT, fname)
    try:
        base = os.path.splitext(fname)[0]
        parts = base.split('_')
        # expect format zoom_lat_lon or zoom_long_lat depending on your naming
        # assume zoom_lat_lon: parts = [zoom, lat, lon]
        if len(parts) >= 3:
            lat = float(parts[-2])
            lon = float(parts[-1])
            point = Point(lon, lat)
            area_id = None
            for _, row in area_df.iterrows():
                if row.geometry.contains(point):
                    area_id = row.OBJECTID
                    break
            if area_id is not None:
                tile_records.append((p, area_id, lat, lon))
    except Exception:
        continue

tile_df = pd.DataFrame(tile_records, columns=['path', 'area_id', 'lat', 'lon'])
matched_areas = set(tile_df['area_id'].tolist())
missing_areas = [aid for aid in area_df['OBJECTID'] if aid not in matched_areas]
print(f"Matched tiles for {len(matched_areas)} areas; missing {len(missing_areas)} areas")

# supplement nearest tile for missing areas
if len(missing_areas) > 0:
    # create list of all tiles coords
    all_tile_coords = []
    for fname in tile_paths:
        try:
            base = os.path.splitext(fname)[0]
            parts = base.split('_')
            if len(parts) >= 3:
                lat = float(parts[-2])
                lon = float(parts[-1])
                all_tile_coords.append((os.path.join(IMAGE_ROOT, fname), lat, lon))
        except Exception:
            continue
    supplement = []
    for aid in tqdm(missing_areas, desc="supplement tiles"):
        row = area_df[area_df['OBJECTID'] == aid].iloc[0]
        centroid = row.geometry.centroid
        c_lat, c_lon = centroid.y, centroid.x
        min_dist = float('inf');
        nearest = None
        for p, tlat, tlon in all_tile_coords:
            d = geodesic((c_lat, c_lon), (tlat, tlon)).meters
            if d < min_dist:
                min_dist = d;
                nearest = (p, aid, tlat, tlon)
        if nearest:
            supplement.append(nearest)
    if len(supplement) > 0:
        sup_df = pd.DataFrame(supplement, columns=['path', 'area_id', 'lat', 'lon'])
        tile_df = pd.concat([tile_df, sup_df], ignore_index=True)
        print(f"Supplemented {len(supplement)} tiles; now have tiles for {tile_df['area_id'].nunique()} areas")

print(f"Total matched satellite images: {len(tile_df)}")

# load POI pivot
poi_df = pd.read_csv(POI_CSV)
poi_df = poi_df[poi_df['area_id'].isin(target_ids)]
poi_pivot = poi_df.groupby(['area_id', 'cate']).size().unstack(fill_value=0)
poi_vecs = poi_pivot.to_dict(orient='index')
poi_cates = sorted(set(poi_df['cate']))
cate2idx = {c: i for i, c in enumerate(poi_cates)}


def get_poi_vector(area_id):
    v = np.zeros(len(cate2idx), dtype=np.float32)
    if area_id in poi_vecs:
        for c, cnt in poi_vecs[area_id].items():
            v[cate2idx[c]] = cnt
    return v


# load KG entities and relations
KG_DIM = 4096

print(f"[KG] Initializing learnable {KG_DIM}-dim embeddings...")

# entity2id 仍然来自 entity2id_NYC.txt
entity2id = {line.strip().split()[0]: int(line.strip().split()[1]) for line in open(ENTITY2ID)}

num_entities = len(entity2id)

# 可训练实体 embedding（4096 dim）
entity_emb = nn.Embedding(num_entities, KG_DIM)
nn.init.xavier_uniform_(entity_emb.weight)

# 找出 area 节点的 index（用于 batch 选取）
area_entities = [k for k in entity2id if k.startswith('Area/')]
area_id2idx = {int(k.split('/')[1]): entity2id[k] for k in area_entities}

print(f"[KG] Total entities: {num_entities}, area entities: {len(area_entities)}")

# 构图（不变）
triples = [line.strip().split() for line in open(URBKG_TXT)]
edges = []
for h, r, t in triples:
    if h in entity2id and t in entity2id:
        edges.append((entity2id[h], entity2id[t]))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE_KG)

x_init = entity_emb.weight  # shape: (num_entities, KG_DIM)

# prepare torch_geometric data
data = GeoData(x=x_init, edge_index=edge_index).to(DEVICE_KG)

# ------------------------------
# === Model components ===
# ------------------------------
print("Loading CLIP vision model (local)...")
# CLIP vision encoder (only)
vision = CLIPVisionModel.from_pretrained(VISION_PATH).to(DEVICE_LLM)    # CLIPVisionModel
# vision = CLIPVisionModel.from_pretrained(VISION_PATH).to(DEVICE_KG)    # CLIPVisionModel
vision.eval()
processor = CLIPImageProcessor.from_pretrained(VISION_PATH)     # CLIPImageProcessor (224, 224)

# Load LLAVA: full multimodal model (includes LLM + projector)
print("Loading LLAVA full model (local)...")
tokenizer = AutoTokenizer.from_pretrained(LLAVA_PATH, use_fast=False)
llava_model = LlavaLlamaForCausalLM.from_pretrained(
    LLAVA_PATH,
    low_cpu_mem_usage=True,
    device_map=str(DEVICE_LLM)
)

LM_HIDDEN = getattr(llava_model.config, "hidden_size", None)
print("LLAVA hidden dim:", LM_HIDDEN)


texts = [
    "The bicycle-flow dataset contains spatio-temporal mobility measurements across 151 administrative regions of New York City throughout 2024. For each region, the data records the number of bicycle trips occurring in every 30-minute interval, forming a fine-grained time series that reflects short-term cycling activity patterns across the city.",
    "The taxi-flow dataset captures citywide transportation dynamics across 151 New York City regions over the full year of 2024. Each entry represents the number of taxi trips within a region during each 30-minute interval, providing high-resolution temporal signals of taxi usage and urban mobility demand.",
    "The crime dataset summarizes public safety conditions across 151 New York City regions in 2024, reporting the number of NYPD-recorded crime incidents within each region every 2 hours. This yields a coarse-grained temporal series that reflects local fluctuations in crime intensity across the city.",
    "The 311 service dataset characterizes civic service demand across 151 New York City regions throughout 2024. It records the count of 311 service requests submitted in each region at a 2-hour interval, capturing variations in community concerns and municipal service needs over time."
]

text_embs = []
max_len = None

for txt in texts:
    enc = tokenizer(txt, return_tensors="pt").to(DEVICE_LLM)
    out = llava_model(
        input_ids=enc.input_ids,
        output_hidden_states=True,
        return_dict=True
    )
    hidden = out.hidden_states[-1].squeeze(0)   # (T, 4096)
    text_embs.append(hidden)

max_len = max(x.shape[0] for x in text_embs)

text_pad = []
for x in text_embs:
    pad_len = max_len - x.shape[0]
    if pad_len > 0:
        x = torch.cat([x, torch.zeros(pad_len, x.shape[1]).to(x.device)], dim=0)
    text_pad.append(x)

text_pad = torch.stack(text_pad)  # (4, max_len, 4096)



# ----------------------------------------------------------------
# === GAT Adapter: 4096 → 4096
# ----------------------------------------------------------------
class KG_GAT_Adapter(nn.Module):
    def __init__(self, in_dim=KG_DIM, hid=1024, out_dim=LM_HIDDEN):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid, heads=4, concat=False)
        self.gat2 = GATConv(hid, hid, heads=4, concat=False)

        # LLaVA LLM hidden size（4096 → 4096）
        self.proj = nn.Sequential(
            nn.Linear(hid, out_dim),
            nn.LayerNorm(out_dim),
            # nn.Tanh()
        )

    def forward(self, x, edge_index):
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        # return self.proj(h)
        return F.normalize(self.proj(h), dim=-1)


gat_adapter = KG_GAT_Adapter(in_dim=KG_DIM, hid=1024, out_dim=LM_HIDDEN).to(DEVICE_KG)


class MLLM_Adapter(nn.Module):
    def __init__(self, in_dim=LM_HIDDEN, out_dim=LM_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


mllm_adapter = MLLM_Adapter(LM_HIDDEN, LM_HIDDEN).to(DEVICE_LLM)

# test
torch.save(mllm_adapter(text_pad), os.path.join(EXP_DIR, f"text_emb.pt"))


# contrastive loss
class CLIPLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp

    def forward(self, e_kg, e_img):
        e_kg = F.normalize(e_kg, dim=-1)
        e_img = F.normalize(e_img, dim=-1)
        logits = torch.matmul(e_kg, e_img.T) / self.temp
        labels = torch.arange(e_kg.size(0), device=e_kg.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


class WeightedCLIPLoss(nn.Module):
    def __init__(self, temp=0.07, beta=0.9):
        """
        beta -> weight for KG->IMG
        (1-beta) -> weight for IMG->KG
        """
        super().__init__()
        self.temp = temp
        self.beta = beta

    def forward(self, e_kg, e_img):
        # e_kg: [N, D]  (GAT-based KG embeddings)
        # e_img: [N, D] (MLLM-anchored image embeddings)

        e_kg = F.normalize(e_kg, dim=-1)
        e_img = F.normalize(e_img, dim=-1)

        logits = torch.matmul(e_kg, e_img.T) / self.temp
        labels = torch.arange(e_kg.size(0), device=e_kg.device)

        # KG -> IMG (pull KG towards IMG semantic space)
        loss_kg2img = F.cross_entropy(logits, labels)

        # IMG -> KG (weaker reverse alignment)
        loss_img2kg = F.cross_entropy(logits.T, labels)

        loss = 0.5 * (
            self.beta * loss_kg2img
            + (1.0 - self.beta) * loss_img2kg
        )

        return loss


loss_fn = WeightedCLIPLoss(temp=TEMP, beta=BETA)

params = list(gat_adapter.parameters()) + list(mllm_adapter.parameters())
if CLIP_FINETUNE:
    params += list(vision.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-6)

# ------------------------------
# === Dataset + Dataloader ===
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])
])


class AreaDataset(Dataset):
    def __init__(self, area_df, tile_df, area_id2idx, transform):
        self.area_df = area_df
        self.tile_df = tile_df
        self.area_id2idx = area_id2idx
        self.area_ids = list(area_df['OBJECTID'])
        self.tile_df = tile_df
        self.transform = transform

    def __len__(self):
        return len(self.area_ids)

    def __getitem__(self, idx):
        aid = self.area_ids[idx]
        tiles = self.tile_df[self.tile_df['area_id'] == aid]
        imgs = []
        coords = []
        for _, r in tiles.iterrows():
            try:
                img = Image.open(r.path).convert('RGB')
                imgs.append(self.transform(img))
                coords.append([r.lat, r.lon])
            except:
                continue
        if len(imgs) == 0:
            imgs = [torch.zeros(3, 224, 224)]
            coords = [[0.0, 0.0]]
        imgs = torch.stack(imgs)  # [N,3,224,224]
        coords = torch.tensor(coords, dtype=torch.float32)  # [N,2]
        poi_vec = torch.tensor(get_poi_vector(aid), dtype=torch.float32)
        return aid, imgs, coords, poi_vec


dataset = AreaDataset(area_df, tile_df, area_id2idx, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda x: x)

# ------------------------------
# === Training loop ===
# ------------------------------
best_loss = float('inf')
no_improve = 0
loss_history = []

for epoch in range(EPOCHS):
    gat_adapter.train()
    mllm_adapter.train()
    if CLIP_FINETUNE:
        vision.train()
    total_loss = 0.0
    # compute full GAT output once per epoch (we let GAT be trained so can't use no_grad)
    # but we can compute e_all each batch after gat_adapter forward
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        area_ids, imgs_list, coords_list, poi_list = zip(*batch)
        B = len(area_ids)
        # prepare KG embeddings for these areas using full entity embedding + gat_adapter
        # forward full entity set through GAT adapter, then select per area
        if batch_idx == 0:
            e_all = gat_adapter(entity_emb.weight.to(DEVICE_KG), data.edge_index.to(DEVICE_KG))
        e_kg_batch = torch.stack([e_all[area_id2idx[aid]] for aid in area_ids]).to(DEVICE_KG)  # [B, LM_HIDDEN]
        # process image tiles per area
        e_img_batch = []
        for imgs, coords, kg_vec in zip(imgs_list, coords_list, e_kg_batch):
            imgs = imgs.to(DEVICE_LLM)  # [N,3,224,224]
            # get CLIP features (we use vision model; optionally fine-tune)
            # 1. 得到 tile_feats
            with torch.no_grad():
                vision_out = vision(imgs)

                if hasattr(vision_out, "pooler_output"):
                    tile_feats = vision_out.pooler_output
                else:
                    tile_feats = vision_out.last_hidden_state.mean(dim=1)

                # 2. 通过 LLaVA 的 projector 得到 language space 特征
                llava_projector = llava_model.model.mm_projector
                tile_proj = llava_projector(tile_feats)  # [N, hidden_size]
                N = tile_proj.shape[0]
                tile_proj = tile_proj.unsqueeze(1)

                dummy_text = " "  # 或者 "image"
                enc = tokenizer(dummy_text, return_tensors="pt")

                input_ids = enc.input_ids.to(DEVICE_LLM).expand(N, -1)

                # 3. 作为 inputs_embeds 输入（注意只能提供 inputs_embeds, 不能给 input_ids）
                outputs = llava_model(
                    input_ids=input_ids,
                    attention_mask=None,
                    inputs_embeds=tile_proj,  # 视觉 token 作为输入
                    output_hidden_states=True,
                    return_dict=True
                )

            tile_emb = outputs.hidden_states[-1].mean(dim=1)
            tile_embs = mllm_adapter(tile_emb)
            # aggregator: KG vector as query, tile_embs as kv
            # kg_vec is [LM_HIDDEN]
            e_img = tile_embs.mean(dim=0)
            e_img_batch.append(e_img)
        e_img_batch = torch.stack(e_img_batch).to(DEVICE_KG)
        # compute CLIP loss
        loss = loss_fn(e_kg_batch, e_img_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += float(loss.item())
    avg_loss = total_loss / max(1, len(loader))
    loss_history.append(avg_loss)
    print(f"Epoch {epoch} avg_loss={avg_loss:.6f}")
    # checkpoint & visualization
    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        with torch.no_grad():
            # ---- 保存 KG embeddings ----
            e_all_kg = gat_adapter(entity_emb.weight.to(DEVICE_KG), data.edge_index.to(DEVICE_KG))
            e_all_kg = e_all_kg.cpu()

            kg_dict = {}
            for aid, idx in area_id2idx.items():
                kg_dict[aid] = e_all_kg[idx]  # (4096)

            torch.save(kg_dict, os.path.join(EXP_DIR, f"kg_emb_epoch{epoch:04d}.pt"))

            # ---- 2. 保存所有 IMG embedding ----
            img_dict = {}

            for (aid, imgs, coords, _) in dataset:
                imgs = imgs.to(DEVICE_LLM)

                vision_out = vision(imgs)
                if hasattr(vision_out, "pooler_output"):
                    tile_feats = vision_out.pooler_output
                else:
                    tile_feats = vision_out.last_hidden_state.mean(dim=1)

                llava_projector = llava_model.model.mm_projector
                tile_proj = llava_projector(tile_feats)
                tile_proj = tile_proj.unsqueeze(1)

                dummy_text = " "
                enc = tokenizer(dummy_text, return_tensors="pt")
                input_ids = enc.input_ids.to(DEVICE_LLM).expand(tile_proj.size(0), -1)

                outputs = llava_model(
                    input_ids=input_ids,
                    inputs_embeds=tile_proj,
                    output_hidden_states=True,
                    return_dict=True
                )

                tile_emb = outputs.hidden_states[-1].mean(dim=1)
                tile_emb = mllm_adapter(tile_emb)
                e_img = tile_emb.mean(dim=0).cpu()

                img_dict[aid] = e_img

            torch.save(img_dict, os.path.join(EXP_DIR, f"img_emb_epoch{epoch:04d}.pt"))

            print(f"[Saved] KG & IMG embeddings at epoch {epoch}")
        # compute similarity matrices on last batch's features
        with torch.no_grad():
            ekg_np = e_kg_batch.cpu().numpy()
            eimg_np = e_img_batch.cpu().numpy()
            plot_tsne(ekg_np, eimg_np, epoch, os.path.join(EXP_DIR, f"tsne_epoch{epoch:04d}.png"))
        torch.save({
            'epoch': epoch,
            'gat_adapter': gat_adapter.state_dict(),
            'optimizer': opt.state_dict(),
            'loss_history': loss_history
        }, os.path.join(EXP_DIR, f"checkpoint_epoch{epoch:04d}.pth"))
        print(f"[Saved] checkpoint_epoch{epoch:04d}.pth")
    # early stop
    if avg_loss + 1e-4 < best_loss:
        best_loss = avg_loss
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered at epoch {epoch} (no improve {EARLY_STOP_PATIENCE})")
        break

# final save
torch.save(mllm_adapter(text_pad), os.path.join(EXP_DIR, f"text_emb.pt"))
