from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiPolygon
import setproctitle
import os

setproctitle.setproctitle('wza_UniDiff_KG_Filtered')

#####################################
# 1. 读取 Borough & Area
#####################################

dataframe1 = gpd.read_file('./Meta_data/NYC/Administrative_data/Borough/Borough.shp', engine="pyogrio")
dataframe1 = dataframe1.to_crs('EPSG:4326')
borough_dataframe = dataframe1[['BoroCode', 'BoroName', 'geometry']]

dataframe2 = gpd.read_file('./Meta_data/NYC/Administrative_data/Area/Area.shp', engine="pyogrio")
dataframe2 = dataframe2.to_crs('EPSG:4326')
area_dataframe = dataframe2[['OBJECTID', 'zone', 'geometry']]

# 过滤过小区域
area_dataframe = area_dataframe[~area_dataframe['OBJECTID'].isin([1, 103, 104])]

#####################################
# 2. 读取目标区域列表并筛选 ### MODIFIED
#####################################
target_area = pd.read_csv('/data/wangziang/UniDiff/data/bike_flow_area_sum.csv')  # 列名为 area_id
target_ids = set(target_area['area_id'].tolist())
print(f"area num: {len(target_ids)}")

# 筛选 area
area_dataframe = area_dataframe[area_dataframe['OBJECTID'].isin(target_ids)].reset_index(drop=True)

#####################################
# Relation 1: Borough Nearby Borough (BNB)
#####################################
BNB = []
for i in tqdm(range(borough_dataframe.shape[0]), desc='BNB'):
    head_borough = borough_dataframe.iloc[i].geometry
    for j in range(borough_dataframe.shape[0]):
        tail_borough = borough_dataframe.iloc[j].geometry
        if head_borough.touches(tail_borough):
            BNB.append(f"Borough/{borough_dataframe.iloc[i].BoroCode} BNB Borough/{borough_dataframe.iloc[j].BoroCode}")

#####################################
# Relation 2: Area Nearby Area (ANA) ### MODIFIED
#####################################
ANA = []
for i in tqdm(range(area_dataframe.shape[0]), desc='ANA'):
    head_area = area_dataframe.iloc[i].geometry
    for j in range(area_dataframe.shape[0]):
        tail_area = area_dataframe.iloc[j].geometry
        if head_area.touches(tail_area):
            ANA.append(f"Area/{area_dataframe.iloc[i].OBJECTID} ANA Area/{area_dataframe.iloc[j].OBJECTID}")

#####################################
# Relation 3–5: POI Relations (filtered by area_id)
#####################################
PLA, PHPC = [], []
poi_dataframe = pd.read_csv('./Processed_data/NYC/NYC_poi.csv')

# 仅保留目标区域内的 POI ### MODIFIED
poi_dataframe = poi_dataframe[poi_dataframe['area_id'].isin(target_ids)]
poi_datanumpy = poi_dataframe[['poi_id', 'borough_id', 'area_id', 'cate']].values

for poi_id, bor_id, area_id, cate in tqdm(poi_datanumpy, desc='POI relations'):
    PLA.append(f"POI/{poi_id} PLA Area/{area_id}")
    PHPC.append(f"POI/{poi_id} PHPC PC/{cate}")

#####################################
# Relation 6–8: Road Relations (filtered by area_id)
#####################################
RLA, RHRC = [], []
road_dataframe = pd.read_csv('./Processed_data/NYC/NYC_road.csv')
road_dataframe = road_dataframe[road_dataframe['area_id'].isin(target_ids)]
road_datanumpy = road_dataframe[['link_id', 'borough_id', 'area_id', 'link_type_name']].values

for link_id, bor_id, area_id, link_type in tqdm(road_datanumpy, desc='Road relations'):
    RLA.append(f"Road/{link_id} RLA Area/{area_id}")
    RHRC.append(f"Road/{link_id} RHRC RC/{link_type}")

#####################################
# Relation 9–11: Junction Relations (filtered by area_id)
#####################################
JLA, JHJC = [], []
junction_dataframe = pd.read_csv('./Processed_data/NYC/NYC_junction.csv')
junction_dataframe = junction_dataframe[junction_dataframe['area_id'].isin(target_ids)]
junction_datanumpy = junction_dataframe[['node_id', 'borough_id', 'area_id', 'osm_highway']].values

for node_id, bor_id, area_id, junc_type in tqdm(junction_datanumpy, desc='Junction relations'):
    JLA.append(f"Junction/{node_id} JLA Area/{area_id}")
    JHJC.append(f"Junction/{node_id} JHJC JC/{junc_type}")

#####################################
# Relation 12: Junction Belongs to Road (JBR)
#####################################
JBR = []
road_datanumpy2 = road_dataframe[['from_node_id', 'to_node_id', 'link_id']].values
for from_node, to_node, link_id in tqdm(road_datanumpy2, desc='JBR'):
    JBR.append(f"Junction/{from_node} JBR Road/{link_id}")
    JBR.append(f"Junction/{to_node} JBR Road/{link_id}")

#####################################
# Relation 13: Area Locates at Borough (ALB)
#####################################
ALB = []
for i in tqdm(range(area_dataframe.shape[0]), desc='ALB'):
    area_geom = area_dataframe.iloc[i].geometry
    area_id = area_dataframe.iloc[i].OBJECTID
    for j in range(borough_dataframe.shape[0]):
        bor_geom = borough_dataframe.iloc[j].geometry
        if area_geom.within(bor_geom) or area_geom.intersects(bor_geom):
            ALB.append(f"Area/{area_id} ALB Borough/{borough_dataframe.iloc[j].BoroCode}")

#####################################
# Combine and Save
#####################################
relations = []
relations.extend(PLA)
relations.extend(RLA)
relations.extend(JLA)
relations.extend(ALB)
relations.extend(JBR)
relations.extend(BNB)
relations.extend(ANA)
relations.extend(PHPC)
relations.extend(RHRC)
relations.extend(JHJC)

os.makedirs('./UrbanKG/NYC', exist_ok=True)
with open('./UrbanKG/NYC/UrbanKG_NYC.txt', 'w') as f:
    for r in relations:
        f.write(r + '\n')

print(f"Urban KG constructed successfully, {len(relations)} triples.")
