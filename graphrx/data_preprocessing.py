import pandas as pd
import torch
import dgl
import numpy as np
from collections import defaultdict

def load_data():
    """
    Loads the DRKG dataset and returns triplets.
    """
    df = pd.read_csv('drkg.tsv', sep='\t', header=None, names=['head', 'relation', 'tail'])
    entities_df = pd.read_csv("embed/entities.tsv", sep="\t", header=None)
    entities_df.columns = ["id", "name"]
    relations_df = pd.read_csv("embed/relations.tsv", sep="\t", header=None)
    relations_df.columns = ["id", "name"]
    
    return df, entities_df, relations_df

def construct_heterograph(df):
    """
    Constructs a heterogeneous graph from the DRKG triplets.
    """
    def get_entity_type(entity_str):
        return entity_str.split("::")[0]

    node_dict = defaultdict(set)
    edge_dict = defaultdict(list)
    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]
        h_type, t_type = get_entity_type(h), get_entity_type(t)
        node_dict[h_type].add(h)
        node_dict[t_type].add(t)
        edge_dict[(h_type, r, t_type)].append((h, t))

    entity_id_maps = {}
    for ntype, entities in node_dict.items():
        entity_id_maps[ntype] = {e: i for i, e in enumerate(sorted(entities))}

    graph_data = {}
    for (src_type, rel, dst_type), edge_list in edge_dict.items():
        src_ids = np.array([entity_id_maps[src_type][h] for h, _ in edge_list])
        dst_ids = np.array([entity_id_maps[dst_type][t] for _, t in edge_list])
        graph_data[(src_type, rel, dst_type)] = (torch.from_numpy(src_ids), torch.from_numpy(dst_ids))

    hg = dgl.heterograph(graph_data)
    return hg

def preprocess_data():
    """
    Load and preprocess the data for GNN training.
    """
    df, entities_df, relations_df = load_data()
    hg = construct_heterograph(df)
    return hg
