import os
import torch
import random
import numpy as np
from torch_geometric.data import Data

def one_hot_aminoacids(mutation, wildtype):
    sequence = "ARNDCEQGHILKMFPSTWYV"
    one_hot = []

    for aa in sequence:
        if aa == mutation:
            one_hot.append(1)
        elif aa == wildtype:
            one_hot.append(-1)
        else:
            one_hot.append(0)

    return torch.tensor(one_hot, dtype=torch.float32)


def networkx_dict_to_data(networkx_list, indices):
    data_networkx = []

    for networkx_dict in networkx_list:
        edge_index = []
        edge_attr = []
        for src, targets in networkx_dict.items():
            for target, edge_data in targets.items():
                edge_index.append([src, target])
                edge_attr.append(edge_data['weight'])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

        data_networkx.append(Data(edge_index=edge_index, edge_attr=edge_attr, **{'indices': indices}))

    return data_networkx

def seed_torch(seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.