from data import get_dataset, get_NAG_data
import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import TransformerModel
from lr import PolynomialDecayLR
import torch.utils.data as Data
import os
import pickle
from adjacency_search import MAS
import dgl
import networkx as nx
import pandas as pd

def parse_args():
    '''
    Generate a parameters parser.
    '''
    # Parse parameters.
    parser = argparse.ArgumentParser()

    # Main parameters.
    parser.add_argument('--save_to', type=str, default='results', help='Result folder.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--max_auxiliary_graph_degree', type=int, default=10, help='Maximum degree of the auxiliary graph constructed from adjacency search.')

    return parser.parse_args()

args = parse_args()

datasets = ["actor","chameleon","citeseer","community","cora","cornell","cycle","grid","photo","shape","squirrel","texas","wisconsin", "computer", "pubmed", "wiki"]
strongly = []
weakly = []

if os.path.exists("graph_connectivity_data_MAS_"+args.save_to+".pkl"):
    with open("graph_connectivity_data_MAS_"+args.save_to+".pkl", "rb") as f:
        data = pickle.load(f)
    strongly, weakly = data
    start = len(strongly)
else:
    start = 0
for i in range(start,len(datasets)):
    dataset = datasets[i]
    print(dataset)
    count_strongly = 0
    count_weakly = 0
    for split_seed in range(20):
        if not os.path.exists(args.save_to):
            os.makedirs(args.save_to)
        if not os.path.exists(args.save_to+'/results'):
            os.makedirs(args.save_to+'/results')
        if not os.path.exists(args.save_to+'/results/exp1_'+str(split_seed)):
            os.makedirs(args.save_to+'/results/exp1_'+str(split_seed))
        if not os.path.exists(args.save_to+'/saved_adjacency_search'):
            os.makedirs(args.save_to+'/saved_adjacency_search')

        start_time1 = time.time()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        # Pre-process data.
        filename_adjacency_search = args.save_to+'/saved_adjacency_search/'+dataset+'_mas.pkl'
        if os.path.exists(filename_adjacency_search):
            with open(filename_adjacency_search, 'rb') as f:
                AS_object = pickle.load(f)
        auxiliary_graph = AS_object.get_auxiliary_graph(args.max_auxiliary_graph_degree).to_networkx()
        if nx.is_strongly_connected(auxiliary_graph):
            count_strongly += 1
        if nx.is_weakly_connected(auxiliary_graph):
            count_weakly += 1
    strongly.append(count_strongly/20)
    weakly.append(count_weakly/20)
    print(strongly[-1], weakly[-1])
    data = strongly,weakly
    with open("graph_connectivity_data_MAS_"+args.save_to+".pkl", "wb") as f:
        pickle.dump(data, f)

    data = {
        "Dataset": datasets[:len(strongly)],
        "Strongly Connected": strongly,
        "Weakly Connected": weakly
    }

    df = pd.DataFrame(data)
    df.to_excel("graph_connectivity_data_MAS_"+args.save_to+".xlsx", index=False)

data = {
    "Dataset": datasets[:len(strongly)],
    "Strongly Connected": strongly,
    "Weakly Connected": weakly
}

df = pd.DataFrame(data)
df.to_excel("graph_connectivity_data_MAS_"+args.save_to+".xlsx", index=False)