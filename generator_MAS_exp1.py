from data import get_dataset
import time
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
from adjacency_search import MAS, get_auxiliary_graph
import dgl

def parse_args():
    '''
    Generate a parameters parser.
    '''
    # Parse parameters.
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    # parser.add_argument('--l', type=int, default=10, help='Maximum length of walks.')
    parser.add_argument('--p', type=float, default=1.0, help='Lazy activation parameter.')
    parser.add_argument('--num_permutations', type=int, default=5, help='Number of permutations.')
    # parser.add_argument('--max_auxiliary_graph_degree', type=int, default=10, help='Maximum degree of the auxiliary graph constructed from adjacency search.')
    # parser.add_argument('--split_seed', type=int, default=0, help='Split seed.')
    parser.add_argument('--train_size', type=float, default=0.5, help='Training proportion.')
    parser.add_argument('--val_size', type=int, default=0.25, help='Validation proportion.')
    
    return parser.parse_args()

args = parse_args()
print(args)

for split_seed in range(1):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Pre-process data.
    if "synthesized" in args.dataset:
        with open("/home/myl/notebooks/graphormer/cascade_rewired1/correct/synthesized/graph_data.pkl", 'rb') as f:
            graphs, all_labels, all_features = pickle.load(f)
        dataset = "synthesized"
        for i in range(len(graphs)):
            print(i)
            graph = graphs[i]
            if not os.path.exists(os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)):
                os.makedirs(os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p))
            filename_adjacency_search = os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)+'/'+dataset+str(i)+'_mas.pkl'
            if not os.path.exists(filename_adjacency_search):
                AS_object = MAS(graph,p=args.p,num_permutations=args.num_permutations)
                with open(filename_adjacency_search, 'wb') as f:
                    pickle.dump(AS_object, f)
    else:
        graph, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.train_size, args.val_size, split_seed)
        graph = dgl.to_networkx(graph).to_undirected()
        k = int(sum(dict(graph.degree()).values()) / len(graph.nodes()))
        if not os.path.exists(os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)):
            os.makedirs(os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p))
        filename_adjacency_search = os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)+'/'+args.dataset+'_mas.pkl'
        if not os.path.exists(filename_adjacency_search):
            AS_object = MAS(graph,p=args.p,num_permutations=args.num_permutations)
            with open(filename_adjacency_search, 'wb') as f:
                pickle.dump(AS_object, f)