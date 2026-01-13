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
from adjacency_search import TAS, get_auxiliary_graph
import dgl

def parse_args():
    '''
    Generate a parameters parser.
    '''
    # Parse parameters.
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--p', type=float, default=1.0, help='Lazy activation parameter.')
    parser.add_argument('--l', type=int, default=10, help='Maximum length of walks.')
    parser.add_argument('--num_start_nodes', type=int, default=5, help='Number of starting neighbors.')
    parser.add_argument('--num_permutations', type=int, default=5, help='Number of permutations.')
    parser.add_argument(
        '--threshold_list',
        type=lambda s: [int(x) for x in s.replace(" ", "").split(',')],
        default=[1, 2, 3, 4, 5],
        help='Comma-separated list of activation thresholds.'
    )    
    
    return parser.parse_args()

args = parse_args()
print(args)

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
        filename_adjacency_search = os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)+'/'+dataset+str(i)+'_tas.pkl'
        if not os.path.exists(filename_adjacency_search):
            AS_object = TAS(graph,p=args.p,l=args.l,num_start_nodes=args.num_start_nodes,maximum_threshold=args.maximum_threshold,num_permutations=args.num_permutations)
            with open(filename_adjacency_search, 'wb') as f:
                pickle.dump(AS_object, f)
else:
    graph, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, 0.5, 0.25, 0)
    graph = dgl.to_networkx(graph).to_undirected()
    k = int(sum(dict(graph.degree()).values()) / len(graph.nodes()))
    if not os.path.exists(os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)):
        os.makedirs(os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p))
    filename_adjacency_search = os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)+'/'+args.dataset+'_tas.pkl'
    if not os.path.exists(filename_adjacency_search):
        AS_object = TAS(graph,p=args.p,l=args.l,num_start_nodes=args.num_start_nodes,threshold_list=args.threshold_list,num_permutations=args.num_permutations)
        with open(filename_adjacency_search, 'wb') as f:
            pickle.dump(AS_object, f)
