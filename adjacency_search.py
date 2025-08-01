import scipy.sparse as sp
import torch
import dgl
import numpy as np
from tqdm import tqdm
import random
from typing import List, Union, Tuple
import numba
from scipy.sparse import csr_matrix, coo_matrix
from torch_geometric.utils import coalesce
import random
import time
import dgl
import networkx as nx

class MAS:
    def __init__(self,
                 g,
                 l=10,
                 p = 1,
                 num_start_nodes=5,
                 num_permutations=5) -> None:
        self.g = g
        self.l = l
        self.p = p
        self.num_start_nodes = num_start_nodes
        self.num_permutations = num_permutations
        self.num_nodes = len(self.g.nodes())
        self.ordered_neighbors = {v: None for v in g.nodes()}
        self.runtime = None
        self.run_MAS()
        self.ordered_neighbors_temp = self.ordered_neighbors.copy()

    def run_MAS(self):
        start_time = time.time()
        progress_bar = tqdm(total=len(self.g.nodes()))
        for v in self.g.nodes():
            neighbours = list(nx.all_neighbors(self.g, v))
            len_n = len(neighbours)
            counter = {i: 0 for i in range(self.num_nodes)}
            if len_n > 0:
                for p in range(self.num_permutations):
                    random.shuffle(neighbours)
                    for i in range(0, len_n, self.num_start_nodes):
                        start = min(i, max(0, len_n - self.num_start_nodes))
                        selected_nodes = [v] + neighbours[start : start + self.num_start_nodes]
                        selected_nodes = self.MAS_inner(selected_nodes)
                        array = np.random.choice([0, 1], size=len(selected_nodes), p=[1 - self.p, self.p])
                        selected_nodes = [n for (n,s) in list(zip(selected_nodes,array)) if s == 1]
                        activated_neighbors = [n for n in selected_nodes if n != v]
                        counter = {key: value + 1 if key in activated_neighbors else value for key, value in counter.items()}
            items = list(counter.items())
            random.shuffle(items)
            counter = dict(items)
            self.ordered_neighbors[v] = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
            progress_bar.update(1)
        progress_bar.close()
        self.runtime = time.time() - start_time

    def MAS_inner(self, nodes):
        count_dict = {}
        activated = set(nodes)
        num_inserted = 0
        for v in nodes:
            for n in list(nx.all_neighbors(self.g, v)):
                if n not in activated:
                    if n not in count_dict:
                        count_dict[n] = 0
                    count_dict[n] += 1
        if len(count_dict) == 0:
            return nodes
        while len(count_dict)>0 and num_inserted<self.l:
            selected = max(count_dict, key=count_dict.get)
            del count_dict[selected]
            activated.add(selected)
            num_inserted += 1
            for n in list(nx.all_neighbors(self.g, selected)):
                if n not in activated:
                    if n not in count_dict:
                        count_dict[n] = 0
                    count_dict[n] += 1
        return activated

    def update_dictionary(self, dictionary):
        self.ordered_neighbors_temp = dictionary

class TAS:
    def __init__(self,
                 g,
                 l=10,
                 p = 1,
                 num_start_nodes=5,
                 num_permutations=5,
                 threshold=5) -> None:
        self.g = g
        self.l = l
        self.p = p
        self.num_start_nodes = num_start_nodes
        self.num_permutations = num_permutations
        self.threshold = threshold
        self.num_nodes = len(self.g.nodes())
        self.ordered_neighbors = {v: None for v in g.nodes()}
        self.runtime = None
        self.run_TAS()
        self.ordered_neighbors_temp = self.ordered_neighbors.copy()

    def run_TAS(self):
        start_time = time.time()
        progress_bar = tqdm(total=len(self.g.nodes()))
        for v in self.g.nodes():
            neighbours = list(nx.all_neighbors(self.g, v))
            len_n = len(neighbours)
            counter = {i: 0 for i in range(self.num_nodes)}
            if len_n > 0:
                for t in range(1,self.threshold+1):
                    for p in range(self.num_permutations):
                        random.shuffle(neighbours)
                        for i in range(0, len_n, self.num_start_nodes):
                            start = min(i, max(0, len_n - self.num_start_nodes))
                            selected_nodes = [v] + neighbours[start : start + self.num_start_nodes]
                            selected_nodes = self.TAS_inner(selected_nodes,t)
                            array = np.random.choice([0, 1], size=len(selected_nodes), p=[1 - self.p, self.p])
                            selected_nodes = [n for (n,s) in list(zip(selected_nodes,array)) if s == 1]
                            activated_neighbors = [n for n in selected_nodes if n != v]
                            counter = {key: value + 1 if key in activated_neighbors else value for key, value in counter.items()}
            items = list(counter.items())
            random.shuffle(items)
            counter = dict(items)
            self.ordered_neighbors[v] = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
            progress_bar.update(1)
        progress_bar.close()
        self.runtime = time.time() - start_time

    def TAS_inner(self, nodes, t):
        T = [min(self.g.degree(u), t) if u not in nodes else 0 for u in range(len(self.g.nodes()))]
        I = nodes
        activated = set(nodes)
        num_inserted = 0
        while len(I)>0 and num_inserted<self.l:
            i = I.pop(0)
            activated.add(i)
            num_inserted += 1
            for n in list(nx.all_neighbors(self.g, i)):
                if n not in I:
                    T[n] -= 1
                    if T[n] == 0:
                        I.append(n)
        return activated
    
    def update_dictionary(self, dictionary):
        self.ordered_neighbors_temp = dictionary

def update_score2(AS_object):
    dictionary = {}
    max_value = 0
    for v in AS_object.g.nodes():
        max_value = max(max(AS_object.ordered_neighbors[v].values()),max_value)
    for v in AS_object.g.nodes():
        dictionary[v] = {key: value/max_value if max_value>0 else 0 for key, value in AS_object.ordered_neighbors[v].items()}
    AS_object.update_dictionary(dictionary)

def update_score3(AS_object):
    dictionary = {}
    for v in AS_object.g.nodes():
        max_value = max(AS_object.ordered_neighbors[v].values())
        dictionary[v] = {key: value/max_value if max_value>0 else 0 for key, value in AS_object.ordered_neighbors[v].items()}
    AS_object.update_dictionary(dictionary)

def get_weights(AS_object,auxiliary_graph):
    edges = auxiliary_graph.edges()
    edges = list(zip(edges[0].tolist(), edges[1].tolist()))
    edges = set([(u,v) if u<v else (v,u) for u,v in edges])
    weights = {i: {} for i in range(AS_object.num_nodes)}
    for u,v in edges:
        weights[u][v] = (AS_object.ordered_neighbors_temp[u][v] + AS_object.ordered_neighbors_temp[v][u]) / 2
        weights[v][u] = weights[u][v]
    for v in AS_object.g.nodes():
        weights[v] = dict(sorted(weights[v].items(), key=lambda item: item[1], reverse=True))
    return weights

def get_auxiliary_graph(AS_object, k):
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(AS_object.g.nodes())
    for v in AS_object.g.nodes():
        non_zero_items = {key: value for key, value in AS_object.ordered_neighbors_temp[v].items() if value != 0 and key != v}
        num_selected = min(k, len(non_zero_items))
        source_nodes = [v]*num_selected
        selected_neighbors = list(non_zero_items.keys())[:num_selected]
        G_nx.add_edges_from(zip(source_nodes, selected_neighbors))
    G_nx = G_nx.to_undirected()
    while not nx.is_connected(G_nx):
        connected_components = list(nx.connected_components(G_nx))
        random_numbers = np.random.choice(np.arange(len(connected_components)), size=2, replace=False)
        first_index = np.random.choice(np.arange(len(connected_components[random_numbers[0]])))
        second_index = np.random.choice(np.arange(len(connected_components[random_numbers[1]])))
        first_node = list(connected_components[random_numbers[0]])[first_index]
        second_node = list(connected_components[random_numbers[1]])[second_index]
        G_nx.add_edge(first_node, second_node)
    return dgl.from_networkx(G_nx)
