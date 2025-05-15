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
import pickle

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

#### NAG ####

def re_features_NAG(adj, features, K):
    nodes_features = torch.empty(features.shape[0], K+1, features.shape[1])
    x = features.to(torch.float32)
    indices = np.vstack((adj.row, adj.col)).T
    values = (adj.val).numpy()
    adj = np.zeros(adj.shape)
    for i,v in list(zip(indices,values)):
        adj[i[0],i[1]] = v
    adj = torch.tensor(adj, dtype=torch.float32)
    for k in range(K+1):
        for i in range(x.shape[0]):
            nodes_features[i, k, :] = x[i]
        x = torch.matmul(adj, x)
    return nodes_features

### VCR ###

def re_features_push_structure(raw_adj_sp, original_adj, features, hops, K, num_supernodes, cluster_dict, weights=None):
    num_nodes = original_adj.shape[0]
    num_members_in_a_cluster = cluster_dict[0].ndata['_ID'].size(0)
    padded_indices = torch.cat([torch.tensor([cluster_dict[0].ndata['_ID'].tolist(), [num_nodes]*num_members_in_a_cluster]),
                                torch.tensor([[num_nodes]*num_members_in_a_cluster, cluster_dict[0].ndata['_ID'].tolist()])],
                                dim=1)
    padded_values = torch.cat([torch.ones(num_members_in_a_cluster), torch.ones(num_members_in_a_cluster)])
    for i in range(1, num_supernodes):
        num_members_in_a_cluster = cluster_dict[i].ndata['_ID'].size(0)
        padded_indices = torch.cat([padded_indices,
                                    torch.tensor([cluster_dict[i].ndata['_ID'].tolist(), [num_nodes+i]*num_members_in_a_cluster]),
                                    torch.tensor([[num_nodes+i]*num_members_in_a_cluster, cluster_dict[i].ndata['_ID'].tolist()])], 
                                    dim=1)
        padded_values = torch.cat([padded_values, torch.ones(num_members_in_a_cluster), torch.ones(num_members_in_a_cluster)])
    padded_indices = torch.cat([original_adj.coalesce().indices(), padded_indices], dim=1)
    padded_values = torch.cat([original_adj.coalesce().val, padded_values])
    adj = torch.sparse_coo_tensor(indices=padded_indices, values=padded_values, size=(num_nodes + num_supernodes, num_nodes + num_supernodes))
    features = torch.nn.functional.pad(features.to_dense(), (0, 0, 0, num_supernodes), value=0)
    nodes_features = torch.empty(features.shape[0], K+1+hops, features.shape[1]+1)
    raw_num_nodes = raw_adj_sp.shape[0]
    # Renormalize.
    raw_adj_sp = raw_adj_sp + sp.eye(raw_adj_sp.shape[0])
    D1 = np.array(raw_adj_sp.sum(axis=1)) ** (-0.5)
    D2 = np.array(raw_adj_sp.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')
    A = raw_adj_sp.dot(D1)
    raw_adj_sp = D2.dot(A)
    # Sparse matrix to tensor.
    raw_adj_sp = raw_adj_sp.tocoo()
    indices = np.vstack((raw_adj_sp.row, raw_adj_sp.col))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(raw_adj_sp.data)
    raw_adj_sp = torch.sparse.DoubleTensor(i, v, torch.Size(adj.shape))
    raw_adj_sp = raw_adj_sp.to(features.device)
    weight = np.sum(range(1, hops+1))
    x = features
    for k in range(hops):
        x = torch.matmul(raw_adj_sp, x)
        for i in range(raw_num_nodes):
            nodes_features[i, k+1, :] = torch.cat((x[i], torch.tensor([(hops-k)/weight]).to(features.device)))
    if weights == None:
        print('Running PPR (structure-based, virtual connection)...')
        _, top_k_of_node, neighbors_weights_of_node = topk_ppr_matrix(edge_index=adj.coalesce().indices(), num_nodes=adj.shape[0], alpha=0.85, eps=1e-7, output_node_indices = np.arange(raw_num_nodes), topk=K, max_itr=10000, normalization='row')
        print('PPR samples top K for each node (structure-based, virtual connection)...')
        progress_bar = tqdm(total=raw_num_nodes)
        for i in range(raw_num_nodes):
            nodes_features[i, 0, :] = torch.cat((features[i], torch.tensor([1]).to(features.device)))
            topk_indices = list(top_k_of_node[i])
            if len(topk_indices) < K:
                others = [v for v in range(num_nodes) if v not in topk_indices]
                random_neighbors = list(random.sample(others, K-len(topk_indices)))
                topk_indices = topk_indices + random_neighbors
            else:
                topk_indices = topk_indices[:K]
            topk_neighbor_weights = list(neighbors_weights_of_node[i])
            if len(topk_neighbor_weights) < K:
                topk_neighbor_weights = topk_neighbor_weights + [0]*(K-len(topk_neighbor_weights))
            else:
                topk_neighbor_weights = topk_neighbor_weights[:K]
            topk_neighbor_weights = torch.tensor(topk_neighbor_weights).view(-1, 1).to(features.device)
            nodes_features[i, hops+1:, :] = torch.cat((features[topk_indices], topk_neighbor_weights), 1)
            progress_bar.update(1)   
        progress_bar.close()
    else:
        progress_bar = tqdm(total=raw_num_nodes)
        for i in range(raw_num_nodes):
            try:
                nodes_features[i, 0, :] = torch.cat((features[i], torch.tensor([1]).to(features.device)))
                topk_indices = list(weights[i].keys())
                if len(topk_indices) < K:
                    others = [v for v in range(num_nodes) if v not in topk_indices]
                    random_neighbors = list(random.sample(others, K-len(topk_indices)))
                    topk_indices = topk_indices + random_neighbors
                else:
                    topk_indices = topk_indices[:K]
                topk_neighbor_weights = list(weights[i].values())
                if len(topk_neighbor_weights) < K:
                    topk_neighbor_weights = topk_neighbor_weights + [0]*(K-len(topk_neighbor_weights))
                else:
                    topk_neighbor_weights = topk_neighbor_weights[:K]
                topk_neighbor_weights = torch.tensor(topk_neighbor_weights).view(-1, 1).to(features.device)
                nodes_features[i, hops+1:, :] = torch.cat((features[topk_indices], topk_neighbor_weights), 1)
                progress_bar.update(1)
            except:
                print(features[topk_indices].shape, nodes_features[i, hops+1:, :].shape)
                raise ValueError("!")
        progress_bar.close()
    return nodes_features

def re_features_push_content(original_adj, processed_features, features, K, num_labels, labelcluster_to_nodes, weights=None):
    # Adding supernodes to original adj and get adj.
    num_nodes = original_adj.shape[0]
    num_members_in_a_cluster = len(labelcluster_to_nodes[0])
    padded_indices = torch.cat([torch.tensor([labelcluster_to_nodes[0].tolist(), [num_nodes]*num_members_in_a_cluster]),
                                torch.tensor([[num_nodes]*num_members_in_a_cluster, labelcluster_to_nodes[0].tolist()])], dim=1)
    padded_values = torch.cat([torch.ones(num_members_in_a_cluster), torch.ones(num_members_in_a_cluster)])
    for i in range(1, num_labels):
        num_members_in_a_cluster = len(labelcluster_to_nodes[i])
        padded_indices = torch.cat([padded_indices,
                                    torch.tensor([labelcluster_to_nodes[i].tolist(), [num_nodes+i]*num_members_in_a_cluster]),
                                    torch.tensor([[num_nodes+i]*num_members_in_a_cluster, labelcluster_to_nodes[i].tolist()])
                                    ], dim=1)
        padded_values = torch.cat([padded_values, torch.ones(num_members_in_a_cluster), torch.ones(num_members_in_a_cluster)])
    padded_indices = torch.cat([original_adj.coalesce().indices(), padded_indices], dim=1)
    padded_values = torch.cat([original_adj.coalesce().val, padded_values])
    adj = torch.sparse_coo_tensor(indices=padded_indices, values=padded_values, size=(num_nodes + num_labels, num_nodes + num_labels))
    features = torch.nn.functional.pad(features.to_dense(), (0, 0, 0, num_labels), value=0)
    nodes_features = torch.empty(num_nodes, K, processed_features.shape[2])
    if weights == None:
        # Run pagerank.
        print('Running PPR (content-based, virtual connection)...')
        _, top_k_of_node, neighbors_weights_of_node = topk_ppr_matrix(edge_index=adj.coalesce().indices(), num_nodes=adj.shape[0], alpha=0.85, eps=1e-7, output_node_indices = np.arange(num_nodes), topk=K, max_itr=10000, normalization='row')
        print('PPR samples Top K for each node (content-based, virtual connection)...')
        progress_bar = tqdm(total=num_nodes)
        for i in range(num_nodes):
            topk_indices = list(top_k_of_node[i])
            if len(topk_indices) < K:
                others = [v for v in range(num_nodes) if v not in topk_indices]
                random_neighbors = list(random.sample(others, K-len(topk_indices)))
                topk_indices = topk_indices + random_neighbors
            else:
                topk_indices = topk_indices[:K]
            topk_neighbor_weights = list(neighbors_weights_of_node[i]) # first k neighbors' weights
            if len(topk_neighbor_weights) < K:
                topk_neighbor_weights = topk_neighbor_weights + [0]*(K-len(topk_neighbor_weights))
            else:
                topk_neighbor_weights = topk_neighbor_weights[:K]
            topk_neighbor_weights = torch.tensor(topk_neighbor_weights).view(-1, 1).to(features.device)
            nodes_features[i, :, :] = torch.cat((features[topk_indices], topk_neighbor_weights), 1)
            progress_bar.update(1)
        progress_bar.close()
    else:
        progress_bar = tqdm(total=num_nodes)
        for i in range(num_nodes):
            topk_indices = list(weights[i].keys())
            if len(topk_indices) < K:
                others = [v for v in range(num_nodes) if v not in topk_indices]
                random_neighbors = list(random.sample(others, K-len(topk_indices)))
                topk_indices = topk_indices + random_neighbors
            else:
                topk_indices = topk_indices[:K]
            topk_neighbor_weights = list(weights[i].values())
            if len(topk_neighbor_weights) < K:
                topk_neighbor_weights = topk_neighbor_weights + [0]*(K-len(topk_neighbor_weights))
            else:
                topk_neighbor_weights = topk_neighbor_weights[:K]
            topk_neighbor_weights = torch.tensor(topk_neighbor_weights).view(-1, 1).to(features.device)
            nodes_features[i, :, :] = torch.cat((features[topk_indices], topk_neighbor_weights), 1)
            progress_bar.update(1)
        progress_bar.close()
    nodes_features = torch.cat((processed_features[:num_nodes, :, :], nodes_features), 1)
    return nodes_features

def topk_ppr_matrix(edge_index: torch.Tensor,
                    num_nodes: int,
                    alpha: float,
                    eps: float,
                    output_node_indices: Union[np.ndarray, torch.LongTensor],
                    topk: int,
                    max_itr: int,
                    normalization='row') -> Tuple[csr_matrix, List[np.ndarray]]:
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""
    if isinstance(output_node_indices, torch.Tensor):
        output_node_indices = output_node_indices.numpy()
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()
    _, indptr, out_degree = np.unique(edge_index_np[0], return_index=True, return_counts=True)
    indptr = np.append(indptr, len(edge_index_np[0]))
    neighbors, weights = calc_ppr_topk_parallel(indptr, edge_index_np[1], out_degree, alpha, eps, output_node_indices, max_itr)
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=int))
    j = np.concatenate(neighbors)
    ppr_matrix = coo_matrix((np.concatenate(weights), (i, j)), (len(output_node_indices), num_nodes)).tocsr()
    neighbors = sparsify(neighbors, weights, topk)
    neighbors = [np.union1d(nei, pr) for nei, pr in zip(neighbors, output_node_indices)]

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg_sqrt = np.sqrt(np.maximum(out_degree, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt
        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = deg_sqrt[output_node_indices[row]] * ppr_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg_inv = 1. / np.maximum(out_degree, 1e-12)
        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = out_degree[output_node_indices[row]] * ppr_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")
    return ppr_matrix, neighbors, weights

def sparsify(neighbors: List[np.ndarray], weights: List[np.ndarray], topk: int): ##
    new_neighbors = []
    for n, w in zip(neighbors, weights):
        idx_topk = np.argsort(w)[-topk:]
        new_neighbor = n[idx_topk]
        new_neighbors.append(new_neighbor)
    return new_neighbors

@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, max_itr) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon, max_itr)
        js[i] = np.array(j)
        vals[i] = np.array(val)
    return js, vals

@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon, max_itr):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    time_to_stop = 0
    while len(q) > 0:
        unode = q.pop()
        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val
            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
        time_to_stop += 1
        if time_to_stop >= max_itr:
            break
    return list(p.keys()), list(p.values())