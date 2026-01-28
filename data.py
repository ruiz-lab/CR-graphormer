import dgl
import torch
import scipy.sparse as sp
import dgl.data as DGLData
from ogb.nodeproppred import NodePropPredDataset
import numpy as np
from dgl.data.utils import split_dataset
import pickle

def get_dataset(dataset_name, train_size=0.5, val_size=0.25, split_seed=0, nclass=5):
    datasets1 = {"stanford", "karate", "cora", "citeseer", "pubmed", "corafull", "aifb", "mutag", "bgs", "am",
                 "computer", "photo", "cs", "physics", "ppi", "reddit", "sbm", "fraud", "fraud-yelp", "fraud-amazon",
                 "shape", "community", "cycle", "grid", "wiki", "flickr", "yelp", "pattern", "cluster", "chameleon",
                 "squirrel", "actor", "cornell", "texas", "wisconsin"}
    datasets2 = {"products", "proteins", "arxiv", "papers", "mag"}
    datasets3 = {"arxiv-year"}
    datasets4 = {"sbm"}
    sets = [datasets1, datasets2, datasets3]
    all_datasets = set().union(*sets)
    dataset_name = dataset_name.lower()
    if dataset_name not in all_datasets:
        raise InterruptedError("Data undefined.")
    if dataset_name in datasets1:
        if dataset_name == "stanford":
            dataset = DGLData.SSTDataset()
        elif dataset_name == "karate":
            dataset = DGLData.KarateClubDataset()
        elif dataset_name == "cora": ##
            dataset = DGLData.CoraGraphDataset()
        elif dataset_name == "citeseer":
            dataset = DGLData.CiteseerGraphDataset()
        elif dataset_name == "pubmed": ##
            dataset = DGLData.PubmedGraphDataset()
        elif dataset_name == "corafull":
            dataset = DGLData.CoraFullDataset()
        elif dataset_name == "aifb":
            dataset = DGLData.AIFBDataset()
        elif dataset_name == "mutag":
            dataset = DGLData.MUTAGDataset()
        elif dataset_name == "bgs":
            dataset = DGLData.BGSDataset()
        elif dataset_name == "am":
            dataset = DGLData.AMDataset()
        elif dataset_name == "computer": ##
            dataset = DGLData.AmazonCoBuyComputerDataset()
        elif dataset_name == "photo":
            dataset = DGLData.AmazonCoBuyPhotoDataset()
        elif dataset_name == "cs":
            dataset = DGLData.CoauthorCSDataset()
        elif dataset_name == "physics":
            dataset = DGLData.CoauthorPhysicsDataset()
        elif dataset_name == "ppi":
            dataset = DGLData.PPIDataset()
        elif dataset_name == "reddit":
            dataset = DGLData.RedditDataset()
        elif dataset_name == "sbm":
            dataset = DGLData.SBMMixtureDataset()
        elif dataset_name == "fraud":
            dataset = DGLData.FraudDataset()
        elif dataset_name == "fraud-yelp":
            dataset = DGLData.FraudYelpDataset()
        elif dataset_name == "fraud-amazon":
            dataset = DGLData.FraudAmazonDataset()
        elif dataset_name == "shape":
            dataset = DGLData.BAShapeDataset()
        elif dataset_name == "community":
            dataset = DGLData.BACommunityDataset()
        elif dataset_name == "cycle":
            dataset = DGLData.TreeCycleDataset()
        elif dataset_name == "grid":
            dataset = DGLData.TreeGridDataset()
        elif dataset_name == "wiki":
            dataset = DGLData.WikiCSDataset()
        elif dataset_name == "flickr":
            dataset = DGLData.FlickrDataset()
        elif dataset_name == "yelp":
            dataset = DGLData.YelpDataset()
        elif dataset_name == "pattern":
            dataset = DGLData.PATTERNDataset()
        elif dataset_name == "cluster":
            dataset = DGLData.CLUSTERDataset()
        elif dataset_name == "chameleon":
            dataset = DGLData.ChameleonDataset()
        elif dataset_name == "squirrel":
            dataset = DGLData.SquirrelDataset()
        elif dataset_name == "actor":
            dataset = DGLData.ActorDataset()
        elif dataset_name == "cornell":
            dataset = DGLData.CornellDataset()
        elif dataset_name == "texas":
            dataset = DGLData.TexasDataset()
        elif dataset_name == "wisconsin":
            dataset = DGLData.WisconsinDataset()
        graph = dataset[0]
        features = torch.as_tensor(graph.ndata["feat"])
        labels = torch.as_tensor(graph.ndata["label"])
        idx_train, idx_val, idx_test = split_dataset(range(len(labels)), frac_list = [train_size, val_size, 1-train_size-val_size], shuffle=True, random_state=split_seed)
    elif dataset_name in datasets3:
        if dataset_name in datasets2:
            graph, labels, idx_train, idx_val, idx_test = load_ogb_dataset(dataset_name,train_size,val_size,split_seed)
        else:
            graph, labels, idx_train, idx_val, idx_test = load_arxiv_year_dataset(nclass,train_size,val_size,split_seed)
        features = torch.as_tensor(graph['node_feat'])
        labels = labels.squeeze()
        num_nodes = features.shape[0]
        adj_tensor = torch.sparse_coo_tensor(graph['edge_index'], torch.ones(graph['edge_index'].shape[1]), (num_nodes, num_nodes)).coalesce()
        values = adj_tensor.values().numpy()
        rows = adj_tensor.indices()[0].numpy()
        cols = adj_tensor.indices()[1].numpy()
        adj_matrix = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
        graph = dgl.from_scipy(adj_matrix)
    return graph, features, labels, idx_train, idx_val, idx_test

def get_NAG_data(graph, features, pe_dim, edge_weights=None):
    if edge_weights ==  None:
        adj = graph.adj()
    else:
        row, col = graph.edges()
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        adj = sp.coo_matrix((edge_weights, (row, col)), shape=(graph.num_nodes(), graph.num_nodes()))
        adj = dgl.from_scipy(adj, eweight_name='w')
        adj.row = torch.tensor(row, dtype=torch.long)
        adj.col = torch.tensor(col, dtype=torch.long)
        adj.val = torch.tensor(edge_weights, dtype=torch.long)
        adj.shape = (graph.num_nodes(), graph.num_nodes())
    graph = dgl.to_bidirected(graph)
    lpe = laplacian_positional_encoding(graph, pe_dim)
    features = torch.cat((features, lpe), dim=1)
    return adj, features

def laplacian_positional_encoding(g, pos_enc_dim):
    """Graph positional encoding v/ Laplacian eigenvectors."""
    # Laplacian.
    A = g.adjacency_matrix().to(dtype=float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N.toarray() * A.to_dense().numpy() * N.toarray()
    # Eigenvectors with scipy.
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # Increasing order.
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    return lap_pos_enc

def get_VCR_data(graph, features, num_supernodes, normalize=False, edge_weights=None, row=None, col=None):
    if edge_weights == None:
        adj = graph.adj()
        raw_adj_sp = sp.coo_matrix((adj.coalesce().val, (adj.coalesce().indices()[0], adj.coalesce().indices()[1])), shape=adj.shape)
    else:
        raw_adj_sp = sp.coo_matrix((edge_weights, (row, col)), shape=(graph.num_nodes(), graph.num_nodes()))
        adj = dgl.from_scipy(raw_adj_sp, eweight_name='w')
        adj.row = torch.tensor(row, dtype=torch.long)
        adj.col = torch.tensor(col, dtype=torch.long)
        adj.val = torch.tensor(edge_weights, dtype=torch.long)
        adj.shape = (graph.num_nodes(), graph.num_nodes())
    graph = dgl.to_bidirected(graph)
    if num_supernodes > 0:
        clusters_dict = dgl.metis_partition(g=graph,
                                            k=num_supernodes,
                                            extra_cached_hops=0,
                                            reshuffle=False,
                                            balance_ntypes=None,
                                            balance_edges=False,
                                            mode='k-way')
    else:
        clusters_dict = None
    features = features.double()
    return raw_adj_sp, adj, features, clusters_dict

def load_ogb_dataset(name, train_size=0.5, val_size=0.25, split_seed=0):
    ogb_dataset = NodePropPredDataset(name="ogbn-"+name)
    graph = ogb_dataset.graph
    labels = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    idx_train, idx_val, idx_test = rand_train_test_idx(name, labels, train_size, val_size, split_seed)
    return graph, labels, idx_train, idx_val, idx_test

def load_arxiv_year_dataset(nclass=5, train_size=0.5, val_size=0.25, split_seed=0):
    name = "arxiv"
    ogb_dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph = ogb_dataset.graph
    labels = torch.as_tensor(even_quantile_labels(graph['node_year'].flatten(), nclass, verbose=False)).reshape(-1, 1)
    idx_train, idx_val, idx_test = rand_train_test_idx(name, labels, train_size, val_size, split_seed)
    return graph, labels, idx_train, idx_val, idx_test

def rand_train_test_idx(name, label, train_prop=.5, valid_prop=.25, split_seed=0):
    """ Randomly splits data into train/valid/test splits. """
    ignore_negative = False
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label
    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    np.random.seed(split_seed)
    perm = torch.as_tensor(np.random.permutation(n))
    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]
    return train_indices, val_indices, test_indices

def even_quantile_labels(vals, nclasses, verbose=True):
    """ Partitions vals into nclasses by a quantile-based split,
    where the first class is less than the 1/nclasses quantile, the 
    second class is less than the 2/nclasses quantile, and so on.
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label