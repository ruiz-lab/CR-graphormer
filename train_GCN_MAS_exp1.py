from data import get_dataset, rand_train_test_idx
import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from GCN import GCN
from lr import PolynomialDecayLR
import torch.utils.data as Data
import os
import pickle
import dgl
import networkx as nx
from torch_geometric.utils import from_networkx
from adjacency_search import get_auxiliary_graph

def parse_args():
    '''
    Generate a parameters parser.
    '''
    # Parse parameters.
    parser = argparse.ArgumentParser()

    # Main parameters.
    parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--save_to', type=str, default='results', help='Result folder.')
    parser.add_argument('--device', type=int, default=1, help='Device cuda ID.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--merge', type=bool, default=False, help='Merge auxiliary graph with original graph.')
    parser.add_argument('--p', type=float, default=1.0, help='Lazy activation parameter.')

    # Model parameters.
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer size.')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of transformer layers.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout.')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Dropout in the attention layer.')
    # parser.add_argument('--split_seed', type=int, default=0, help='Split seed.')

    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000, help='Used for optimizer learning rate scheduling.')
    parser.add_argument('--warmup_updates', type=int, default=500, help='Warmup steps.')
    parser.add_argument('--peak_lr', type=float, default=0.001, help='Peak learning rate.')
    parser.add_argument('--end_lr', type=float, default=0.0001, help='End learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay.')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')
    parser.add_argument('--train_size', type=float, default=0.5, help='Training proportion.')
    parser.add_argument('--val_size', type=int, default=0.25, help='Validation proportion.')
    
    return parser.parse_args()

args = parse_args()
print(args)

method = "GCN_mas"
method = method.lower()

for split_seed in range(20):

    filename = args.save_to+'/results/exp1_'+str(split_seed)+'/'+args.dataset+'_'+method+'.pkl'
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    if not os.path.exists(args.save_to+'/results'):
        os.makedirs(args.save_to+'/results')
    if not os.path.exists(args.save_to+'/results/exp1_'+str(split_seed)):
        os.makedirs(args.save_to+'/results/exp1_'+str(split_seed))
    if not os.path.exists(filename):

        print(os.getcwd(), filename)

        start_time1 = time.time()
        device = args.device
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        # Pre-process data.
        if "synthesized" in args.dataset:
            with open("/home/myl/notebooks/graphormer/cascade_rewired1/correct/synthesized/graph_data.pkl", 'rb') as f:
                graphs, all_labels, all_features = pickle.load(f)
            i = int(args.dataset[11:])
            graph = graphs[i]
            k = int(sum(dict(graph.degree()).values()) / len(graph.nodes()))
            features = torch.tensor(all_features[i], dtype=torch.float32).to(device)
            labels = torch.as_tensor(all_labels[i], dtype=torch.long).to(device)
            idx_train, idx_val, idx_test = rand_train_test_idx("synthesized", labels, split_seed=split_seed)
        else:
            graph, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.train_size, args.val_size, split_seed)
            graph = dgl.to_networkx(graph).to_undirected()
            k = int(sum(dict(graph.degree()).values()) / len(graph.nodes()))
            features = features.float().to(device)
            labels = labels.to(device)

        filename_adjacency_search = os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)+'/'+args.dataset+'_mas.pkl'
        if not os.path.exists('saved_adjacency_search_p'+str(args.p)):
            os.makedirs('saved_adjacency_search_p'+str(args.p))
        copy_to = 'saved_adjacency_search_p'+str(args.p)+'/'+args.dataset+'_mas.pkl'
        if os.path.exists(copy_to):
            try:
                with open(copy_to, 'rb') as f:
                    AS_object = pickle.load(f)
            except:
                try:
                    with open(filename_adjacency_search, 'rb') as f:
                        AS_object = pickle.load(f)
                    if 'version2' in os.getcwd():
                        update_score2(AS_object)
                    elif 'version3' in os.getcwd():
                        update_score3(AS_object)
                    with open(copy_to, 'wb') as f:
                        pickle.dump(AS_object, f)
                except:
                    raise ValueError("Trouble loading the source file or the source file does not exist.")
        else:
            try:
                with open(filename_adjacency_search, 'rb') as f:
                    AS_object = pickle.load(f)
                if 'version2' in os.getcwd():
                    update_score2(AS_object)
                elif 'version3' in os.getcwd():
                    update_score3(AS_object)
                with open(copy_to, 'wb') as f:
                    pickle.dump(AS_object, f)
            except:
                raise ValueError("Trouble loading the source file or the source file does not exist.")
        auxiliary_graph = get_auxiliary_graph(AS_object, k)
        if args.merge:
            auxiliary_graph = nx.compose(graph, auxiliary_graph)
        data = from_networkx(auxiliary_graph)
        edge_index = data.edge_index.to(device)

        del graph, auxiliary_graph

        # Model configuration.
        model = GCN(in_dim=features.shape[1],
                    out_dim=labels.max().item() + 1).to(device)
        print(model)
        print('Total parameters:', sum(p.numel() for p in model.parameters()))

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
        lr_scheduler = PolynomialDecayLR(optimizer,
                                        warmup_updates=args.warmup_updates,
                                        tot_updates=args.tot_updates,
                                        lr=args.peak_lr,
                                        end_lr=args.end_lr,
                                        power=1.0)

        def train_valid_epoch(epoch,dictionary):
            model.train()
            optimizer.zero_grad()
            output = model(features,edge_index)
            # print("Output shape:", output.shape)        # should be [num_nodes, num_classes]
            # print("Unique labels:", labels.unique())
            # print("Labels min:", labels.min().item())
            # print("Labels max:", labels.max().item())
            # print("Train idx range:", idx_train.min().item(), idx_train.max().item())
            # print("Num classes:", output.shape[1])
            # print(labels[idx_train].min(), labels[idx_train].max())
            # print(labels[idx_train].dtype)
            # print("len(labels):", len(labels))
            # print("max(idx_train):", idx_train.max().item())
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
            acc_train = utils.accuracy_batch(output[idx_train], labels[idx_train]).item()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
            acc_val = utils.accuracy_batch(output[idx_val], labels[idx_val]).item()

            loss_train_val = loss_train.item()/len(idx_train)
            acc_train = acc_train/len(idx_train)
            loss_val = loss_val/len(idx_val)
            acc_val = acc_val/len(idx_val)
            print('Epoch: {:04d}'.format(epoch+1),
                'Training Loss: {:.4f}'.format(loss_train_val),
                'Training Accuracy: {:.4f}'.format(acc_train),
                'Validation Loss: {:.4f}'.format(loss_val),
                'Validation Accuracy: {:.4f}'.format(acc_val))
            dictionary['training_results'][0].append(loss_train_val)
            dictionary['training_results'][1].append(acc_train)
            dictionary['training_results'][2].append(loss_val)
            dictionary['training_results'][3].append(acc_val)
            return loss_val, acc_val

        def test(dictionary):
            model.eval()
            output = model(features,edge_index)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test]).item()
            acc_test = utils.accuracy_batch(output[idx_test], labels[idx_test]).item()
            loss_test = loss_test/len(idx_test)
            acc_test = acc_test/len(idx_test)
            print('Test Loss: {:.4f}'.format(loss_test))
            print('Test Accuracy: {:.4f}'.format(acc_test))
            dictionary['test_loss'] = loss_test
            dictionary['test_accuracy'] = acc_test

        dictionary = {'training_results': [[],[],[],[]],
                    'best_epoch': None,
                    'test_loss': None,
                    'test_accuracy': None,
                    'training_runtime': None,
                    'total_runtime': None}

        best_val_loss = float('inf')
        best_epoch = 0
        no_improvement_count = 0

        start_time2 = time.time()
        for epoch in range(args.epochs):
            loss_val, acc_val = train_valid_epoch(epoch, dictionary)
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                best_epoch = epoch
                no_improvement_count = 0
                torch.save(model.state_dict(), args.save_to+'/results/exp1_'+str(split_seed)+'/best_model_'+method+'_'+args.dataset+'.pth')
            else:
                no_improvement_count += 1
            if no_improvement_count >= args.patience:
                model.load_state_dict(torch.load(args.save_to+'/results/exp1_'+str(split_seed)+'/best_model_'+method+'_'+args.dataset+'.pth'))
                model = model.to(device)
                break

        training_runtime = time.time() - start_time2
        print('Training Runtime: {:.4f}s'.format(training_runtime))
        dictionary['training_runtime'] = training_runtime
        print('Loading Epoch {}'.format(best_epoch+1))
        dictionary['best_epoch'] = best_epoch+1
        test(dictionary)
        total_runtime = time.time() - start_time1
        print('Total Runtime: {:.4f}s'.format(total_runtime))
        dictionary['total_runtime'] = total_runtime

        with open(filename, 'wb') as f:
            pickle.dump(dictionary, f)