from data import get_dataset, rand_train_test_idx, get_VCR_data
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
import dgl
from adjacency_search import get_auxiliary_graph, get_weights, globally_normalize, locally_normalize

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
    parser.add_argument('--adj_renorm', default=False, help='Whether needs to normalize input adjacency.')
    parser.add_argument('--merge', type=bool, default=False, help='Merge auxiliary graph with original graph.')
    parser.add_argument('--p', type=float, default=1.0, help='Lazy activation parameter.')
    parser.add_argument('--k', type=int, default=5, help='k in top-k.')
    parser.add_argument('--normalization', type=str, default=None, help='Frequency normalization.')
    
    # Model parameters.
    parser.add_argument('--num_structure_tokens', type=int, default=10, help='Number of structure-aware virtually connected neighbors.')
    parser.add_argument('--num_content_tokens', type=int, default=10, help='Number of content-aware virtually connected neighbors.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer size.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of transformer heads.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout.')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Dropout in the attention layer.')
    # parser.add_argument('--split_seed', type=int, default=0, help='Split seed.')

    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000, help='Used for optimizer learning rate scheduling.')
    parser.add_argument('--warmup_updates', type=int, default=500, help='Warmup steps.')
    parser.add_argument('--peak_lr', type=float, default=0.01, help='Peak learning rate.')
    parser.add_argument('--end_lr', type=float, default=0.0001, help='End learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay.')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')
    parser.add_argument('--train_size', type=float, default=0.5, help='Training proportion.')
    parser.add_argument('--val_size', type=int, default=0.25, help='Validation proportion.')
    
    return parser.parse_args()

args = parse_args()
print(args)

method = "CR_tas"
method = method.lower()

for split_seed in range(20):

    try:

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
                # k = int(sum(dict(graph.degree()).values()) / len(graph.nodes()))
                features = torch.tensor(all_features[i][:len(graph.nodes())], dtype=torch.float32)
                labels = torch.as_tensor(all_labels[i], dtype=torch.long)
                idx_train, idx_val, idx_test = rand_train_test_idx("synthesized", labels, split_seed=split_seed)
            else:
                graph, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.train_size, args.val_size, split_seed)
                graph = dgl.to_networkx(graph).to_undirected()
                # k = int(sum(dict(graph.degree()).values()) / len(graph.nodes()))
            
            filename_adjacency_search = os.path.dirname(os.getcwd())+'/saved_adjacency_search_p'+str(args.p)+'/'+args.dataset+'_tas.pkl'
            if args.normalization == None:
                if os.path.exists(filename_adjacency_search):
                    try:
                        with open(filename_adjacency_search, 'rb') as f:
                            AS_object = pickle.load(f)
                    except:
                        raise ValueError("Trouble loading the source file or the source file does not exist.")
                else:
                    raise ValueError("Trouble loading the source file or the source file does not exist.")
            else:
                if args.normalization == "global":
                    copy_to = 'globally_normalized/'+args.dataset+'_tas.pkl'
                    if not os.path.exists('globally_normalized'):
                        os.makedirs('globally_normalized')
                elif args.normalization == "local":
                    copy_to = 'locally_normalized/'+args.dataset+'_tas.pkl'
                    if not os.path.exists('locally_normalized'):
                        os.makedirs('locally_normalized')
                else:
                    raise ValueError("Invalid normalization method ('global' or 'local' or NoneType only).")
                    
                if os.path.exists(copy_to):
                    try:
                        with open(copy_to, 'rb') as f:
                            AS_object = pickle.load(f)
                    except:
                        try:
                            with open(filename_adjacency_search, 'rb') as f:
                                AS_object = pickle.load(f)
                            if args.normalization == "global":
                                globally_normalize(AS_object)
                            elif args.normalization == "local":
                                locally_normalize(AS_object)
                            with open(copy_to, 'wb') as f:
                                pickle.dump(AS_object, f)
                        except:
                            raise ValueError("Trouble loading the source file or the source file does not exist.")
                else:
                    try:
                        with open(filename_adjacency_search, 'rb') as f:
                            AS_object = pickle.load(f)
                        if args.normalization == "global":
                            globally_normalize(AS_object)
                        elif args.normalization == "local":
                            locally_normalize(AS_object)
                        with open(copy_to, 'wb') as f:
                            pickle.dump(AS_object, f)
                    except:
                        raise ValueError("Trouble loading the source file or the source file does not exist.")
            auxiliary_graph = get_auxiliary_graph(AS_object, args.k)
            if args.merge:
                auxiliary_graph = nx.compose(graph, auxiliary_graph)
            auxiliary_graph = dgl.from_networkx(auxiliary_graph)
            raw_adj_sp, original_adj, features, cluster_dict = get_VCR_data(auxiliary_graph, features, 20, normalize=args.adj_renorm)
            weights = get_weights(AS_object, auxiliary_graph)
            processed_features = utils.re_features_push_structure(raw_adj_sp, original_adj, features, 1, args.num_structure_tokens, weights)
            processed_features = processed_features.to(device)
            labels = labels.to(device)

            del graph, auxiliary_graph, weights, AS_object

            batch_data_train = Data.TensorDataset(processed_features[idx_train], labels[idx_train])
            batch_data_val = Data.TensorDataset(processed_features[idx_val], labels[idx_val])
            batch_data_test = Data.TensorDataset(processed_features[idx_test], labels[idx_test])

            train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle = True)
            val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle = True)
            test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle = True)

            # Model configuration.
            model = TransformerModel(1 + args.num_structure_tokens, 
                                    n_class=labels.max().item() + 1, 
                                    input_dim=features.shape[1] + 1,
                                    hidden_dim=args.hidden_dim,
                                    n_layers=args.n_layers,
                                    num_heads=args.n_heads,
                                    dropout_rate=args.dropout,
                                    attention_dropout_rate=args.attention_dropout).to(device)
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
                loss_train_b = 0
                acc_train_b = 0
                for _, item in enumerate(train_data_loader):
                    nodes_features = item[0].to(device)
                    labels = item[1].to(device)
                    optimizer.zero_grad()
                    output = model(nodes_features)
                    loss_train = F.nll_loss(output, labels)
                    loss_train.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    loss_train_b += loss_train.item()
                    acc_train = utils.accuracy_batch(output, labels)
                    acc_train_b += acc_train.item()
                
                model.eval()
                loss_val = 0
                acc_val = 0
                for _, item in enumerate(val_data_loader):
                    nodes_features = item[0].to(device)
                    labels = item[1].to(device)
                    output = model(nodes_features)
                    loss_val += F.nll_loss(output, labels).item()
                    acc_val += utils.accuracy_batch(output, labels).item()

                loss_train_b = loss_train_b/len(idx_train)
                acc_train_b = acc_train_b/len(idx_train)
                loss_val = loss_val/len(idx_val)
                acc_val = acc_val/len(idx_val)
                print('Epoch: {:04d}'.format(epoch+1),
                    'Training Loss: {:.4f}'.format(loss_train_b),
                    'Training Accuracy: {:.4f}'.format(acc_train_b),
                    'Validation Loss: {:.4f}'.format(loss_val),
                    'Validation Accuracy: {:.4f}'.format(acc_val))
                dictionary['training_results'][0].append(loss_train_b)
                dictionary['training_results'][1].append(acc_train_b)
                dictionary['training_results'][2].append(loss_val)
                dictionary['training_results'][3].append(acc_val)
                return loss_val, acc_val

            def test(dictionary):
                loss_test = 0
                acc_test = 0
                for _, item in enumerate(test_data_loader):
                    nodes_features = item[0].to(device)
                    labels = item[1].to(device)
                    model.eval()
                    output = model(nodes_features)
                    loss_test += F.nll_loss(output, labels).item()
                    acc_test += utils.accuracy_batch(output, labels).item()
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

    except:
        pass