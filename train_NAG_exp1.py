from data import get_dataset, rand_train_test_idx, get_NAG_data
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

# Training settings.
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
    parser.add_argument('--split_seed', type=int, default=0, help='Data split seed.')

    # Model parameters.
    parser.add_argument('--hops', type=int, default=3, help='Hops of neighbors to be calculated.')
    parser.add_argument('--pe_dim', type=int, default=3, help='Position embedding size.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer size.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of transformer heads.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout.')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Dropout in the attention layer.')

    # Training parameters.
    parser.add_argument('--train_size', type=float, default=0.5, help='Training proportion.')
    parser.add_argument('--val_size', type=int, default=0.25, help='Validation proportion.')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')
    parser.add_argument('--tot_updates',  type=int, default=1000, help='Used for optimizer learning rate scheduling.')
    parser.add_argument('--warmup_updates', type=int, default=500, help='Warmup steps.')
    parser.add_argument('--peak_lr', type=float, default=0.01, help='Peak learning rate.')
    parser.add_argument('--end_lr', type=float, default=0.0001, help='End learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay.')
    
    return parser.parse_args()

args = parse_args()
print(args)

method = "NAG"
method = method.lower()

try:

    filename = args.save_to+'/results/exp1_'+str(args.split_seed)+'/'+args.dataset+'_'+method+'.pkl'
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    if not os.path.exists(args.save_to+'/results'):
        os.makedirs(args.save_to+'/results')
    if not os.path.exists(args.save_to+'/results/exp1_'+str(args.split_seed)):
        os.makedirs(args.save_to+'/results/exp1_'+str(args.split_seed))
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
            graph = dgl.from_networkx(graphs[i])
            features = torch.tensor(all_features[i][:len(graph.nodes())], dtype=torch.float32)
            labels = torch.as_tensor(all_labels[i], dtype=torch.long)
            idx_train, idx_val, idx_test = rand_train_test_idx("synthesized", labels, split_seed=args.split_seed)
        else:
            graph, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.train_size, args.val_size, args.split_seed)
        adj, features = get_NAG_data(graph, features, args.pe_dim)
        processed_features = utils.re_features_NAG(adj, features, args.hops).to(device)
        labels = labels.to(device)

        del graph

        batch_data_train = Data.TensorDataset(processed_features[idx_train], labels[idx_train])
        batch_data_val = Data.TensorDataset(processed_features[idx_val], labels[idx_val])
        batch_data_test = Data.TensorDataset(processed_features[idx_test], labels[idx_test])

        train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle = True)
        val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle = True)
        test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle = True)

        # Model configuration.
        model = TransformerModel(args.hops, 
                                n_class=labels.max().item() + 1, 
                                input_dim=features.shape[1],
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
                torch.save(model.state_dict(), args.save_to+'/results/exp1_'+str(args.split_seed)+'/best_model_'+method+'_'+args.dataset+'.pth')
            else:
                no_improvement_count += 1
            if no_improvement_count >= args.patience:
                model.load_state_dict(torch.load(args.save_to+'/results/exp1_'+str(args.split_seed)+'/best_model_'+method+'_'+args.dataset+'.pth'))
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