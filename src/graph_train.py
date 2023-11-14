import os
import torch
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utility import one_hot_aminoacids, seed_torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from focal_loss.focal_loss import FocalLoss
from sklearn.model_selection import StratifiedKFold
from metrics import custom_metrics, train_metrics, test_metrics, label_metrics, save_auc, save_bac, save_mcc, save_spec
from models import DataLoaderGraph, LossWrapper, GNNModel, GNNModelOneHot, GNNModelAttention, GNNModelSimple
from dataloader import dataset_preparation_proteinbert

parser = argparse.ArgumentParser(description=('dataloader.py prepare dataset for training, validation, test'))
parser.add_argument('--prepare_data', required=False, default='//', help='Tell if necessary to extract files for dataset creation')
parser.add_argument('--metrics', required=False, default='True', help='Tell if necessary to extract metrics')
parser.add_argument('--difference', required=False, default='True', help='Tell if necessary to extract metrics')
parser.add_argument('--one_hot', required=False, default='True', help='Tell if necessary to use one hot encoding')
parser.add_argument('--trials', required=True, default=0, help='Tell which trials you are performing')
parser.add_argument('--epochs', required=False, default=100, help='Tell how many epochs you need to do')
parser.add_argument('--global_metrics', required=False, default='True', help='Tell if it is necessary to extract global metrics')
parser.add_argument('--fold_mapping', required=False, default='False', help='Tell if we are dealing with the paper')
parser.add_argument('--train', required=False, default='True', help='Tell if we are dealing with the paper')

args = parser.parse_args()


def train_test_graph(num_epochs, dimension, train_loader, test_loader, device, group):
    input_size = 20
    output_size = 1

    model = GNNModelAttention(input_size, dimension[-2], output_size)
    model.to(device)

    loss = FocalLoss(gamma=4)
    criterion = LossWrapper(loss=loss, ignore_index=-999)
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    train_acc, test_acc, auc_train, auc_test, mcc_train, mcc_test, spec_train, spec_test = [], [], [], [], [], [], [], []

    for epoch in range(int(num_epochs)):
        running_loss = 0.0
        metrics = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": [], "auc": []}

        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels, data, indices = (batch[1]).to(device), (batch[2]).to(device), batch[0].to(device), batch[3]
            tensor_one_hot = []

            if args.one_hot == 'True' and args.difference == 'True':
                for mapping in indices:
                    mutation = mapping.split('_')[-1]
                    wildtype = mapping.split('_')[2]
                    one_hot = one_hot_aminoacids(wildtype=wildtype, mutation=mutation)
                    tensor_one_hot.append(one_hot)

                tensor_one_hot = torch.stack(tensor_one_hot, dim=0)
                tensor_one_hot = tensor_one_hot.to(device)
                outputs = model(data, inputs, tensor_one_hot)
            else:
                outputs = model(data, inputs)

            m = nn.Sigmoid()
            try:
                loss = criterion(m(outputs), labels.long())
            except Exception as error:
                print('TRAIN')
                print(f'{error} and {inputs}')

            loss.backward() # type: ignore
            optimizer.step()

            running_loss += loss.item() # type: ignore

            ignore_indices = (labels != -999)
            y_pred = (torch.sigmoid(outputs) > 0.5).float()            

            if args.global_metrics == 'True':
                y_pred_masked = torch.masked_select(y_pred, ignore_indices)
                y_true_masked = torch.masked_select(labels, ignore_indices)
                mcc, prec, rec, spec, balanced_acc, f1_score, auc = custom_metrics(y_true_masked, y_pred_masked, outputs, labels)
                metrics = train_metrics(metrics, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
            else:
                spec, balanced, mcc, auc = label_metrics(labels, y_pred)
                metrics = test_metrics(metrics, mcc, spec, balanced, auc)

        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {running_loss}, Matthews Mean: {np.mean(metrics['mcc'])}, Precision Mean: {np.mean(metrics['prec'])}, Recall Mean: {np.mean(metrics['rec'])}, Specificity: {np.mean(metrics['spec'])}, Balanced Accuracy: {np.mean(metrics['balanced_acc'])}, F1 Score Mean: {np.mean(metrics['f1_score'])}")
        auc_train.append(np.mean(metrics["auc"]))
        train_acc.append(np.mean(metrics['balanced_acc']))
        mcc_train.append(np.mean(metrics['mcc']))
        spec_train.append(np.mean(metrics['spec']))

        model.eval()

        with torch.no_grad():
            total_loss = 0
            metrics_test = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": [], "auc": []}

            for test_batch in test_loader:
                batch_X, batch_y, batch_data, batch_indices = (test_batch[1]).to(device), (test_batch[2]).to(device), test_batch[0].to(device), test_batch[3]
                tensor_one_hot = []

                if args.one_hot == 'True' and args.difference == 'True':
                    tensor_one_hot = []

                    for mapping in batch_indices:
                        mutation = mapping.split('_')[-1]
                        wildtype = mapping.split('_')[2]
                        one_hot = one_hot_aminoacids(wildtype=wildtype, mutation=mutation)
                        tensor_one_hot.append(one_hot)

                    tensor_one_hot = torch.stack(tensor_one_hot, dim=0)
                    tensor_one_hot = tensor_one_hot.to(device)
                    predictions = model(batch_data, batch_X, tensor_one_hot)
                else:
                    predictions = model(batch_data, batch_X)

                m = nn.Sigmoid()
                try:
                    total_loss += criterion(m(predictions), batch_y.long()).item()
                except Exception as error:
                    print('TEST')
                    print(f'{error} and {batch_X}')

                ignore_indices = (batch_y != -999)
                y_pred = (torch.sigmoid(predictions) > 0.5).float()

                if args.global_metrics == 'True':
                    y_pred_masked = torch.masked_select(y_pred, ignore_indices)
                    y_true_masked = torch.masked_select(batch_y, ignore_indices)
                    mcc, prec, rec, spec, balanced_acc, f1_score, auc = custom_metrics(y_true_masked, y_pred_masked, predictions, batch_y)
                    metrics_test = train_metrics(metrics_test, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
                else:
                    spec, balanced_acc, mcc, auc = label_metrics(batch_y, y_pred)
                    metrics_test = test_metrics(metrics_test, mcc, spec, balanced_acc, auc)

            avg_loss = total_loss / len(test_loader)

            print(f"Test Loss: {avg_loss:.4f}, Matthews Test Mean: {np.mean(metrics_test['mcc'])}, Specificity: {np.mean(metrics_test['spec'])}, Balanced Accuracy: {np.mean(metrics_test['balanced_acc'])}")
            auc_test.append(np.mean(metrics_test['auc']))
            test_acc.append(np.mean(metrics_test['balanced_acc']))
            mcc_test.append(np.mean(metrics_test['mcc']))
            spec_test.append(np.mean(metrics_test['spec']))
            print('\n')

    save_auc(num_epochs, auc_train, auc_test, group, args)
    save_bac(num_epochs, train_acc, test_acc, group, args)
    save_mcc(num_epochs, mcc_train, mcc_test, group, args)
    save_spec(num_epochs, spec_train, spec_test, group, args)


def main_train(device):
    seed_torch(seed=0)
    # Extract Features matrix and prepare Dataset
    print('-----------------Extracting Dataset-----------------')
    if args.prepare_data == 'True':
        dataset_preparation_proteinbert(args, os.listdir('dataset/fasta'))

    if args.train == 'True':
        print('-----------------Prepare Dataset--------------------')
        with open('dataset/prepared/data_processed.pickle', 'rb') as handle:
            X_temp = pickle.load(handle)
        with open('dataset/prepared/labels.pickle', 'rb') as handle:
            y_temp = pickle.load(handle)

        with open('dataset/prepared/mapping.pickle', 'rb') as handle:
            mapping = pickle.load(handle)
        with open('dataset/prepared/uniprot_to_pdb.pickle', 'rb') as handle:
            uniprot_to_pdb = pickle.load(handle)

        print('-----------------Create Folding---------------------')
        X = np.array([val1 for (_, val1) in X_temp])
        y = np.array([val2 for (_, val2) in y_temp])
        graphs = {}
        
        for data in os.listdir('embedding/graphs'):
            graph = pickle.load(open(f'embedding/graphs/{data}', 'rb'))
            graphs[data.split('.pickle')[0]] = {'edge_index': torch.tensor(list(graph.edges)).t().contiguous() , 
                                                'features': torch.tensor([graph._node[u]['features'] for u in graph.nodes()], dtype=torch.float)}

        graphs_mapped = []
        for graphName, graphValues in graphs.items():
            for maps in mapping:
                if graphName in uniprot_to_pdb.keys() and maps[1] == uniprot_to_pdb[graphName]:
                    graphs_mapped.append(Data(x=graphValues['features'], edge_index=graphValues['edge_index']))
                    break
        
        n_splits = 4
        group = 0
        kfold = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
        y = y.reshape(-1,1)
        fold = kfold.split(X, y)


        for train_index, test_index in fold:
            X_train, X_test = torch.tensor(X[train_index]),  torch.tensor(X[test_index])
            y_train, y_test =  torch.tensor(y[train_index]),  torch.tensor(y[test_index])
            graphs_train, graphs_test = [graphs_mapped[idx] for idx in train_index], [graphs_mapped[idx] for idx in test_index]

            train_indices, test_indices = np.array(mapping)[train_index][:,1], np.array(mapping)[test_index][:,1]

            train_dataset = DataLoaderGraph(graphs_train, X_train, y_train, train_indices)
            test_dataset = DataLoaderGraph(graphs_test, X_test, y_test, test_indices)

            batch_size = 64
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # type: ignore
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # type: ignore
            print(f'-----------------Start Training Group {group}-------------')
            train_test_graph(args.epochs, X.shape, train_loader, test_loader, device, group)
            group += 1


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    if not os.path.exists('dataset/prepared'):
        os.mkdir('dataset/prepared')
    
    main_train(device)