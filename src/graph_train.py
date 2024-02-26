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
from sklearn.model_selection import StratifiedKFold, GroupKFold
from metrics import custom_metrics, train_metrics, test_metrics, label_metrics, save_auc, save_bac, save_mcc, save_spec
from models import DataLoaderGraph, LossWrapper, GNNModel, GNNModelOneHot, GNNModelAttention, GNNModelSimple, GAttention
from dataloader import dataset_preparation_proteinbert, mapping_split, dataset_preparation_esm

parser = argparse.ArgumentParser(description=('dataloader.py prepare dataset for training, validation, test'))
parser.add_argument('--prepare_data', required=False, default='False', help='Tell if necessary to extract files for dataset creation')
parser.add_argument('--metrics', required=False, default='True', help='Tell if necessary to extract metrics')
parser.add_argument('--difference', required=False, default='True', help='Tell if necessary to extract metrics')
parser.add_argument('--one_hot', required=False, default='True', help='Tell if necessary to use one hot encoding')
parser.add_argument('--trials', required=True, default=0, help='Tell which trials you are performing')
parser.add_argument('--epochs', required=False, default=300, help='Tell how many epochs you need to do')
parser.add_argument('--global_metrics', required=False, default='True', help='Tell if it is necessary to extract global metrics')
parser.add_argument('--fold_mapping', required=False, default='True', help='Tell if we are dealing with the HPMPdb or Not')
parser.add_argument('--train', required=False, default='True', help='Tell if we are dealing with the paper')
parser.add_argument('--multiclass', required=False, default='True', help='Tell if we are dealing with a multiclass problem')

args = parser.parse_args()


def train_test_graph(num_epochs, dimension, train_loader, test_loader, device, group):
    input_size = 20
    output_size = 3

    model = GAttention(input_size, dimension[-2], output_size)
    model.to(device)

    class_weights = torch.tensor([1.0, 1.0 / 15, 1.0 / 30]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-998)
    #criterion = LossWrapper(loss=loss, ignore_index=-998)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)#, weight_decay=1e-3)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, min_lr=1e-20, threshold_mode='rel')

    
    train_acc, test_acc, auc_train, auc_test, mcc_train, mcc_test, spec_train, spec_test = [], [], [], [], [], [], [], []
    mcc, prec, rec, spec, balanced_acc, f1_score, auc = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5

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

            #m = nn.Sigmoid()
                
            try:
                loss = criterion(outputs.to(device), labels.long().view(-1) + 1)#.long())
            except Exception as error:
                print('TRAIN')
                print(f'{error} and {inputs}')

            loss.backward() # type: ignore
            optimizer.step()

            running_loss += loss.item() # type: ignore

            ignore_indices = (labels != -999)
            if args.multiclass == 'True':
                y_pred = torch.nn.functional.softmax(outputs).float()
            else:
                y_pred = (torch.sigmoid(outputs) > 0.5).float()

            if args.global_metrics == 'True':
                y_pred_masked = torch.masked_select(y_pred, ignore_indices.repeat(labels.size()[1], outputs.size()[1])).view(-1,outputs.size()[1])
                y_true_masked = torch.masked_select(labels, ignore_indices)
                if len(y_pred_masked) == 0 or len(y_true_masked) == 0:
                    metrics = train_metrics(metrics, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
                else:
                    mcc, prec, rec, spec, balanced_acc, f1_score, auc = custom_metrics(y_true_masked, y_pred_masked, outputs, labels, args)
                    metrics = train_metrics(metrics, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
            else:
                spec, balanced, mcc, auc = label_metrics(labels, y_pred, args)
                metrics = test_metrics(metrics, mcc, spec, balanced, auc)
        
        #scheduler.step(loss) #type: ignore

        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {running_loss}, Matthews Mean: {np.mean(metrics['mcc'])}, Precision Mean: {np.mean(metrics['prec'])}, Recall Mean: {np.mean(metrics['rec'])}, Specificity: {np.mean(metrics['spec'])}, Balanced Accuracy: {np.mean(metrics['balanced_acc'])}, F1 Score Mean: {np.mean(metrics['f1_score'])}")
        auc_train.append(np.mean(metrics["auc"]))
        train_acc.append(np.mean(metrics['balanced_acc']))
        mcc_train.append(np.mean(metrics['mcc']))
        spec_train.append(np.mean(metrics['spec']))

        model.eval()

        with torch.no_grad():
            total_loss = 0
            metrics_test = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": [], "auc": []}
            mcc, prec, rec, spec, balanced_acc, f1_score, auc = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5

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

                #m = nn.Sigmoid()

                try:
                    total_loss += criterion(predictions.to(device), batch_y.long().view(-1) + 1).item()
                except Exception as error:
                    print('TEST')
                    print(f'{error} and {batch_X}')

                ignore_indices = (batch_y != -999)
                if args.multiclass == 'True':
                    y_pred = torch.nn.functional.softmax(predictions).float()
                else:
                    y_pred = (torch.sigmoid(predictions) > 0.5).float()

                if args.global_metrics == 'True':
                    y_pred_masked = torch.masked_select(y_pred, ignore_indices.repeat(batch_y.size()[1], predictions.size()[1])).view(-1,predictions.size()[1])
                    y_true_masked = torch.masked_select(batch_y, ignore_indices)
                    if len(y_pred_masked) == 0 or len(y_true_masked) == 0:
                        metrics_test = train_metrics(metrics_test, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
                    else:
                        mcc, prec, rec, spec, balanced_acc, f1_score, auc = custom_metrics(y_true_masked, y_pred_masked, predictions, batch_y, args)
                        metrics_test = train_metrics(metrics_test, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
                else:
                    spec, balanced_acc, mcc, auc = label_metrics(batch_y, y_pred, args)
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
    elif args.prepare_data == 'ESM':
        dataset_preparation_esm(args, os.listdir('dataset/fasta'))

    if args.train == 'True':
        print('-----------------Prepare Dataset--------------------')
        with open('dataset/prepared/data_processed.pickle', 'rb') as handle:
            X = dict(pickle.load(handle))
        with open('dataset/prepared/labels.pickle', 'rb') as handle:
            y = dict(pickle.load(handle))

        with open('dataset/prepared/mapping.pickle', 'rb') as handle:
            mapping = dict(pickle.load(handle))
        with open('dataset/prepared/uniprot_to_pdb.pickle', 'rb') as handle:
            uniprot_to_pdb = pickle.load(handle)

        print('-----------------Create Folding---------------------')
        graphs = {}

        for data in os.listdir('embedding/graphs'):
            graph = pickle.load(open(f'embedding/graphs/{data}', 'rb'))
            graphs[data.split('.pickle')[0]] = {'edge_index': torch.tensor(list(graph.edges)).t().contiguous(), 
                                                'features': torch.tensor([graph._node[u]['features'] for u in graph.nodes()], dtype=torch.float)}

        graphs_mapped = []
        for graphName, graphValues in graphs.items():
            if graphName in uniprot_to_pdb.keys():
                graphs_mapped.append((graphName, Data(x=graphValues['features'], edge_index=graphValues['edge_index'])))

        graphs_mapped = dict(graphs_mapped)
        for key, value in uniprot_to_pdb.items():
            if value in list(X.keys()) and key not in list(graphs_mapped.keys()):
                del X[value]
                del y[value]
                del mapping[value]

        X = np.array([val1 for _, val1 in X.items()])
        y = np.array([val1 for _, val1 in y.items()])
        graphs_mapped = [val1 for _, val1 in graphs_mapped.items()]
        mapping = [(val1, _) for _, val1 in mapping.items()]

        if args.fold_mapping == 'True':
            mapping = np.array(mapping_split(os.listdir('split'), mapping, args))
            IDs = mapping[:,2].astype(int)

            n_splits = 4
            group = 0
            group_kfold = GroupKFold(n_splits=n_splits)
            fold = group_kfold.split(X, y, groups=IDs)
        else:
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

            batch_size = 32
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