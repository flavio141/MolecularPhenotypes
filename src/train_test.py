import os
import torch
import argparse
import pickle
import numpy as np
import torch.nn as nn
import networkx as nx
import torch.optim as optim

from utility import one_hot_aminoacids
from torch.utils.data import DataLoader
from focal_loss.focal_loss import FocalLoss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from metrics import custom_metrics, train_metrics, test_metrics, label_metrics, save_auc, save_bac
from models import NeuralNetwork, CustomMatrixDataset, MatrixDataset, LossWrapper, TRAM, TRAM_Att, TRAM_Att_solo, TRAM_Att_solo_one_hot, NN1, NN2, NN3, NN4, NN5, NN6
from dataloader import dataset_preparation, mapping_split, dataset_preparation_proteinbert
from paper import train_test_paper

parser = argparse.ArgumentParser(description=('dataloader.py prepare dataset for training, validation, test'))
parser.add_argument('--prepare_data', required=False, default='True', help='Tell if necessary to extract files for dataset creation')
parser.add_argument('--metrics', required=False, default='True', help='Tell if necessary to extract metrics')
parser.add_argument('--difference', required=True, default='True', help='Tell if necessary to extract metrics')
parser.add_argument('--one_hot', required=False, default='True', help='Tell if necessary to use one hot encoding')
parser.add_argument('--trials', required=True, default=0, help='Tell which trials you are performing')
parser.add_argument('--epochs', required=False, default=10, help='Tell how many epochs you need to do')
parser.add_argument('--global_metrics', required=True, default='False', help='Tell if it is necessary to extract global metrics')
parser.add_argument('--fold_mapping', required=True, default='False', help='Tell if we are dealing with the paper')
parser.add_argument('--train', required=False, default='True', help='Tell if we are dealing with the paper')

args = parser.parse_args()


def train_test(num_epochs, dimension, train_loader, test_loader, device, group):
    input_size = dimension[-1]
    output_size = 3

    model = NN5(input_size, dimension[-2], output_size)
    model.to(device)

    loss = FocalLoss(gamma=2)
    criterion = LossWrapper(loss=loss, ignore_index=-999)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    train_acc, test_acc, auc_train, auc_test = [], [], [], []

    for epoch in range(int(num_epochs)):
        running_loss = 0.0
        metrics = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": [], "auc": []}

        model.train()
        for inputs, labels, indices in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            tensor_one_hot = []

            if args.one_hot == 'True' and args.difference == 'True':
                for mapping in indices:
                    mutation = mapping.split('_')[-1]
                    wildtype = mapping.split('_')[2]
                    one_hot = one_hot_aminoacids(wildtype=wildtype, mutation=mutation)
                    tensor_one_hot.append(one_hot)

                tensor_one_hot = torch.stack(tensor_one_hot, dim=0)
                tensor_one_hot = tensor_one_hot.to(device)
                outputs = model(inputs, tensor_one_hot)
            elif args.one_hot == 'False' and args.difference == 'True':
                outputs = model(inputs)
            elif args.one_hot == 'True' and args.difference == 'False':
                for mapping in indices:
                    mutation = mapping.split('_')[-1]
                    wildtype = mapping.split('_')[2]
                    one_hot = one_hot_aminoacids(wildtype=wildtype, mutation=mutation)
                    tensor_one_hot.append(one_hot)

                tensor_one_hot = torch.stack(tensor_one_hot, dim=0)
                tensor_one_hot = tensor_one_hot.to(device)
                outputs = model(tensor_one_hot)
            else:
                split_tensors = torch.split(inputs, split_size_or_sections=1, dim=1)
                outputs = model(split_tensors[0], split_tensors[1])

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

        if args.global_metrics == 'True':
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {running_loss}, Matthews Mean: {np.mean(metrics['mcc'])}, Precision Mean: {np.mean(metrics['prec'])}, Recall Mean: {np.mean(metrics['rec'])}, Specificity: {np.mean(metrics['spec'])}, Balanced Accuracy: {np.mean(metrics['balanced_acc'])}, F1 Score Mean: {np.mean(metrics['f1_score'])}")
            auc_train.append(np.mean(metrics["auc"]))
            train_acc.append(np.mean(metrics['balanced_acc']))
        else:
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {running_loss}, AUC: {torch.nanmean(torch.stack(metrics['auc']), dim=0).tolist()}")
            auc_train.append(torch.nanmean(torch.stack(metrics["auc"]), dim=0))
            train_acc.append(torch.nanmean(torch.stack(metrics["balanced_acc"]), dim=0))

        model.eval()

        with torch.no_grad():
            total_loss = 0
            metrics_test = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": [], "auc": []}

            for batch_X, batch_y, batch_indices in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
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
                    predictions = model(batch_X, tensor_one_hot)
                elif args.one_hot == 'False' and args.difference == 'True':
                    predictions = model(batch_X)
                elif args.one_hot == 'True' and args.difference == 'False':
                    for mapping in batch_indices:
                        mutation = mapping.split('_')[-1]
                        wildtype = mapping.split('_')[2]
                        one_hot = one_hot_aminoacids(wildtype=wildtype, mutation=mutation)
                        tensor_one_hot.append(one_hot)

                    tensor_one_hot = torch.stack(tensor_one_hot, dim=0)
                    tensor_one_hot = tensor_one_hot.to(device)
                    predictions = model(tensor_one_hot)
                else:
                    split_tensors = torch.split(batch_X, split_size_or_sections=1, dim=1)
                    predictions = model(split_tensors[0], split_tensors[1])
                
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

            if args.global_metrics == 'True':
                print(f"Test Loss: {avg_loss:.4f}, Matthews Test Mean: {np.mean(metrics_test['mcc'])}, Specificity: {np.mean(metrics_test['spec'])}, Balanced Accuracy: {np.mean(metrics_test['balanced_acc'])}")
                auc_test.append(np.mean(metrics_test['auc']))
                test_acc.append(np.mean(metrics_test['balanced_acc']))
                print('\n')
            else:           
                print(f"Test Loss: {avg_loss:.4f}, AUC: {torch.nanmean(torch.stack(metrics_test['auc']), dim=0).tolist()}")
                auc_test.append(torch.nanmean(torch.stack(metrics_test["auc"]), dim=0))
                test_acc.append(torch.nanmean(torch.stack(metrics_test["balanced_acc"]), dim=0))
                print('\n')

    save_auc(num_epochs, auc_train, auc_test, group, args)



def main_train(device):
    # Extract Features matrix and prepare Dataset
    if args.prepare_data == 'True':
        print('-----------------Extracting Dataset-----------------')
        dataset_preparation(args)
    elif args.prepare_data == 'False':
        pass
    else:
        dataset_preparation_proteinbert(args, os.listdir('dataset/fasta'))

    if args.train == 'True':
        print('-----------------Prepare Dataset-----------------')
        X = np.array(np.load('dataset/prepared/data_processed.npy'))
        y = np.array(np.load('dataset/prepared/labels.npy'))
        #list_of_dicts = pickle.load('embedding/additional_features/graphs.pickle')
        #graphs = [nx.Graph(graph) for graph in list_of_dicts]

        if args.fold_mapping == 'True':
            mapping = np.array(mapping_split(os.listdir('split'), args))
            IDs = mapping[:,2]

            n_splits = 5
            group = 0
            group_kfold = GroupKFold(n_splits=n_splits)
            fold = group_kfold.split(X, y, groups=IDs)
        else:
            mapping = np.array(mapping_split(os.listdir('split'), args))

            n_splits = 4
            group = 0
            kfold = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
            y = y.reshape(-1,1)
            fold = kfold.split(X, y)
            

        for train_index, test_index in fold:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #graphs_train, graphs_test = graphs[train_index], graphs[test_index]

            if args.fold_mapping == 'True':
                train_indices, test_indices = mapping[train_index][:,1], mapping[test_index][:,1]

                train_dataset = CustomMatrixDataset(X_train, y_train, train_indices)
                test_dataset = CustomMatrixDataset(X_test, y_test, test_indices)
            else:
                train_indices, test_indices = mapping[train_index][:,1], mapping[test_index][:,1]

                train_dataset = MatrixDataset(X_train, y_train, train_indices)
                test_dataset = MatrixDataset(X_test, y_test, test_indices) 

            batch_size = 64
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            print(f'-----------------Start Training Group {group}------------------')
            if args.fold_mapping == 'True':
                train_test(args.epochs, X.shape, train_loader, test_loader, device, group)
            else:
                train_test_paper(args.epochs, X.shape, train_loader, test_loader, device, group, args)
            group += 1


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    if not os.path.exists('dataset/prepared'):
        os.mkdir('dataset/prepared')
    
    main_train(device)