import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from focal_loss.focal_loss import FocalLoss
from sklearn.model_selection import GroupKFold
from metrics import custom_metrics, train_metrics, label_metrics, save_auc, save_bac
from models import NeuralNetwork, CustomMatrixDataset, LossWrapper, TRAM, TRAM_Att, TRAM_Att_solo, NN2, NN3
from dataloader import dataset_preparation, mapping_split

parser = argparse.ArgumentParser(description=('dataloader.py prepare dataset for training, validation, test'))
parser.add_argument('--prepare_data', required=False, default=False, help='Tell if necessary to extract files for dataset creation')
parser.add_argument('--metrics', required=False, default=True, help='Tell if necessary to extract metrics')
parser.add_argument('--difference', required=True, default=True, help='Tell if necessary to extract metrics')

args = parser.parse_args()
global_metrics = False
difference = True


def train_test(num_epochs, dimension, train_loader, test_loader, device, group):
    input_size = dimension[-1]
    output_size = 15

    model = NN2(input_size, dimension[-2], output_size)
    model.to(device)

    loss = FocalLoss(gamma=2)
    criterion = LossWrapper(loss=loss, ignore_index=-999)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_acc, test_acc, auc_train, auc_test = [], [], [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        metrics = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": [], "auc": []}

        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device) 

            if difference:
                outputs = model(inputs)
            else:
                split_tensors = torch.split(inputs, split_size_or_sections=1, dim=1)
                outputs = model(split_tensors[0], split_tensors[1])

            m = nn.Sigmoid()
            try:
                loss = criterion(m(outputs), labels.long())
            except Exception as error:
                print('TRAIN')
                print(f'{error} and {inputs}')

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            ignore_indices = (labels != -999)
            y_pred = (torch.sigmoid(outputs) > 0.5).float()            

            if global_metrics:
                y_pred_masked = torch.masked_select(y_pred, ignore_indices)
                y_true_masked = torch.masked_select(labels, ignore_indices)
                mcc, prec, rec, spec, balanced_acc, f1_score, auc = custom_metrics(y_true_masked, y_pred_masked, outputs, labels)
                metrics = train_metrics(metrics, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
            else:
                spec, balanced, mcc, auc = label_metrics(labels, y_pred)
                metrics["spec"].append(spec)
                metrics["balanced_acc"].append(balanced)
                metrics["mcc"].append(mcc)
                metrics["auc"].append(auc)

        if global_metrics:
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
            metrics = {"mcc": [], "spec": [], "balanced_acc": [], "auc": []}

            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                if difference:
                    predictions = model(batch_X)
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
                
                if global_metrics:
                    y_pred_masked = torch.masked_select(y_pred, ignore_indices)
                    y_true_masked = torch.masked_select(batch_y, ignore_indices)
                    mcc, prec, rec, spec, balanced_acc, f1_score, auc = custom_metrics(y_true_masked, y_pred_masked, predictions, batch_y)
                    metrics = train_metrics(metrics, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
                else:
                    spec, balanced, mcc, auc = label_metrics(batch_y, y_pred)
                    metrics["spec"].append(spec)
                    metrics["balanced_acc"].append(balanced)
                    metrics["mcc"].append(mcc)
                    metrics["auc"].append(auc)

            avg_loss = total_loss / len(test_loader)
            
            if global_metrics:
                print(f"Test Loss: {avg_loss:.4f}, Matthews Test Mean: {np.mean(metrics['mcc'])}, Specificity: {np.mean(metrics['spec'])}, Balanced Accuracy: {np.mean(metrics['balanced_acc'])}")
                auc_test.append(np.mean(metrics['auc']))
                test_acc.append(np.mean(metrics['balanced_acc']))
                print('\n')
            else:           
                print(f"Test Loss: {avg_loss:.4f}, AUC: {torch.nanmean(torch.stack(metrics['auc']), dim=0).tolist()}")
                auc_test.append(torch.nanmean(torch.stack(metrics["auc"]), dim=0))
                test_acc.append(torch.nanmean(torch.stack(metrics["balanced_acc"]), dim=0))
                print('\n')

    #save_bac(num_epochs, train_acc, test_acc, group)
    save_auc(num_epochs, auc_train, auc_test, group, global_metrics)


def main_train(device):
    # Extract Features matrix and prepare Dataset
    if args.prepare_data == True:
        print('-----------------Extracting Dataset-----------------')
        dataset_preparation(args.difference)

    print('-----------------Prepare Dataset-----------------')
    X = np.array(np.load('dataset/prepared/data_processed.npy'))
    y = np.array(np.load('dataset/prepared/labels.npy'))
    mapping = np.array(mapping_split(os.listdir('split')))


    # Prepare Group Folding
    IDs = mapping[:,2]

    n_splits = 5
    group = 0
    group_kfold = GroupKFold(n_splits=n_splits)

    for train_index, test_index in group_kfold.split(X, y, groups=IDs):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_dataset = CustomMatrixDataset(X_train, y_train)
        test_dataset = CustomMatrixDataset(X_test, y_test)

        batch_size = 10
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        print(f'-----------------Start Training Group {group}------------------')
        train_test(10, X.shape, train_loader, test_loader, device, group)
        group += 1


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if not os.path.exists('dataset/prepared'):
        os.mkdir('dataset/prepared')
    
    main_train(device)