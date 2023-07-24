import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from metrics import custom_metrics
from models import NeuralNetwork, CustomMatrixDataset, LossWrapper
from dataloader import dataset_preparation, mapping_split

parser = argparse.ArgumentParser(description=('dataloader.py prepare dataset for training, validation, test'))
parser.add_argument('--prepare_data', required=False, default=False, help='Tell if necessary to extract files for dataset creation')

args = parser.parse_args()

def train_test(num_epochs, rows, train_loader, test_loader, device):
    input_size = 1024
    output_size = 15

    model = NeuralNetwork(input_size, rows, output_size)
    model.to(device)

    loss = nn.BCEWithLogitsLoss()
    criterion = LossWrapper(loss=loss, ignore_index=-999)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        metrics = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": []}

        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device) 
            
            split_tensors = torch.split(inputs, split_size_or_sections=1, dim=1)
            outputs = model(split_tensors[0], split_tensors[1])
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            ignore_indices = (labels != -999)
            y_pred = (torch.sigmoid(outputs) > 0.5).float()
            y_pred_masked = torch.masked_select(y_pred, ignore_indices)
            y_true_masked = torch.masked_select(labels, ignore_indices)
            mcc, prec, rec, spec, balanced_acc, f1_score = custom_metrics(y_pred_masked, y_true_masked)
            metrics["mcc"].append(mcc)
            metrics["prec"].append(prec)
            metrics["rec"].append(rec)
            metrics["balanced_acc"].append(balanced_acc)
            metrics["spec"].append(spec)
            metrics["f1_score"].append(f1_score)

        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {running_loss}, Matthews Mean: {np.mean(metrics['mcc'])}, Precision Mean: {np.mean(metrics['prec'])}, Recall Mean: {np.mean(metrics['rec'])}, Specificity: {np.mean(metrics['spec'])}, Balanced Accuracy: {np.mean(metrics['balanced_acc'])}, F1 Score Mean: {np.mean(metrics['f1_score'])}")

    model.eval()
    with torch.no_grad():
        total_loss = 0
        metrics = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": []}

        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            split_tensors = torch.split(batch_X, split_size_or_sections=1, dim=1)
            predictions = model(split_tensors[0], split_tensors[1])
            total_loss += criterion(predictions, batch_y).item()

            ignore_indices = (batch_y != -999)
            y_pred = (torch.sigmoid(predictions) > 0.5).float()
            y_pred_masked = torch.masked_select(y_pred, ignore_indices)
            y_true_masked = torch.masked_select(batch_y, ignore_indices)
            mcc, prec, rec, spec, balanced_acc, f1_score = custom_metrics(y_pred_masked, y_true_masked)
            metrics["mcc"].append(mcc)
            metrics["prec"].append(prec)
            metrics["rec"].append(rec)
            metrics["balanced_acc"].append(balanced_acc)
            metrics["spec"].append(spec)
            metrics["f1_score"].append(f1_score)
        
        
        ignore_indices = (batch_y != -999)
        y_pred = (torch.sigmoid(predictions) > 0.5).float()
        y_pred_masked = torch.masked_select(y_pred, ignore_indices)
        y_true_masked = torch.masked_select(batch_y, ignore_indices)

        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}, Matthews Test Mean: {np.mean(metrics['mcc'])}, Precision Test Mean: {np.mean(metrics['prec'])}, Recall Test Mean: {np.mean(metrics['rec'])}, Specificity: {np.mean(metrics['spec'])}, Balanced Accuracy: {np.mean(metrics['balanced_acc'])}, F1 Score Test Mean: {np.mean(metrics['f1_score'])}")
        print('\n')


def main_train(device):
    if args.prepare_data == True:
        dataset_preparation()
    
    print('-----------------Prepare Dataset-----------------')
    X = np.array(np.load('dataset/prepared/data_processed.npy'))
    y = np.array(np.load('dataset/prepared/labels.npy'))
    mapping = np.array(mapping_split(os.listdir('split')))

    IDs = mapping[:,2]

    n_splits = 5
    group = 0
    group_kfold = GroupKFold(n_splits=n_splits)

    for train_index, test_index in group_kfold.split(X, y, groups=IDs):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_dataset = CustomMatrixDataset(X_train, y_train)
        test_dataset = CustomMatrixDataset(X_test, y_test)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        print(f'-----------------Start Training Group {group}------------------')
        group += 1
        train_test(50, X.shape[2], train_loader, test_loader, device)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if not os.path.exists('dataset/prepared'):
        os.mkdir('dataset/prepared')
    
    main_train(device)