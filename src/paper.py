import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from focal_loss.focal_loss import FocalLoss
from metrics import custom_metrics, train_metrics, label_metrics, save_auc
from models import LossWrapper, TRAM_Att_solo, NN1, NN2, NN3

def train_test_paper(num_epochs, dimension, train_loader, test_loader, device, group, args):
    input_size = dimension[-1]
    output_size = 1

    model = TRAM_Att_solo(input_size, dimension[-2], output_size)
    model.to(device)

    loss = FocalLoss(gamma=2)
    criterion = LossWrapper(loss=loss, ignore_index=-999)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    train_acc, test_acc, auc_train, auc_test = [], [], [], []

    for epoch in range(int(num_epochs)):
        running_loss = 0.0
        metrics = {"mcc": [], "prec": [], "rec": [], "spec": [], "balanced_acc": [], "f1_score": [], "auc": []}

        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)

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

            if args.global_metrics:
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

        if args.global_metrics:
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

            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                
                m = nn.Sigmoid()
                try:
                    total_loss += criterion(m(predictions), batch_y.long()).item()
                except Exception as error:
                    print('TEST')
                    print(f'{error} and {batch_X}')

                ignore_indices = (batch_y != -999)
                y_pred = (torch.sigmoid(predictions) > 0.5).float()

                if args.global_metrics:
                    y_pred_masked = torch.masked_select(y_pred, ignore_indices)
                    y_true_masked = torch.masked_select(batch_y, ignore_indices)
                    mcc, prec, rec, spec, balanced_acc, f1_score, auc = custom_metrics(y_true_masked, y_pred_masked, predictions, batch_y)
                    metrics_test = train_metrics(metrics_test, mcc, prec, rec, spec, balanced_acc, f1_score, auc)
                else:
                    spec, balanced, mcc, auc = label_metrics(batch_y, y_pred)
                    metrics_test["spec"].append(spec)
                    metrics_test["balanced_acc"].append(balanced)
                    metrics_test["mcc"].append(mcc)
                    metrics_test["auc"].append(auc)

            avg_loss = total_loss / len(test_loader)
            
            if args.global_metrics:
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