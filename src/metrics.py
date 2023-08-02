import torch
import numpy as np
import openpyxl as xl
import matplotlib.pyplot as plt
from torchmetrics import AUROC

auroc = AUROC(task='binary', ignore_index=-999)


def custom_metrics(y_true, y_pred, digits, labels):
    true_positives = torch.logical_and(y_true == 1., y_pred == 1.).sum().item()
    true_negatives = torch.logical_and(y_true == 0., y_pred == 0.).sum().item()
    false_positives = torch.logical_and(y_true == 0., y_pred == 1.).sum().item()
    false_negatives = torch.logical_and(y_true == 1., y_pred == 0.).sum().item()

    # Calcola il coefficiente di Matthews
    mcc = ((true_positives * true_negatives) - (false_positives * false_negatives)) / (
        (true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives) + 1e-08) ** 0.5
    
    prec = true_positives / (true_positives + false_positives + 1e-08)

    rec = true_positives / (true_positives + false_negatives + 1e-08)

    spec = true_negatives / (true_negatives + false_positives + 1e-08)

    f1_score = (2 * prec * rec) / (prec + rec + 1e-08)

    balanced_acc = (rec + spec) / 2

    auc = auroc(digits, labels).item()

    return mcc, prec, rec, spec, balanced_acc, f1_score, auc


def label_metrics(y_true, y_pred):
    spec_per_column = []
    balanced_per_column = []
    matthews_per_column = []
    auc_per_column = []

    y_true_processed = y_true.clone()
    y_true_processed[y_true_processed == -999] = float('nan')

    for col in range(y_true_processed.size(1)):
        label_column = y_true_processed[:, col]
        pred_column = y_pred[:, col]
        
        valid_indices = ~torch.isnan(label_column)
        pred_column = pred_column[valid_indices]
        label_column = label_column[valid_indices]

        if len(pred_column) > 0:
            mcc, _, _, spec, balanced_acc, _, auc = custom_metrics(label_column, pred_column, y_pred, y_true)

            spec_per_column.append(torch.tensor(spec))
            balanced_per_column.append(torch.tensor(balanced_acc))
            matthews_per_column.append(torch.tensor(mcc))
            auc_per_column.append(torch.tensor(auc))

        else:
            spec_per_column.append(torch.tensor(float('nan')))
            balanced_per_column.append(torch.tensor(float('nan')))
            matthews_per_column.append(torch.tensor(float('nan')))
            auc_per_column.append(torch.tensor(float('nan')))

    return torch.stack(spec_per_column), torch.stack(balanced_per_column), torch.stack(matthews_per_column), torch.stack(auc_per_column)


def train_metrics(metrics, mcc, prec, rec, spec, balanced_acc, f1_score, auc):
    metrics["mcc"].append(mcc)
    metrics["prec"].append(prec)
    metrics["rec"].append(rec)
    metrics["balanced_acc"].append(balanced_acc)
    metrics["spec"].append(spec)
    metrics["f1_score"].append(f1_score)
    metrics["auc"].append(auc)

    return metrics


def save_bac(num_epochs, train_acc, test_acc, group, global_metric):
    if global_metric:
        plt.plot(range(num_epochs), train_acc, label='Train Balanced Accuracy', color='blue')
        plt.plot(range(num_epochs), test_acc, label='Test Balanced Accuracy', color='red')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Balanced Accuracy - Train vs. Test')
        plt.legend()
        plt.show()
        plt.savefig(f'plots/train_vs_test_{group}.png')

    plt.clf()
    plt.close()

def save_auc(num_epochs, auc_train, auc_test, group, global_metrics):
    if global_metrics:
        plt.plot(range(num_epochs), auc_train, label='Train AUC', color='blue')
        plt.plot(range(num_epochs), auc_test, label='Test AUC', color='red')

        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.title('AUC per Epochs - Train vs. Test')
        plt.legend()
        plt.show()
        plt.savefig(f'plots/AUCTrain_vs_AUCTest_{group}.png')

        plt.clf()
        plt.close()
    else:
        wb = xl.Workbook()
        ws = wb.active
        ws.append([f'Label AUC: {label}'for label in range(15)])

        for train in auc_train:
            ws.append(train.tolist())

        ws.append([])
        ws.append([f'Label Test AUC: {label}'for label in range(15)])

        for test in auc_test:
            ws.append(test.tolist())
        
        wb.save(f'results/Fold{group}_Results.xlsx')
        wb.close()
