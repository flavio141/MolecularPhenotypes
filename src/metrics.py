import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import AUROC
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, recall_score, f1_score

auroc = AUROC(task='binary', ignore_index=-999)


def custom_metrics(y_true, y_pred, digits, labels, args):
    if args.multiclass == 'True':
        y_true, y_pred = y_true.cpu().detach().numpy(), np.argmax(y_pred.cpu().detach().numpy(), axis=1) - 1
        mcc = matthews_corrcoef(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, average='weighted')
        f1Score = f1_score(y_true, y_pred, average='weighted')
        auc = 0
        prec = 0
        spec = 0

    else:
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

        f1Score = (2 * prec * rec) / (prec + rec + 1e-08)

        balanced_acc = (rec + spec) / 2

        auc = auroc(digits, labels).item()

    return mcc, prec, rec, spec, balanced_acc, f1Score, auc


def label_metrics(y_true, y_pred, args):
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
            mcc, _, _, spec, balanced_acc, _, auc = custom_metrics(label_column, pred_column, y_pred, y_true, args)

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


def test_metrics(metrics_test, mcc, spec, balanced_acc, auc):
    metrics_test["spec"].append(spec)
    metrics_test["balanced_acc"].append(balanced_acc)
    metrics_test["mcc"].append(mcc)
    metrics_test["auc"].append(auc)

    return metrics_test


def save_bac(num_epochs, train_acc, test_acc, group, args):
    if not os.path.exists(f'plots/trials_{args.trials}'):
        os.mkdir(f'plots/trials_{args.trials}')
    
    plt.plot(range(int(num_epochs)), train_acc, label='Train Balanced Accuracy', color='blue')
    plt.plot(range(int(num_epochs)), test_acc, label='Test Balanced Accuracy', color='red')

    plt.text(0.8, 0.9, f'Best Train: {max(train_acc):.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.text(0.8, 0.8, f'Best Test: {max(test_acc):.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Balanced Accuracy - Train vs. Test')
    plt.legend()
    plt.show()
    plt.savefig(f'plots/trials_{args.trials}/BACTrain_vs_BACTest_{group}.png')

    plt.clf()
    plt.close()


def save_spec(num_epochs, spec_train, spec_test, group, args):
    if not os.path.exists(f'plots/trials_{args.trials}'):
        os.mkdir(f'plots/trials_{args.trials}')
    
    plt.plot(range(int(num_epochs)), spec_train, label='Train Balanced Accuracy', color='blue')
    plt.plot(range(int(num_epochs)), spec_test, label='Test Balanced Accuracy', color='red')

    plt.text(0.8, 0.9, f'Best Train: {max(spec_train):.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.text(0.8, 0.8, f'Best Test: {max(spec_test):.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Specificity - Train vs. Test')
    plt.legend()
    plt.show()
    plt.savefig(f'plots/trials_{args.trials}/SPECTrain_vs_SPECTest_{group}.png')

    plt.clf()
    plt.close()

def save_mcc(num_epochs, mcc_train, mcc_test, group, args):
    if not os.path.exists(f'plots/trials_{args.trials}'):
        os.mkdir(f'plots/trials_{args.trials}')
    
    plt.plot(range(int(num_epochs)), mcc_train, label='Train MCC', color='blue')
    plt.plot(range(int(num_epochs)), mcc_test, label='Test MCC', color='red')

    plt.text(0.8, 0.9, f'Best Train: {max(mcc_train):.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.text(0.8, 0.8, f'Best Test: {max(mcc_test):.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xlabel('Epochs')
    plt.ylabel('MCC')
    plt.title('Matthews - Train vs. Test')
    plt.legend()
    plt.show()
    plt.savefig(f'plots/trials_{args.trials}/MCCTrain_vs_MCCTest_{group}.png')

    plt.clf()
    plt.close()


def save_auc(num_epochs, auc_train, auc_test, group, args):
    if not os.path.exists(f'plots/trials_{args.trials}'):
        os.mkdir(f'plots/trials_{args.trials}')

    plt.plot(range(int(num_epochs)), auc_train, label='Train AUC', color='blue')
    plt.plot(range(int(num_epochs)), auc_test, label='Test AUC', color='red')


    plt.text(0.8, 0.9, f'Best Train: {max(auc_train):.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.text(0.8, 0.8, f'Best Test: {max(auc_test):.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('AUC per Epochs - Train vs. Test')
    plt.legend()
    plt.show()
    plt.savefig(f'plots/trials_{args.trials}/AUCTrain_vs_AUCTest_{group}.png')

    plt.clf()
    plt.close()
