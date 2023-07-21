import torch

def matthews_coefficient(y_true_class, y_pred_class, labels):
    confusion_matrix = torch.zeros(labels, labels)
    for i in range(len(y_true_class)):
        confusion_matrix[y_true_class[i]][y_pred_class[i]] += 1

    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator

    # Matthews Mean
    mcc_mean = mcc.mean()

    print("Matthews Coefficient (MCC) Mean:", mcc_mean.item())