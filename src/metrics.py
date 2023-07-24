import torch

def custom_metrics(y_true, y_pred):
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

    return mcc, prec, rec, spec, balanced_acc, f1_score