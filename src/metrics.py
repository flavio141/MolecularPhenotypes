import os
import torch
import openpyxl

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


def save_metrics(results, process=None):
    mapping = {"mcc": 2, "balanced_acc": 3, "spec": 4, "prec": 5, "rec": 6, "f1_score": 7}

    if process == 'train':
        if not os.path.exists('results/training_test.xlsx'):
            wb = openpyxl.Workbook()
            sheet = wb.active
            sheet.title = "Train"

            sheet["A1"] = "Epoch"
            sheet["B1"] = "Matthews Coefficient"
            sheet["C1"] = "Balanced Accuracy"
            sheet["D1"] = "Specificity"
            sheet["E1"] = "Precision"
            sheet["F1"] = "Recall"
            sheet["G1"] = "F1 Score"

            for keys, subkeys in results.items():
                row = 2
                column = mapping[keys]
                for item in subkeys:
                    sheet.cell(row=row, column=column, value=item)
                    row += 1

            wb.save("results/training_test.xlsx")
        else:
            wb = openpyxl.load_workbook('results/training_test.xlsx')
            sheet = wb["Train"]

            for keys, subkeys in results.items():
                row = sheet.max_row + 1
                column = mapping[keys]
                for item in subkeys:
                    sheet.cell(row=row, column=column, value=item)
                    row += 1
            
            wb.save("results/training_test.xlsx")
            wb.close()

    elif process == 'test':
        wb = openpyxl.load_workbook('results/training_test.xlsx')
        sheet = wb["Test"]

        sheet["A1"] = "Matthews Coefficient"
        sheet["B1"] = "Balanced Accuracy"
        sheet["C1"] = "Specificity"

        for keys, subkeys in results.items():
            row = sheet.max_row + 1
            column = mapping[keys]
            for item in subkeys:
                sheet.cell(row=row, column=column, value=item)
                row += 1
        
        wb.save("results/training_test.xlsx")
        wb.close()
