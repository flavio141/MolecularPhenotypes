import openpyxl

def save_similarity(mutation_similarity):
    wb = openpyxl.Workbook()
    sheet = wb.active

    sheet["A1"] = "PDB wt"
    sheet["B1"] = "PDB mutated"
    sheet["C1"] = "Similarity: Cosine"
    sheet["D1"] = "Similarity: Pearson"

    row = 2
    for keys, subkeys in mutation_similarity.items():
        for subkey, value in subkeys.items():
            sheet.cell(row=row, column=1, value=keys)
            sheet.cell(row=row, column=2, value=subkey)
            sheet.cell(row=row, column=3, value=value[0])
            sheet.cell(row=row, column=4, value=value[1])
            row += 1

    wb.save("additional_features/similarities.xlsx")


def save_percentage(perc):
    wb = openpyxl.Workbook()
    sheet = wb.active

    sheet["A1"] = "PDB wt"
    sheet["B1"] = "Percentage Position"

    row = 2
    for keys, subkeys in perc.items():
        sheet.cell(row=row, column=1, value=keys)
        sheet.cell(row=row, column=2, value=subkeys)
        row += 1

    wb.save("additional_features/percentage.xlsx")