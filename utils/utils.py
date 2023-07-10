import openpyxl

def save_excel(mutation_similarity):
    wb = openpyxl.Workbook()

    # Selezionare il foglio di lavoro attivo
    sheet = wb.active

    # Scrivere le intestazioni delle colonne
    sheet["A1"] = "PDB wt"
    sheet["B1"] = "PDB mutated"
    sheet["C1"] = "Similarity"

    # Scrivere i dati nel foglio di lavoro
    row = 2
    for keys, subkeys in mutation_similarity.items():
        for subkey, value in subkeys.items():
            sheet.cell(row=row, column=1, value=keys)
            sheet.cell(row=row, column=2, value=subkey)
            sheet.cell(row=row, column=3, value=value)
            row += 1

    # Salvare il workbook come file Excel
    wb.save("dataset/similarities.xlsx")
