import os
import argparse
from src.preprocessing import extract_data_wt, extract_data_mut

parser = argparse.ArgumentParser(description=('main.py extract with UniProtID and PDB files all the features for the wildtype Protein'))
parser.add_argument('--extract_data_wt', required=False, default=False, help='Tell if necessary to extract information using UniProtID and PDB files')
parser.add_argument('--extract_data_mut', required=False, default=True, help='Tell if necessary to extract information using UniProtID and PDB files for mutated proteins')


folders = ['dataset/fasta', 'dataset/pdb', 'dataset/cif', 'dataset/pdb_temp',
           'dataset/fasta_mut', 
           'embedding/fastaEmb_wt', 'embedding/distmap_wt', 'embedding/structural_wt', 'embedding/results_wt']

def main():
    args = parser.parse_args()

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if args.extract_data_wt == True:
        extract_data_wt(args.extract_data_wt)

    if args.extract_data_mut == True:
        extract_data_mut(args.extract_data_mut)


if __name__ == "__main__":
    main()