import os
import argparse
from src.preprocessing import extract_data_wt
from src.processing import extract_data_mut

parser = argparse.ArgumentParser(description=('main.py extract with UniProtID the FASTA information'))
parser.add_argument('--extract_data', required=False, default=True, help='Tell if necessary to extract FASTA information from UniProtID')

folders = ['dataset/fasta_wt', 'dataset/pdb_wt', 'dataset/cif_wt', 'dataset/pdb_temp',
           'embedding/fastaEmb_wt', 'embedding/distmap_wt', 'embedding/results_wt']

def main():
    args = parser.parse_args()

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if args.extract_data == True:
        extract_data_wt(args.extract_data)
        extract_data_mut()


if __name__ == "__main__":
    main()