import os
import argparse
from src.preprocessing import extract_data

parser = argparse.ArgumentParser(description=('main.py extract with UniProtID the FASTA information'))
parser.add_argument('--extract_data', required=False, default=True, help='Tell if necessary to extract FASTA information from UniProtID')

folders = ['dataset/fasta', 'dataset/pdb', 'dataset/cif', 'embedding']

def main():
    args = parser.parse_args()

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if args.extract_data == True:
        extract_data(args.extract_data)

if __name__ == "__main__":
    main()