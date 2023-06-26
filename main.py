import os
import argparse
from src.preprocessing import read_data

parser = argparse.ArgumentParser(description=('main.py extract with UniProtID the FASTA information'))
parser.add_argument('--extract_data', required=False, default=True, help='Tell if necessary to extract FASTA information from UniProtID')


def main():
    args = parser.parse_args()

    if not os.path.exists('dataset/fasta'):
        os.makedirs('dataset/fasta')

    if not os.path.exists('dataset/pdb'):
        os.makedirs('dataset/pdb')

    if args.extract_data == True:
        read_data(args.extract_data)

if __name__ == "__main__":
    main()