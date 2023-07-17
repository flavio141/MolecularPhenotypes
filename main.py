import os
import sys
import argparse
from src.unirep_emb import UniRep_embedding
from src.preprocessing import extract_data_wt, extract_data_mut, similarity, percentage

sys.path.append("UniRep")

parser = argparse.ArgumentParser(description=('main.py extract with UniProtID and PDB files all the features for the wildtype Protein'))
parser.add_argument('--extract_data_wt', required=False, default=False, help='Tell if necessary to extract information using UniProtID and PDB files')
parser.add_argument('--extract_data_mut', required=False, default=False, help='Tell if necessary to extract information using UniProtID and PDB files for mutated proteins')
parser.add_argument('--extract_similarity', required=False, default=True, help='Tell if necessary to extract similarities between protein features mutated and wildtypes')
parser.add_argument('--extract_percentage', required=False, default=False, help='Tell if necessary to extract the percentage postion of the mutation')


folders = ['dataset/fasta', 'dataset/pdb', 'dataset/cif', 'dataset/pdb_temp',
           'dataset/fasta_mut', 'embedding/fastaEmb_mut', 'embedding/additional_features',
           'embedding/fastaEmb_wt', 'embedding/distmap_wt', 'embedding/structural_wt', 'embedding/results_wt']

def main():
    args = parser.parse_args()

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    extract_data_wt(args.extract_data_wt)
    extract_data_mut(args.extract_data_mut)


    if args.extract_similarity == True:
        similarity()
    if args.extract_percentage == True:
        percentage()

    UniRep_embedding(os.listdir('dataset/fasta'))


if __name__ == "__main__":
    main()