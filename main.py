import argparse
from src.preprocessing import features

parser = argparse.ArgumentParser(description=('main.py extract with UniProtID and PDB files all the features for the wildtype Protein'))
parser.add_argument('--extract_data_wt', required=False, default=True, help='Tell if necessary to extract information using UniProtID and PDB files')
parser.add_argument('--extract_data_mut', required=False, default=False, help='Tell if necessary to extract information using UniProtID and PDB files for mutated proteins')
parser.add_argument('--extract_unirep', required=False, default=False, help='Tell if necessary to extract information using UniRep')
parser.add_argument('--extract_similarity', required=False, default=True, help='Tell if necessary to extract similarities between protein features mutated and wildtypes')


folders = ['dataset/fasta', 'dataset/pdb_chain', 'dataset/cif', 'dataset/pdb_full',
           'dataset/fasta_mut', 'embedding/fastaEmb_mut', 'embedding/additional_features',
           'embedding/fastaEmb_wt', 'embedding/graphs']

args = parser.parse_args()


def main():
    features(args, folders)


if __name__ == "__main__":
    main()