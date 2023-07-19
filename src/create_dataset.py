import os
import pickle
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('dataset/database.tsv', sep='\t')

def dataset_preparation():
    wildtype_mutated = []
    labels = []

    for mutated, count in zip(os.listdir('embedding/fastaEmb_mut'), tqdm(range(0, len(os.listdir('embedding/fastaEmb_mut'))), desc= 'Creating dictionary with matrices')):
        pdb_id = '_'.join(mutated.split('_')[:2])
        values = mutated.split('_')

        with open(f'embedding/fastaEmb_wt/{pdb_id}.embeddings.pkl', 'rb') as wt:
            wt_matrix = pickle.load(wt)

        with open(f'embedding/fastaEmb_mut/{mutated}', 'rb') as mut:
            mut_matrix = pickle.load(mut)

        matrix_tuple = (list(wt_matrix.values())[0], list(mut_matrix.values())[0])
        label = tuple(map(tuple, data.loc[(data['pdb_id'] == pdb_id.replace('_',':')) & 
                                        (data['mutation'] == values[4].split('.')[0]) & 
                                        (data['position'] == int(values[3]))].values))[0]
        
        wildtype_mutated.append(matrix_tuple)
        labels.append(label[-15:])

    return wildtype_mutated, labels
