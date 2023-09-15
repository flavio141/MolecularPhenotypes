import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
from jax_unirep import get_reps

data = pd.read_csv('dataset/SNV.tsv', sep='\t')

def UniRep_embedding(fastas, fastas_mut):
    unirep_emb = {}

    for fasta, count in zip(fastas, tqdm(range(0, len(fastas)), desc='Extract features with UniRep')):
        sequence = ''
        with open(f'dataset/fasta/{fasta}', 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    continue
                sequence += line
        
        embedding, h_final, c_final = get_reps(sequence)
        unirep_emb[fasta.split('.')[0]] = [embedding]
        
        fasta_present = [fasta_mut for fasta_mut in fastas_mut if fasta.split('.')[0] in fasta_mut]

        for fasta_mut in fasta_present:
            sequence = ''
            with open(f'dataset/fasta_mut/{fasta_mut}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('>'):
                        continue
                    sequence += line
            
            embedding_mut, h_final, c_final = get_reps(sequence)
            unirep_emb[fasta.split('.')[0]].append(embedding_mut)

    with open(f'embedding/additional_features/unirep.pkl', 'wb') as unirep:
        pickle.dump(unirep_emb, unirep, protocol=pickle.HIGHEST_PROTOCOL)


def process_unirep(difference):
    with open('embedding/additional_features/unirep.pkl', 'rb') as file:
        unirep = pickle.load(file)

    fastas_mut = os.listdir('dataset/fasta_mut')
    wildtype_mutated, protein_to_mut, labels = [], [], []

    try:
        pbar = tqdm(total=len(unirep))
        for key, value in unirep.items():
            features_wt = unirep[key][0]

            fasta_present = [fasta_mut for fasta_mut in fastas_mut if key in fasta_mut]

            for count, mut in enumerate(fasta_present):
                values = mut.split('_')
                pdb_id = '_'.join(mut.split('_')[:2])
                features_mut = unirep[key][count + 1]


                matrix_tuple = (features_wt, features_mut)
                information = tuple(map(tuple, data.loc[(data['pdb_id'] == pdb_id.replace('_',':')) & 
                                            (data['mutation'].str.contains(values[4].split('.')[0])) & 
                                            (data['position'] == int(values[3]))].values))[0]

                if difference:
                    wildtype_mutated.append((features_wt - features_mut))
                else:
                    wildtype_mutated.append(matrix_tuple)
                
                protein_to_mut.append((information[0], mut.split('.')[0]))
                labels.append(np.nan_to_num(np.abs(information[-15:]), nan=-999))
            
            pbar.update()
        pbar.close()
    except Exception as error:
        print(error)

    assert len(protein_to_mut) == len(wildtype_mutated) == len(labels)
    
    np.save('dataset/prepared/mapping_unirep.npy', protein_to_mut)
    np.save('dataset/prepared/data_processed_unirep.npy', wildtype_mutated)
    np.save('dataset/prepared/labels_unirep.npy', labels)


if __name__ == "__main__":
    process_unirep(difference=True)