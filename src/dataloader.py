import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('dataset/database.tsv', sep='\t')

def extract_rows_around_position(matrix, row_index, num_rows_to_extract=25):
    num_rows, num_cols = matrix.shape
    rows_before = min(row_index, num_rows_to_extract)
    rows_after = min(num_rows - row_index - 1, num_rows_to_extract)
    
    rows_zeros_before = num_rows_to_extract - rows_before
    rows_zeros_after = num_rows_to_extract - rows_after
    
    extracted_rows = []
    if rows_before > 0:
        extracted_rows.extend(matrix[row_index - rows_before:row_index])
    extracted_rows.extend(matrix[row_index:row_index + rows_after + 1])
    
    zero_row = np.zeros((1, num_cols))[0]
    if rows_zeros_before > 0:
        extracted_rows = [zero_row] * rows_zeros_before + extracted_rows
    if rows_zeros_after > 0:
        extracted_rows = extracted_rows + [zero_row] * rows_zeros_after
    
    result_matrix = np.vstack(extracted_rows)
    return result_matrix


def dataset_preparation(args):
    wildtype_mutated = []
    protein_to_mut = []
    labels = []

    for mutated, _ in zip(os.listdir('embedding/fastaEmb_mut'), tqdm(range(0, len(os.listdir('embedding/fastaEmb_mut'))), desc= 'Creating dictionary with matrices')):
        pdb_id = '_'.join(mutated.split('_')[:2])
        values = mutated.split('_')

        with open(f'embedding/fastaEmb_wt/{pdb_id}.embeddings.pkl', 'rb') as wt:
            wt_matrix = pickle.load(wt)

        with open(f'embedding/fastaEmb_mut/{mutated}', 'rb') as mut:
            mut_matrix = pickle.load(mut)

        features_wt = extract_rows_around_position(list(wt_matrix.values())[0], int(values[3]))
        features_mut = extract_rows_around_position(list(mut_matrix.values())[0], int(values[3]))

        matrix_tuple = (features_wt, features_mut)
        information = tuple(map(tuple, data.loc[(data['pdb_id'] == pdb_id.replace('_',':')) & 
                                          (data['mutation'].str.contains(values[4].split('.')[0])) & 
                                          (data['position'] == int(values[3]))].values))[0]

        if args.difference:
            wildtype_mutated.append((features_wt - features_mut))
        else:
            wildtype_mutated.append(matrix_tuple)
        
        protein_to_mut.append((information[0], mutated.split('.')[0]))
        if args.fold_mapping == 'True':
            labels.append(np.nan_to_num(np.abs(information[-15:]), nan=-999))
        else:
            labels.append(np.nan_to_num(np.abs(int(information[-1].replace('Neutral', '0').replace('Deleterious', '1'))), nan=-999))

    assert len(protein_to_mut) == len(wildtype_mutated) == len(labels)
    
    np.save('dataset/prepared/mapping.npy', protein_to_mut)
    np.save('dataset/prepared/data_processed.npy', wildtype_mutated)
    np.save('dataset/prepared/labels.npy', labels)


def mapping_split(split_files, mapping, args):
    protein_mut_split = []
    protein_to_mut = mapping

    for num, split in enumerate(split_files):
        with open(f'split/{split}', 'r') as file:
            splits = file.read().split('\n')

        for idx, tup in enumerate(protein_to_mut):
            if tup[0] in splits:
                tup = list(tup)
                tup.insert(2, num)
                protein_mut_split.append(tuple(tup))

    return protein_mut_split


def dataset_preparation_proteinbert(args, fastas):
    with open(f'embedding/additional_features/protbert.pkl', 'rb') as wt:
        matrices = pickle.load(wt)

    wildtype_mutated = []
    protein_to_mut = []
    labels = []
    uniprot_to_pdb = {}

    for fasta, _ in zip(fastas, tqdm(range(0, len(fastas)), desc= 'Creating dictionary with matrices')):
        for mutation in matrices[fasta.split('.')[0]]:
            pdb_id = '_'.join(mutation[0].split('_')[:2])
            values = mutation[0].split('_')

            information = tuple(map(tuple, data.loc[(data['pdb_id'] == pdb_id.replace('_',':')) & 
                                            (data['mutation'].str.contains(values[4].split('.')[0])) & 
                                            (data['position'] == int(values[3]))].values))[0]

            if args.fold_mapping != 'True': 
                if information[6] > 0:
                    if ',' in information[4]:
                        for mut in information[4].split(','):
                            uniprot_to_pdb['_'.join([information[5].replace(':', '_'), information[2], str(information[6]), mut])] = mutation[0]
                    else:
                        uniprot_to_pdb['_'.join([information[5].replace(':', '_'), information[2], str(information[6]), information[4]])] = mutation[0]
                    
                    protein_to_mut.append((information[0], mutation[0]))
                    features = extract_rows_around_position(mutation[1].reshape(-1, mutation[1].shape[-1]), int(values[3]))
                    wildtype_mutated.append((information[0], features))
                    labels.append((information[0],np.nan_to_num(np.abs(int(information[-1].replace('Neutral', '0').replace('Deleterious', '1'))), nan=-999)))
            elif args.fold_mapping == 'True' and information[5] > 0:
                if ',' in information[3]:
                    for mut in information[3].split(','):
                        uniprot_to_pdb['_'.join([information[4].replace(':', '_'), information[1], str(information[5]), mut])] = mutation[0]
                else:
                    uniprot_to_pdb['_'.join([information[4].replace(':', '_'), information[1], str(information[5]), information[3]])] = mutation[0]

                protein_to_mut.append((mutation[0], information[0]))
                features = extract_rows_around_position(mutation[1].reshape(-1, mutation[1].shape[-1]), int(values[3]))
                wildtype_mutated.append((mutation[0], features))
                labels.append((mutation[0], np.nan_to_num(np.abs(information[-15:-14]), nan=-999)))
            else:
                pass


    assert len(protein_to_mut) == len(wildtype_mutated) == len(labels)
    
    with open('dataset/prepared/mapping.pickle', 'wb') as handle:
        pickle.dump(protein_to_mut, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('dataset/prepared/data_processed.pickle', 'wb') as handle:
        pickle.dump(wildtype_mutated, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('dataset/prepared/labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('dataset/prepared/uniprot_to_pdb.pickle', 'wb') as handle:
        pickle.dump(uniprot_to_pdb, handle, protocol=pickle.HIGHEST_PROTOCOL)

def dataset_preparation_esm(args, fastas):
    with open(f'embedding/additional_features/esm2.pkl', 'rb') as wt:
        matrices = pickle.load(wt)

    wildtype_mutated = []
    protein_to_mut = []
    labels = []
    uniprot_to_pdb = {}

    for fasta, _ in zip(fastas, tqdm(range(0, len(fastas)), desc= 'Creating dictionary with matrices ESM')):
        for mutation in matrices[fasta.split('.')[0]]:
            pdb_id = '_'.join(mutation[0].split('_')[:2])
            values = mutation[0].split('_')

            information = tuple(map(tuple, data.loc[(data['pdb_id'] == pdb_id.replace('_',':')) & 
                                            (data['mutation'].str.contains(values[4].split('.')[0])) & 
                                            (data['position'] == int(values[3]))].values))[0]

            if args.fold_mapping != 'True': 
                if information[6] > 0:
                    if ',' in information[4]:
                        for mut in information[4].split(','):
                            uniprot_to_pdb['_'.join([information[5].replace(':', '_'), information[2], str(information[6]), mut])] = mutation[0]
                    else:
                        uniprot_to_pdb['_'.join([information[5].replace(':', '_'), information[2], str(information[6]), information[4]])] = mutation[0]
                    
                    protein_to_mut.append((information[0], mutation[0]))
                    features = extract_rows_around_position(mutation[1].reshape(-1, mutation[1].shape[-1]), int(values[3]))
                    wildtype_mutated.append((information[0], features))
                    labels.append((information[0],np.nan_to_num(np.abs(int(information[-1].replace('Neutral', '0').replace('Deleterious', '1'))), nan=-999)))
            elif args.fold_mapping == 'True' and information[5] > 0:
                if ',' in information[3]:
                    for mut in information[3].split(','):
                        uniprot_to_pdb['_'.join([information[4].replace(':', '_'), information[1], str(information[5]), mut])] = mutation[0]
                else:
                    uniprot_to_pdb['_'.join([information[4].replace(':', '_'), information[1], str(information[5]), information[3]])] = mutation[0]

                protein_to_mut.append((mutation[0], information[0]))
                features = extract_rows_around_position(mutation[1].reshape(-1, mutation[1].shape[-1]), int(values[3]))
                wildtype_mutated.append((mutation[0], features))
                labels.append((mutation[0], np.nan_to_num(information[-15:-14], nan=-999)))
            else:
                pass


    assert len(protein_to_mut) == len(wildtype_mutated) == len(labels)
    
    with open('dataset/prepared/mapping.pickle', 'wb') as handle:
        pickle.dump(protein_to_mut, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('dataset/prepared/data_processed.pickle', 'wb') as handle:
        pickle.dump(wildtype_mutated, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('dataset/prepared/labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('dataset/prepared/uniprot_to_pdb.pickle', 'wb') as handle:
        pickle.dump(uniprot_to_pdb, handle, protocol=pickle.HIGHEST_PROTOCOL)