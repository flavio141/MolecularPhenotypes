import pickle
from tqdm import tqdm
from jax_unirep import get_reps

def UniRep_embedding(fastas):
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
        unirep_emb[fasta.split('.')[0]] = embedding

    with open(f'embedding/additional_features/unirep.pkl', 'wb') as unirep:
        pickle.dump(unirep_emb, unirep, protocol=pickle.HIGHEST_PROTOCOL)
