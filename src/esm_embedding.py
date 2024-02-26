import os
import esm
import torch
import pickle
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('dataset/database.tsv', sep='\t')


def embedding_esm(fastas, fastas_mut):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    cuda1 = torch.device('cuda:1')

    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()#esm.pretrained.esm1v_t33_650M_UR90S_5()
    #model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    esm_emb = {}

    with torch.no_grad():
        for fasta, _ in zip(fastas, tqdm(range(0, len(fastas)), desc='Extract features with ESM')):
            seqs = ''
            with open(f'dataset/fasta/{fasta}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('>'):
                        continue
                    seqs += line

            esm_emb[fasta.split('.')[0]] = []
            fasta_present = [fasta_mut for fasta_mut in fastas_mut if fasta.split('.')[0] in fasta_mut]

            data = [('wt', seqs)]

            for fasta_mut in fasta_present:
                seqs = ''
                with open(f'dataset/fasta_mut/{fasta_mut}', 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line.startswith('>'):
                            continue
                        seqs += line
                if len(seqs) > 800:
                    continue
                data.append(('mt', seqs)) # type: ignore
                _, _, batch_tokens = batch_converter(data)

                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                
                token_representations = results["representations"][33]
                esm_emb[fasta.split('.')[0]].append((fasta_mut.split('.')[0], (token_representations[0] - token_representations[1])))

                data.pop()
    
    
    with open(f'embedding/additional_features/esm2.pkl', 'wb') as esmFile:
        pickle.dump(esm_emb, esmFile, protocol=pickle.HIGHEST_PROTOCOL)

embedding_esm(os.listdir('dataset/fasta'), os.listdir('dataset/fasta_mut'))