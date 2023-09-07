import os
import esm
import torch
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('dataset/SNV.tsv', sep='\t')


def embedding_esm(fastas, fastas_mut):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    esm_emb = {}

    for fasta, count in zip(fastas, tqdm(range(0, len(fastas)), desc='Extract features with ESM')):
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

            data.append(('mt', seqs))
            _, _, batch_tokens = batch_converter(data)


            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            
            token_representations = results["representations"][33]
            esm_emb[fasta.split('.')[0]].append((fasta_mut.split('.')[0], (token_representations[0] - token_representations[1])))

            data.pop()

embedding_esm(os.listdir('dataset/fasta'), os.listdir('dataset/fasta_mut'))