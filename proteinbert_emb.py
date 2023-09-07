import os
import pickle
import pandas as pd
from tqdm import tqdm
from protein_bert.proteinbert.existing_model_loading import load_pretrained_model
from protein_bert.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

data = pd.read_csv('dataset/SNV.tsv', sep='\t')


def extract_features_proteinbert(fastas, fastas_mut):
    pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = '/data/fsartori/MolecularPhenotypes/model', validate_downloading=False)
    proteinbert_emb = {}


    for fasta, count in zip(fastas, tqdm(range(0, len(fastas)), desc='Extract features with ProteinBERT')):
        seqs = ''
        with open(f'dataset/fasta/{fasta}', 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    continue
                seqs += line

        model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(len(seqs)+2))
        X = input_encoder.encode_X([seqs], len(seqs))
        local_representations_wt, global_representations_wt = model.predict(X, batch_size = 1)


        proteinbert_emb[fasta.split('.')[0]] = []
        fasta_present = [fasta_mut for fasta_mut in fastas_mut if fasta.split('.')[0] in fasta_mut]

        for fasta_mut in fasta_present:
            seqs = ''
            with open(f'dataset/fasta_mut/{fasta_mut}', 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('>'):
                        continue
                    seqs += line

            X = input_encoder.encode_X([seqs], len(seqs))
            local_representations, global_representations= model.predict(X, batch_size = 1)

            proteinbert_emb[fasta.split('.')[0]].append((fasta_mut.split('.')[0], local_representations_wt.reshape(-1, local_representations_wt.shape[-1]) - local_representations.reshape(-1, local_representations.shape[-1])))
    
    with open(f'embedding/additional_features/protbert.pkl', 'wb') as protbert:
        pickle.dump(proteinbert_emb, protbert, protocol=pickle.HIGHEST_PROTOCOL)

extract_features_proteinbert(os.listdir('dataset/fasta'), os.listdir('dataset/fasta_mut'))