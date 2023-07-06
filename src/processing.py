import pandas as pd

data = pd.read_csv('dataset/database.tsv', sep='\t')


def fasta_mutated(wildtype, position, mutation, pdb):
    for wt, pos, mut, pdb in zip(wildtype, position, mutation, pdb):
        pdb_id = pdb.split(':')[0]
        chain = pdb.split(':')[1]

        with open(f'dataset/fasta/{pdb_id}_{chain}.fasta', 'r') as fasta:
            fasta_original = fasta.read()
        
        if fasta_original.split('\n')[1][pos - 1] == wt:
            if ',' in mut:
                for m in mut.split(','):
                    mutation = m
                    fasta_seq = fasta_original.split('\n')[1][:pos - 1] + mutation + fasta_original.split('\n')[1][pos:]
                    fasta_mut = fasta_original.split('\n')[0] + '\n' + fasta_seq

                    with open(f'dataset/fasta_mut/{pdb_id}_{chain}_{m}.fasta', 'w') as mutated:
                        mutated.write(fasta_mut)
            else:
                mutation = mut
                fasta_seq = fasta_original.split('\n')[1][:pos - 1] + mutation + fasta_original.split('\n')[1][pos:]
                fasta_mut = fasta_original.split('\n')[0] + '\n' + fasta_seq

                with open(f'dataset/fasta_mut/{pdb_id}_{chain}_{mut}.fasta', 'w') as mutated:
                    mutated.write(fasta_mut)
        else:
            original = fasta_original.split("\n")[1][pos - 1]
            print(f'No match between {wt} and the amino acids at position {original}:{pos}')


def extract_data_mut(extract_data_mut=False):
    if extract_data_mut:
        fasta_mutated(data['wildtype'], data['position'], data['mutation'], data['pdb_id'])
