import os
import shutil
import urllib
import pandas as pd
import requests as r

from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, PDBList
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

data = pd.read_csv('dataset/database.tsv', sep='\t')

def download_cif_file(cIDs, pdbs):
    for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting CIF and Converting')):
        pdb_id = cID.split(':')[0]
        cif_file = '{}.cif'.format(pdb_id)
        print(f'Download {cif_file}')

        if (pdb_id + '_' + cID.split(':')[1] + '.pdb') in os.listdir('dataset/pdb'):
            continue

        url = 'https://files.rcsb.org/download/{}.cif'.format(pdb_id)
        response = urllib.request.urlopen(url)
        cif_data = response.read().decode('utf-8')
        
        with open(os.path.join('dataset/cif',cif_file), 'w') as output_handle:
            output_handle.write(cif_data)
        
        chains = [pdb for pdb in pdbs if cID.split('.')[0] in pdb]

        for chain in chains:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(pdb_id, os.path.join('dataset/cif', cif_file))

            selected_chain = None
            for element in structure[0]:
                if element.get_id() == chain.split(':')[1]:
                    selected_chain = element
                    if len(selected_chain.get_id()) > 1:
                        selected_chain.id = '-'
                    break
            
            pdb_io = PDBIO()

            pdb_io.set_structure(selected_chain)
            pdb_save = f'dataset/pdb/{pdb_id}_{chain.split(":")[1]}.pdb'
            pdb_io.save(pdb_save)


def download_pdb_file(cIDs):
    try:
        not_pdb = []

        for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting PDB using UniProt ID')):
            pdb_id = cID.split(':')[0]
            chain_id = cID.split(':')[1]
            pdb_list = PDBList(verbose=False)

            if (pdb_id + '_' + chain_id + '.pdb') in os.listdir('dataset/pdb'):
                continue
    
            pdb_url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
            
            pdb_response = r.get(pdb_url)
            if pdb_response.status_code == 200:
                with open(os.path.join('dataset/pdb_temp', f'{pdb_id}.pdb'), 'wb') as file:
                    file.write(pdb_response.content)
            else:
                not_pdb.append(f'{pdb_id}:{chain_id}.pdb')
                continue

            pdb_list.retrieve_pdb_file(pdb_id, pdir="dataset/pdb_temp", file_format="pdb")
            parser = PDBParser(QUIET=True)

            structure = parser.get_structure(pdb_id, f"dataset/pdb_temp/{pdb_id}.pdb")

            selected_chain = None
            for chain in structure[0]:
                if chain.get_id() == chain_id:
                    selected_chain = chain
                    break

            io = PDBIO()
            io.set_structure(selected_chain)
            io.save(f"dataset/pdb/{pdb_id}_{chain_id}.pdb")

        return not_pdb
    except r.exceptions.RequestException as e:
        print(f'Error {cID}: {e}')


def download_fasta_file(cIDs):
    for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting FASTA')):
        pdb_id = cID.split(':')[0]
        chain = cID.split(':')[1]
        fasta_file = '{}_{}.fasta'.format(pdb_id, chain)
        fasta_sequences = []

        if fasta_file in os.listdir('dataset/fasta'):
            continue

        if (pdb_id + '.pdb') in os.listdir('dataset/pdb_temp'):
            for record in SeqIO.parse(f'dataset/pdb_temp/{pdb_id}.pdb', 'pdb-seqres'):
                if record.annotations['chain'] == chain:
                    fasta_header = f">{record.id}|{record.annotations['chain']}"
                    fasta_sequence = str(record.seq)
                    fasta_entry = f'{fasta_header}\n{fasta_sequence}'
                    fasta_sequences.append(fasta_entry)
        else:
            mmcif_dict = MMCIF2Dict(f'dataset/cif/{pdb_id}.cif')
            
            
            for entity_id in mmcif_dict['_entity.id']:
                entity_type = mmcif_dict['_entity.type'][int(entity_id) - 1]
                
                if entity_type == 'polymer':
                    sequence = mmcif_dict['_entity_poly.pdbx_seq_one_letter_code'][int(entity_id) - 1]
                    sequence = sequence.replace('.', '')
                    sequence = sequence.replace('\n', '')
                    
                    chain_id = mmcif_dict['_entity_poly.pdbx_strand_id'][int(entity_id) - 1]
                    if chain_id == chain:
                        fasta_header = f">{mmcif_dict['_entry.id'][0]}:{chain_id}|{chain_id}"
                        
                        fasta_entry = f"{fasta_header}\n{sequence}"
                        fasta_sequences.append(fasta_entry)
        
        with open(os.path.join('dataset/fasta',fasta_file), 'w') as fasta_file:
            fasta_file.write('\n'.join(fasta_sequences))
    
    shutil.rmtree('dataset/pdb_temp')


def feature_extraction(cIDs):
    for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting Features')):
        pdb = '_'.join(cID.split(':'))
        path_fasta = f'dataset/fasta/{pdb}.fasta'
        path_fasta_emb = f'embedding/fastaEmb_wt/{pdb}.embeddings.pkl'
        os.system(f'python GCN-for-Structure-and-Function/scripts/seqvec_embedder.py --input={path_fasta} --output={path_fasta_emb}')

        path_pdb = f'dataset/pdb/{pdb}.pdb'
        path_pdb_emb = f'embedding/distmap_wt/{pdb}.distmap.npy'
        os.system(f'python GCN-for-Structure-and-Function/scripts/convert_pdb_to_distmap.py {path_pdb} {path_pdb_emb}')

        # Extract structural features
        #dsspexe = 'dssp/dssp'
        #strucfile = f'embedding/structural/{pdb}.strucfeats.pkl'
        #os.system(f'python GCN-for-Structure-and-Function/scripts/get_structural_feats.py dataset/pdb/{pdb}.pdb {dsspexe} {strucfile}')

        # Create dictionary with all needed features
        #output = f'embedding/results_wt/{pdb}.pkl'
        #os.system(f'python GCN-for-Structure-and-Function/scripts/generate_feats.py {pdb} {path_fasta} {path_fasta_emb} {path_pdb_emb} {output}')


def extract_data_wt(extract_data=False):
    if not os.path.exists('dataset'):
        raise

    data.drop(columns=['phenotypic_annotation'], inplace=True)

    nans = {'wildtype' : data['wildtype'].isnull().sum(), 
            'position': data['position'].isnull().sum(), 
            'mutation': data['mutation'].isnull().sum()
        }
    
    if nans['wildtype'] == 0 or nans['position'] == 0 or nans['mutation'] == 0:
        data.dropna(subset = [min(nans, key=nans.get)])

    # Download all PDB files
    not_pdb = download_pdb_file(data['pdb_id'].unique())

    # Download PDB files too large as CIF and then convert them
    if not_pdb:
        download_cif_file(list(set(not_pdb)), data['pdb_id'].unique())
    
    # Extract FASTA
    download_fasta_file(data['pdb_id'].unique())

    # Extract Features
    feature_extraction(data['pdb_id'].unique())
