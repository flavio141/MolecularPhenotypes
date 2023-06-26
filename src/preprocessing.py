import os
import gemmi
import urllib
import pandas as pd
import requests as r

from tqdm import tqdm
from Bio.PDB import PDBParser, Polypeptide, MMCIFParser, PDBIO
from utils.classes import ChainSelector
from utils.logger import loggerError, logger

def download_cif_file(cIDs, pdbs):
    for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting CIF and Converting')):
        pdb_id = cID.split('_')[0]

        url = 'https://files.rcsb.org/download/{}.cif'.format(pdb_id)
        response = urllib.request.urlopen(url)
        cif_data = response.read().decode('utf-8')
        
        cif_file = '{}.cif'.format(pdb_id)
        with open(os.path.join('dataset/cif',cif_file), 'w') as output_handle:
            output_handle.write(cif_data)
        
        chains = [pdb for pdb in pdbs if cID in pdb]

        for chain in chains:
            parser = MMCIFParser()

            structure = parser.get_structure("structure", os.path.join('dataset/cif', cif_file))
            chain_structure = structure[0][chain]
            pdb_io = PDBIO()

            pdb_io.set_structure(chain_structure)
            pdb_save = f'dataset/pdb/{pdb_id}_{chain}.pdb'
            pdb_io.save(pdb_save)


def download_pdb_file(cIDs):
    try:
        not_pdb = []

        for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting PDB using UniProt ID')):
            pdb_id = cID.split(':')[0]
            chain = cID.split(':')[1]
            if (pdb_id + '.pdb') in os.listdir('dataset/pdb'):
                continue

            pdb_url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
            filename = f'{pdb_id}_{chain}.pdb'
            
            pdb_response = r.get(pdb_url)
            if pdb_response.status_code == 200:
                with open(os.path.join('dataset/pdb', filename), 'wb') as file:
                    file.write(pdb_response.content)
            else:
                loggerError.error(f"{filename}")
                not_pdb.append(filename)
        
        return not_pdb
    except r.exceptions.RequestException as e:
        print(f'Error {cID}: {e}')


def download_fasta_file(cIDs):
    for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting FASTA')):
        pdb_id = cID.split(':')[0]
        chain = cID.split(':')[1]
        url = 'https://www.rcsb.org/fasta/entry/{}.fasta?entity={}'.format(pdb_id, chain)
        
        response = urllib.request.urlopen(url)
        fasta_data = response.read().decode('utf-8')
        
        fasta_file = '{}_{}.fasta'.format(pdb_id, chain)
        with open(os.path.join('dataset/fasta',fasta_file), 'w') as output_handle:
            output_handle.write(fasta_data)


def feature_extraction(cIDs):
    for cID in cIDs:
        path_fasta = f'dataset/fasta/{cID}.fasta'
        path_fasta_emb = f'embedding/fastaEmb/{cID}.embeddings.pkl'
        os.system(f'python GCN-for-Structure-and-Function/scripts/seqvec_embedder.py --input={path_fasta} --output={path_fasta_emb}')

        path_pdb = f'dataset/pdb/{cID}.pdb'
        path_pdb_emb = f'embedding/distmap/{cID}.distmap.npy'
        os.system(f'python GCN-for-Structure-and-Function/scripts/convert_pdb_to_distmap.py {path_pdb} {path_pdb_emb}')

        # Extract structural features
        #PDBFILE=example/${PROTEINID}.pdb
        #DSSPEXE=scripts/dssp.exe
        #STRUCFILE=example/${PROTEINID}.strucfeats.pkl

        #python scripts/get_structural_feats.py ${PDBFILE} ${DSSPEXE} ${STRUCFILE}

        # Create dictionary with all needed features
        #LABELSFILE=datasets/data_pdb/Yterms.pkl
        #OUTFILE=example/${PROTEINID}.pkl

        #python scripts/generate_feats.py ${PROTEINID} ${FASTAFILE} ${EMBFILE} ${DMAPFILE} ${LABELSFILE} ${OUTFILE}


def extract_data(extract_data=False):
    if not os.path.exists('dataset'):
        raise

    data = pd.read_csv('dataset/database.tsv', sep='\t')
    data.drop(columns=['phenotypic_annotation'], inplace=True)

    # Find NaN values in the columns of the name of the SAV
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
    #feature_extraction(data['uniprot_id'].unique())
