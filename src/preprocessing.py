import os
import time
import pandas as pd
import requests as r

from Bio import SeqIO
from tqdm import tqdm
from io import StringIO
from Bio.PDB import PDBList
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.MMCIFParser import MMCIFParser
from utils.logger import loggerError, logger

def download_cif_file(cIDs):
    try:
        for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting CIF and Converting')):
            pdb_id = cID.split('.')[0]
            pdbl = PDBList()
            pdbl.retrieve_pdb_file(pdb_id, pdir="dataset/cif", file_format="mmCif")

            cif_file_path = os.path.join('dataset/cif',f'{pdb_id.lower()}.cif')
            pdb_file_path = os.path.join('dataset/pdb',f'{pdb_id.lower()}.pdb')

            parser = MMCIFParser()
            structure = parser.get_structure(pdb_id, cif_file_path)

            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            pdb_io.save(pdb_file_path)
    except Exception as error:
        logger.error(f'{error}')

def download_pdb_file(cIDs):
    try:
        not_pdb = []

        for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting PDB using UniProt ID')):
            pdb_id = cID.split(':')[0]
            if (pdb_id + '.pdb') in os.listdir('dataset/pdb'):
                continue

            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            filename = f"{pdb_id}.pdb"
            
            pdb_response = r.get(pdb_url)
            if pdb_response.status_code == 200:
                with open(os.path.join('dataset/pdb', filename), 'wb') as file:
                    file.write(pdb_response.content)
            else:
                loggerError.error(f"{filename}")
                not_pdb.append(filename)
        
        return not_pdb
    except r.exceptions.RequestException as e:
        print(f"Errore durante la richiesta per l'ID UniProt {cID}: {e}")


def download_fasta_file(cIDs, extract_data=False):
    pSeq = []

    if extract_data == True:
        for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting FASTA from UniProt ID')):
            if (cID + '.fasta') in os.listdir('dataset/fasta'):
                continue
            
            baseUrl="http://www.uniprot.org/uniprot/"
            currentUrl=baseUrl+cID+".fasta"
            response = r.post(currentUrl)
            cData=''.join(response.text)

            Seq=list(SeqIO.parse(StringIO(cData),'fasta'))
            pSeq.extend(Seq)
            SeqIO.write(Seq, f"dataset/fasta/{cID}.fasta", "fasta")
            time.sleep(1)
    else:
        pass
    
    return pSeq


def read_data(extract_data=False):
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

    download_fasta_file(data['uniprot_id'].unique(), extract_data)
    not_pdb = download_pdb_file(data['pdb_id'].unique())

    if not_pdb:
        download_cif_file(not_pdb)
