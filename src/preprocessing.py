import os
import time
import pandas as pd
import requests as r

from Bio import SeqIO
from tqdm import tqdm
from io import StringIO

def download_pdb_file(cIDs):
    try:
        for cID, step in zip(cIDs, tqdm(range(0, len(cIDs)), desc= 'Extracting PDB using UniProt ID')):
            pdb_id = cID.split(':')[0]
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            filename = f"{pdb_id}.pdb"
            
            pdb_response = r.get(pdb_url)
            if pdb_response.status_code == 200:
                with open(os.path.join('dataset/pdb', filename), 'wb') as file:
                    file.write(pdb_response.content)
            else:
                print(f"Errore durante il download del file {filename}.")
    except r.exceptions.RequestException as e:
        print(f"Errore durante la richiesta per l'ID UniProt {cID}: {e}")



def retrieve_fasta(cIDs, extract_data=False):
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

    retrieve_fasta(data['uniprot_id'].unique(), extract_data)
    download_pdb_file(data['pdb_id'].unique())
