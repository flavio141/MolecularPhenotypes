import os
import shutil
import urllib
import pandas as pd
import requests as r

data = pd.read_csv('dataset/database.tsv', sep='\t')

def fasta_mutated(wildtype, position, mutation):
    for wt, pos, mut in zip(wildtype, position, mutation):
        pass

def extract_data_mut():
    fasta_mutated(data['wildtype'], data['position'], data['mutation'])
