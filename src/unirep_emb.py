import os
import numpy as np
import tensorflow as tf

USE_FULL_1900_DIM_MODEL = True


def UniRep_embedding(fastas):
    if USE_FULL_1900_DIM_MODEL:
        os.system('aws s3 sync --no-sign-request --quiet s3://unirep-public/1900_weights/1900_weights/')
        from UniRep.unirep import babbler1900 as babbler
        MODEL_WEIGHT_PATH = "./1900_weights"
    else:
        os.system('aws s3 sync --no-sign-request --quiet s3://unirep-public/64_weights/64_weights/')
        from UniRep.unirep import babbler64 as babbler
        MODEL_WEIGHT_PATH = "./64_weights"

    model = babbler(batch_size=12, model_path=MODEL_WEIGHT_PATH)

    for fasta in fastas:
        sequence = ''
        with open(f'dataset/fasta/{fasta}', 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    continue
                sequence += line
        
        embedding = model.get_rep(np.array(model.format_seq(sequence)))

