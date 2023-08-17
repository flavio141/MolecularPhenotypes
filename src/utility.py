import torch

def one_hot_aminoacids(mutation, wildtype):
    sequence = "ARNDCEQGHILKMFPSTWYV"
    one_hot = []

    for aa in sequence:
        if aa == mutation:
            one_hot.append(1)
        elif aa == wildtype:
            one_hot.append(-1)
        else:
            one_hot.append(0)

    return torch.tensor(one_hot, dtype=torch.float32)
