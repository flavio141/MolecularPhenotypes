import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim_wt, hidden_dim_mut, aggregation_dim, output):
        super(MyNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_wt = hidden_dim_wt
        self.hidden_dim_mut = hidden_dim_mut
        self.aggregation_dim = aggregation_dim
        self.output = output
        
        # Layers
        self.fc1_wt = nn.Linear(self.input_dim, self.hidden_dim_wt)
        self.fc1_mut = nn.Linear(self.input_dim, self.hidden_dim_mut)
        self.aggregation_layer = nn.Linear(self.hidden_dim_wt + self.hidden_dim_mut, self.aggregation_dim)
        self.fc2 = nn.Linear(self.aggregation_dim, self.output)

    def forward(self, x_wt, x_mut):
        out_wt = F.relu(self.fc1_wt(x_wt))
        out_mut = F.relu(self.fc1_mut(x_mut))
        
        combined_output = torch.cat((out_wt, out_mut), dim=1)
        aggregated_output = F.relu(self.aggregation_layer(combined_output))
        
        output = F.softmax(self.fc2(aggregated_output), dim=1)
        return output
    

class CustomMatrixDataset(Dataset):
    def __init__(self, wildtype_mutated_matrices, labels_list):
        self.data_pairs = wildtype_mutated_matrices
        self.labels_list = labels_list

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        wildtype, mutated = self.data_pairs[index]
        labels = self.labels_list[index]
        return wildtype, mutated, labels