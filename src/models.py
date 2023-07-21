import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, rows, hidden_dim, aggregation_dim, output):
        super(NeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggregation_dim = aggregation_dim
        self.output = output
        
        # Layers
        self.fc1_wt = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc1_mut = nn.Linear(self.input_dim, self.hidden_dim)
        self.aggregation_layer = nn.Linear(self.hidden_dim * 2, self.aggregation_dim)
        self.fc2 = nn.Linear(self.aggregation_dim * rows, self.output)

    def forward(self, x_wt, x_mut):
        out_wt = F.relu(self.fc1_wt(x_wt))
        out_mut = F.relu(self.fc1_mut(x_mut))
        
        combined_output = torch.cat((out_wt, out_mut), dim=3)
        combined_output = combined_output.reshape(combined_output.shape[0], combined_output.shape[2], combined_output.shape[3])
        aggregated_output = F.relu(self.aggregation_layer(combined_output))
        
        output = self.fc2(aggregated_output.reshape(aggregated_output.shape[0], -1))
        return output
    

class CustomMatrixDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LossWrapper(torch.nn.Module):
	def __init__(self, loss:torch.nn.Module, ignore_index:int):
		super(LossWrapper, self).__init__()
		self.loss = loss
		self.ignore_index = ignore_index
	
	def __call__(self, input, target):
		input = input.view(-1)
		target = target.view(-1)
		if self.ignore_index != None:
			mask = target.ne(self.ignore_index)
			input = torch.masked_select(input, mask)
			target = torch.masked_select(target, mask)
		
		r = self.loss(input, target)
		return r
