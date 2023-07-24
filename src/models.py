import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(NeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.output = output
        
        # Layers
        self.fc1_wt = nn.Linear(self.input_dim, 256)
        self.fc1_mut = nn.Linear(self.input_dim, 256)
        self.aggregation_layer = nn.Linear(256 * 2, 128)
        self.fc2 = nn.Linear(128 * rows, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.output)


    def forward(self, x_wt, x_mut):
        out_wt = F.leaky_relu(self.fc1_wt(x_wt))
        out_mut = F.leaky_relu(self.fc1_mut(x_mut))
        
        combined_output = torch.cat((out_wt, out_mut), dim=3)
        combined_output = combined_output.reshape(combined_output.shape[0], combined_output.shape[2], combined_output.shape[3])
        aggregated_output = F.leaky_relu(self.aggregation_layer(combined_output))
        
        out_fc2 = self.fc2(aggregated_output.reshape(aggregated_output.shape[0], -1))
        out_fc3 = F.leaky_relu(self.fc3(out_fc2))
        output = F.leaky_relu(self.fc4(out_fc3))
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
