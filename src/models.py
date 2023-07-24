import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, rows, output, l2_lambda=0.01):
        super(NeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.output = output

        self.l2_lambda = l2_lambda
        
        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1_wt = nn.Linear(self.input_dim, 256)
        self.fc1_mut = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.aggregation_layer = nn.Linear(256 * 2, 128)
        self.bna = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128 * rows, self.output)


    def forward(self, x_wt, x_mut):
        x_wt = x_wt.reshape(x_wt.shape[0], x_wt.shape[2], x_wt.shape[3])
        x_mut = x_mut.reshape(x_mut.shape[0], x_mut.shape[2], x_mut.shape[3])

        x_wt_processed = self.fc1_wt(x_wt).permute(0,2,1)
        x_mut_processed = self.fc1_mut(x_mut).permute(0,2,1)

        bn_x_wt = self.bn1(x_wt_processed).permute(0,2,1)
        bn_x_mut = self.bn1(x_mut_processed).permute(0,2,1)

        out_wt = self.dropout(F.leaky_relu(bn_x_wt))
        out_mut = self.dropout(F.leaky_relu(bn_x_mut))
        
        combined_output = torch.cat((out_wt, out_mut), dim=2)
        aggregated_output = self.dropout(F.leaky_relu(self.bna(self.aggregation_layer(combined_output).permute(0,2,1)).permute(0,2,1)))
        
        output = self.fc2(aggregated_output.reshape(aggregated_output.shape[0], -1))
        return output
    

    def l2_regularization_loss(self):
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg
    

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
