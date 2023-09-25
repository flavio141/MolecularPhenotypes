import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utility import one_hot_aminoacids

# ATTENTION NEURAL NETWORKS
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
    

class TRAM(nn.Module):
    def __init__(self, input_dim, rows, output, l2_lambda=0.01):
        super(TRAM, self).__init__()
        self.input_dim = input_dim
        self.output = output

        self.l2_lambda = l2_lambda

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1_wt = nn.Linear(self.input_dim, 64)
        self.fc1_mut = nn.Linear(self.input_dim, 64)
        self.bn1_wt = nn.BatchNorm1d(64)
        self.bn1_mut = nn.BatchNorm1d(64)

        self.fc2_wt = nn.Linear(64 * rows, 256)
        self.fc2_mut = nn.Linear(64 * rows, 256)
        self.bn2_wt = nn.BatchNorm1d(256)
        self.bn2_mut = nn.BatchNorm1d(256)

        self.aggregation_layer = nn.Linear(256 * 2, 128)
        self.bna = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, self.output)


    def forward(self, x_wt, x_mut):
        x_wt = x_wt.reshape(x_wt.shape[0], x_wt.shape[2], x_wt.shape[3])
        x_mut = x_mut.reshape(x_mut.shape[0], x_mut.shape[2], x_mut.shape[3])

        out1_wt = self.dropout(F.leaky_relu(self.bn1_wt(self.fc1_wt(x_wt).permute(0,2,1)).permute(0,2,1)))
        out1_mut = self.dropout(F.leaky_relu(self.bn1_mut(self.fc1_mut(x_mut).permute(0,2,1)).permute(0,2,1)))

        out1_wt = torch.flatten(out1_wt, start_dim=1, end_dim=2)
        out1_mut = torch.flatten(out1_mut, start_dim=1, end_dim=2)

        out2_wt = self.dropout(F.leaky_relu(self.bn2_wt(self.fc2_wt(out1_wt))))
        out2_mut = self.dropout(F.leaky_relu(self.bn2_mut(self.fc2_mut(out1_mut))))
        
        combined_output = torch.cat((out2_wt, out2_mut), dim=1)
        aggregated_output = self.dropout(F.leaky_relu(self.bna(self.aggregation_layer(combined_output))))
        
        output = self.fc3(aggregated_output)
        return output
    

    def l2_regularization_loss(self):
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg


class TRAM_Att(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(TRAM_Att, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1_wt = nn.Linear(self.input_dim, 256)
        self.fc1_mut = nn.Linear(self.input_dim, 256)
        
        # Attention Mechanism
        self.attention_a = nn.Linear(256, 64)
        self.attention_b = nn.Linear(256, 64)
        self.attention_c = nn.Linear(64, 1)

        self.fc2_wt = nn.Linear(256 * rows, 128)
        self.fc2_mut = nn.Linear(256, 128)

        self.aggregation_layer = nn.Linear(128 * 2, 64)

        self.fc3 = nn.Linear(64, self.output)


    def forward(self, x_wt, x_mut):
        x_wt = x_wt.reshape(x_wt.shape[0], x_wt.shape[2], x_wt.shape[3])
        x_mut = x_mut.reshape(x_mut.shape[0], x_mut.shape[2], x_mut.shape[3])

        out1_wt = self.dropout(F.leaky_relu(self.fc1_wt(x_wt)))
        out1_mut = self.dropout(F.leaky_relu(self.fc1_mut(x_mut)))

        attention_amut = self.dropout(F.tanh(self.attention_a(out1_mut)))
        attention_bmut = self.dropout(F.sigmoid(self.attention_b(out1_mut)))
        A = attention_amut.mul(attention_bmut)
        attention_cmut = F.softmax(torch.transpose(self.attention_c(A), 2, 1), dim=1)

        out1_mut = torch.matmul(attention_cmut, out1_mut)

        out1_wt = torch.flatten(out1_wt, start_dim=1, end_dim=2)
        out1_mut = torch.flatten(out1_mut, start_dim=1, end_dim=2)

        out2_wt = self.dropout(F.leaky_relu(self.fc2_wt(out1_wt)))
        out2_mut = self.dropout(F.leaky_relu(self.fc2_mut(out1_mut)))
        
        combined_output = torch.cat((out2_wt, out2_mut), dim=1)
        aggregated_output = self.dropout(F.leaky_relu(self.aggregation_layer(combined_output)))
        
        output = self.fc3(aggregated_output)
        return output


class TRAM_Att_solo(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(TRAM_Att_solo, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 256)
        
        # Attention Mechanism
        self.attention_a = nn.Linear(256, 64)
        self.attention_b = nn.Linear(256, 64)
        self.attention_c = nn.Linear(64, 1)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, self.output)


    def forward(self, x):
        out1 = self.dropout(F.leaky_relu(self.fc1(x)))

        attention_amut = self.dropout(F.tanh(self.attention_a(out1)))
        attention_bmut = self.dropout(F.sigmoid(self.attention_b(out1)))
        A = attention_amut.mul(attention_bmut)
        attention_cmut = F.softmax(torch.transpose(self.attention_c(A), 2, 1), dim=1)

        out1 = torch.matmul(attention_cmut, out1)

        out1 = torch.flatten(out1, start_dim=1, end_dim=2)

        out2 = self.dropout(F.leaky_relu(self.fc2(out1)))
        
        output = self.fc3(out2)
        return output


class TRAM_Att_solo_one_hot(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(TRAM_Att_solo_one_hot, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 256)
        
        # Attention Mechanism
        self.attention_a = nn.Linear(256, 64)
        self.attention_b = nn.Linear(256, 64)
        self.attention_c = nn.Linear(64, 1)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128 + 20, self.output)


    def forward(self, x, tensor_one_hot):
        out1 = self.dropout(F.leaky_relu(self.fc1(x)))

        attention_amut = self.dropout(F.tanh(self.attention_a(out1)))
        attention_bmut = self.dropout(F.sigmoid(self.attention_b(out1)))
        A = attention_amut.mul(attention_bmut)
        attention_cmut = F.softmax(torch.transpose(self.attention_c(A), 2, 1), dim=1)

        out1 = torch.matmul(attention_cmut, out1)

        out1 = torch.flatten(out1, start_dim=1, end_dim=2)

        out2 = self.dropout(F.leaky_relu(self.fc2(out1)))
        out2 = torch.hstack((out2, tensor_one_hot))
        
        output = self.fc3(out2)
        return output
    

# FEED FORWARD NEURAL NETWORKS
class NN1(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(NN1, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128 * rows, self.output)


    def forward(self, x):
        out1 = self.dropout(F.leaky_relu(self.bn1(self.fc1(x).permute(0,2,1)).permute(0,2,1)))

        out2 = self.dropout(F.leaky_relu(self.bn2(self.fc2(out1).permute(0,2,1)).permute(0,2,1)))

        out2 = torch.flatten(out2, start_dim=1, end_dim=2)
        
        output = self.fc3(out2)
        return output
    

class NN2(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(NN2, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear((128 * rows) + 20, self.output)


    def forward(self, x, tensor_one_hot):
        out1 = self.dropout(F.leaky_relu(self.bn1(self.fc1(x).permute(0,2,1)).permute(0,2,1)))

        out2 = self.dropout(F.leaky_relu(self.bn2(self.fc2(out1).permute(0,2,1)).permute(0,2,1)))

        out2 = torch.flatten(out2, start_dim=1, end_dim=2)
        out2 = torch.hstack((out2, tensor_one_hot))
        
        output = self.fc3(out2)
        return output
    

class NN3(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(NN3, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256 * rows, self.output)


    def forward(self, x):
        out1 = self.dropout(F.leaky_relu(self.bn1(self.fc1(x).permute(0,2,1)).permute(0,2,1)))

        out1 = torch.flatten(out1, start_dim=1, end_dim=2)
        
        output = self.fc2(out1)
        return output


class NN4(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(NN4, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256 * rows, self.output)


    def forward(self, x):
        out1 = self.dropout(F.leaky_relu(self.bn1(self.fc1(x).permute(0,2,1)).permute(0,2,1)))

        out1 = torch.flatten(out1, start_dim=1, end_dim=2)
        
        output = self.fc2(out1)
        return output
    

class NN5(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(NN5, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.LayerNorm([rows, 256])

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm([rows, 128])

        self.fc3 = nn.Linear((128 * rows) + 20, self.output)


    def forward(self, x, tensor_one_hot):
        out1 = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))

        out2 = self.dropout(F.leaky_relu(self.bn2(self.fc2(out1))))

        out2 = torch.flatten(out2, start_dim=1, end_dim=2)
        out2 = torch.hstack((out2, tensor_one_hot))
        
        output = self.fc3(out2)
        return output


class NN6(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(NN6, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 10)
        self.fc2 = nn.Linear(10, self.output)


    def forward(self, x):
        out1 = self.dropout(F.leaky_relu(self.fc1(x)))
        output = self.dropout(F.leaky_relu(self.fc2(out1)))
        return output
    

class NN7(nn.Module):
    def __init__(self, input_dim, rows, output):
        super(NN7, self).__init__()
        self.input_dim = input_dim
        self.output = output

        # Layers
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.input_dim, 10)
        self.fc2 = nn.Linear(10, self.output)


    def forward(self, x):
        out1 = self.dropout(F.leaky_relu(self.fc1(x)))
        output = self.dropout(F.leaky_relu(self.fc2(out1)))
        return output


# GRAPH NEURAL NETWORKS
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, features): # Se viene da NetworkX
        adjacency_matrix = graph.adjacency_matrix() 
        adjacency_matrix = torch.sparse.FloatTensor( #type: ignore
            torch.LongTensor(adjacency_matrix.nonzero()).t(),
            torch.FloatTensor([1.0] * adjacency_matrix.nnz),
            torch.Size(adjacency_matrix.shape),
        )
        
        support = torch.mm(features, self.weight)
        output = torch.sparse.mm(adjacency_matrix, support)
        return output


class NNG(nn.Module):
    def __init__(self, input_dim, rows, output_dim):
        super(NNG, self).__init__()
        self.graph_conv1 = GraphConvolution(input_dim, 128)
        self.graph_conv2 = GraphConvolution(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, graph, features):
        x = self.graph_conv1(graph, features)
        x = self.relu(x)
        x = self.graph_conv2(graph, x)
        return x
    

class CustomMatrixDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.indices = indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


class MatrixDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.indices = indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


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
