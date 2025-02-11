import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAT, Sequential, global_add_pool, global_mean_pool
from torch.utils.data import Dataset


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
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, rows, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = Sequential('x, edge_index', [
            (GCNConv(in_channels, 8), 'x, edge_index -> x'),
            nn.LeakyReLU(inplace=True),
            (GCNConv(8, 8), 'x, edge_index -> x'),
            nn.LeakyReLU(inplace=True)
            ])

        self.linear = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.flattening = nn.Sequential(
            nn.Linear(7680, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        self.output = nn.Linear(60, out_channels)

    def forward(self, data, features, one_hot):
        xGCN = global_mean_pool(self.conv1(data.x, data.edge_index), batch=data.batch)
        xLinear = self.linear(features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])).flatten(start_dim=1)

        x = self.flattening(xLinear)
        x = torch.hstack((xGCN, x, one_hot))
        output = self.output(x)
        return output



class GNNModelOneHot(torch.nn.Module):
    def __init__(self, in_channels, rows, out_channels):
        super(GNNModelOneHot, self).__init__()
        self.conv1 = Sequential('x, edge_index', [
            (GCNConv(in_channels, 8), 'x, edge_index -> x'),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(0.2),
            (GCNConv(8, 8), 'x, edge_index -> x'),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(0.2)
            ])

        self.linear = nn.Sequential(
            nn.Linear(2560, 256),
            nn.BatchNorm1d(rows),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(rows),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
            nn.BatchNorm1d(rows),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        self.flattening = nn.Sequential(
            nn.Linear(rows * 8, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        self.output = nn.Linear(60, out_channels)

    def forward(self, data, features, one_hot):
        xGCN = global_mean_pool(self.conv1(data.x, data.edge_index), batch=data.batch)
        xLinear = self.linear(features).flatten(start_dim=1)

        x = self.flattening(xLinear)
        x = torch.hstack((xGCN, x, one_hot))
        output = self.output(x)
        return output


class GAttention(torch.nn.Module):
    def __init__(self, in_channels, rows, out_channels):
        super(GAttention, self).__init__()
        self.dropout = 0.3
        self.conv1 = GAT(in_channels=in_channels, hidden_channels=6, num_layers=1, out_channels=4, act='leakyrelu', v2=True, jk='lstm')

        self.linear = nn.Sequential(
            nn.Linear(2560, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.flattening = nn.Sequential(
            nn.Linear(rows * 8, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.output = nn.Linear(56, out_channels)

    def forward(self, data, features, one_hot):
        xGCN = global_mean_pool(self.conv1(data.x, data.edge_index), batch=data.batch)
        xLinear = self.linear(features).flatten(start_dim=1)

        x = self.flattening(xLinear)
        x = torch.hstack((xGCN, x, one_hot))
        output = self.output(x)
        return output


class GNNModelAttention(torch.nn.Module):
    def __init__(self, in_channels, rows, out_channels):
        super(GNNModelAttention, self).__init__()
        # GraphConvolution
        self.conv1 = GAT(in_channels=in_channels, hidden_channels=6, num_layers=1, out_channels=8, act='leakyrelu', v2=True, jk='lstm')

        #self.conv1 = Sequential('x, edge_index', [
        #    (GCNConv(in_channels, 12), 'x, edge_index -> x'),
        #    nn.LeakyReLU(inplace=True),
        #    nn.Dropout(0.2),
        #    (GCNConv(12, 8), 'x, edge_index -> x'),
        #    nn.LeakyReLU(inplace=True),
        #    nn.Dropout(0.2)
        #    ])

        # Layer Linear
        self.linear = nn.Sequential(
            nn.Linear(2560, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention Mechanism
        self.attention_a = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(256, 64),
            nn.Sigmoid(),
            nn.Dropout(0.2)
        )
        self.attention_c = nn.Linear(64, 1)

        # Flattening
        self.flattening = nn.Sequential(
            nn.Linear(256, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # Output
        self.output = nn.Linear(36, out_channels)

    def forward(self, data, features, one_hot):
        xGCN = global_mean_pool(self.conv1(data.x, data.edge_index), batch=data.batch)
        xLinearFirst = self.linear(features)

        attention_amut = self.attention_a(xLinearFirst)
        attention_bmut = self.attention_b(xLinearFirst)
        A = attention_amut.mul(attention_bmut)
        attention_cmut = F.softmax(torch.transpose(self.attention_c(A), 2, 1), dim=1)

        xLinearAttention = torch.matmul(attention_cmut, xLinearFirst)
        
        x = self.flattening(xLinearAttention).flatten(start_dim=1)
        x = torch.hstack((xGCN, x, one_hot))
        output = self.output(x)
        return output


class GNNModelSimple(torch.nn.Module):
    def __init__(self, in_channels, rows, out_channels):
        super(GNNModelSimple, self).__init__()
        self.conv1 = Sequential('x, edge_index', [
            (GCNConv(in_channels, 12), 'x, edge_index -> x'),
            nn.LeakyReLU(inplace=True),
            (GCNConv(12, 12), 'x, edge_index -> x'),
            nn.LeakyReLU(inplace=True)
            ])
        
        self.attention_a = nn.Sequential(
            nn.Linear(12, 4),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(12, 4),
            nn.Sigmoid(),
            nn.Dropout(0.2)
        )
        self.attention_c = nn.Linear(4, 1)
        
        self.output = nn.Linear(13, out_channels)

    def forward(self, data, features):
        xGCN = global_add_pool(self.conv1(data.x, data.edge_index), batch=data.batch)

        attention_amut = self.attention_a(xGCN)
        attention_bmut = self.attention_b(xGCN)
        A = attention_amut.mul(attention_bmut)
        attention_cmut = F.softmax(self.attention_c(A), dim=1)

        xGCNAttention = torch.hstack((F.softmax(self.attention_c(A), dim=1), xGCN))

        output = self.output(xGCNAttention)
        return output

# DATASET
class CustomMatrixDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.indices = indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]
    

class DataLoaderGraph(Dataset):
    def __init__(self, graphs, features, labels, indices):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.graphs = graphs
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.indices = indices

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.graphs[idx], self.features[idx], self.labels[idx], self.indices[idx]


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
