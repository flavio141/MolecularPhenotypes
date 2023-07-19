import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from create_dataset import dataset_preparation
from models import MyNeuralNetwork, CustomMatrixDataset

def train(num_epochs, train_loader):
    input_size = 10
    hidden_size = 20
    output_size = 5

    model = MyNeuralNetwork(input_size, hidden_size, output_size)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")


wildtype_mutated, labels = dataset_preparation()

dataset = CustomMatrixDataset(wildtype_mutated, labels)
#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
