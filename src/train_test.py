import torch.nn as nn
import torch.optim as optim
from models import Feedforward


def train(num_epochs, train_loader):
    input_size = 10
    hidden_size = 20
    output_size = 5

    model = Feedforward(input_size, hidden_size, output_size)

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