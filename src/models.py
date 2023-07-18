import torch.nn as nn

class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size[1], self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_rows, num_cols = x.size()
        x = x.view(batch_size * num_rows, num_cols)

        # Primo Layer
        hidden = self.fc1(x)
        relu = self.relu(hidden)

        # Secondo Layer
        output = self.fc2(relu)
        output = self.sigmoid(output)

        output = output.view(batch_size, num_rows, -1)
        return output