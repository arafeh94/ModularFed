import torch
from torch import nn, optim


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, l1_dims, l2_dims, n_action):
        super().__init__()
        self.l2_dims = l2_dims
        self.l1_dims = l1_dims
        self.n_action = n_action
        self.input_dims = input_dims
        self.l1 = nn.Linear(self.input_dims, self.l1_dims)
        self.l2 = nn.Linear(self.l1_dims, self.l2_dims)
        self.l3 = nn.Linear(self.l2_dims, self.n_action)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))
        return x
