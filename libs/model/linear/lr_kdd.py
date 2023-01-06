import torch


class KDD_LR(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(KDD_LR, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, 288)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(288, 120)
        self.l3 = torch.nn.Linear(120, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.dropout(x)
        x = self.softmax(self.l3(x))
        return x
