import torch
from torch import nn

from libs.lr_finder import LRFinder
from libs.model.linear.lr import LogisticRegression
from src import manifest
from src.data.data_provider import PickleDataProvider

mnist = PickleDataProvider(manifest.datasets_urls['mnist']).collect().as_tensor().split(0.8)
train = mnist[0]
validate = mnist[1]
model = LogisticRegression(28 * 28, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
finder = LRFinder(model, optimizer, criterion, 'cuda')
finder.range_test(train, validate)
finder.plot()
