from typing import Tuple, Dict

import torch
from torch import nn
from tqdm import tqdm

from src.data.data_container import DataContainer
from src.federated.federated import FederatedLearning
from src.federated.protocols import Trainer, TrainerParams


class TorchTrainer(Trainer):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, model: nn.Module, train_data: DataContainer, context: FederatedLearning.Context,
              config: TrainerParams) -> Tuple[any, int]:
        model.to(self.device)
        model.train()
        optimizer = config.get_optimizer()(model)
        criterion = config.get_criterion()

        epoch_loss = []
        epochs = range(config.epochs)
        if 'verbose' in config.args and config.args['verbose']:
            epochs = tqdm(epochs)
        for epoch in epochs:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data.batch(config.batch_size)):
                x = x.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        weights = model.cpu().state_dict()
        return weights, len(train_data)


class CPUTrainer(TorchTrainer):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'


class TorchChunkTrainer(TorchTrainer):
    def train(self, model: nn.Module, train_data: DataContainer, context: FederatedLearning.Context,
              config: TrainerParams, chunk_ratio=0.1) -> Tuple[any, int]:
        round_id = context.round_id
        data_size = int(len(train_data) * chunk_ratio)
        data_from = (int(round_id * data_size)) % len(train_data)
        data_to = int(data_from + data_size)
        x = train_data.x[data_from:data_to]
        y = train_data.y[data_from:data_to]
        chunk = DataContainer(x, y)
        return super(TorchChunkTrainer, self).train(model, chunk, round_id, config)
