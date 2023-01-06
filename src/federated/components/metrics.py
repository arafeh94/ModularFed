import torch
from torch import nn

from src.data.data_container import DataContainer
from src.federated.protocols import Trainer, ModelInfer


class AccLoss(ModelInfer):
    def __init__(self, batch_size: int, criterion, device=None):
        super().__init__(batch_size, criterion)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer(self, model: nn.Module, test_data: DataContainer):
        model.to(self.device)
        model.eval()
        test_loss = test_acc = test_total = 0.
        criterion = self.criterion
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data.batch(self.batch_size)):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc / test_total, test_loss / test_total
