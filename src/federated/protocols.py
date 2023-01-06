import typing
from abc import abstractmethod, ABC
from typing import Tuple, List
from torch import nn, Tensor

from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.federated.components import params
from src.federated.events import FederatedSubscriber

if typing.TYPE_CHECKING:
    from src.federated.federated import FederatedLearning


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, trainers_models_weight_dict: Dict[int, nn.ModuleDict], sample_size: Dict[int, int],
                  round_id: int) -> nn.ModuleDict:
        pass


class ModelInfer(ABC):
    def __init__(self, batch_size: int, criterion):
        self.batch_size = batch_size
        self.criterion = criterion
        if isinstance(criterion, str):
            self.criterion = params.criterion(criterion)

    @abstractmethod
    def infer(self, model: nn.Module, test_data: DataContainer):
        pass


class TrainerParams:

    def __init__(self, trainer_class: type, batch_size: int, epochs: int,
                 criterion: str, optimizer: str, **kwargs):
        self.epochs = epochs
        self.criterion = criterion
        self.batch_size = batch_size
        self.trainer_class = trainer_class
        self.optimizer = optimizer
        self.args = kwargs

    def get_optimizer(self):
        return params.optimizer(self.optimizer, **self.args)

    def get_criterion(self):
        return params.criterion(self.criterion, **self.args)


class Trainer(ABC):
    @abstractmethod
    def train(self, model: nn.Module, train_data: DataContainer, context, config: TrainerParams) -> Tuple[
        Dict[str, Tensor], int]:
        pass


class ClientSelector(ABC):
    @abstractmethod
    def select(self, client_ids: List[int], context: 'FederatedLearning.Context') -> List[int]:
        pass


class ModelBasedClientSelector(ClientSelector, ABC):
    def __init__(self):
        self.weights = Dict()
        self.samples = Dict()

    def select(self, client_ids: List[int], context: 'FederatedLearning.Context') -> List[int]:
        return self.model_based_select(client_ids, self.weights, self.samples, context)

    def attach(self, federated: 'FederatedLearning'):
        federated.add_subscriber(self._ModelBasedSelectorSubscriber(self))

    def _update(self, weights_dict: Dict, sample_dict: Dict):
        for cl_id, weight in weights_dict.items():
            self.weights[cl_id] = weight
        for cl_id, sample in sample_dict.items():
            self.samples[cl_id] = sample

    @abstractmethod
    def model_based_select(self, client_ids, clients_weights: Dict[int, any], sample_sizes: Dict[int, int],
                           context: 'FederatedLearning.Context'):
        pass

    class _ModelBasedSelectorSubscriber(FederatedSubscriber):
        def __init__(self, parent_ref: 'ModelBasedClientSelector'):
            super().__init__()
            self.parent_ref = parent_ref

        def on_training_end(self, params):
            weights, sample_size, fed_context = params['trainers_weights'], params['sample_size'], params['context']
            self.parent_ref._update(weights, sample_size)


class ClientScanner(ABC):
    @abstractmethod
    def scan(self) -> typing.Dict[int, typing.Any]:
        pass
