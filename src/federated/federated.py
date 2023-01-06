import copy
import logging
import math
import time
from collections import defaultdict
from functools import reduce
from src.apis.broadcaster import Broadcaster
from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.federated.components.client_scanners import DefaultScanner
from src.federated.events import Events
from src.federated.protocols import Aggregator, ClientSelector, ModelInfer, TrainerParams, ClientScanner
from src.federated.components.trainer_manager import TrainerManager


class FederatedLearning(Broadcaster):

    def __init__(self, trainer_manager: TrainerManager, trainer_config: TrainerParams, aggregator: Aggregator,
                 client_selector: ClientSelector, metrics: ModelInfer, trainers_data_dict: Dict[int, DataContainer],
                 initial_model: callable, num_rounds=10, desired_accuracy=0.99, train_ratio=0.8,
                 accepted_accuracy_margin=False, test_data: DataContainer = None, zero_client_exception=True, **kwargs):
        super().__init__()
        self.trainer_config = trainer_config
        self.trainer_manager = trainer_manager
        self.aggregator = aggregator
        self.client_selector = client_selector
        self.metrics = metrics
        self.aam = accepted_accuracy_margin
        self.trainers_data_dict = trainers_data_dict
        self.desired_accuracy = desired_accuracy
        self.initial_model = initial_model
        self.train_ratio = train_ratio
        self.num_rounds = num_rounds
        self.args = kwargs
        self.events = {}
        self._check_params()
        self.context = FederatedLearning.Context()
        self.test_data = test_data
        self.trainers_train = self.trainers_data_dict
        self.is_finished = False
        self.zero_client_exception = zero_client_exception
        self.logger = logging.getLogger('FederatedLearning')
        if self.test_data is None:
            self.test_data = Dict()
            self.trainers_train = Dict()
            for trainer_id, data in trainers_data_dict.items():
                data = data.shuffle().as_tensor()
                train, test = data.split(train_ratio)
                self.trainers_train[trainer_id] = train
                self.test_data[trainer_id] = test
        if self.trainer_manager.scanner is None:
            self.trainer_manager.scanner = DefaultScanner(self.trainers_data_dict)
        self.trainer_manager.trainer_started = lambda trainer_id: \
            self.broadcast(Events.ET_TRAINER_STARTED, trainer_id=trainer_id)
        self.trainer_manager.trainer_finished = lambda trainer_id, weights, sample_size: \
            self.broadcast(Events.ET_TRAINER_FINISHED, trainer_id=trainer_id, weights=weights, sample_size=sample_size)

    def start(self):
        self.init()
        while True:
            is_done = self.one_round()
            if is_done:
                break
        return self.context.model

    def init(self):
        self.broadcast(Events.ET_FED_START, **self.configs())
        self.context.build(self)
        self.broadcast(Events.ET_INIT, global_model=self.context.model)

    def one_round(self):
        if self.is_finished:
            return self.is_finished
        self.broadcast(Events.ET_ROUND_START, round=self.context.round_id)
        available_clients = list(self.trainer_manager.scanner.scan().keys())
        trainers_ids = self.client_selector.select(available_clients, self.context)
        if len(trainers_ids) > 0:
            self.broadcast(Events.ET_TRAINER_SELECTED, trainers_ids=trainers_ids)
            trainers_train_data = self.trainers_train.select(trainers_ids)
            self.broadcast(Events.ET_TRAIN_START, trainers_data=trainers_train_data)
            trainers_weights, sample_size_dict = self.train(trainers_train_data)
            self.broadcast(Events.ET_TRAIN_END, trainers_weights=trainers_weights, sample_size=sample_size_dict)
            global_weights = self.aggregator.aggregate(trainers_weights, sample_size_dict, self.context.round_id)
            temporary_model = self.context.model_copy(global_weights)
            self.broadcast(Events.ET_AGGREGATION_END, global_weights=global_weights, global_model=self.context.model)
            accuracy, loss, local_acc, local_loss = self.infer(temporary_model, self.test_data)
            model_status = self.context.update_model(temporary_model, accuracy, self.aam)
            self.broadcast(Events.ET_MODEL_STATUS, model_status=model_status, accuracy=accuracy)
            accuracy = accuracy if model_status else self.context.highest_accuracy()
            self.context.store(acc=accuracy, loss=loss, local_acc=local_acc, local_loss=local_loss, status=model_status)
            self.broadcast(Events.ET_ROUND_FINISHED, round=self.context.round_id, accuracy=accuracy, loss=loss,
                           local_acc=local_acc, local_loss=local_loss)
        else:
            if self.zero_client_exception:
                raise Exception('no client selected for the current rounds')
            self.broadcast(Events.ET_MODEL_STATUS, model_status=False, accuracy=0)
            accuracy = self.context.latest_accuracy()
            loss = self.context.latest_loss()
            self.context.store(acc=accuracy, loss=loss, local_acc={}, local_loss={}, status=False)
            self.broadcast(Events.ET_ROUND_FINISHED, round=self.context.round_id, accuracy=accuracy, loss=loss,
                           local_acc={}, local_loss={})

        self.context.new_round()
        is_done = self.context.stop(self, accuracy)
        if is_done:
            self.is_finished = True
            self.broadcast(Events.ET_FED_END, aggregated_model=self.context.model)
        return is_done

    def train(self, trainers_train_data: Dict[int, DataContainer]):
        for trainer_id, train_data in trainers_train_data.items():
            self.broadcast(Events.ET_TRAINER_STARTED, trainer_id=trainer_id, train_data=train_data)
            model_copy = copy.deepcopy(self.context.model)
            self.trainer_manager.train_req(trainer_id, model_copy, train_data, self.context, self.trainer_config)
        return self.trainer_manager.resolve()

    def infer(self, model, test_data: Dict[int, DataContainer] or DataContainer):
        if isinstance(test_data, DataContainer):
            acc, loss = self.metrics.infer(model, test_data)
            self.context.store(acc=acc, loss=loss, local_acc=[], local_loss=[])
            return acc, loss, {}, {}
        else:
            local_accuracy = {}
            local_loss = {}
            sample_size = {}
            for trainer_id, test_data in test_data.items():
                acc, loss = self.metrics.infer(model, test_data)
                local_accuracy[trainer_id] = acc
                local_loss[trainer_id] = loss
                sample_size[trainer_id] = len(test_data)
            weighted_accuracy = [local_accuracy[tid] * sample_size[tid] for tid in local_accuracy]
            weighted_loss = [local_loss[tid] * sample_size[tid] for tid in local_loss]
            total_accuracy = sum(weighted_accuracy) / sum(sample_size.values())
            total_loss = sum(weighted_loss) / sum(sample_size.values())
            return total_accuracy, total_loss, local_accuracy, local_loss

    def compare(self, other, verbose=1):
        local_history = self.context.history
        other_history = other.context.history
        performance_history = defaultdict(lambda: [])
        diff = {}
        for round_id, first_data in local_history.items():
            if round_id not in other_history:
                continue
            second_data = other_history[round_id]
            for item in first_data:
                if type(first_data[item]) in [int, float, str]:
                    performance_history[item].append(first_data[item] - second_data[item])
        for item, val in performance_history.items():
            diff[item] = math.fsum(val) / len(val)
        if verbose == 1:
            return diff
        else:
            return diff, performance_history

    def finished(self):
        return self.is_finished

    def _check_params(self):
        pass

    def configs(self):
        named = {
            'trainer_config': self.trainer_config,
            'trainer_manager': self.trainer_manager,
            'aggregator': self.aggregator,
            'client_selector': self.client_selector,
            'metrics': self.metrics,
            'accepted_accuracy_margin': self.aam,
            'trainers_data_dict': self.trainers_data_dict,
            'desired_accuracy': self.desired_accuracy,
            'initial_model': self.initial_model,
            'train_ratio': self.train_ratio,
            'num_rounds': self.num_rounds,
            'context': self.context,
            'test_data': self.test_data,
            'trainers_train': self.trainers_train,
            'is_finished': self.is_finished,
        }
        return reduce(lambda x, y: dict(x, **y), (named, self.args))

    def broadcast(self, event_name: str, **kwargs):
        args = reduce(lambda x, y: dict(x, **y), ({'context': self.context}, kwargs))
        super(FederatedLearning, self).broadcast(event_name, **args)

    class Context:
        def __init__(self):
            self.round_id = 0
            self.model = None
            self.history = Dict()
            self.timestamp = time.time()

        def load_weights(self, weights):
            self.model.load_state_dict(weights)

        def model_copy(self, new_weights=None):
            acopy = copy.deepcopy(self.model)
            if new_weights is not None:
                acopy.load_state_dict(new_weights)
            return acopy

        def new_round(self):
            self.round_id += 1

        def highest_accuracy(self):
            highest = 0
            for rnd in self.history:
                if 'acc' in self.history[rnd] and self.history[rnd]['acc'] > highest:
                    highest = self.history[rnd]['acc']
            return highest

        def last_entry(self):
            if len(self.history) == 0:
                return Dict()
            return self.history[list(self.history)[-1]]

        def latest_accuracy(self):
            if len(self.history) == 0:
                return 0
            return self.history[list(self.history)[-1]]['acc']

        def latest_loss(self):
            if len(self.history) == 0:
                return 0
            return self.history[list(self.history)[-1]]['loss']

        def stop(self, federated, acc: float):
            return (0 < federated.num_rounds <= self.round_id) or acc >= federated.desired_accuracy

        def build(self, federated):
            self.reset()
            self.model = federated.initial_model() if callable(federated.initial_model) else federated.initial_model

        def reset(self):
            self.round_id = 0
            self.history.clear()

        def store(self, **kwargs):
            """

            Args:
                **kwargs: kwargs is used to pass any key value similar to function, use it like store(acc=0.79)

            Returns:

            """
            if self.round_id not in self.history:
                self.history[self.round_id] = Dict()
            self.history[self.round_id].update(kwargs)

        def describe(self):
            return f"created at {self.timestamp}"

        def update_model(self, temporary_model, accuracy, accepted_accuracy_margin):
            highest_accuracy = self.highest_accuracy()
            is_model_accepted = not accepted_accuracy_margin or (accuracy >= highest_accuracy) or (
                    abs(accuracy - highest_accuracy) < accepted_accuracy_margin)
            if is_model_accepted:
                self.model = temporary_model
            return is_model_accepted

        def get_field(self, field_name):
            return [items[field_name] for rnd, items in self.history.items()]
