import copy
import logging
import os
import pickle
import typing
from abc import abstractmethod, ABC
from time import sleep

import dill as dill
import numpy as np
import torch
import tqdm
from sklearn import decomposition
from torch import nn

T = typing.TypeVar('T')
B = typing.TypeVar('B')
K = typing.TypeVar('K')
V = typing.TypeVar('V')


class Functional(typing.Generic[T]):
    @abstractmethod
    def for_each(self, func: typing.Callable[[T], typing.NoReturn]):
        pass

    @abstractmethod
    def filter(self, predicate: typing.Callable[[T], bool]):
        pass

    @abstractmethod
    def map(self, func: typing.Callable[[T], T]):
        pass

    @abstractmethod
    def reduce(self, func: typing.Callable[[B, T], B]):
        pass

    @abstractmethod
    def select(self, keys):
        pass

    @abstractmethod
    def concat(self, other):
        pass


class Dict(typing.Dict[K, V], Functional):

    # noinspection PyDefaultArgument
    def __init__(self, iter_map: typing.Dict[K, V] = {}):
        super().__init__(iter_map)

    def for_each(self, func: typing.Callable) -> typing.NoReturn:
        for key, val in self.items():
            func(key, val)

    def select(self, keys) -> 'Dict[K, V]':
        return Dict({key: self[key] for key in keys})

    def filter(self, predicate: typing.Callable[[K, V], bool]) -> 'Dict[K, V]':
        new_dict = Dict()
        for key, val in self.items():
            if predicate(key, val):
                new_dict[key] = self[key]
        return new_dict

    def map(self, func: typing.Callable[[K, V], V]) -> 'Dict[K, V]':
        new_dict = Dict()
        for key, val in self.items():
            new_val = func(key, val)
            new_dict[key] = new_val
        return new_dict

    def reduce(self, func: typing.Callable[[B, K, V], B]) -> 'B':
        first_item = None
        for key, val in self.items():
            first_item = func(first_item, key, val)
        return first_item

    def but(self, keys):
        new_dict = {}
        for item, val in self.items():
            if item not in keys:
                new_dict[item] = val
        return new_dict

    def concat(self, other):
        self.update(other)


class Array(typing.List[V], Functional):
    def __init__(self, iter_=None):
        iter_ = iter_ if iter_ is not None else []
        super().__init__(iter_)

    def for_each(self, func: typing.Callable[[V], None]) -> typing.NoReturn:
        for item in self:
            func(item)

    def filter(self, predicate: typing.Callable[[V], bool]) -> 'Array[V]':
        new_a = Array()
        for item in self:
            if predicate(item):
                new_a.append(item)
        return new_a

    def map(self, func: typing.Callable[[V], V]) -> 'Array[V]':
        new_a = Array()
        for item in self:
            na = func(item)
            new_a.append(na)
        return new_a

    def reduce(self, func: typing.Callable[[B, V], B]) -> 'B':
        first = None
        for item in self:
            first = func(first, item)
        return first

    def select(self, indexes: typing.List[V]) -> 'Array[V]':
        new_a = Array()
        for index in indexes:
            new_a.append(self[index])
        return new_a

    def concat(self, other: 'Array[V]') -> 'Array[V]':
        return Array(self.copy().extend(other))


class Serializable:
    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = logging.getLogger('Serializable')

    def save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        to_save = {}
        for key, item in self.__dict__.items():
            if not callable(item):
                to_save[key] = item
        self._flush(to_save)

    def _flush(self, to_save):
        try:
            with open(self.file_path, 'wb') as fop:
                dill.dump(to_save, fop)
        except Exception as e:
            self.logger.info(e)
            sleep(1)
            self._flush(to_save)

    def load(self):
        if self.exists():
            try:
                with open(self.file_path, 'rb') as fop:
                    for key, item in dill.load(fop).items():
                        if 'file_path' in key:
                            continue
                        self.__dict__[key] = item
                return True
            except Exception as e:
                self.logger.info(e)
                sleep(1)
                self.load()

    def exists(self):
        return os.path.exists(self.file_path)

    def sync(self, func, *params):
        self.load()
        func(*params)
        self.save()


# noinspection PyUnresolvedReferences
class TorchModel:
    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger('TorchModel')

    def train(self, batched, **kwargs):
        r"""
        Args:
            batched:
            **kwargs:
                epochs (int)
                lr (float)
                momentum (float)
                optimizer (Optimizer)
                criterion (_WeightedLoss)
                verbose (int)
        Returns:

        """
        model = self.model
        epochs = kwargs.get('epochs', 1)
        learn_rate = kwargs.get('lr', 0.01)
        momentum = kwargs.get('momentum', 0)
        optimizer = kwargs.get('optimizer', torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum))
        criterion = kwargs.get('criterion', nn.CrossEntropyLoss())
        device = kwargs['device'] if 'device' in kwargs else ('cuda' if torch.cuda.is_available() else 'cpu')
        verbose = kwargs.get('verbose', 1)
        model.to(device)
        model.train()
        data_size = len(batched) * len(batched[0][0])
        iterator = tqdm.tqdm(range(epochs), 'training')
        for _ in iterator:
            correct = 0
            for batch_idx, (x, labels) in enumerate(batched):
                x = x.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                correct += (log_probs.max(dim=1)[1] == labels).float().sum()
            accuracy = round(100 * (float(correct) / data_size), 2)
            iterator.set_postfix_str(f"accuracy: {accuracy}")
        weights = model.cpu().state_dict()
        return weights

    def infer(self, batched, **kwargs):
        verbose = kwargs.get('verbose', 1)
        model = self.model
        model.eval()
        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            iterator = tqdm.tqdm(enumerate(batched), 'inferring') if verbose else enumerate(batched)
            for batch_idx, (x, target) in iterator:
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc / test_total, test_loss / test_total

    def log(self, msg, level=1):
        self.logger.info(msg)

    def weights(self):
        return self.model.state_dict()

    def copy(self):
        return copy.deepcopy(self)

    def load(self, weights):
        self.model.load_state_dict(weights)

    def save(self, file_path):
        save_file = open(file_path, 'wb')
        pickle.dump(self.model, save_file)

    @staticmethod
    def open(file_path):
        if os.path.isfile(file_path):
            model = pickle.load(open(file_path, 'rb'))
            return TorchModel(model)
        else:
            return False

    def flatten(self):
        all_weights = []
        for _, weight in self.weights().items():
            all_weights.extend(weight.flatten().tolist())
        return np.array(all_weights)

    def compress(self, output_dim, n_components):
        weights = self.flatten().reshape(output_dim, -1)
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(weights)
        weights = pca.transform(weights)
        return weights.flatten()

    def extract(self):
        return self.model

    def state(self):
        return self.extract().state_dict()


def first(items: list):
    return next(filter(lambda x: x, items))
