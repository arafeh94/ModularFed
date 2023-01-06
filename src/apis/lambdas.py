import typing

import numpy as np

from src.data.data_container import DataContainer


def reshape(shape: typing.Tuple) -> typing.Callable:
    def _inner(a, b):
        if isinstance(b, DataContainer):
            return b.reshape(shape)
        return np.reshape(a, shape), b

    return _inner


def transpose(shape: typing.Tuple) -> typing.Callable:
    def _inner(a, b):
        if isinstance(b, DataContainer):
            return b.transpose(shape)
        return np.transpose(a, shape), b

    return _inner


def take_only_features(nb_features) -> typing.Callable:
    return lambda cid, data: DataContainer(data.x[:, 0:nb_features], data.y)


def empty(_, value) -> bool:
    return len(value) > 0


def as_numpy(_, val: DataContainer) -> np.array:
    return val.as_numpy()


def as_list(_, val: DataContainer) -> np.array:
    return val.as_list()


def as_tensor(_, val: DataContainer) -> 'Tensor':
    if not isinstance(val, DataContainer):
        raise Exception(f'as tensor lambdas work only on dictionaries with DataContainer as value, '
                        f'current value type is {type(val)}. try using .as_tensor() instead')
    return val.as_tensor()


def dict2dc(dc: DataContainer, key: int, val: DataContainer) -> DataContainer:
    dc = DataContainer([], []) if dc is None else dc
    return dc.concat(val)


def dc_split(percentage, take0or1) -> typing.Callable:
    return lambda cid, data: data.split(percentage)[take0or1]


# noinspection PyPep8Naming
class reducers:
    @staticmethod
    def dict2dc(dc: DataContainer, key: int, val: DataContainer) -> DataContainer:
        return dict2dc(dc, key, val)


# noinspection PyPep8Naming
class mappers:
    @staticmethod
    def as_numpy(_, val: DataContainer) -> np.array:
        return as_numpy(_, val)

    @staticmethod
    def as_tensor(_, val: DataContainer) -> 'Tensor':
        return as_tensor(_, val)

    @staticmethod
    def dc_split(percentage, take0or1) -> typing.Callable:
        return dc_split(percentage, take0or1)

    @staticmethod
    def reshape(shape: typing.Tuple) -> typing.Callable:
        return reshape(shape)

    @staticmethod
    def transpose(shape: typing.Tuple) -> typing.Callable:
        return transpose(shape)
