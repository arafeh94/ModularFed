from src.apis import lambdas
from src.apis.extensions import Dict
from src.data.data_container import DataContainer


def cifar10_rgb(data):
    if isinstance(data, Dict):
        return data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    if isinstance(data, DataContainer):
        return data.map(lambdas.reshape((32, 32, 3))).map(lambdas.transpose((2, 0, 1)))

