import os
import pickle
import typing
from src import manifest
import logging
from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.data.data_distributor import Distributor, LabelDistributor, SizeDistributor, UniqueDistributor
from src.data.data_provider import PickleDataProvider

logger = logging.getLogger('data_loader')


def get_dataset_path(dataset_name):
    # try to load dataset from our dropbox urls, if it not exists it might be either a name in dataset path or a hole
    # path. if it is a path we load it, otherwise we check if the dataset path contains such a dataset
    # if all fails we raise an exception
    file_path = manifest.DATA_PATH + dataset_name + ".pkl"
    if os.path.exists(file_path):
        return file_path
    if os.path.exists(dataset_name):
        return dataset_name
    if dataset_name in manifest.datasets_urls:
        return manifest.datasets_urls[dataset_name]
    raise Exception('unknown dataset is requested')


def load_tag(tag):
    file_path = manifest.DATA_PATH + tag + ".pkl"
    data = pickle.load(open(file_path, 'rb'))
    return data


def preload(dataset, distributor: Distributor = None, tag=None, transformer=None) -> typing.Union[
    Dict[int, DataContainer], DataContainer]:
    """
    Args:
        tag: file name without postfix (file type, auto-filled with .pkl)
        dataset: dataset used, should be exists inside urls
        distributor: distribution function, dg.distribute_shards or dg.distribute_size ...
        transformer: transform data

    Returns: clients data of type typing.Dict[int, DataContainer]

    """

    tag = tag or (dataset + '_' + distributor.id() if distributor else dataset)
    file_path = manifest.DATA_PATH + tag + ".pkl"
    logger.info(f'searching for {file_path}...')
    data = None
    if os.path.exists(file_path):
        logger.info('distributed data file exists, loading...')
        data = pickle.load(open(file_path, 'rb'))
        logger.info(data)
    else:
        logger.info('distributed data file does not exists, distributing...')
        data = PickleDataProvider(get_dataset_path(dataset)).collect(file_name=dataset)
        if distributor:
            data = distributor.distribute(data)
        data = transformer(data) if callable(transformer) else data
        pickle.dump(data, open(file_path, 'wb'))
    return data


def mnist() -> DataContainer:
    return preload('mnist', None)


def cifar10() -> DataContainer:
    return preload('cifar10', None)


def fall() -> DataContainer:
    return preload('fall_all_merged', None)


def kdd() -> DataContainer:
    return preload('kdd', None)


def mnist_10shards_100c_400min_400max() -> Dict[int, DataContainer]:
    return preload('mnist', LabelDistributor(100, 10, 400, 400))


def cifar10_10shards_100c_400min_400max() -> Dict[int, DataContainer]:
    return preload('cifar10', LabelDistributor(100, 10, 400, 400))


def cifar10_10shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('cifar10', LabelDistributor(100, 10, 600, 600))


def mnist_2shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('mnist', LabelDistributor(100, 2, 600, 600))


def cifar10_2shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('cifar10', LabelDistributor(100, 2, 600, 600))


def cifar10_1shard_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('cifar10', LabelDistributor(100, 1, 600, 600))


def mnist_1shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('mnist', LabelDistributor(100, 1, 600, 600))


def femnist_2shards_100c_600min_600max() -> Dict[int, DataContainer]:
    return preload('femnist', LabelDistributor(100, 2, 600, 600))


def femnist_100c_2000min_2000max() -> Dict[int, DataContainer]:
    return preload('femnist', SizeDistributor(100, 2000, 2000))


def femnist_2shards_100c_2000min_2000max() -> Dict[int, DataContainer]:
    return preload('femnist', LabelDistributor(100, 2, 2000, 2000))


def kdd_100c_400min_400max() -> Dict[int, DataContainer]:
    return preload('kdd', SizeDistributor(100, 400, 400))


def femnist_1shard_62c_2000min_2000max() -> Dict[int, DataContainer]:
    return preload('femnist', UniqueDistributor(62, 2000, 2000))


def femnist_1shard_62c_200min_2000max() -> Dict[int, DataContainer]:
    return preload('femnist', UniqueDistributor(62, 200, 2000))
