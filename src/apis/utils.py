import hashlib
import logging
import typing
from datetime import datetime, timedelta
from functools import reduce
import numpy as np
import torch
from sklearn import decomposition
from sklearn.cluster import KMeans

from src.apis.extensions import Dict

logger = logging.getLogger('utils')


def smooth(vals, sigma=2):
    from scipy.ndimage import gaussian_filter1d
    return list(gaussian_filter1d(vals, sigma=sigma))


def hash_string(string: str):
    full_hash = str.encode(string)
    return hashlib.md5(full_hash).hexdigest()


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))


# noinspection PyUnresolvedReferences
def fed_avg(runs: typing.List['FederatedLearning.Context']):
    from collections import defaultdict
    import numpy as np
    avg_acc = defaultdict(list)
    avg_loss = defaultdict(list)
    for run in runs:
        for round_id, performance in run.history.items():
            avg_acc[round_id].append(performance['acc'])
            avg_loss[round_id].append(performance['loss'])

    for round_id in avg_acc:
        avg_acc[round_id] = np.average(avg_acc[round_id])
        avg_loss[round_id] = np.average(avg_loss[round_id])
    return avg_acc, avg_loss


def cluster(client_weights: Dict, cluster_size=10, compress_weights=True):
    logger.info("Clustering Models --Started")
    weights = []
    client_ids = []
    clustered = {}
    for client_id, stats in client_weights.items():
        client_ids.append(client_id)
        weights.append(compress(flatten_weights(stats), 4)
                       if compress_weights else flatten_weights(stats))
    kmeans: KMeans = KMeans(n_clusters=cluster_size).fit(weights)
    logger.info(kmeans.labels_)
    for i, label in enumerate(kmeans.labels_):
        clustered[client_ids[i]] = label
    logger.info("Clustering Models --Finished")
    return clustered


def compress(weights, n_components):
    pca = decomposition.PCA(n_components)
    pca.fit(weights)
    weights = pca.transform(weights)
    return weights.flatten()


def flatten_weights(weights):
    weight_vectors = []
    for _, weight in weights.items():
        weight_vectors.extend(weight.flatten().tolist())
    return np.array(weight_vectors)


def dict_select(idx, dict_ref):
    new_dict = {}
    for i in idx:
        new_dict[i] = dict_ref[i]
    return new_dict


def models_state(models):
    if isinstance(models, list):
        return [model.state_dict() for model in models]
    if isinstance(models, dict):
        return Dict(models).map(lambda _, model: model.state_dict())


def timed_func(seconds, callable: typing.Callable):
    stop = datetime.now() + timedelta(seconds=seconds)
    while datetime.now() < stop:
        callable()


def enable_logging(file_name=None, level=logging.INFO):
    if file_name:
        logging.basicConfig(filename=file_name, filemode='w', datefmt='%H:%M:%S', level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        logger.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
            model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            logger.info(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            logger.info(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True
