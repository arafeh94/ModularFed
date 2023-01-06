import statistics
import sys
from collections import defaultdict
from statistics import variance
from typing import Union

from src.apis import lambdas
from src.apis.extensions import Dict
from src.data.data_container import DataContainer


def norm_mean(values):
    max_val = max(values)
    if max_val == 0:
        return 0
    norm = [i / max(values) for i in values]
    return statistics.mean(norm)


def iidness(data: Dict[int, DataContainer], labels, aggregation_method=statistics.mean, by_label=False):
    client_label_sizes = {}
    if not isinstance(labels, list):
        labels = range(labels)
    for cid, dc in data.items():
        client_label_sizes[cid] = label_count(dc)
        for lbl in labels:
            if lbl not in client_label_sizes[cid]:
                client_label_sizes[cid][lbl] = 0

    label_variances = {}
    for label in labels:
        clients_label_count = []
        for cid, client_label_size in client_label_sizes.items():
            clients_label_count.append(client_label_size[label])
        label_variances[label] = statistics.stdev(clients_label_count)
    results = aggregation_method(label_variances.values())
    if by_label:
        results = results, label_variances
    return results


def label_count(data: DataContainer):
    labels = defaultdict(int)
    for x, y in data.iter():
        labels[y] += 1
    return Dict(labels)
