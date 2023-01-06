import matplotlib.pyplot as plt
import numpy as np

from src import manifest
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def variance(data, each=30):
    var_data = []
    results = []
    while len(var_data) < each:
        var_data.append(data[len(var_data)])
        results.append(0)
    for d in data[each:]:
        results.append(np.var(var_data))
        var_data.pop(0)
        var_data.append(d)
    return normalize(results)


graphs = Graphs(FedDB('../experiments/perf.db'))
print(graphs)
plt.rcParams.update({'font.size': 22})
plt.grid()
graphs.plot([
    {
        'session_id': 'cifar',
        'field': 'acc',
        'config': {'color': 'b'},
        'transform': utils.smooth
    },
    {
        'session_id': 'cifar',
        'field': 'wd',
        'config': {'color': 'r'},
        'transform': normalize
    },
    {
        'session_id': 'cifar',
        'field': 'wd',
        'config': {'color': 'g'},
        'transform': variance
    },
], animated=True)
