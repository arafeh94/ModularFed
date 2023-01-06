import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import manifest
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('../experiments/res.db'))
plt.rcParams.update({'font.size': 28})
plt.grid()
print(graphs)
graphs.plot([
    {
        'session_id': 's1',
        'field': 'acc',
        'config': {'color': 'b', 'label': 'IID'},
    },
    {
        'session_id': 's2',
        'field': 'acc',
        'config': {'color': 'r', 'label': 'Non-IID', 'linestyle': 'dotted'},
    },
], xlabel='Round', ylabel='Accuracy')
