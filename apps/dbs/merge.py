import matplotlib.pyplot as plt
import numpy as np

from src import manifest
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

db1 = FedDB('../experiments/perf2.db')
db2 = FedDB('../../compares/perf.db')
db1.merge(db2)
