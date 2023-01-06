import os
from pathlib import Path
import shutil

from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

FedDB('../basic.db').merge(FedDB('../genetic2.db'))
shutil.copyfile('../basic.db', './all.db')
