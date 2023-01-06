from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

db1 = FedDB('all.db')
graphs = Graphs(db1)
print(graphs)

graphs.plot([
    {
        'session_id': 't1647774412',
        'field': 'acc',
        'config': {'color': 'b', 'label': 'basic'},
    },
    {
        'session_id': 't1647774452',
        'field': 'acc',
        'config': {'color': 'y', 'label': 'ours'},
    },
])
