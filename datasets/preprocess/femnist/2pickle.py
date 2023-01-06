import logging

import h5py

from src.data.data_container import DataContainer
from src.data.data_provider import PickleDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

path_train = "../../raw/femnist/fed_emnist_train.h5"
path_test = "../../raw/femnist/fed_emnist_test.h5"

f = h5py.File(path_train, 'r')
x = []
y = []
for name in f:
    for user in f[name]:
        h5_x = f[name][user]['pixels']
        h5_y = f[name][user]['label']
        logging.info(f"processing user {user} - num raw {len(h5_x)}")
        for i in range(len(h5_x)):
            x.append(f[name][user]['pixels'][i].flatten().tolist())
            y.append(f[name][user]['label'][i])

dc = DataContainer(x, y)
print("saving...")
PickleDataProvider.save(dc, '../../pickles/femnist.pkl')
