import pickle
from os import listdir
from os.path import isfile, join

from src.apis.rw import IODict
from src.federated.federated import FederatedLearning

path = './cache'
path_files = [f for f in listdir(path) if isfile(join(path, f))]
for file in path_files:
    fp = IODict(path + '/' + file)
    context: FederatedLearning.Context = fp.read('context')
    print(fp.cached)
