from src.data.data_container import DataContainer
from src.data.data_distributor import Distributor


class Dataset:
    def __init__(self, data: DataContainer, distribution: Distributor):
        self.data = data
        self.distribution = distribution
