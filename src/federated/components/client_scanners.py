from src.apis.extensions import Dict
from src.federated.protocols import ClientScanner


class DefaultScanner(ClientScanner):
    def __init__(self, client_data: Dict):
        self.client_data = client_data

    def scan(self):
        return self.client_data
