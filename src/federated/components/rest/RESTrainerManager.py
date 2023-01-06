import asyncio
import logging
import pickle
from abc import abstractmethod
from asyncio import Future
from threading import Thread
from typing import Dict

import requests

from src.federated.components.rest.RESTDriver import RESTDriver
from src.federated.components.trainer_manager import TrainerManager
from src.federated.events import FederatedSubscriber
from src.federated.protocols import TrainerParams, ClientScanner
from requests_futures.sessions import FuturesSession as Session


class RESTrainerManager(TrainerManager, FederatedSubscriber):
    def __init__(self, driver: RESTDriver, host_scanner: ClientScanner, crash_on_error=True):
        super().__init__(scanner=host_scanner)
        self.crash_on_error = crash_on_error
        self.logger = logging.getLogger('RESTrainerManager')
        self.driver = driver
        self.requests: Dict[Future] = {}
        self.headers = {'Content-Type': 'application/octet-stream'}

    def train_req(self, trainer_id, model, train_data, context, config: TrainerParams):
        trainer_iport_map = self.scanner.scan()
        if trainer_id not in trainer_iport_map:
            self.logger.warning(f'Requesting non existent trainer {trainer_id}')
            if self.crash_on_error:
                raise Exception(f'Requesting non existent trainer {trainer_id}')
        ip, port = trainer_iport_map[trainer_id]
        data = pickle.dumps([model, train_data, context, config])
        url = f'http://{ip}:{port}/train'
        self.logger.info(f'Sending API rest request to worker {trainer_id}-@-{url}')
        req = Session().post(url=url, data=data, headers=self.headers)
        self.requests[trainer_id] = req

    def resolve(self):
        trainers_trained_weights = {}
        trainers_sample_size = {}
        for trainer_id, req in self.requests.items():
            result = req.result()
            self.logger.debug(f'received worker request, {trainer_id}:{result}')
            trained_weights, sample_size = pickle.loads(result.content)
            self.notify_trainer_finished(trainer_id, trained_weights, sample_size)
            trainers_trained_weights[trainer_id] = trained_weights
            trainers_sample_size[trainer_id] = sample_size
        self.reset()
        return trainers_trained_weights, trainers_sample_size

    def reset(self):
        self.used_procs = []
        self.requests = {}

    def on_init(self, params):
        self.init_server()

    def init_server(self):
        thread = Thread(target=self.driver.serve, args=({},))
        thread.start()


class FileScanner(ClientScanner, FederatedSubscriber):
    def __init__(self, file):
        super().__init__()
        self.file = file

    def scan(self):
        iports = {}
        with open(self.file, 'r') as file:
            for item in file:
                client, url = item.split(',')
                iports[int(client)] = url.split(':')
        return iports
