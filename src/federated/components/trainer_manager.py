from abc import ABC, abstractmethod
from typing import Union

from src.apis.broadcaster import Subscriber
from src.apis.mpi import Comm
from src.federated.events import FederatedSubscriber
from src.federated.protocols import Trainer, TrainerParams


class TrainerManager:
    def __init__(self, scanner):
        self.scanner = scanner
        self.trainer_started = None
        self.trainer_finished = None

    @abstractmethod
    def train_req(self, trainer_id, model, train_data, context, config: TrainerParams):
        pass

    @abstractmethod
    def resolve(self):
        pass

    def notify_trainer_started(self, trainer_id):
        if callable(self.trainer_started):
            self.trainer_started(trainer_id=trainer_id)

    def notify_trainer_finished(self, trainer_id, weights, sample_size):
        if callable(self.trainer_finished):
            self.trainer_finished(trainer_id=trainer_id, weights=weights, sample_size=sample_size)


class SeqTrainerManager(TrainerManager):
    class TrainerProvider(ABC):
        @abstractmethod
        def collect(self, trainer_id, config: TrainerParams) -> Trainer:
            pass

    def __init__(self, trainer_provider: TrainerProvider = None):
        super().__init__(None)
        self.train_requests = {}
        self.trainer_provider = trainer_provider
        if trainer_provider is None:
            self.trainer_provider = SharedTrainerProvider()

    def train_req(self, trainer_id, model, train_data, context, config):
        trainer = self.trainer_provider.collect(trainer_id, config)
        request = [trainer.train, model, train_data, context, config]
        self.train_requests[trainer_id] = request
        return request

    def resolve(self):
        trainers_trained_weights = {}
        trainers_sample_size = {}
        for trainer_id, request in self.train_requests.items():
            train_func, model, train_data, context, config = request
            self.notify_trainer_started(trainer_id)
            trained_weights, sample_size = train_func(model, train_data, context, config)
            self.notify_trainer_finished(trainer_id, trained_weights, sample_size)
            trainers_trained_weights[trainer_id] = trained_weights
            trainers_sample_size[trainer_id] = sample_size
        self.train_requests = {}
        return trainers_trained_weights, trainers_sample_size


class SharedTrainerProvider(SeqTrainerManager.TrainerProvider):
    def __init__(self):
        self.trainers = {}

    def collect(self, trainer_id, config: TrainerParams) -> Trainer:
        return self._trainer(trainer_id, config)

    def _create(self, trainer_id, config: TrainerParams) -> Trainer:
        trainer = config.trainer_class()
        self.trainers[trainer_id] = trainer
        return trainer

    def _trainer(self, trainer_id, config):
        if trainer_id not in self.trainers.keys():
            self.trainers[trainer_id] = self._create(trainer_id, config)
        return self.trainers[trainer_id]


class MPITrainerManager(TrainerManager):

    def __init__(self):
        super().__init__(None)
        self.comm = Comm()
        self.procs = [i + 1 for i in range(self.comm.size() - 1)]
        self.used_procs = []
        self.requests = {}

    def train_req(self, trainer_id, model, train_data, context, config):
        pid = self.get_proc()
        self.comm.send(pid, (model, train_data, context, config), 1)
        self.notify_trainer_started(trainer_id)
        req = self.comm.irecv(pid, 2)
        self.requests[trainer_id] = req

    def get_proc(self):
        for proc in self.procs:
            if proc not in self.used_procs:
                self.used_procs.append(proc)
                return proc
        raise Exception("No more available processes to answer the request. "
                        "Increase mpi nb proc or decrease selected clients number for each round")

    def reset(self):
        self.used_procs = []
        self.requests = {}

    def resolve(self):
        trainers_trained_weights = {}
        trainers_sample_size = {}
        for trainer_id, req in self.requests.items():
            trained_weights, sample_size = req.wait()
            self.notify_trainer_finished(trainer_id, trained_weights, sample_size)
            trainers_trained_weights[trainer_id] = trained_weights
            trainers_sample_size[trainer_id] = sample_size
        self.reset()
        return trainers_trained_weights, trainers_sample_size

    @staticmethod
    def mpi_trainer_listener(comm: Comm):
        trainer: Union[Trainer, None] = None
        while True:
            model, train_data, context, config = comm.recv(0, 1)
            trainer = trainer or config.trainer_class()
            trained_weights, sample_size = trainer.train(model, train_data, context, config)
            comm.send(0, (trained_weights, sample_size), 2)


class StrictMPITrainerManager(MPITrainerManager):

    def __init__(self, client_rank_mapping, skip_on_running=True):
        super().__init__()
        self.select_trainer_id = None
        self.client_rank_mapping = client_rank_mapping
        self.skip_on_running = skip_on_running

    def train_req(self, trainer_id, model, train_data, context, config):
        self.select_trainer_id = trainer_id
        super().train_req(trainer_id, model, train_data, context, config)

    def get_proc(self):
        if self.select_trainer_id not in self.client_rank_mapping:
            raise Exception(f"Requested trainer [{self.select_trainer_id}] is not mapped to an MPI process, "
                            "make sure client_rank_mapping are properly defined")
        if self.client_rank_mapping[self.select_trainer_id] in self.used_procs and not self.skip_on_running:
            raise Exception(f"Requested trainer {self.select_trainer_id} is already working on a task,"
                            "make sure you not selecting the same client twice for the same round,"
                            "use skip_on_running=True to bypass this error")
        return self.client_rank_mapping[self.select_trainer_id]

    @staticmethod
    def default_map(size):
        res = {}
        for i in range(1, size + 1):
            res[i - 1] = i
        return res
