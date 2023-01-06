# mpiexec -n 2 python distributed_averaging.py
import sys

sys.path.append('../../')
from src.apis import utils
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from src.federated.components.rest.RESTrainer import RESTrainerServer
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
import logging
from torch import nn
from src.federated.protocols import TrainerParams
from src.federated.components import metrics, client_selectors, aggregators, trainers, rest
from libs.model.linear.lr import LogisticRegression
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.rest.RESTrainerManager import RESTrainerManager, FileScanner
from src.federated.components.rest.RESTDriver import RESTFalconDriver

utils.enable_logging()
kind = sys.argv[1]
host, port = sys.argv[2].split(':')
hosts = sys.argv[3] if len(sys.argv) > 3 else None
print(kind, host, port)
if kind == 'server':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    logger.info('Generating Data --Started')
    client_data = preload('kdd', ShardDistributor(300, 2)).select(range(40))
    logger.info('Generating Data --Ended')
    scanner = FileScanner(hosts)
    trainer_manager = RESTrainerManager(RESTFalconDriver(host, port), scanner)
    trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=20,
                                   optimizer='sgd', criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(50, criterion=nn.CrossEntropyLoss()),
        client_selector=client_selectors.Random(1),
        trainers_data_dict=client_data,
        initial_model=lambda: LogisticRegression(41, 2),
        num_rounds=0,
        desired_accuracy=0.99
    )

    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    logger.info("----------------------")
    logger.info("start federated 1")
    logger.info("----------------------")
    federated.start()
else:
    trainer = RESTrainerServer(RESTFalconDriver(host, port))
    trainer.init_server()
