import logging
import sys


sys.path.append('../../')
from libs.model.cv.cnn import CNN_OriginalFedAvg
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
from torch import nn
from src.apis.mpi import Comm
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import MPITrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

comm = Comm()
if comm.pid() == 0:
    logger.info('Generating Data --Started')
    client_data = data_loader.femnist_100c_2000min_2000max()
    logger.info('Generating Data --Ended')

    trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=150, optimizer='sgd',
                                   criterion='cel', lr=0.001)

    federated = FederatedLearning(
        trainer_manager=MPITrainerManager(),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
        client_selector=client_selectors.Random(0.1),
        trainers_data_dict=client_data,
        initial_model=lambda: CNN_OriginalFedAvg(False),
        num_rounds=100,
        desired_accuracy=0.99,
    )

    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND, Timer.TRAINER]))

    logger.info("----------------------")
    logger.info("start federated 1")
    logger.info("----------------------")
    federated.start()
else:
    MPITrainerManager.mpi_trainer_listener(comm)
