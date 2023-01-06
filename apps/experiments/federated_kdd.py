import logging

from torch import nn

from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.linear.lr import LogisticRegression
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = data_loader.kdd_100c_400min_400max()
logger.info('Generating Data --Ended')

trainer_config = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=20, optimizer='sgd',
                               criterion='cel', lr=0.1)
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_config,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(5),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(41, 2),
    num_rounds=50,
    desired_accuracy=0.99
)

federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
