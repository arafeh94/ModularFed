# federated learning distributed training based on web http rest api
# advantage, secure communication over any network interface
# run distributed_rest_trainer.py first to initialize a server for each trainer
# run distributed_rest_server.py second to start sending training requests to the trainers
# drawback, the server should have a map which consists of each trainer ip+port


import sys
from os.path import dirname

sys.path.append(dirname(__file__) + '../../../')

from src.federated.components.rest import comon
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
import logging
from torch import nn
from src.federated.protocols import TrainerParams
from src.federated.components import metrics, client_selectors, aggregators, trainers, rest
from libs.model.linear.lr import LogisticRegression
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.rest.RESTrainerManager import RESTrainerManager
from src.federated.components.rest.RESTDriver import RESTFalconDriver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload('mnist', ShardDistributor(300, 5)).select(range(40))
logger.info('Generating Data --Ended')

trainer_manager = RESTrainerManager(RESTFalconDriver('localhost', '8080'), comon.SEQ(client_data.keys()))
trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=20, optimizer='sgd',
                               criterion='cel', lr=0.1)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(2),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=0,
    desired_accuracy=0.99
)

federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
