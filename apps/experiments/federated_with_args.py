# py federated_with_args.py -e 25 -b 50 -r 100 -s 2 -d mnist -cr 0.1 -lr 0.1 -t mnist10 -mn 600 -mx 600 -cln 100

import logging
import sys

sys.path.append('../../')
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
from torch import nn
from libs.model.cv.cnn import Cifar10Model
from src.apis import lambdas
from src.apis.federated_args import FederatedArgs
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from libs.model.linear.lr import LogisticRegression
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

args = FederatedArgs()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(args.dataset, LabelDistributor(args.clients, args.shard, args.min, args.max))
logger.info('Generating Data --Ended')

if args.dataset == 'cifar10':
    client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    initial_model = Cifar10Model()
else:
    initial_model = LogisticRegression(28 * 28, 10)

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=args.batch, epochs=args.epoch,
                               optimizer='sgd', criterion='cel', lr=args.learn_rate)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=args.batch, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(args.clients_ratio),
    trainers_data_dict=client_data,
    initial_model=lambda: initial_model,
    num_rounds=args.round,
    desired_accuracy=0.99,
    accepted_accuracy_margin=0.01
)

federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
