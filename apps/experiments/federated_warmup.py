import logging
import sys


sys.path.append('../../')

from src.federated.subscribers.analysis import ShowWeightDivergence
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
from torch import nn
from src.apis import lambdas, files
from src.apis.extensions import TorchModel
from libs.model.linear.lr import LogisticRegression
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
tag = 'federated_test_basic_2'
is_warmup = True

logger.info('Generating Data --Started')
client_data = data_loader.mnist_2shards_100c_600min_600max()
logger.info('Generating Data --Ended')

if is_warmup:
    warmup_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.2)[0]).reduce(lambdas.dict2dc).as_tensor()
    client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.2)[1]).map(lambdas.as_tensor)
    initial_model = TorchModel(LogisticRegression(28 * 28, 10))
    initial_model.train(warmup_data.batch(50), epochs=500)
    initial_model = initial_model.extract()
else:
    initial_model = LogisticRegression(28 * 28, 10)

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=1, optimizer='sgd',
                               criterion='cel', lr=0.1)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(0.1),
    trainers_data_dict=client_data,
    initial_model=lambda: initial_model,
    num_rounds=50,
    desired_accuracy=0.99,
)
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(ShowWeightDivergence(save_dir="./pct", plot_type='linear'))
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()

files.accuracies.save_accuracy(federated, tag)
