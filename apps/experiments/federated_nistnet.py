import logging
import sys

from src.federated.events import Events

sys.path.append('../../')

from libs.model.linear.mnist_net import MnistNet
from src.federated.subscribers.trackers import BandwidthTracker
from src.federated.subscribers.fed_plots import RoundAccuracy, RoundLoss
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

client_data = preload('mnist', ShardDistributor(300, 2)).select(range(20))

# trainers configuration
trainer_params = TrainerParams(
    trainer_class=trainers.TorchTrainer,
    batch_size=50, epochs=10, optimizer='sgd',
    criterion='cel', lr=0.1)

# fl parameters
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
    client_selector=client_selectors.Random(1),
    trainers_data_dict=client_data,
    initial_model=lambda: MnistNet(28 * 28, 32, 10),
    num_rounds=100,
    desired_accuracy=0.99
)

federated.start()

# (subscribers)
federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
# federated.add_subscriber(RoundAccuracy(plot_ratio=5))
# federated.add_subscriber(RoundLoss(plot_ratio=5))
# federated.add_subscriber(SQLiteLogger('exp', 'res.db'))
# federated.add_subscriber(BandwidthTracker())

logger.info("----------------------")
logger.info("start federated learning")
logger.info("----------------------")
federated.start()
