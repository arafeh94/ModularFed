import logging

from torch import nn

import src
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from libs.model.linear.lr import LogisticRegression
from src.data.data_provider import PickleDataProvider
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.fedruns import FedRuns
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../datasets/pickles/mnist_2shards_70c_600mn_600mx.pkl'
test_file = '../../datasets/pickles/test_data.pkl'

logger.info('Generating Data --Started')
client_data = data_loader.mnist_10shards_100c_400min_400max()
logger.info('Generating Data --Ended')

federated_configs = {
    'first': {
        'batch_size': 8,
        'epochs': 10,
        'criterion': 'cel',
        'optimizer': 'sgd',
        'lr': 0.1,
        'num_clients': 10,
        'num_rounds': 3,
        'desired_accuracy': 0.99,
        'model_init': lambda: LogisticRegression(28 * 28, 10),
        'clients_data': client_data,
    },
    'second': {
        'batch_size': 8,
        'epochs': 3,
        'criterion': 'cel',
        'optimizer': 'sgd',
        'lr': 0.1,
        'num_clients': 10,
        'num_rounds': 3,
        'desired_accuracy': 0.99,
        'model_init': lambda: LogisticRegression(28 * 28, 10),
        'clients_data': client_data,
    },
    'third': {
        'batch_size': 18,
        'epochs': 2,
        'criterion': 'cel',
        'optimizer': 'sgd',
        'lr': 0.1,
        'num_clients': 10,
        'num_rounds': 10,
        'desired_accuracy': 0.99,
        'model_init': lambda: LogisticRegression(28 * 28, 10),
        'clients_data': client_data,
    },
}

federated_runs = {}

for name, federated_params in federated_configs.items():
    trainer_manager = SeqTrainerManager()
    trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=federated_params['batch_size'],
                                   epochs=federated_params['epochs'], optimizer=federated_params['optimizer'],
                                   criterion=federated_params['criterion'], lr=federated_params['lr'])

    federated = FederatedLearning(
        trainer_manager=trainer_manager,
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=federated_params['batch_size'], criterion=nn.CrossEntropyLoss()),
        client_selector=client_selectors.Random(federated_params['num_clients']),
        trainers_data_dict=federated_params['clients_data'],
        initial_model=federated_params['model_init'],
        num_rounds=federated_params['num_rounds'],
        desired_accuracy=federated_params['desired_accuracy']
    )

    federated.add_subscriber(subscribers.FederatedLogger([Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(subscribers.CustomModelTestPlug(PickleDataProvider(test_file).collect().as_tensor(),
                                                             federated_params['batch_size'], False))

    logger.info("----------------------")
    logger.info("start federated " + name)
    logger.info("----------------------")
    federated.start()
    federated_runs[name] = federated.context

runs = FedRuns(federated_runs)
runs.compare_all()
runs.plot()
