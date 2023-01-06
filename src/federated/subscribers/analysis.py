import json
import logging
import os
import time
from collections import defaultdict
from random import randint

import numpy as np
from matplotlib import pyplot as plt
from src.apis import plots, utils
from src.apis.extensions import Dict
from src.data.data_container import DataContainer
from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning


class ClientSelectionCounter(FederatedSubscriber):
    def __init__(self, save_dir=None):
        super().__init__()
        self.client_counter = defaultdict(int)
        self.save_dir = save_dir

    def on_trainers_selected(self, params):
        trainers_ids, context = params['trainers_ids'], params['context']
        for trainer_id in trainers_ids:
            self.client_counter[trainer_id] += 1
        context.store(selection_counter=json.dumps(self.client_counter))
        self.plot(show=False)

    def on_federated_ended(self, params):
        logging.getLogger('selection_counter').info(self.client_counter)
        self.plot(show=True)

    def plot(self, show=True):
        plt.bar(self.client_counter.keys(), self.client_counter.values())
        plt.savefig(f"{self.save_dir}") if self.save_dir else ()
        if show:
            plt.show()


class ShowDataDistribution(FederatedSubscriber):
    def __init__(self, label_count, save_dir=None):
        super().__init__()
        self.label_count = label_count
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def on_federated_started(self, params):
        clients_data: Dict[int, DataContainer] = params['trainers_data_dict']
        self.plot(clients_data, self.label_count, self.save_dir)

    @staticmethod
    def plot(clients_data: Dict[int, DataContainer], label_count: int, save_dir=None, text_color=None,
             sub_title=None, title='Clients Data Distribution'):
        """
        Args:
            clients_data:
            label_count:
            save_dir:
            text_color:
            sub_title:
            title:

        Returns: an image with x:client - y:class

        """
        tick = time.time()
        logger = logging.getLogger('data_distribution')
        logger.info('building data distribution...')
        ids = list(clients_data.keys())
        id_mapper = lambda id: ids.index(id)

        client_label_count = np.zeros((len(clients_data), label_count))
        for client_id, data in clients_data.items():
            for y in data.y:
                client_label_count[id_mapper(client_id)][int(y)] += 1
        save_dir = f"{save_dir}/data_distribution.png" if save_dir is not None else None
        client_label_count = np.transpose(client_label_count)
        plots.heatmap(client_label_count, title, sub_title, save_dir, text_color=text_color)
        logger.info(f'building data distribution finished {time.time() - tick}')


class ShowWeightDivergence(FederatedSubscriber):
    def __init__(self, show_log=False, include_global_weights=False, save_dir=None, plot_type='matrix', caching=False):
        """
        show the weight divergence of the model after each round
        Args:
            show_log: print out the log to console
            include_global_weights:
            save_dir: a place to save the images
            plot_type: ['matrix', 'linear']

        """
        super().__init__()
        self.logger = logging.getLogger('weights_divergence')
        self.show_log = show_log
        self.include_global_weights = include_global_weights
        self.trainers_weights = None
        self.global_weights = None
        self.save_dir = save_dir
        self.round_id = 0
        self.plot_type = plot_type
        self.caching = caching
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_training_end(self, params):
        self.trainers_weights = params['trainers_weights']

    def on_aggregation_end(self, params):
        self.global_weights = params['global_weights']

    def on_round_end(self, params):
        context: 'FederatedLearning.Context' = params['context']
        tick = time.time()
        self.logger.info('extracting weights...')
        self.round_id = params['context'].round_id
        save_dir = f"./{self.save_dir}/round_{self.round_id}_wd.png" if self.save_dir is not None else None
        acc = params['accuracy']
        trainers_weights = self.trainers_weights
        if self.include_global_weights:
            trainers_weights[len(trainers_weights)] = self.global_weights
        ids = list(trainers_weights.keys())
        self.logger.info(f'building weights divergence finished {time.time() - tick}')
        if self.plot_type == 'matrix':
            id_mapper = lambda id: ids.index(id)
            heatmap = np.zeros((len(trainers_weights), len(trainers_weights)))
            for trainer_id, weights in trainers_weights.items():
                for trainer_id_1, weights_1 in trainers_weights.items():
                    w0 = utils.flatten_weights(weights)
                    w1 = utils.flatten_weights(weights_1)
                    heatmap[id_mapper(trainer_id)][id_mapper(trainer_id_1)] = np.var(np.subtract(w0, w1))
            plots.heatmap(heatmap, 'Weight Divergence', f'Acc {round(acc, 4)}', save_dir)
            context.store(heatmap=json.dumps(heatmap))
            if self.show_log:
                self.logger.info(heatmap)
        elif self.plot_type == 'linear':
            weight_dict = defaultdict(lambda: [])
            for trainer_id, weights in trainers_weights.items():
                weights = utils.flatten_weights(weights)
                weights = np.reshape(weights, (5, -1))
                weights = utils.compress(weights, 5)
                weight_dict[trainer_id] = weights.tolist()
            context.store(pca=json.dumps(weight_dict)) if self.caching else None
            plots.linear(weight_dict, "Model's Weights", f'R: {self.round_id}', save_dir)
        else:
            raise Exception('plot type should be a string with a value either "linear" or "matrix"')
