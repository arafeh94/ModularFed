import logging
from typing import Union

import tqdm

from src.apis.extensions import Dict
from src.federated.events import Events, FederatedSubscriber
from src.federated.federated import FederatedLearning


class TqdmLogger(FederatedSubscriber):
    def __init__(self):
        super().__init__()
        self.tqdm: tqdm.tqdm = None
        self.logger = logging.getLogger('tqdm')
        self.logger.propagate = False
        empty_handler = logging.StreamHandler()
        empty_handler.setFormatter(logging.Formatter(''))
        self.logger.addHandler(empty_handler)

    def on_federated_started(self, params):
        self.tqdm = tqdm.tqdm(total=params['num_rounds'], desc='INFO:tqdm:FL Progress')
        self.logger.info('')

    def on_round_start(self, params):
        self.tqdm.update()
        self.logger.info('')


class FederatedLogger(FederatedSubscriber):
    def __init__(self, only=None):
        super().__init__(only)
        self.trainers_data_dict = None
        self.logger = logging.getLogger('federated')

    def on_federated_started(self, params):
        params = Dict(params).but(['context'])
        self.logger.info('federated learning started')

    def on_federated_ended(self, params):
        context: FederatedLearning.Context = params['context']
        self.logger.info(f'federated learning ended')
        self.logger.info(f'"""final accuracy {context.latest_accuracy()}"""')

    def on_init(self, params):
        params = Dict(params).but(['context'])
        self.logger.info(f'federated learning initialized with initial model {params}')

    def on_training_start(self, params):
        params = Dict(params).but(['context'])
        self.logger.info(f"training started {params}")

    def on_training_end(self, params):
        params = Dict(params).but(['context'])
        self.logger.info(f"training ended {params}")

    def on_aggregation_end(self, params):
        params = Dict(params).but(['context'])
        self.logger.info(f"aggregation ended {params}")

    def on_round_end(self, params):
        params = Dict(params).but(['context'])
        self.logger.info(f'federated learning ended {params}')
        self.logger.info("----------------------------------------")

    def on_round_start(self, params):
        params = Dict(params).but(['context'])
        self.logger.info(f"round started {params}")

    def on_trainer_start(self, params):
        params = Dict(params).but(['context', 'train_data'])
        self.logger.info(f"trainer started {params}")

    def on_trainer_end(self, params):
        params = Dict(params).but(['context', 'trained_model'])
        self.logger.info(f"trainer ended {params}")

    def on_model_status_eval(self, params):
        params = Dict(params).but(['context', 'trained_model'])
        self.logger.info(f"model status: {params}")

    def force(self) -> []:
        return [Events.ET_FED_START, Events.ET_MODEL_STATUS, Events.ET_FED_END]

    def on_trainers_selected(self, params):
        params = Dict(params).but(['context'])
        self.logger.info(f"selected clients {params}")
