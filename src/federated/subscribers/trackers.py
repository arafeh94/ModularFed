import logging
import sys

from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning


class BandwidthTracker(FederatedSubscriber):
    def __init__(self, print_log=False):
        super().__init__()
        self.logger = logging.getLogger('band_tr') if print_log else None

    def log(self, msg):
        if self.logger:
            self.logger.log(logging.INFO, msg)

    def on_trainer_end(self, params):
        weights = params['weights']
        weights_size = sys.getsizeof(weights)
        context: 'FederatedLearning.Context' = params['context']
        old = {}
        if 'models_size' in context.last_entry():
            old = context.last_entry()['models_size']
        old[params['trainer_id']] = weights_size
        context.store(models_size=old)
        self.log(old)
