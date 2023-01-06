import logging
import time
from collections import defaultdict
from src.federated.events import Events, FederatedSubscriber
from src.federated.federated import FederatedLearning


class Timer(FederatedSubscriber):
    TRAINER = 'trainer'
    FEDERATED = 'federated'
    ROUND = 'round'
    TRAINING = 'training'
    AGGREGATION = 'aggregation'

    def __init__(self, show_only=None):
        super().__init__(None)
        self.ticks = defaultdict(lambda: 0)
        self.show_only = show_only
        self.logger = logging.getLogger("fed_timer")
        if show_only is not None:
            for item in show_only:
                if item not in [Timer.FEDERATED, Timer.ROUND, Timer.AGGREGATION, Timer.TRAINING]:
                    Exception('requested timer does not exists')

    def tick(self, name, is_done):
        now = time.time()
        now_cpu = time.process_time()
        if is_done:
            dif = now - self.ticks[name]
            dif_cpu = now_cpu - self.ticks[name + '_cpu']
            if self.show_only is not None and name not in self.show_only:
                return
            self.logger.info(f'{name}, elapsed: {round(dif, 3)}s')
            self.logger.info(f'{name}, elapsed: {round(dif_cpu, 3)}s of cpu time')
        else:
            self.ticks[name] = now
            self.ticks[name + '_cpu'] = now_cpu

    def on_federated_started(self, params):
        self.tick('federated', False)

    def on_federated_ended(self, params):
        self.tick(self.FEDERATED, True)

    def on_training_start(self, params):
        self.tick(self.TRAINING, False)

    def on_training_end(self, params):
        self.tick(self.TRAINING, True)
        self.tick(self.AGGREGATION, False)

    def on_round_start(self, params):
        self.tick(self.ROUND, False)

    def on_round_end(self, params):
        self.tick(self.ROUND, True)
        context: 'FederatedLearning.Context' = params['context']
        context.store(round_time=self.ticks[Timer.ROUND])
        context.store(aggregation_time=self.ticks[Timer.AGGREGATION])
        context.store(training_time=self.ticks[Timer.TRAINING])

    def on_aggregation_end(self, params):
        self.tick(self.AGGREGATION, True)

    def on_trainer_start(self, params):
        self.tick(self.TRAINER, False)

    def on_trainer_end(self, params):
        self.tick(self.TRAINER, True)

    def force(self) -> []:
        return [Events.ET_FED_START, Events.ET_TRAINER_FINISHED, Events.ET_TRAINER_STARTED, Events.ET_TRAIN_START,
                Events.ET_TRAIN_END, Events.ET_AGGREGATION_END, Events.ET_INIT, Events.ET_ROUND_START,
                Events.ET_ROUND_FINISHED]
