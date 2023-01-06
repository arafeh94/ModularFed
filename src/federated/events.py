import typing

from src.apis.broadcaster import Subscriber


class Events:
    ET_FED_START = 'federated_learning_start'
    ET_INIT = 'init'
    ET_ROUND_START = 'round_start'
    ET_TRAINER_SELECTED = 'trainers_selected'
    ET_TRAIN_START = 'training_started'
    ET_TRAIN_END = 'training_finished'
    ET_AGGREGATION_END = 'aggregation_finished'
    ET_ROUND_FINISHED = 'round_finished'
    ET_FED_END = 'federated_learning_end'
    ET_TRAINER_STARTED = 'trainer_started'
    ET_TRAINER_FINISHED = 'trainer_ended'
    ET_MODEL_STATUS = 'model_status'


class FederatedSubscriber(Subscriber):

    def __init__(self, only: None or [] = None):
        super().__init__(only)

    def on_federated_started(self, params):
        """
        :param params: params['context']
        :return:
        """
        pass

    def on_federated_ended(self, params):
        """

        :param params: params['aggregated_model'], params['context']
        :return:
        """
        pass

    def on_init(self, params):
        """
        :param params: params['global_model'], params['context']
        :return:
        """
        pass

    def on_training_start(self, params):
        """

        :param params: params['trainers_data'], params['context']
        :return:
        """
        pass

    def on_training_end(self, params):
        """

        :param params: params['trainers_weights'], params['sample_size'], params['context']
        :return:
        """
        pass

    def on_aggregation_end(self, params):
        """

        :param params: params['global_weights'], params['global_model'], params['context']
        :return:
        """
        pass

    def on_round_end(self, params):
        """

        :param params: params['round'], params['accuracy'], params['loss'], params['local_acc'], params['local_loss'],
            params['context']
        :return:
        """
        pass

    def on_round_start(self, params):
        """

        :param params: params['round'], params['context']
        :return:
        """
        pass

    def on_trainers_selected(self, params):
        """

        :param params: params['trainers_ids'], params['context']
        :return:
        """
        pass

    def on_trainer_start(self, params):
        """

       :param params: params['trainer_id'], params['context']
       :return:
       """
        pass

    def on_trainer_end(self, params):
        """

       :param params: params['trainer_id'], params['weights'], params['sample_size'], params['context']
       :return:
       """
        pass

    def on_model_status_eval(self, params):
        """

         :param params: params['model_status'], params['context']
         :return:
         """
        pass

    def force(self) -> []:
        return []

    # noinspection PyUnresolvedReferences
    def attach(self, federated_learning: 'FederatedLearning'):
        federated_learning.add_subscriber(self)

    def map_events(self) -> typing.Dict[str, typing.Callable]:
        return {
            Events.ET_FED_START: self.on_federated_started,
            Events.ET_INIT: self.on_init,
            Events.ET_ROUND_START: self.on_round_start,
            Events.ET_TRAINER_SELECTED: self.on_trainers_selected,
            Events.ET_TRAIN_START: self.on_training_start,
            Events.ET_TRAIN_END: self.on_training_end,
            Events.ET_AGGREGATION_END: self.on_aggregation_end,
            Events.ET_ROUND_FINISHED: self.on_round_end,
            Events.ET_FED_END: self.on_federated_ended,
            Events.ET_TRAINER_STARTED: self.on_trainer_start,
            Events.ET_TRAINER_FINISHED: self.on_trainer_end,
            Events.ET_MODEL_STATUS: self.on_model_status_eval,
        }
