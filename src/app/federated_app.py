import logging

from src.apis.broadcaster import Subscriber
from src.app.session import Session
from src.app.settings import Settings
from src.data.data_loader import preload
from src.federated.components import aggregators, client_selectors, metrics
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer

logging.basicConfig(level=logging.INFO)


class FederatedApp:
    def __init__(self, settings: Settings, **kwargs):
        self.settings = settings
        self.logger = logging.getLogger('main')
        self.log_level = settings.get('log_level', kwargs, absent_ok=True) or logging.INFO
        self.load_default_subscribers = kwargs.get('load_default_subscribers', True)
        self.kwargs = kwargs

    def _get_distributed_data(self):
        if 'data' in self.kwargs:
            distributed_data = self.kwargs['data']
        else:
            dataset_name = self.settings.get('data.dataset', self.kwargs, absent_ok=False)
            distributor = self.settings.get('data.distributor', self.kwargs, absent_ok=False)
            transformer = self.settings.get('data.transformer', self.kwargs, absent_ok=True)
            distributed_data = preload(dataset_name, distributor, transformer=transformer)
        return distributed_data

    def init_federated(self, session):
        model = self.settings.get('model', self.kwargs, absent_ok=False)
        desired_accuracy = self.settings.get('desired_accuracy', self.kwargs, absent_ok=True) or 0.99
        trainer_params = self.settings.get("trainer_config", self.kwargs)
        aggregator = self.settings.get('aggregator', self.kwargs, absent_ok=True) or aggregators.AVGAggregator()
        metric = metrics.AccLoss(
            batch_size=self.settings.get('trainer_config.batch_size', self.kwargs, absent_ok=False),
            criterion=self.settings.get('trainer_config.criterion', self.kwargs),
            device=self.settings.get('device', self.kwargs) or None)
        selector = self.settings.get('client_selector', self.kwargs) or client_selectors.Random(
            self.settings.get('client_ratio', self.kwargs, absent_ok=False))
        distributed_data = self._get_distributed_data()
        rounds = self.settings.get('rounds', self.kwargs)
        federated = FederatedLearning(
            trainer_manager=self._trainer_manager(),
            trainer_config=trainer_params,
            aggregator=aggregator,
            metrics=metric,
            client_selector=selector,
            trainers_data_dict=distributed_data,
            initial_model=lambda: model,
            num_rounds=rounds,
            desired_accuracy=desired_accuracy,
            accepted_accuracy_margin=self.settings.get('accepted_accuracy_margin', self.kwargs) or -1,
        )
        return federated

    def _trainer_manager(self):
        return SeqTrainerManager()

    def _start(self, subscribers=None):
        if subscribers and not isinstance(subscribers, list):
            subscribers = [subscribers]
        session = Session(self.settings)
        federated = self.init_federated(session)
        configs_subscribers: list = session.settings.get('subscribers', self.kwargs, absent_ok=True) or []
        subscribers = configs_subscribers + subscribers if subscribers else configs_subscribers
        self._attach_subscribers(federated, subscribers, session)
        federated.start()
        return federated

    def start(self, *subscribers):
        for index, st in enumerate(self.settings):
            self.logger.info(f'starting config {index}: {str(st.get_config())}')
            return self._start([s for s in subscribers] if subscribers else None)

    def _attach_subscribers(self, federated: FederatedLearning, subscribers, session):
        self.logger.info('attaching subscribers...')
        subscribers = self._check_subscribers(subscribers, session)
        for subs in subscribers:
            self.logger.info(f'attaching: {type(subs)}')
            federated.add_subscriber(subs)

    def _check_subscribers(self, subscribers, session):
        attaching_subscribers = self._default_subscribers(session) if self.load_default_subscribers else []
        attaching_subscribers = (subscribers or []) + attaching_subscribers
        for subscriber in attaching_subscribers:
            if not isinstance(subscriber, Subscriber):
                raise Exception(f'unsupported subscriber of type {type(subscriber)}')
        return attaching_subscribers

    def _default_subscribers(self, session):
        return [
            FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]),
            Timer([Timer.FEDERATED, Timer.ROUND]),
            # Resumable(io=session.cache),
            # SQLiteLogger(session.session_id(), db_path='./cache/perf.db', config=str(session.settings.get_config()))
        ]
