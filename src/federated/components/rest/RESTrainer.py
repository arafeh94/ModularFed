import pickle
from threading import Thread

from src.federated.components.rest.RESTDriver import RESTDriver
from src.federated.events import FederatedSubscriber
from src.federated.protocols import Trainer


class RESTrainerServer(FederatedSubscriber):

    def __init__(self, driver: RESTDriver):
        super().__init__()
        self.driver = driver

    def on_init(self, params):
        self.init_server()

    class Train:
        def __init__(self, trainer: 'RESTrainerServer'):
            self.trainer = trainer

        def on_post(self, req, resp):
            model, train_data, context, config = pickle.load(req.stream)
            trainer: Trainer = config.trainer_class()
            result = trainer.train(model, train_data, context, config)
            result = pickle.dumps(result)
            resp.content_type = req.content_type
            resp.method = 'post'
            resp.data = result
            return resp

    class Whoami:
        def on_get(self, req, resp):
            resp.text = 'Federated Learning Trainer'
            return resp.text

    def init_server(self):
        thread = Thread(target=self.driver.serve, args=(
            {
                '/train': self.Train(self),
                '/whoami': self.Whoami()
            },))
        thread.start()
