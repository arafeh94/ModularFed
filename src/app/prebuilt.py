import json

from src.app.federated_app import FederatedApp
from src.app.settings import Settings


def FastFed(**kwargs):
    configs = """
        {
      "session_id": "fast_fed",
      "cache": {
        "class": "src.app.cache.Cache"
      },
      "data": {
        "dataset": "mnist",
        "distributor": {
          "class": "src.data.data_distributor.LabelDistributor",
          "num_clients": 50,
          "min_size": 1200,
          "max_size": 1200,
          "label_per_client": 8
        }
      },
      "model": {
        "class": "libs.model.linear.lr.LogisticRegression",
        "input_dim": 784,
        "output_dim": 10
      },
      "trainer_config": {
        "class": "src.federated.protocols.TrainerParams",
        "trainer_class": {
          "refer": "src.federated.components.trainers.TorchTrainer"
        },
        "epochs": 1,
        "lr": 0.01,
        "batch_size": 999,
        "optimizer": "sgd",
        "criterion": "cel"
      },
      "aggregator": {
        "class": "src.federated.components.aggregators.AVGAggregator"
      },
      "rounds": 100,
      "client_ratio": 0.5,
      "desired_accuracy": 2,
      "device": "cuda"
    }
    """
    settings = json.loads(configs)
    settings = {**settings, **kwargs}
    settings = Settings(settings)
    return FederatedApp(settings, **kwargs)
