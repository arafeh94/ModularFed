# federated learning distributed training based on web http rest api
# advantage, secure communication over any network interface
# run distributed_rest_trainer.py first to initialize a server for each trainer
# run distributed_rest_server.py second to start sending training requests to trainers
# drawback, the server should have a map which consists of each trainer ip+port


from src.apis import utils
from src.federated.components.rest.RESTrainer import RESTrainerServer
from src.federated.components.rest.RESTDriver import RESTFalconDriver
from src.federated.components.trainers import TorchTrainer

utils.enable_logging()
for i in range(40):
    trainer = RESTrainerServer(RESTFalconDriver('localhost', 8081 + i))
    trainer.init_server()
