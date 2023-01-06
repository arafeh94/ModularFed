import asyncio
from time import sleep

from src.apis import utils
from src.federated.components.rest.RESTDriver import RESTFalconDriver
from src.federated.components.rest.RESTrainer import RESTrainer
from src.federated.components.rest.RESTrainerManager import RESTrainerManager
from src.federated.components.trainers import TorchTrainer

utils.enable_logging()

trainer1 = RESTrainer(RESTFalconDriver('localhost', '8081'), TorchTrainer())
trainer2 = RESTrainer(RESTFalconDriver('localhost', '8082'), TorchTrainer())
mng = RESTrainerManager(RESTFalconDriver('localhost', 8080), {'1': ('localhost', 8081), '2': ('localhost', 8082)})
mng.init_server()
trainer1.init_server()
trainer2.init_server()
sleep(1)
mng.train_req('1', 'asdasd', 'asdad', 'asdasd', 'asdasd')
mng.resolve()
