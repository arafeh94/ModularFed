import sys

sys.path.append('../../../')
from src.app.settings import Settings
from src.app.distributed_app import DistributedApp

if __name__ == '__main__':
    settings = Settings.from_json_file('mnist.json')
    app = DistributedApp(settings, np=6)
    app.start()
