import logging
import os
import sys
import platform
from src.apis.mpi import Comm
from src.app.federated_app import FederatedApp
from src.app.settings import Settings
from src.federated.components.trainer_manager import MPITrainerManager
from src.federated.subscribers.adapters import MPIStop

logging.basicConfig(level=logging.INFO)


class MPIApp(FederatedApp):
    def __init__(self, settings: Settings, comm: Comm):
        super().__init__(settings)
        self.comm = comm

    def _trainer_manager(self):
        return MPITrainerManager()

    def _default_subscribers(self, session):
        subs = super(MPIApp, self)._default_subscribers(session)
        return subs + [MPIStop(self.comm)]


class DistributedApp:
    WINDOWS_OS = 'windows'
    LINUX_OS = 'linux'
    WINDOWS_MPI = 'windows-mpi'
    OPEN_MPI = 'open-mpi'
    AUTO = 'auto'

    def __init__(self, settings: Settings, np=2, os=AUTO):
        self.settings = settings
        self.comm = Comm()
        self.script = sys.argv[0]
        self.is_child = len(sys.argv) > 1 and sys.argv[1] == 'child'
        self.np = np
        self.os = os
        self.logger = logging.getLogger(f'distributed:{self.comm.pid()}')

    def start(self):
        if self.is_child:
            if self.comm.pid() == 0:
                self.logger.info(f'server start: {self.settings}')
                app = MPIApp(self.settings, self.comm)
                app.start()
            else:
                self.logger.info(f'child start: {self.comm.pid()}')
                MPITrainerManager.mpi_trainer_listener(self.comm)
        else:
            os.system(self._exec())

    def _exec(self):
        os = self.os
        if os == DistributedApp.AUTO:
            os = platform.system().lower()
        if os == DistributedApp.WINDOWS_OS or os == DistributedApp.WINDOWS_MPI:
            return f'mpiexec -n {self.np} py {self.script} child'
        else:
            return f'mpirun -np {self.np} py {self.script} child'
