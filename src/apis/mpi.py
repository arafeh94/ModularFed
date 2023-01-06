class Comm:
    def __init__(self):
        try:
            from mpi4py import MPI
        except Exception as e:
            raise Exception(str(e) + ' No MPI module found, please install using: pip install mpi4py, '
                                     'also make sure host support mpi module')
        self.mpi: MPI.Intracomm = MPI.COMM_WORLD

    def size(self):
        return self.mpi.size

    def pid(self):
        return self.mpi.rank

    def send(self, pid, message, tag=0):
        self.mpi.send(message, pid, tag)

    def recv(self, src=None, tag=None):
        return self.mpi.recv(source=src, tag=tag)

    def isend(self, pid, message, tag=0):
        return self.mpi.send(message, pid, tag)

    def irecv(self, src=None, tag=None):
        return self.mpi.irecv(9_999_999, source=src, tag=tag)

    def stop(self):
        self.mpi.Abort()
