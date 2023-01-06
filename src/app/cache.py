from src.apis.rw import IODict


class Cache(IODict):
    def __init__(self):
        super().__init__(file_path=None)
