import logging
from abc import abstractmethod

from src.apis.extensions import Dict


class RESTDriver:
    def __init__(self):
        pass

    @abstractmethod
    def serve(self, router: dict):
        pass


class RESTFalconDriver(RESTDriver):
    def __init__(self, ip, port):
        self.logger = logging.getLogger('RESTFalconDriver')
        super().__init__()
        import falcon
        self.ip = ip
        self.port = int(port)
        self.falcon_server = falcon.App()

    def serve(self, router: dict):
        try:
            for route, controller in router.items():
                self.falcon_server.add_route(route, controller)
            from wsgiref.simple_server import make_server
            with make_server(self.ip, self.port, self.falcon_server) as httpd:
                self.logger.info(f'Serving on http://{self.ip}:{self.port}')
                httpd.serve_forever()
        except Exception as e:
            raise e
