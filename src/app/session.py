import copy
import hashlib
import random
from datetime import date

from src.apis import utils
from src.app.cache import Cache
from src.app.settings import Settings


class Session:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache: Cache = settings.get('cache', absent_ok=True) or Cache()
        self._create_cache()

    def write(self, key, obj):
        self.cache.write(key, obj)

    def read(self, key):
        return self.cache.read(key, absent_ok=True)

    def _hash(self):
        config_copy = copy.deepcopy(self.settings.get_config())
        if 'rounds' in config_copy:
            del config_copy['rounds']
        hashed = utils.hash_string(str(config_copy))
        return hashed

    def _generate_id(self):
        return f'session{random.randint(0, 99999)}'

    def session_id(self):
        return self.read('session_id')

    def _create_cache(self):
        self.cache.file_path = f'./cache/{self._hash()}'
        if self.cache.exists():
            self.cache.write('updated_at', f'{date.today()}')
        else:
            session_id = self.settings.get("session_id", absent_ok=True) or self._generate_id()
            self.cache.write("created_at", str(date.today()))
            self.cache.write('updated_at', str(date.today()))
            self.cache.write('session_configs', self.settings.get_config())
            self.cache.write('session_id', session_id)
            self.cache.write('hash', self._hash())
