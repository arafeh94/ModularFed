import copy
import importlib
import json
import typing


class Clazz:
    def __init__(self, clz_str: str):
        self.module_name = clz_str[0:clz_str.rfind('.')]
        self.class_name = clz_str[clz_str.rfind('.') + 1:len(clz_str)]

    def create(self, params: dict = None):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.class_name)
        instance = class_(**params) if params else class_(**params)
        return instance

    def refer(self):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.class_name)
        return class_

    @staticmethod
    def is_class(val: str):
        try:
            clz = Clazz(val)
            clz.create()
            return True
        except:
            return False


class Settings:
    def __init__(self, config):
        self.configs = self._init(config)
        self.cursor = 0
        self.zero_stop = True

    def __iter__(self):
        self.cursor = 0
        self.zero_stop = True
        return self

    def __next__(self):
        if self.zero_stop:
            self.zero_stop = False
            return self

        self.cursor = self.cursor + 1
        if self.cursor >= len(self.configs):
            raise StopIteration
        return self

    def get_config(self):
        return self.configs[self.cursor]

    def set_cursor(self, cursor: int):
        self.cursor = cursor

    def _init(self, config):
        if not isinstance(config, list):
            config = [config]
        return config

    def get(self, key, force_args=None, absent_ok=True) -> typing.Any:
        if force_args and key in force_args:
            return force_args[key]
        try:
            val = self._extract(key)
            return self._create(val)
        except Exception as e:
            if absent_ok:
                return None
            else:
                raise e

    def _extract(self, key: str, absent_ok=False):
        if '.' in key:
            paths = key.split('.')
            level = self.get_config()
            for path in paths:
                if path in level:
                    level = level[path]
                else:
                    raise KeyError(paths)
            return level
        if key not in self.get_config():
            raise KeyError(key)
        return self.get_config()[key]

    def _has_children(self, level):
        if isinstance(level, dict):
            if ('class' in level) or ('refer' in level):
                return False
            return True
        return False

    def __len__(self):
        return len(self.configs)

    def _create(self, obj) -> typing.Any:
        if isinstance(obj, dict):
            if 'refer' in obj:
                class_ref = obj.get('refer')
                return Clazz(class_ref).refer()
            if 'class' in obj:
                class_name = obj.get('class')
                obj_params = {}
                for key, item in obj.items():
                    if key != 'class':
                        obj_params[key] = self._create(item)
                return Clazz(class_name).create(obj_params)
        if isinstance(obj, list):
            initializations = []
            for o in obj:
                initialized = self._create(o)
                initializations.append(initialized)
            return initializations
        return obj

    @staticmethod
    def from_json_file(file_path):
        configs = json.load(open(file_path, 'r'))
        return Settings(configs)

    @staticmethod
    def from_json(file_path):
        configs = json.loads(file_path)
        return Settings(configs)
