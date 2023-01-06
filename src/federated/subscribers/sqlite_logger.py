import logging
import os.path
import re
import sqlite3
from pathlib import Path
from sqlite3 import OperationalError

from src import manifest
from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning


# noinspection SqlNoDataSourceInspection
class SQLiteLogger(FederatedSubscriber):
    def __init__(self, id, db_path=manifest.DB_PATH, config=''):
        super().__init__()
        self.id = id
        self.con = sqlite3.connect(db_path, timeout=1000)
        self.check_table_creation = True
        self._logger = logging.getLogger('sqlite')
        self.tag = str(config)
        self._check_table_name()

    def on_federated_started(self, params):
        self.init()

    def init(self):
        query = 'create table if not exists session (session_id text primary key, config text)'
        self._execute(query)
        query = f"insert or replace into session values (?,?)"
        self._execute(query, [self.id, self.tag])

    def _create_table(self, **kwargs):
        if self.check_table_creation:
            self.check_table_creation = False
            params = self._extract_params(**kwargs)
            sub_query = ''
            for param in params:
                sub_query += f'{param[0]} {param[1]},'
            sub_query = sub_query.rstrip(',')
            query = f'''
            create table if not exists {self.id} (
                {sub_query}
            )
            '''
            self._execute(query)

    def _insert(self, params):
        sub_query = ' '.join(['?,' for _ in range(len(params))]).rstrip(',')
        query = f'insert OR replace into {self.id} values ({sub_query})'
        values = list(map(lambda v: str(v) if isinstance(v, (list, dict)) else v, params.values()))
        self._execute(query, values)

    def _execute(self, query, params=None):
        cursor = self.con.cursor()
        self._logger.debug(f'executing {query}')
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        self.con.commit()

    def _extract_params(self, **kwargs):
        def param_map(val):
            if isinstance(val, int):
                return 'INTEGER'
            elif isinstance(val, str):
                return 'TEXT'
            elif isinstance(val, float):
                return 'FLOAT'
            else:
                return 'text'

        params = [('round_id', 'INTEGER PRIMARY KEY')]
        for key, val in kwargs.items():
            params.append((key, param_map(val)))
        return params

    def log(self, round_id, **kwargs):
        self._create_table(**kwargs)
        record = {'round_id': round_id, **kwargs}
        self._insert(record)

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        last_record: dict = context.history[context.round_id]
        self.log(context.round_id, **last_record)

    def _check_table_name(self):
        if self.id is None:
            self.id = 'None'
        if self.id[0].isdigit():
            self.id = f't{self.id}'
