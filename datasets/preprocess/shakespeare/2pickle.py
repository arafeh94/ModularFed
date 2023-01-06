import json
import mysql.connector

from libs import language_tools
from src.data.data_container import DataContainer
from src.data.data_provider import PickleDataProvider

file = open('../../raw/shakespeare/shakespeare_all_data.json', 'r')
shakespear = json.load(file)

user_data = shakespear['user_data']

finished = 0
all_x = []
all_y = []
for user_id, data in user_data.items():
    print(f"start with {user_id}")
    for x, y in zip(data['x'], data['y']):
        all_x.append(language_tools.word_to_indices(x))
        all_y.append(language_tools.letter_to_index(y))
    finished += 1
    print(f"finished with {user_id}")
    print(f"finished: {finished / len(user_data) * 100}%")
dc = DataContainer(all_x, all_y)
PickleDataProvider.save(dc, '../../pickles/shakespeare.pkl')
