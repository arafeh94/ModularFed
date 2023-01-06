import json
import mysql.connector

db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='shakespeare'
)
cursor = db.cursor()

file = open('../../raw/shakespeare/shakespeare_all_data.json', 'r')
shakespear = json.load(file)

user_data = shakespear['user_data']

finished = 0
for user_id, data in user_data.items():
    print(f"start with {user_id}")
    for x, y in zip(data['x'], data['y']):
        query = "insert into shakespeare.sample values (null, %s, %s, %s, %s)"
        cursor.execute(query, (user_id, x, y, 0))
    finished += 1
    print(f"finished with {user_id}")
    print(f"finished: {finished / len(user_data) * 100}%")
db.commit()
