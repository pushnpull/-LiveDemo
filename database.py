import mysql.connector

class Database:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self._connection = None

    def __enter__(self):
        self._connection = mysql.connector.connect(
            host=self.host,
            port= 3306,
            user=self.user,
            password=self.password,
            database=self.database
        )
        return self._connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.close()

def execute_query(query, params=None):
    with Database(host="localhost", user="root", password="root", database="chilling") as connection:
        cursor = connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()