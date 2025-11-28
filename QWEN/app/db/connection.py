import psycopg2
import psycopg2.extras

DATABASE_URL = "dbname=medelse user=tristan password= host=localhost port=5432"

def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    return conn
