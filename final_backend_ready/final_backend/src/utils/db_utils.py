import psycopg2

def get_database_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        database="SecureDocAI",
        user="postgres",
        password="5103"
    )