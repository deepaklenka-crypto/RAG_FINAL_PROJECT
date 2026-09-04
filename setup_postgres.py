"""
PostgreSQL Database Setup Script:
Connects with user 'postgres' and password 'sa', creates database 'rag' if not present,
and initializes all tables.
"""

import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine

def setup():
    user = "postgres"
    password = "sa"
    host = "localhost"
    port = 5432
    target_db = "rag"

    print(f"Connecting to PostgreSQL at {host}:{port} as user '{user}'...")
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=user,
            password=password,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (target_db,))
        exists = cur.fetchone()
        if not exists:
            print(f"Creating database '{target_db}'...")
            cur.execute(f'CREATE DATABASE "{target_db}";')
            print(f"Database '{target_db}' created successfully.")
        else:
            print(f"Database '{target_db}' already exists.")

        cur.close()
        conn.close()

        # Initialize schema inside 'rag' database
        pg_url = f"postgresql://{user}:{password}@{host}:{port}/{target_db}"
        print(f"Testing connection to target database '{target_db}'...")
        engine = create_engine(pg_url)
        with engine.connect() as connection:
            print("Successfully connected to 'rag' database!")

        # Import and init tables
        import database
        database.DATABASE_URL = pg_url
        database.engine = engine
        database.SessionLocal.configure(bind=engine)
        database.init_db()
        print("Initialized all database tables in PostgreSQL 'rag' database:")
        print(" - documents")
        print(" - document_chunks")
        print(" - graph_entities")
        print(" - graph_relations")
        print(" - query_logs")
        print(" - evaluation_logs")
        print(" - benchmark_logs")

    except Exception as e:
        print("Error connecting to PostgreSQL:", e)
        sys.exit(1)

if __name__ == "__main__":
    setup()
