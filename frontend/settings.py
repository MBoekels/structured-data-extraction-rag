import os

# Correctly set the base directory and construct the database path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "rag_app.db")

API_BASE = "http://127.0.0.1:8000"