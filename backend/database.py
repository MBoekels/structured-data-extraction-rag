import sqlite3
import json
import logging
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles all interactions with the SQLite database."""

    def __init__(self, db_url: str = settings.DATABASE_URL):
        """
        Initializes the database manager.
        
        Args:
            db_url (str): The connection string for the database.
        """
        # The db_url for sqlite is 'sqlite:///path/to/db.file'
        # We need to extract the file path.
        db_path_str = db_url.replace("sqlite:///", "")
        self.db_path = Path(db_path_str)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection = None
        self._connect()
        self._setup_tables()

    def _connect(self):
        """Establishes a connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Successfully connected to database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def _setup_tables(self):
        """Creates the necessary tables if they don't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pdfs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING, SUCCESS, FAILED
                    docling_json_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (company_id) REFERENCES companies (id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    output_schema TEXT, -- JSON string
                    query_embedding_json TEXT, -- JSON string for the embedding
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER NOT NULL,
                    company_id INTEGER NOT NULL,
                    llm_output TEXT, -- JSON string
                    source_chunks TEXT, -- JSON string of source metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES queries (id),
                    FOREIGN KEY (company_id) REFERENCES companies (id)
                )
            """)
        logger.info("Database tables verified/created successfully.")

    def get_or_create_company(self, company_name: str, description: str | None = None) -> int:
        """Gets the ID of a company, creating it if it doesn't exist."""
        with self.conn:
            cursor = self.conn.execute("SELECT id FROM companies WHERE name = ?", (company_name,))
            row = cursor.fetchone()
            if row:
                return row['id']
            else:
                cursor = self.conn.execute(
                    "INSERT INTO companies (name, description) VALUES (?, ?)", 
                    (company_name, description)
                )
                return cursor.lastrowid

    def get_all_companies(self) -> list[dict]:
        """Retrieves all companies from the database."""
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM companies")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_company_by_id(self, company_id: int) -> dict | None:
        """Retrieves a company by its ID."""
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_pdf_info(self, company_id: int, file_path: str) -> int:
        """Saves information about a processed PDF."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO pdfs (company_id, file_path) VALUES (?, ?)",
                (company_id, file_path)
            )
            return cursor.lastrowid

    def update_pdf_docling_path(self, pdf_id: int, docling_path: str):
        """Updates the pdfs table with the path to the serialized docling output."""
        with self.conn:
            self.conn.execute(
                "UPDATE pdfs SET docling_json_path = ? WHERE id = ?",
                (docling_path, pdf_id)
            )

    def get_pdf_info_by_name(self, company_id: int, file_path: str) -> dict | None:
        """
        Retrieves PDF info, including its status, by company and file name.
        Returns a dictionary or None if not found.
        """
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM pdfs WHERE company_id = ? AND file_path = ?",
                (company_id, file_path)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_pdfs_for_company(self, company_id: int) -> list[dict]:
        """Retrieves all PDFs for a given company."""
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM pdfs WHERE company_id = ?", (company_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_pdf_by_id(self, pdf_id: int) -> dict | None:
        """Retrieves a PDF by its ID."""
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM pdfs WHERE id = ?", (pdf_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_pdf_status(self, pdf_id: int, status: str):
        """Updates the ingestion status of a PDF (e.g., 'SUCCESS', 'FAILED')."""
        with self.conn:
            self.conn.execute(
                "UPDATE pdfs SET status = ? WHERE id = ?",
                (status, pdf_id)
            )

    def delete_pdf_info(self, pdf_id: int):
        """Deletes a PDF record from the database."""
        with self.conn:
            self.conn.execute("DELETE FROM pdfs WHERE id = ?", (pdf_id,))

    def save_query_and_schema(self, query_text: str, schema: dict, query_embedding: list[float]) -> int:
        """Saves a user query, its schema, and its embedding."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO queries (query_text, output_schema, query_embedding_json) VALUES (?, ?, ?)",
                (query_text, json.dumps(schema), json.dumps(query_embedding))
            )
            return cursor.lastrowid

    def get_all_queries(self) -> list[dict]:
        """Retrieves all queries from the database."""
        with self.conn:
            cursor = self.conn.execute("SELECT * FROM queries")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_query_info(self, query_id: int) -> dict | None:
        """Retrieves query information by its ID."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT query_text, output_schema, query_embedding_json FROM queries WHERE id = ?",
                (query_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'query_text': row['query_text'],
                    'schema_json': json.loads(row['output_schema']),
                    'query_embedding': json.loads(row['query_embedding_json'])
                }
            return None

    def save_llm_result(self, query_id: int, company_id: int, llm_output: dict, source_chunks: list):
        """Saves the final LLM output and the source chunks used."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO results (query_id, company_id, llm_output, source_chunks) VALUES (?, ?, ?, ?)",
                (query_id, company_id, json.dumps(llm_output), json.dumps(source_chunks))
            )

    def reset_database(self):
        """Drops all tables and recreates them."""
        with self.conn:
            # Drop tables in reverse order of creation due to foreign keys
            self.conn.execute("DROP TABLE IF EXISTS results")
            self.conn.execute("DROP TABLE IF EXISTS queries")
            self.conn.execute("DROP TABLE IF EXISTS pdfs")
            self.conn.execute("DROP TABLE IF EXISTS companies")
        logger.info("All tables have been dropped.")
        # Recreate tables
        self._setup_tables()

    def delete_company(self, company_id: int):
        """Deletes a company from the database."""
        with self.conn:
            self.conn.execute("DELETE FROM companies WHERE id = ?", (company_id,))

    def delete_query(self, query_id: int):
        """Deletes a query from the database."""
        with self.conn:
            self.conn.execute("DELETE FROM queries WHERE id = ?", (query_id,))

    def delete_results_by_company_id(self, company_id: int):
        """Deletes all results for a specific company."""
        with self.conn:
            self.conn.execute("DELETE FROM results WHERE company_id = ?", (company_id,))

    def delete_results_by_query_id(self, query_id: int):
        """Deletes all results for a specific query."""
        with self.conn:
            self.conn.execute("DELETE FROM results WHERE query_id = ?", (query_id,))
