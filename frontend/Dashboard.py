import streamlit as st
import requests
import sqlite3

from settings import API_BASE, DB_PATH

st.title("📊 Dashboard")

def get_companies():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM companies")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_queries():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, query_text, output_schema, created_at FROM queries")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_all_pdfs():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, company_id, file_path, status, created_at FROM pdfs")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# Initialize empty data
companies = []
queries = []
pdfs = []

error_messages = []

# Fetch data with graceful fallback
try:
    companies = get_companies()
except (sqlite3.Error, Exception) as e:
    error_messages.append(f"Failed to fetch companies: {e}")

try:
    queries = get_queries()
except (sqlite3.Error, Exception) as e:
    error_messages.append(f"Failed to fetch queries: {e}")

try:
    pdfs = get_all_pdfs()
except (sqlite3.Error, Exception) as e:
    error_messages.append(f"Failed to fetch PDFs: {e}")

# Show errors if any
for err in error_messages:
    st.error(err)

num_companies = len(companies)
num_queries = len(queries)
num_pdfs = len(pdfs)
num_success_pdfs = sum(1 for pdf in pdfs if pdf.get("status") == "SUCCESS")

# Show summary stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Companies", num_companies)
col2.metric("Queries", num_queries)
col3.metric("PDFs Ingested", num_pdfs)
col4.metric("PDFs Processed", num_success_pdfs)

st.markdown("---")

# Navigation buttons
st.subheader("Quick Links")
col1, col2, col3 = st.columns(3)
if col1.button("Manage Companies"):
    st.query_params(page="companies")
if col2.button("Manage Queries"):
    st.query_params(page="queries")
if col3.button("View Results"):
    st.query_params(page="results")

st.info("Use the sidebar to navigate between pages.")
