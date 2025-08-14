# pages/2_Queries.py
import streamlit as st
import requests
import sqlite3
import json

from settings import API_BASE, DB_PATH

st.title("🔍 Manage Queries & Trigger Analysis")

# --- Data Fetching Functions ---
@st.cache_data(ttl=30)
def fetch_queries():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, query_text, output_schema, created_at FROM queries")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

@st.cache_data(ttl=30)
def fetch_companies():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM companies")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# --- API Functions ---
def create_query(query_text):
    resp = requests.post(f"{API_BASE}/query/", data={"query_text": query_text})
    resp.raise_for_status()
    return resp.json()

def delete_query(query_id):
    resp = requests.delete(f"{API_BASE}/query/{query_id}")
    resp.raise_for_status()
    return resp.json()

def run_rag_analysis(query_id, company_id):
    resp = requests.post(f"{API_BASE}/query/rag_analysis", data={"query_id": query_id, "company_id": company_id})
    resp.raise_for_status()
    return resp.json()

# --- UI ---

# Display existing queries
st.subheader("Existing Queries")
queries = fetch_queries()
if not queries:
    st.info("No queries found. Please add a new query.")
else:
    for query in queries:
        with st.expander(f"Query ID: {query['id']} - {query['query_text']}"):
            st.write("**Output Schema:**")
            st.json(json.loads(query['output_schema']))
            if st.button("Delete Query", key=f"delete_{query['id']}"):
                try:
                    delete_query(query['id'])
                    st.success(f"Query '{query['query_text']}' deleted successfully.")
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Failed to delete query: {e}")

st.markdown("---")

# Add new query
st.subheader("Add New Query")
with st.form("create_query_form", clear_on_submit=True):
    query_text = st.text_area("Query Text", help="Enter the natural language query to be answered.")
    submitted = st.form_submit_button("Create Query and Generate Schema")
    if submitted:
        if not query_text.strip():
            st.warning("Query text cannot be empty.")
        else:
            with st.spinner("Generating schema for your query..."):
                try:
                    create_query(query_text.strip())
                    st.success("Query created successfully and schema generated!")
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Failed to create query: {e.response.text}")

st.markdown("---")

# Run RAG Analysis
st.subheader("Run RAG Analysis")
companies = fetch_companies()
if not companies or not queries:
    st.warning("Please create at least one company and one query before running an analysis.")
else:
    company_options = {c['name']: c['id'] for c in companies}
    query_options = {q['query_text']: q['id'] for q in queries}

    selected_company_name = st.selectbox("Select Company for Analysis", options=list(company_options.keys()))
    selected_query_text = st.selectbox("Select Query for Analysis", options=list(query_options.keys()))

    if st.button("Run Analysis"):
        if selected_company_name and selected_query_text:
            company_id = company_options[selected_company_name]
            query_id = query_options[selected_query_text]
            with st.spinner(f"Running analysis for '{selected_company_name}' with query '{selected_query_text}'..."):
                try:
                    result = run_rag_analysis(query_id, company_id)
                    st.success("Analysis complete! View the results on the Results page.")
                    st.json(result)
                except requests.RequestException as e:
                    st.error(f"Analysis failed: {e.response.text}")
        else:
            st.warning("Please select both a company and a query.")
