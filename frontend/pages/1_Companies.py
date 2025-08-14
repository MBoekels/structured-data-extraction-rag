# pages/1_Companies.py
import streamlit as st
import requests
import sqlite3

from settings import API_BASE, DB_PATH

st.title("🏢 Manage Companies")

def fetch_companies():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description FROM companies")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def fetch_pdfs_for_company(company_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path, status, created_at FROM pdfs WHERE company_id = ?", (company_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def create_company(name, description):
    resp = requests.post(f"{API_BASE}/company/", data={"company_name": name, "description": description})
    resp.raise_for_status()
    return resp.json()

def delete_company(company_id):
    resp = requests.delete(f"{API_BASE}/company/{company_id}/")
    resp.raise_for_status()
    return resp.json()

def ingest_pdf(company_id, file):
    files = {'file': (file.name, file, file.type)}
    data = {'company_id': company_id}
    resp = requests.post(f"{API_BASE}/ingest/", files=files, data=data)
    resp.raise_for_status()
    return resp.json()

# --- UI ---

# Fetch and display companies
st.subheader("Existing Companies")
companies = fetch_companies()
if not companies:
    st.info("No companies found. Please add a new company.")
else:
    for company in companies:
        with st.expander(f"{company['name']}"):
            st.write(f"**Description:** {company.get('description', 'N/A')}")
            
            pdfs = fetch_pdfs_for_company(company['id'])
            if pdfs:
                st.write("**Uploaded PDFs:**")
                for pdf in pdfs:
                    st.write(f"- {pdf['file_path']} (Status: {pdf['status']})")
            else:
                st.write("No PDFs uploaded for this company yet.")

            if st.button("Delete Company", key=f"delete_{company['id']}"):
                try:
                    delete_company(company['id'])
                    st.success(f"Company '{company['name']}' deleted successfully.")
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Failed to delete company: {e}")

st.markdown("---")

# Add new company
st.subheader("Add New Company")
with st.form("create_company_form", clear_on_submit=True):
    name = st.text_input("Company Name", max_chars=100)
    description = st.text_area("Description", max_chars=300)
    submitted = st.form_submit_button("Create Company")
    if submitted:
        if not name.strip():
            st.warning("Company Name is required.")
        else:
            try:
                create_company(name.strip(), description.strip())
                st.success(f"Company '{name}' created successfully!")
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Failed to create company: {e}")

st.markdown("---")

# Upload PDF
st.subheader("Upload PDF to a Company")
if not companies:
    st.warning("Please create a company first before uploading a PDF.")
else:
    company_options = {c['name']: c['id'] for c in companies}
    selected_company_name = st.selectbox("Select Company", options=list(company_options.keys()))
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if st.button("Upload and Ingest PDF"):
        if uploaded_file is not None and selected_company_name:
            company_id = company_options[selected_company_name]
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    ingest_pdf(company_id, uploaded_file)
                    st.success(f"Successfully ingested {uploaded_file.name} for {selected_company_name}.")
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Failed to ingest PDF: {e.response.text}")
        else:
            st.warning("Please select a company and a PDF file.")