import pytest
import json


def test_get_or_create_company(test_db_manager):
    """Test creating and retrieving a company."""
    company_name = "TestCorp"
    company_id_1 = test_db_manager.get_or_create_company(company_name)
    assert isinstance(company_id_1, int)

    # Test retrieving the same company
    company_id_2 = test_db_manager.get_or_create_company(company_name)
    assert company_id_1 == company_id_2


def test_save_pdf_info(test_db_manager):
    """Test saving PDF information."""
    company_id = test_db_manager.get_or_create_company("PDFCorp")
    pdf_id = test_db_manager.save_pdf_info(company_id, "/path/to/test.pdf")
    assert isinstance(pdf_id, int)

    # Verify it was saved
    with test_db_manager.conn:
        cursor = test_db_manager.conn.execute("SELECT * FROM pdfs WHERE id = ?", (pdf_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["company_id"] == company_id
        assert row["file_path"] == "/path/to/test.pdf"


def test_save_query_and_result(test_db_manager):
    """Test saving a query and its corresponding result."""
    company_id = test_db_manager.get_or_create_company("QueryCorp")
    query_text = "What is the revenue?"
    schema = {"revenue": "float"}
    embedding = [0.1, 0.2, 0.3]
    query_id = test_db_manager.save_query_and_schema(query_text, schema, embedding)
    assert isinstance(query_id, int)

    llm_output = {"revenue": 100.5}
    source_chunks = [{"chunk_id": 0, "document_id": "doc1"}]
    test_db_manager.save_llm_result(query_id, company_id, llm_output, source_chunks)

    with test_db_manager.conn:
        cursor = test_db_manager.conn.execute("SELECT * FROM results WHERE query_id = ?", (query_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["company_id"] == company_id
        assert json.loads(row["llm_output"]) == llm_output
        assert json.loads(row["source_chunks"]) == source_chunks