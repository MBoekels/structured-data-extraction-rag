import pytest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
from fastapi import UploadFile, HTTPException
from io import BytesIO
import json
import numpy as np

# Import the functions to be tested
from backend.main import _extract_json_from_llm_output, _calculate_cosine_similarity


@pytest.mark.anyio
async def test_create_company(test_client, monkeypatch):
    """Test the /company/ endpoint."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_or_create_company.return_value = 1
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.post("/company/", data={"company_name": "TestCorp"})

    assert response.status_code == 201
    json_response = response.json()
    assert json_response["message"] == "Company created successfully."
    assert json_response["company_id"] == 1
    assert json_response["company_name"] == "TestCorp"
    mock_db_manager.get_or_create_company.assert_called_with("TestCorp")


@pytest.mark.anyio
async def test_create_company_error(test_client, monkeypatch):
    """Test the /company/ endpoint error handling."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_or_create_company.side_effect = Exception("DB Error")
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.post("/company/", data={"company_name": "TestCorp"})

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Error"}


@pytest.mark.anyio
async def test_ingest_pipeline(test_client, monkeypatch):
    """Test the full /ingest/ pipeline."""
    # Mock external services and file system operations
    mock_db_manager = MagicMock()
    mock_db_manager.save_pdf_info.return_value = 101
    mock_db_manager.get_pdf_info_by_name.return_value = None # No existing PDF

    mock_vector_store_manager = MagicMock()
    mock_vector_store_manager.ingest_chunks = MagicMock()

    mock_doc_model = MagicMock()
    mock_doc_model.model_dump.return_value = {"key": "value"}  # Mock the serialization output
    mock_parse_doc = MagicMock(return_value=pd.DataFrame([{"doc": mock_doc_model}]))
    mock_chunk_doc = MagicMock(return_value=pd.DataFrame([{"text": "chunk1", "metadata": {}}]))
    mock_save_docling = MagicMock(return_value="path/to/docling.json")

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.vector_store_manager", mock_vector_store_manager)
    monkeypatch.setattr("backend.main.parse_document_with_docling", mock_parse_doc)
    monkeypatch.setattr("backend.main.chunk_document", mock_chunk_doc)
    monkeypatch.setattr("backend.main.save_docling_document", mock_save_docling)

    # Mock file upload
    file_content = b"dummy pdf content"
    file = BytesIO(file_content)
    
    response = await test_client.post(
        "/ingest/",
        data={"company_id": 1},
        files={"file": ("test.pdf", file, "application/pdf")},
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Document ingested successfully."
    assert json_response["company_id"] == 1
    assert json_response["pdf_id"] == 101
    assert json_response["num_chunks"] == 1

    mock_db_manager.save_pdf_info.assert_called_once_with(1, "test.pdf")
    mock_parse_doc.assert_called_once()
    mock_chunk_doc.assert_called_once()
    mock_vector_store_manager.ingest_chunks.assert_called_once()
    mock_db_manager.update_pdf_status.assert_called_once_with(101, 'SUCCESS')


@pytest.mark.anyio
async def test_ingest_pipeline_existing_pdf_success(test_client, monkeypatch):
    """Test ingest pipeline when PDF already exists and docling can be loaded."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_pdf_info_by_name.return_value = {'id': 101, 'docling_json_path': 'path/to/existing_docling.json'}
    mock_db_manager.save_pdf_info.return_value = 101 # Still returns pdf_id even if existing

    mock_vector_store_manager = MagicMock()
    mock_vector_store_manager.ingest_chunks = MagicMock()

    mock_doc_model = MagicMock()
    mock_doc_model.model_dump.return_value = {"key": "value"}
    mock_load_docling = MagicMock(return_value=pd.DataFrame([{"doc": mock_doc_model}]))
    mock_chunk_doc = MagicMock(return_value=pd.DataFrame([{"text": "chunk1", "metadata": {}}]))

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.vector_store_manager", mock_vector_store_manager)
    monkeypatch.setattr("backend.main.load_docling_document", mock_load_docling)
    monkeypatch.setattr("backend.main.chunk_document", mock_chunk_doc)

    file_content = b"dummy pdf content"
    file = BytesIO(file_content)
    
    response = await test_client.post(
        "/ingest/",
        data={"company_id": 1},
        files={"file": ("test.pdf", file, "application/pdf")},
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Document ingested successfully."
    assert json_response["company_id"] == 1
    assert json_response["pdf_id"] == 101
    assert json_response["num_chunks"] == 1

    mock_db_manager.get_pdf_info_by_name.assert_called_once_with(1, "test.pdf")
    mock_load_docling.assert_called_once_with('path/to/existing_docling.json')
    mock_db_manager.save_pdf_info.assert_not_called() # Should not be called if existing
    mock_db_manager.update_pdf_docling_path.assert_not_called() # Should not be called if existing
    mock_db_manager.delete_pdf_info.assert_not_called() # Should not be called if existing
    mock_vector_store_manager.delete_chunks_by_pdf_id.assert_not_called() # Should not be called if existing
    mock_chunk_doc.assert_called_once()
    mock_vector_store_manager.ingest_chunks.assert_called_once()
    mock_db_manager.update_pdf_status.assert_called_once_with(101, 'SUCCESS')


@pytest.mark.anyio
async def test_ingest_pipeline_existing_pdf_reingest(test_client, monkeypatch):
    """Test ingest pipeline when PDF exists but docling cannot be loaded, triggering re-ingestion."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_pdf_info_by_name.return_value = {'id': 101, 'docling_json_path': 'path/to/non_existent_docling.json'}
    mock_db_manager.save_pdf_info.return_value = 102 # New PDF ID for re-ingestion

    mock_vector_store_manager = MagicMock()
    mock_vector_store_manager.ingest_chunks = MagicMock()
    mock_vector_store_manager.delete_chunks_by_pdf_id = MagicMock()

    mock_doc_model = MagicMock()
    mock_doc_model.model_dump.return_value = {"key": "value"}
    mock_load_docling = MagicMock(side_effect=Exception("File not found")) # Simulate docling load failure
    mock_parse_doc = MagicMock(return_value=pd.DataFrame([{"doc": mock_doc_model}]))
    mock_chunk_doc = MagicMock(return_value=pd.DataFrame([{"text": "chunk1", "metadata": {}}]))
    mock_save_docling = MagicMock(return_value="path/to/new_docling.json")

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.vector_store_manager", mock_vector_store_manager)
    monkeypatch.setattr("backend.main.load_docling_document", mock_load_docling)
    monkeypatch.setattr("backend.main.parse_document_with_docling", mock_parse_doc)
    monkeypatch.setattr("backend.main.chunk_document", mock_chunk_doc)
    monkeypatch.setattr("backend.main.save_docling_document", mock_save_docling)

    file_content = b"dummy pdf content"
    file = BytesIO(file_content)
    
    response = await test_client.post(
        "/ingest/",
        data={"company_id": 1},
        files={"file": ("test.pdf", file, "application/pdf")},
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Document ingested successfully."
    assert json_response["company_id"] == 1
    assert json_response["pdf_id"] == 102 # New PDF ID
    assert json_response["num_chunks"] == 1

    mock_db_manager.get_pdf_info_by_name.assert_called_once_with(1, "test.pdf")
    mock_load_docling.assert_called_once_with('path/to/non_existent_docling.json')
    mock_db_manager.delete_pdf_info.assert_called_once_with(101) # Old PDF info deleted
    mock_vector_store_manager.delete_chunks_by_pdf_id.assert_called_once_with(1, 101) # Old chunks deleted
    mock_parse_doc.assert_called_once() # Re-parsing
    mock_db_manager.save_pdf_info.assert_called_once_with(1, "test.pdf") # New PDF info saved
    mock_db_manager.update_pdf_docling_path.assert_called_once_with(102, "path/to/new_docling.json")
    mock_chunk_doc.assert_called_once()
    mock_vector_store_manager.ingest_chunks.assert_called_once()
    mock_db_manager.update_pdf_status.assert_called_once_with(102, 'SUCCESS')


@pytest.mark.anyio
async def test_ingest_pipeline_parse_error(test_client, monkeypatch):
    """Test ingest pipeline error during document parsing."""
    mock_db_manager = MagicMock()
    mock_db_manager.save_pdf_info.return_value = 101
    mock_db_manager.get_pdf_info_by_name.return_value = None
    mock_db_manager.update_pdf_status = MagicMock()

    mock_parse_doc = MagicMock(side_effect=Exception("Parsing Error"))

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.parse_document_with_docling", mock_parse_doc)

    file_content = b"dummy pdf content"
    file = BytesIO(file_content)
    
    response = await test_client.post(
        "/ingest/",
        data={"company_id": 1},
        files={"file": ("test.pdf", file, "application/pdf")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Parsing Error"}
    mock_db_manager.save_pdf_info.assert_called_once()
    mock_db_manager.update_pdf_status.assert_called_once_with(101, 'FAILED')


@pytest.mark.anyio
async def test_ingest_pipeline_chunk_error(test_client, monkeypatch):
    """Test ingest pipeline error during chunking."""
    mock_db_manager = MagicMock()
    mock_db_manager.save_pdf_info.return_value = 101
    mock_db_manager.get_pdf_info_by_name.return_value = None
    mock_db_manager.update_pdf_status = MagicMock()

    mock_doc_model = MagicMock()
    mock_doc_model.model_dump.return_value = {"key": "value"}
    mock_parse_doc = MagicMock(return_value=pd.DataFrame([{"doc": mock_doc_model}]))
    mock_chunk_doc = MagicMock(side_effect=Exception("Chunking Error"))

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.parse_document_with_docling", mock_parse_doc)
    monkeypatch.setattr("backend.main.chunk_document", mock_chunk_doc)

    file_content = b"dummy pdf content"
    file = BytesIO(file_content)
    
    response = await test_client.post(
        "/ingest/",
        data={"company_id": 1},
        files={"file": ("test.pdf", file, "application/pdf")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Chunking Error"}
    mock_db_manager.save_pdf_info.assert_called_once()
    mock_db_manager.update_pdf_status.assert_called_once_with(101, 'FAILED')


@pytest.mark.anyio
async def test_ingest_pipeline_ingest_error(test_client, monkeypatch):
    """Test ingest pipeline error during vector store ingestion."""
    mock_db_manager = MagicMock()
    mock_db_manager.save_pdf_info.return_value = 101
    mock_db_manager.get_pdf_info_by_name.return_value = None
    mock_db_manager.update_pdf_status = MagicMock()

    mock_vector_store_manager = MagicMock()
    mock_vector_store_manager.ingest_chunks.side_effect = Exception("Ingestion Error")

    mock_doc_model = MagicMock()
    mock_doc_model.model_dump.return_value = {"key": "value"}
    mock_parse_doc = MagicMock(return_value=pd.DataFrame([{"doc": mock_doc_model}]))
    mock_chunk_doc = MagicMock(return_value=pd.DataFrame([{"text": "chunk1", "metadata": {}}]))

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.vector_store_manager", mock_vector_store_manager)
    monkeypatch.setattr("backend.main.parse_document_with_docling", mock_parse_doc)
    monkeypatch.setattr("backend.main.chunk_document", mock_chunk_doc)

    file_content = b"dummy pdf content"
    file = BytesIO(file_content)
    
    response = await test_client.post(
        "/ingest/",
        data={"company_id": 1},
        files={"file": ("test.pdf", file, "application/pdf")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Ingestion Error"}
    mock_db_manager.save_pdf_info.assert_called_once()
    mock_db_manager.update_pdf_status.assert_called_once_with(101, 'FAILED')


@pytest.mark.anyio
async def test_initiate_query_pipeline(test_client, monkeypatch):
    """Test the /query/initiate pipeline."""
    mock_db_manager = MagicMock()
    mock_db_manager.save_query_and_schema.return_value = 123  # Mock query_id

    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_query.return_value = [0.1] * 384

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.embedding_model", mock_embedding_model)

    mock_schema_chain = MagicMock()
    mock_schema_chain.invoke.return_value = '{"revenue": "float", "year": "integer"}'

    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__.return_value = mock_schema_chain

    with patch("backend.main.PromptTemplate.from_template", return_value=mock_prompt_instance) as mock_from_template,\
         patch("backend.main.get_llm") as mock_get_llm:
        response = await test_client.post(
            "/query/",
            data={"query_text": "What is the revenue?"},
        )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["query_id"] == 123
    assert json_response["message"] == "Query initiated successfully. Ready for RAG analysis."

    mock_db_manager.save_query_and_schema.assert_called_once_with(
        "What is the revenue?", {"revenue": "float", "year": "integer"}, [0.1] * 384
    )
    mock_embedding_model.embed_query.assert_called_once_with("What is the revenue?")


@pytest.mark.anyio
async def test_initiate_query_pipeline_llm_json_error(test_client, monkeypatch):
    """Test initiate query pipeline when LLM fails to generate valid JSON."""
    mock_db_manager = MagicMock()
    mock_db_manager.save_query_and_schema.return_value = 123

    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_query.return_value = [0.1] * 384

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.embedding_model", mock_embedding_model)

    mock_schema_chain = MagicMock()
    mock_schema_chain.invoke.return_value = 'invalid json' # LLM returns invalid JSON

    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__.return_value = mock_schema_chain

    with patch("backend.main.PromptTemplate.from_template", return_value=mock_prompt_instance),\
         patch("backend.main.get_llm"):
        response = await test_client.post(
            "/query/",
            data={"query_text": "What is the revenue?"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "LLM failed to generate a valid JSON schema."}
    mock_db_manager.save_query_and_schema.assert_not_called()


@pytest.mark.anyio
async def test_initiate_query_pipeline_db_error(test_client, monkeypatch):
    """Test initiate query pipeline when database save fails."""
    mock_db_manager = MagicMock()
    mock_db_manager.save_query_and_schema.side_effect = Exception("DB Save Error")

    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_query.return_value = [0.1] * 384

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.embedding_model", mock_embedding_model)

    mock_schema_chain = MagicMock()
    mock_schema_chain.invoke.return_value = '{"revenue": "float"}'

    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__.return_value = mock_schema_chain

    with patch("backend.main.PromptTemplate.from_template", return_value=mock_prompt_instance),\
         patch("backend.main.get_llm"):
        response = await test_client.post(
            "/query/",
            data={"query_text": "What is the revenue?"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Save Error"}
    mock_db_manager.save_query_and_schema.assert_called_once()


@pytest.mark.anyio
async def test_company_rag_analysis(test_client, monkeypatch):
    """Test the /query/rag_analysis pipeline."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_query_info.return_value = {
        "query_text": "What is the revenue?",
        "schema_json": {"revenue": "float", "year": "integer"},
        "query_embedding": [0.1] * 384,
    }
    mock_db_manager.save_llm_result = MagicMock()

    mock_vector_store_manager = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "The company revenue was $1M in 2023."
    mock_doc.metadata = {"source": "report.pdf", "page": 5, "pdf_id": 101, "bboxes": []}
    mock_vector = [0.1] * 384
    mock_vector_store_manager.search.return_value = [(mock_doc, mock_vector)]

    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_query.return_value = [0.2] * 384

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.embedding_model", mock_embedding_model)
    monkeypatch.setattr("backend.main.vector_store_manager", mock_vector_store_manager)

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = '{"revenue": 1000000.0, "year": 2023}'

    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__.return_value = mock_rag_chain

    with patch("backend.main.PromptTemplate.from_template", return_value=mock_prompt_instance) as mock_from_template,\
         patch("backend.main.get_llm") as mock_get_llm:
        response = await test_client.post(
            "/query/rag_analysis",
            data={"query_id": 123, "company_id": 1},
        )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["query_id"] == 123
    assert json_response["answer"]["revenue"] == 1000000.0

    mock_db_manager.get_query_info.assert_called_once_with(123)
    mock_vector_store_manager.search.assert_called_once_with(
        "What is the revenue?", 1, k=5, query_vector=[0.1] * 384
    )
    mock_embedding_model.embed_query.assert_called_once()
    mock_db_manager.save_llm_result.assert_called_once_with(
        123, 1, {"revenue": 1000000.0, "year": 2023}, ANY
    )


@pytest.mark.anyio
async def test_company_rag_analysis_query_not_found(test_client, monkeypatch):
    """Test RAG analysis when query ID is not found."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_query_info.return_value = None # Query not found
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.post(
        "/query/rag_analysis",
        data={"query_id": 999, "company_id": 1},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Query with ID 999 not found."}
    mock_db_manager.get_query_info.assert_called_once_with(999)


@pytest.mark.anyio
async def test_company_rag_analysis_no_relevant_documents(test_client, monkeypatch):
    """Test RAG analysis when no relevant documents are found."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_query_info.return_value = {
        "query_text": "What is the revenue?",
        "schema_json": {"revenue": "float"},
        "query_embedding": [0.1] * 384,
    }
    mock_vector_store_manager = MagicMock()
    mock_vector_store_manager.search.return_value = [] # No relevant documents

    mock_embedding_model = MagicMock() # Define mock_embedding_model here
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.embedding_model", mock_embedding_model)
    monkeypatch.setattr("backend.main.vector_store_manager", mock_vector_store_manager)

    response = await test_client.post(
        "/query/rag_analysis",
        data={"query_id": 123, "company_id": 1},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "No relevant documents found for the query."}
    mock_db_manager.get_query_info.assert_called_once_with(123)
    mock_vector_store_manager.search.assert_called_once()


@pytest.mark.anyio
async def test_company_rag_analysis_llm_json_error(test_client, monkeypatch):
    """Test RAG analysis when LLM fails to generate valid JSON for answer."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_query_info.return_value = {
        "query_text": "What is the revenue?",
        "schema_json": {"revenue": "float"},
        "query_embedding": [0.1] * 384,
    }
    mock_vector_store_manager = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "The company revenue was $1M in 2023."
    mock_doc.metadata = {"source": "report.pdf", "page": 5, "pdf_id": 101, "bboxes": []}
    mock_vector_store_manager.search.return_value = [(mock_doc, [0.1] * 384)]

    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_query.return_value = [0.2] * 384

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.embedding_model", mock_embedding_model)
    monkeypatch.setattr("backend.main.vector_store_manager", mock_vector_store_manager)

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = 'invalid json response' # LLM returns invalid JSON

    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__.return_value = mock_rag_chain

    with patch("backend.main.PromptTemplate.from_template", return_value=mock_prompt_instance),\
         patch("backend.main.get_llm"):
        response = await test_client.post(
            "/query/rag_analysis",
            data={"query_id": 123, "company_id": 1},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "LLM failed to generate a valid JSON response."}
    mock_db_manager.save_llm_result.assert_not_called()


@pytest.mark.anyio
async def test_company_rag_analysis_db_save_error(test_client, monkeypatch):
    """Test RAG analysis when database save fails."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_query_info.return_value = {
        "query_text": "What is the revenue?",
        "schema_json": {"revenue": "float"},
        "query_embedding": [0.1] * 384,
    }
    mock_db_manager.save_llm_result.side_effect = Exception("DB Save Error")

    mock_vector_store_manager = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "The company revenue was $1M in 2023."
    mock_doc.metadata = {"source": "report.pdf", "page": 5, "pdf_id": 101, "bboxes": []}
    mock_vector_store_manager.search.return_value = [(mock_doc, [0.1] * 384)]

    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_query.return_value = [0.2] * 384

    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)
    monkeypatch.setattr("backend.main.embedding_model", mock_embedding_model)
    monkeypatch.setattr("backend.main.vector_store_manager", mock_vector_store_manager)

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = '{"revenue": 1000000.0}'

    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__.return_value = mock_rag_chain

    with patch("backend.main.PromptTemplate.from_template", return_value=mock_prompt_instance),\
         patch("backend.main.get_llm"):
        response = await test_client.post(
            "/query/rag_analysis",
            data={"query_id": 123, "company_id": 1},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Save Error"}
    mock_db_manager.save_llm_result.assert_called_once()


@pytest.mark.anyio
async def test_get_all_companies(test_client, monkeypatch):
    """Test the GET /company/ endpoint."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_all_companies.return_value = [{"id": 1, "name": "TestCorp"}]
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/company/")

    assert response.status_code == 200
    assert response.json() == [{"id": 1, "name": "TestCorp"}]


@pytest.mark.anyio
async def test_get_all_companies_error(test_client, monkeypatch):
    """Test GET /company/ error handling."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_all_companies.side_effect = Exception("DB Error")
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/company/")

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Error"}


@pytest.mark.anyio
async def test_get_company(test_client, monkeypatch):
    """Test the GET /company/{company_id} endpoint."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_company_by_id.return_value = {"id": 1, "name": "TestCorp"}
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/company/1")

    assert response.status_code == 200
    assert response.json() == {"id": 1, "name": "TestCorp"}


@pytest.mark.anyio
async def test_get_company_not_found(test_client, monkeypatch):
    """Test GET /company/{company_id} when company is not found."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_company_by_id.return_value = None
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/company/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


@pytest.mark.anyio
async def test_get_company_error(test_client, monkeypatch):
    """Test GET /company/{company_id} error handling."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_company_by_id.side_effect = Exception("DB Error")
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/company/1")

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Error"}


@pytest.mark.anyio
async def test_get_all_pdfs_for_company(test_client, monkeypatch):
    """Test the GET /company/{company_id}/pdfs/ endpoint."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_all_pdfs_for_company.return_value = [
        {"id": 101, "file_path": "test.pdf", "company_id": 1}
    ]
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/company/1/pdfs/")

    assert response.status_code == 200
    assert response.json() == [
        {"id": 101, "file_path": "test.pdf", "company_id": 1}
    ]


@pytest.mark.anyio
async def test_get_all_pdfs_for_company_error(test_client, monkeypatch):
    """Test GET /company/{company_id}/pdfs/ error handling."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_all_pdfs_for_company.side_effect = Exception("DB Error")
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/company/1/pdfs/")

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Error"}


@pytest.mark.anyio
async def test_get_pdf(test_client, monkeypatch):
    """Test the GET /pdfs/{pdf_id} endpoint."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_pdf_by_id.return_value = {"id": 101, "file_path": "test.pdf", "company_id": 1}
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/pdfs/101")

    assert response.status_code == 200
    assert response.json() == {"id": 101, "file_path": "test.pdf", "company_id": 1}


@pytest.mark.anyio
async def test_get_pdf_not_found(test_client, monkeypatch):
    """Test GET /pdfs/{pdf_id} when PDF is not found."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_pdf_by_id.return_value = None
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/pdfs/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "PDF not found"}


@pytest.mark.anyio
async def test_get_pdf_error(test_client, monkeypatch):
    """Test GET /pdfs/{pdf_id} error handling."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_pdf_by_id.side_effect = Exception("DB Error")
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/pdfs/101")

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Error"}


@pytest.mark.anyio
async def test_get_all_queries(test_client, monkeypatch):
    """Test the GET /query/ endpoint."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_all_queries.return_value = [
        {"id": 123, "query_text": "What is the revenue?"}
    ]
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/query/")

    assert response.status_code == 200
    assert response.json() == [
        {"id": 123, "query_text": "What is the revenue?"}
    ]


@pytest.mark.anyio
async def test_get_all_queries_error(test_client, monkeypatch):
    """Test GET /query/ error handling."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_all_queries.side_effect = Exception("DB Error")
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/query/")

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Error"}


@pytest.mark.anyio
async def test_get_query(test_client, monkeypatch):
    """Test the GET /query/{query_id} endpoint."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_query_info.return_value = {
        "query_text": "What is the revenue?",
        "schema_json": {"revenue": "float", "year": "integer"},
        "query_embedding": [0.1] * 384,
    }
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/query/123")

    assert response.status_code == 200
    assert response.json() == {
        "query_text": "What is the revenue?",
        "schema_json": {"revenue": "float", "year": "integer"},
        "query_embedding": [0.1] * 384,
    }


@pytest.mark.anyio
async def test_get_query_not_found(test_client, monkeypatch):
    """Test GET /query/{query_id} when query is not found."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_query_info.return_value = None
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/query/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Query not found"}


@pytest.mark.anyio
async def test_get_query_error(test_client, monkeypatch):
    """Test GET /query/{query_id} error handling."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_query_info.side_effect = Exception("DB Error")
    monkeypatch.setattr("backend.main.db_manager", mock_db_manager)

    response = await test_client.get("/query/123")

    assert response.status_code == 500
    assert response.json() == {"detail": "DB Error"}


def test_extract_json_from_llm_output_valid_json():
    """Test _extract_json_from_llm_output with valid JSON."""
    text = '{"key": "value"}'
    result = _extract_json_from_llm_output(text)
    assert result == {"key": "value"}

def test_extract_json_from_llm_output_json_with_markdown():
    """Test _extract_json_from_llm_output with JSON in markdown block."""
    text = """```json
        {
            "key": "value",
            "number": 123
        }
        ```"""
    result = _extract_json_from_llm_output(text)
    assert result == {"key": "value", "number": 123}

def test_extract_json_from_llm_output_invalid_json():
    """Test _extract_json_from_llm_output with invalid JSON."""
    text = "invalid json string"
    result = _extract_json_from_llm_output(text)
    assert result is None

def test_extract_json_from_llm_output_empty_string():
    """Test _extract_json_from_llm_output with an empty string."""
    text = ""
    result = _extract_json_from_llm_output(text)
    assert result is None

def test_calculate_cosine_similarity_identical_vectors():
    """Test _calculate_cosine_similarity with identical vectors."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]
    result = _calculate_cosine_similarity(vec1, vec2)
    assert np.isclose(result, 1.0)

def test_calculate_cosine_similarity_orthogonal_vectors():
    """Test _calculate_cosine_similarity with orthogonal vectors."""
    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]
    result = _calculate_cosine_similarity(vec1, vec2)
    assert np.isclose(result, 0.0)

def test_calculate_cosine_similarity_opposite_vectors():
    """Test _calculate_cosine_similarity with opposite vectors."""
    vec1 = [1.0, 1.0]
    vec2 = [-1.0, -1.0]
    result = _calculate_cosine_similarity(vec1, vec2)
    assert np.isclose(result, -1.0)

def test_calculate_cosine_similarity_zero_vector():
    """Test _calculate_cosine_similarity with a zero vector."""
    vec1 = [0.0, 0.0]
    vec2 = [1.0, 1.0]
    result = _calculate_cosine_similarity(vec1, vec2)
    assert np.isclose(result, 0.0)

def test_calculate_cosine_similarity_both_zero_vectors():
    """Test _calculate_cosine_similarity with both zero vectors."""
    vec1 = [0.0, 0.0]
    vec2 = [0.0, 0.0]
    result = _calculate_cosine_similarity(vec1, vec2)
    assert np.isclose(result, 0.0)
