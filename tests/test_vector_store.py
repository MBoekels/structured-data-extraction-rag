import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from qdrant_client import models
from backend.vector_store import VectorStoreManager
from langchain_core.documents import Document


@pytest.fixture
def mock_embedding_model():
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
    mock.embed_query.return_value = [0.3] * 384
    return mock


@pytest.fixture
def mock_qdrant_client():
    mock = MagicMock()
    # Mock get_collection to raise an exception initially, then return a mock collection
    mock.get_collection.side_effect = [Exception("Collection not found"), MagicMock()]
    mock.recreate_collection.return_value = None
    mock.upsert.return_value = None
    mock.delete.return_value = None
    # Mock search to return some hits
    mock_hit1 = MagicMock()
    mock_hit1.payload = {"text": "chunk1", "metadata": {"page": 1}}
    mock_hit1.vector = [0.4] * 384
    mock_hit2 = MagicMock()
    mock_hit2.payload = {"text": "chunk2", "metadata": {"page": 2}}
    mock_hit2.vector = [0.5] * 384
    mock.search.return_value = [mock_hit1, mock_hit2]
    return mock


@pytest.fixture
def vector_store_manager(mock_embedding_model, mock_qdrant_client):
    with patch("backend.vector_store.QdrantClient", return_value=mock_qdrant_client):
        manager = VectorStoreManager(embedding_model=mock_embedding_model)
        return manager


def test_ingest_chunks(vector_store_manager, mock_qdrant_client, mock_embedding_model):
    chunks_df = pd.DataFrame({
        "text": ["chunk1", "chunk2"],
        "metadata": [{"page": 1}, {"page": 2}]
    })
    company_id = 1
    pdf_id = 101

    vector_store_manager.ingest_chunks(chunks_df, company_id, pdf_id)

    collection_name = f"company_report_chunks_{company_id}"
    mock_embedding_model.embed_documents.assert_called_once_with(["chunk1", "chunk2"])
    mock_qdrant_client.recreate_collection.assert_called_once()
    mock_qdrant_client.upsert.assert_called_once()
    # Verify payload includes text and metadata with pdf_id
    upsert_calls = mock_qdrant_client.upsert.call_args[1]["points"]
    assert len(upsert_calls) == 2
    assert upsert_calls[0].payload["text"] == "chunk1"
    assert upsert_calls[0].payload["metadata"]["page"] == 1
    assert upsert_calls[0].payload["metadata"]["pdf_id"] == 101


def test_search_with_query_string(vector_store_manager, mock_qdrant_client, mock_embedding_model):
    query = "test query"
    company_id = 1
    k = 2

    results = vector_store_manager.search(query, company_id, k)

    mock_embedding_model.embed_query.assert_called_once_with(query)
    mock_qdrant_client.search.assert_called_once()
    assert len(results) == 2
    assert isinstance(results[0][0], Document)
    assert results[0][0].page_content == "chunk1"
    assert results[0][1] == [0.4] * 384


def test_search_with_precomputed_vector(vector_store_manager, mock_qdrant_client, mock_embedding_model):
    query = "test query"
    company_id = 1
    k = 2
    precomputed_vector = [0.9] * 384

    results = vector_store_manager.search(query, company_id, k, query_vector=precomputed_vector)

    mock_embedding_model.embed_query.assert_not_called()  # Should not be called if vector is provided
    mock_qdrant_client.search.assert_called_once_with(
        collection_name=f"company_report_chunks_{company_id}",
        query_vector=precomputed_vector,
        limit=k,
        with_vectors=True
    )
    assert len(results) == 2


def test_delete_chunks_by_pdf_id(vector_store_manager, mock_qdrant_client):
    company_id = 1
    pdf_id = 101

    vector_store_manager.delete_chunks_by_pdf_id(company_id, pdf_id)

    collection_name = f"company_report_chunks_{company_id}"
    mock_qdrant_client.delete.assert_called_once()
    delete_selector = mock_qdrant_client.delete.call_args[1]["points_selector"]
    assert isinstance(delete_selector, models.FilterSelector)
    assert delete_selector.filter.must[0].key == "metadata.pdf_id"
    assert delete_selector.filter.must[0].match.value == pdf_id
