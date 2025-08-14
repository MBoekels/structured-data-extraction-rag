import pytest
from unittest.mock import patch
from backend.llm_services import get_embedding_model, get_llm
from config.settings import settings


@patch("backend.llm_services.HuggingFaceEmbeddings")
def test_get_embedding_model_success(mock_embeddings, mocker):
    """Test successful creation of the embedding model."""
    mocker.patch("backend.llm_services.settings.USE_OPENAI", False)
    mocker.patch("backend.llm_services.settings.HUGGINGFACEHUB_API_TOKEN", "fake-token")
    get_embedding_model()
    mock_embeddings.assert_called_once()

@patch("backend.llm_services.HuggingFacePipeline")
def test_get_llm_success(mock_llm, mocker):
    """Test successful creation of the LLM."""
    mocker.patch("backend.llm_services.settings.USE_OPENAI", False)
    mocker.patch("backend.llm_services.settings.HUGGINGFACEHUB_API_TOKEN", "fake-token")
    get_llm()
    mock_llm.assert_called_once()
    assert "system_prompt" not in mock_llm.call_args.kwargs

@patch("backend.llm_services.OpenAIEmbeddings")
def test_get_openai_embedding_model_success(mock_openai_embeddings, mocker):
    """Test successful creation of the OpenAI embedding model."""
    mocker.patch("backend.llm_services.settings.USE_OPENAI", True)
    mocker.patch("backend.llm_services.settings.OPENAI_API_KEY", "fake-openai-key")
    get_embedding_model()
    mock_openai_embeddings.assert_called_once_with(
        openai_api_key="fake-openai-key",
        model=settings.OPENAI_EMBEDDING_MODEL_ID,
        model_kwargs={'service_tier': 'flex'}
    )

@patch("backend.llm_services.ChatOpenAI")
def test_get_openai_llm_success(mock_chat_openai, mocker):
    """Test successful creation of the OpenAI LLM."""
    mocker.patch("backend.llm_services.settings.USE_OPENAI", True)
    mocker.patch("backend.llm_services.settings.OPENAI_API_KEY", "fake-openai-key")
    get_llm()
    mock_chat_openai.assert_called_once_with(
        openai_api_key="fake-openai-key",
        model=settings.OPENAI_LLM_MODEL_ID,
        temperature=0.1,
        max_tokens=1024,
        service_tier='flex'
    )