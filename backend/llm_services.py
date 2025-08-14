from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

from openai import OpenAI
from config.settings import settings
import tiktoken
from transformers import AutoTokenizer

def get_embedding_client():
    """
    Initializes and returns the embedding model based on configuration.
    """
    # Modify OpenAI's API key and API base to use vLLM's API server.
    
    if settings.USE_OPENAI:
        openai_api_key = settings.OPENAI_API_KEY
        openai_api_base = None
    else:
        # Modify OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
    return OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            max_retries=2
        )

def get_llm():
    """
    Initializes and returns the LLM based on the configuration.

    Args:
        max_new_tokens (int): The maximum number of new tokens to generate.
    """
    if settings.USE_OPENAI:
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_LLM_MODEL_ID,
            temperature=0.1,
            max_retries=2,
        )
    else:
        return ChatDeepSeek(
            api_key=settings.DEEPSEEK_API_KEY,
            model=settings.DEEPSEEK_LLM_MODEL_ID,
            temperature=0.1,
            max_retries=2
        )

def get_tokenizer():
    """
    Initializes and returns the appropriate tokenizer based on configuration.
    """
    if settings.USE_OPENAI:
        # OpenAI models use tiktoken
        return tiktoken.encoding_for_model(settings.OPENAI_LLM_MODEL_ID)
    else:
        # For Qwen3 embeddings, use its tokenizer
        return AutoTokenizer.from_pretrained(settings.VLLM_EMBEDDING_MODEL_ID)

def get_max_tokens():
    if settings.USE_OPENAI:
        # OpenAI models use tiktoken
        return 8191 # number of tokens supported by OpenAI Embedding models
    else:
        return 8191 # number of tokens supported by Qwen: 32768. Keeping this a little lower to ensure proper processing later on
