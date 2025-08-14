import logging
import pandas as pd
import uuid
from qdrant_client import QdrantClient, models
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAI
from langchain_core.documents import Document
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages interactions with the Qdrant vector store."""

    def __init__(self, embedding_client: OpenAI):
        """
        Initializes the VectorStoreManager.

        Args:
            embedding_client: The embedding model to use for vectorization.
        """
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.embedding_client = embedding_client

    def _get_collection_name(self, company_id: int) -> str:
        """Generates a unique collection name for a company."""
        return f"company_report_chunks_{company_id}"

    def ingest_chunks(self, chunks_df: pd.DataFrame, company_id: int, pdf_id: int):
        """
        Ingests document chunks into the Qdrant vector store.

        Args:
            chunks_df (pd.DataFrame): DataFrame containing 'text' and metadata.
            company_id (int): The ID of the company the document belongs to.
            pdf_id (int): The ID of the PDF the chunks came from.
        """
        collection_name = self._get_collection_name(company_id)
        logger.info(f"Starting ingestion for collection: {collection_name}")
        
        texts = chunks_df["text"].tolist()
        metadatas = chunks_df["metadata"].tolist()
        
        # Add pdf_id to each chunk's metadata
        for metadata_dict in metadatas:
            metadata_dict['pdf_id'] = pdf_id
        
        logger.info(f"Generating embeddings for {len(texts)} chunks in batches...")
        
        vectors = []
        batch_size = 500  # Increased batch size to minimize API calls
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embedding_response = self.embedding_client.embeddings.create(input=batch_texts, model=settings.EMBEDDING_MODEL_ID)
            for embedding_obj in embedding_response.data:
                vectors.append(embedding_obj.embedding)
            logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")

        logger.info("Embeddings generated.")
        
        # Ensure the collection exists, creating it if necessary.
        try:
            self.client.get_collection(collection_name=collection_name)
        except Exception:
            logger.info(f"Collection '{collection_name}' not found. Creating new collection.")
            if not vectors:
                logger.warning("No vectors to ingest, cannot create collection.")
                return
            
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]),  # Infer dimension from the first vector
                    distance=models.Distance.COSINE,
                ),
            )
            
        logger.info(f"Upserting {len(vectors)} points to Qdrant in batches...")
        
        points = [
            models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": text, "metadata": metadata})
            for vector, text, metadata in zip(vectors, texts, metadatas)
        ]
        
        # Upsert in batches for performance and reliability with large datasets
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(points) - 1)//batch_size + 1}")
            
        logger.info(f"Successfully ingested {len(chunks_df)} chunks into '{collection_name}'.")

    def search(self, query: str, company_id: int, k: int = 5, query_vector: list[float] = None) -> list[tuple[Document, list[float]]]:
        """
        Performs a similarity search, returning documents and their vectors.

        Args:
            query (str): The search query.
            company_id (int): The ID of the company to search within.
            k (int): The number of results to return.
            query_vector (list[float], optional): The query vector to use for the search. Defaults to None.

        Returns:
            list[tuple[Document, list[float]]]: A list of tuples, where each
            tuple contains a Document object and its corresponding vector.
        """
        collection_name = self._get_collection_name(company_id)
        logger.info(f"Searching in collection '{collection_name}' for query: '{query}'")

        if query_vector is None:
            embedding_response = self.embedding_client.embeddings.create(input=query, model=settings.EMBEDDING_MODEL_ID)
            query_vector = embedding_response.data[0].embedding

        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=k,
            with_vectors=True  # Crucial for retrieving the vectors
        )

        # Reconstruct documents and pair them with their vectors, ensuring vector exists.
        results = []
        for hit in hits:
            if hit.vector:
                doc = Document(page_content=hit.payload.get("text", ""), metadata=hit.payload.get("metadata", {}))
                results.append((doc, hit.vector))
        return results

    def delete_chunks_by_pdf_id(self, company_id: int, pdf_id: int):
        """
        Deletes all chunks associated with a specific pdf_id from the collection.
        This is crucial for ensuring data consistency when re-ingesting a failed document.
        """
        collection_name = self._get_collection_name(company_id)
        logger.info(f"Deleting chunks for pdf_id {pdf_id} from collection '{collection_name}'")

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.pdf_id",
                                match=models.MatchValue(value=pdf_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logger.info(f"Successfully deleted chunks for pdf_id {pdf_id}.")
        except Exception as e:
            # It's possible the collection doesn't exist or other client-side errors.
            # We log a warning but don't raise an error to allow the ingestion to proceed.
            logger.warning(f"Could not delete chunks for pdf_id {pdf_id}. This might be expected. Error: {e}")