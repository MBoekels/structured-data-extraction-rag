import pandas as pd
import json
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, ConversionStatus, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from langchain_core.documents import Document as LangchainDocument
from config.settings import settings
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from backend.llm_services import get_max_tokens, get_tokenizer

logger = logging.getLogger(__name__)

def parse_document_with_docling(file_path: str) -> pd.DataFrame:
    """
    Processes a single document file using Docling to extract its content.

    Args:
        file_path (str): The path to the document file.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed DoclingDocument.
    """
    logger.info(f"Processing file with Docling: {file_path}")

    # Explicitly configure Docling to disable OCR for better performance,
    # as we assume the PDFs are text-based.
    pdf_pipeline_options = PdfPipelineOptions(do_ocr=False)
    pdf_format_option = PdfFormatOption(pipeline_options=pdf_pipeline_options)
    format_options = {
        InputFormat.PDF: pdf_format_option,
        InputFormat.IMAGE: pdf_format_option,
    }
    converter = DocumentConverter(format_options=format_options)
    results = converter.convert_all([file_path])
    
    processed_data = []
    for res in results:
        if res.status == ConversionStatus.SUCCESS:
            processed_data.append({
                "doc": res.document,
                "file_path": file_path
            })
        else:
            logger.error(f"Docling failed to process {res.input.file}: {res.status}")

    if not processed_data:
        raise RuntimeError("Docling failed to process the input file.")

    return pd.DataFrame(processed_data)

class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # configuring a different table serializer
        )

def chunk_document(docling_df: pd.DataFrame) -> pd.DataFrame:
    """
    Chunks a DoclingDocument using a HybridChunker with table serialization.

    Args:
        docling_df (pd.DataFrame): DataFrame containing the DoclingDocument.

    Returns:
        pd.DataFrame: A DataFrame with columns 'text' and 'metadata'.
    """
    logger.info("Chunking document with table-aware strategy...")
    
    tokenizer: BaseTokenizer = HuggingFaceTokenizer(
        tokenizer=get_tokenizer(),
        max_tokens=get_max_tokens()
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=MDTableSerializerProvider()
    )

    doc = docling_df.iloc[0]['doc']
    chunks = []
    for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        doc_items = chunk.meta.doc_items
        # Get a sorted, unique list of page numbers this chunk appears on.
        pages = sorted(list(set(item.prov[0].page_no for item in doc_items)))
        # Get a list of bounding boxes for all items in the chunk.
        # The bbox format from docling is [x0, y0, x1, y1].
        bboxes = [item.prov[0].bbox for item in doc_items]


        enriched_text = chunker.contextualize(chunk=chunk)
        metadata = {
            "chunk_id": i,
            "document_id": f"{doc.origin.binary_hash}",
            "pages": pages,
            "bboxes": bboxes,
        }
        chunks.append({"text": enriched_text, "metadata": metadata})
    
    logger.info(f"Document split into {len(chunks)} chunks.")
    return pd.DataFrame(chunks)

def format_context_from_evaluated_chunks(evaluated_chunks_data: list[tuple]) -> tuple[str, list]:    
    """Formats evaluated chunks data into a context string and extracts metadata."""    
    context_parts = []    
    source_metadata_list = []    
    for evaluation_json, chunk_content, chunk_metadata in evaluated_chunks_data:        
        context_parts.append(chunk_content)        
        source_metadata_list.append(chunk_metadata)    
        context = """\n\n---\n\n""".join(context_parts)
    return context, source_metadata_list

def save_docling_document(docling_df: pd.DataFrame, pdf_id: int) -> str:
    """
    Serializes and saves the DoclingDocument to a JSON file.

    Args:
        docling_df (pd.DataFrame): DataFrame containing the DoclingDocument.
        pdf_id (int): The ID of the PDF to use in the filename.

    Returns:
        str: The path to the saved JSON file.
    """
    cache_dir = Path("data/docling_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"docling_output_{pdf_id}.json"

    # We assume a single document in the dataframe.
    doc = docling_df.iloc[0]['doc']
    # DoclingDocument is a Pydantic model, so we can use model_dump for serialization.
    # Using mode='json' ensures complex types like datetime are handled correctly.
    doc_dict = doc.model_dump(mode='json')

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved serialized DoclingDocument to {output_path}")
    return str(output_path)

def load_docling_document(docling_json_path: str) -> pd.DataFrame:
    """
    Loads a serialized DoclingDocument from a JSON file.

    Args:
        docling_json_path (str): The path to the cached JSON file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded DoclingDocument.
    """
    logger.info(f"Loading cached DoclingDocument from {docling_json_path}")
    file_path = Path(docling_json_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cache file not found: {docling_json_path}")


    # Reconstruct the Pydantic model from the dictionary
    doc = DoclingDocument.load_from_json(file_path)
    
    # Wrap the loaded document in a DataFrame to match the output of the parsing function.
    processed_data = [{
        "doc": doc,
        "file_path": file_path
    }]
    return pd.DataFrame(processed_data)
